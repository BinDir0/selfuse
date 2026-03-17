import os
import re
import argparse
import subprocess
import sys
import time

def parse_datasets(yaml_path):
    datasets = []
    current_dataset = {}
    
    with open(yaml_path, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        stripped_line = line.strip()
        # Skip empty lines or lines starting with #
        if not stripped_line or stripped_line.startswith('#'):
            continue
            
        # Match lines like "  - name: ..."
        name_match = re.match(r'-\s*name:\s*(.*)', stripped_line)
        path_match = re.match(r'path:\s*(.*)', stripped_line)
        mapping_match = re.match(r'mapping:\s*(.*)', stripped_line)
        weight_match = re.match(r'weight:\s*(.*)', stripped_line)
        
        if name_match:
            if current_dataset:
                datasets.append(current_dataset)
            current_dataset = {'name': name_match.group(1).strip()}
        elif path_match and current_dataset:
            current_dataset['path'] = path_match.group(1).strip()
        elif mapping_match and current_dataset:
            current_dataset['mapping'] = mapping_match.group(1).strip()
        elif weight_match and current_dataset:
            current_dataset['weight'] = weight_match.group(1).strip()
            
    if current_dataset:
        datasets.append(current_dataset)
        
    # No filtering by name, just use all valid datasets found
    target_datasets = datasets
    
    if not target_datasets:
        print("No valid datasets found (check if they are commented out).")
        return []
    
    return target_datasets

def get_training_command(dataset):
    name = dataset['name']
    path = dataset['path']
    mapping = dataset['mapping']
    weight = dataset.get('weight', 1)
    
    # Use standard Hydra list syntax but without quoting the whole list as a string
    # We rely on shell to pass the brackets correctly.
    # We escape the $ in mapping.
    mapping_escaped = mapping.replace('$', '\\$')
    
    # We construct the argument like: vla_dataset_paths=[{name:'foo',...}]
    # But we need to be careful about spaces. There are no spaces in our values so it should be fine.
    # We wrap the whole argument in single quotes to protect it from shell expansion
    
    dataset_override = f"vla_dataset_paths='[{{name:\"{name}\",path:\"{path}\",mapping:\"{mapping_escaped}\",weight:{weight}}}]'"
    
    exp_name_override = f"name=legendvla_{name.replace(' ', '_')}"
    
    # Environment variables from pretrain_legendvla_deepspeed.sh
    env_vars = [
        "export NCCL_DEBUG=INFO",
        "export NCCL_TIMEOUT=3600000",
        "export NCCL_ASYNC_ERROR_HANDLING=1",
        "export NCCL_SOCKET_IFNAME=eth0",
        "export NCCL_IB_GID_INDEX=3",
        "export NCCL_IB_DISABLE=0",
        "export NCCL_IB_HCA=mlx5_bond_0,mlx5_bond_1,mlx5_bond_2,mlx5_bond_3,mlx5_bond_4,mlx5_bond_5,mlx5_bond_6,mlx5_bond_7",
        "export NCCL_NET_GDR_LEVEL=2",
        "export NCCL_IB_QPS_PER_CONNECTION=4",
        "export NCCL_IB_TC=160",
        "export NCCL_IB_TIMEOUT=600",
        "export NCCL_PXN_DISABLE=0",
        "export NCCL_MIN_CTAS=4",
        "export PYTORCH_CUDA_ALLOC_CONF=\"expandable_segments:True\"",
        "export TORCHINDUCTOR_FX_GRAPH_CACHE=1",
        "export TORCHINDUCTOR_FORCE_CUDA_CODE_CACHE=1",
        "export TOKENIZERS_PARALLELISM=false",
        "export PYTHONUNBUFFERED=1"
    ]
    
    env_prefix = "; ".join(env_vars)
    
    # Use the current python interpreter path to ensure we use the same environment
    python_exec = sys.executable
    
    cmd = [
        "accelerate", "launch",
        "--config_file", "src/config/acc_config.yaml",
        "train.py",
        "experiment=legendvla_qwen3_vl",
        dataset_override,
        exp_name_override
    ]
    
    # Combine env vars and command
    return f"{env_prefix}; {' '.join(cmd)}"

def main():
    parser = argparse.ArgumentParser(description="Auto finetune scheduler")
    parser.add_argument("--hosts", type=str, default=None, help="Comma separated list of host IPs (e.g. '192.168.1.11,192.168.1.12'). If not provided, runs locally.")
    parser.add_argument("--dry_run", action="store_true", help="Print commands without executing")
    parser.add_argument("--log_dir", type=str, default="logs/auto_finetune", help="Directory to store remote logs")
    args = parser.parse_args()

    yaml_path = "src/config/experiment/vla_dataset_paths.yaml"
    datasets = parse_datasets(yaml_path)
    
    if not datasets:
        return

    print(f"Found {len(datasets)} active datasets.")

    # Determine nodes configuration
    if args.hosts:
        hosts = [h.strip() for h in args.hosts.split(',')]
        num_nodes = len(hosts)
        print(f"Running in LAUNCHER mode for {num_nodes} hosts: {hosts}")
    else:
        hosts = None
        num_nodes = 1
        node_rank = 0
        print(f"Running in LOCAL mode (all datasets)")

    # Create log dir if not exists
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir, exist_ok=True)

    # Distribute datasets
    chunk_size = len(datasets) // num_nodes
    remainder = len(datasets) % num_nodes

    # If in launcher mode, loop through all hosts
    if hosts:
        cwd = os.getcwd()
        
        # First, clean up all hosts
        print("\n>>> Cleaning up existing processes on all hosts...")
        for host in hosts:
            print(f"Cleaning {host}...")
            try:
                # Kill python processes related to training
                # We use || true to ignore errors if no process is found
                cleanup_cmd = f"ssh {host} \"pkill -9 -f 'train.py' || true; pkill -9 -f 'accelerate' || true\""
                if not args.dry_run:
                    subprocess.run(cleanup_cmd, shell=True, check=False)
            except Exception as e:
                print(f"Warning: Failed to cleanup {host}: {e}")
        
        # Wait a bit for cleanup to finish and ports to be released
        if not args.dry_run:
            print("Waiting 5 seconds for cleanup to settle...")
            time.sleep(5)
            
        for rank, host in enumerate(hosts):
            start_idx = rank * chunk_size + min(rank, remainder)
            end_idx = start_idx + chunk_size + (1 if rank < remainder else 0)
            my_datasets = datasets[start_idx:end_idx]
            
            if not my_datasets:
                print(f"Host {host} (Rank {rank}) has no datasets assigned.")
                continue

            print(f"\n>>> Preparing commands for Host: {host} (Datasets: {len(my_datasets)})")
            
            # Chain commands with '&&' so they run sequentially
            cmd_chain = []
            for d in my_datasets:
                cmd_chain.append(get_training_command(d))
            
            full_cmd_str = " && ".join(cmd_chain)
            
            # Wrap in nohup and background
            # We assume the remote machine has the same path structure
            log_file = os.path.join(args.log_dir, f"node_{rank}_{host}.log")
            
            # Construct SSH command using tmux
            safe_cmd_str = full_cmd_str.replace("'", "'\\''")
            safe_cmd_str = safe_cmd_str.replace('"', '\\"')
            
            remote_cmd = (
                f"ssh {host} \""
                f"tmux has-session -t finetune 2>/dev/null || tmux new-session -d -s finetune; "
                f"tmux send-keys -t finetune 'conda activate legendvla' C-m; "
                f"tmux send-keys -t finetune 'cd {cwd}' C-m; "
                f"tmux send-keys -t finetune '{safe_cmd_str} > {log_file} 2>&1' C-m"
                f"\""
            )
            
            print(f"Launching on {host}...")
            if args.dry_run:
                print(f"[DRY RUN] {remote_cmd}")
            else:
                try:
                    subprocess.run(remote_cmd, shell=True, check=True)
                    print(f"Successfully launched on {host}. Logs: {log_file}")
                except subprocess.CalledProcessError as e:
                    print(f"Failed to launch on {host}: {e}")

    # If in local mode (legacy/single execution)
    else:
        start_idx = node_rank * chunk_size + min(node_rank, remainder)
        end_idx = start_idx + chunk_size + (1 if node_rank < remainder else 0)
        my_datasets = datasets[start_idx:end_idx]
        
        print(f"Node {node_rank} processing {len(my_datasets)} datasets.")
        
        for d in my_datasets:
            cmd = get_training_command(d)
            print(f"\n>>> Running: {d['name']}")
            print(f"Command: {cmd}")
            
            if not args.dry_run:
                try:
                    subprocess.run(cmd, shell=True, check=True)
                except subprocess.CalledProcessError as e:
                    print(f"Error running dataset {d['name']}: {e}")

if __name__ == "__main__":
    main()