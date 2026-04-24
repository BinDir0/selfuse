import os
import sys
import resource

# Raise NOFILE soft to hard so DataLoader IPC shm-fd allocations do not
# hit the low soft limit inherited from ssh non-interactive sessions.
_nofile_soft, _nofile_hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (_nofile_hard, _nofile_hard))

# Raise MEMLOCK soft to hard so NCCL/RDMA pinned memory registration has
# room to grow (Linux default 64MB cap is too small).  Only helps when
# the container's hard limit is already raised; otherwise a no-op.
_memlock_soft, _memlock_hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
try:
    resource.setrlimit(resource.RLIMIT_MEMLOCK, (_memlock_hard, _memlock_hard))
except (ValueError, OSError):
    pass

# ================== debugpy 调试配置 ==================
# 通过环境变量 ENABLE_DEBUGPY=1 来启用调试
# 通过环境变量 DEBUGPY_PORT=5678 来指定调试端口（默认 5678）
# 通过环境变量 DEBUGPY_WAIT=1 来控制是否等待调试器连接（默认等待）
# =====================================================

# if os.environ.get("ENABLE_DEBUGPY", "0") == "1":
#     # 1. 获取本地 Rank，避免所有卡都监听端口
#     local_rank = int(os.environ.get("LOCAL_RANK", -1))
    
#     # 只在 rank 0 或单卡训练时启用调试
#     if local_rank in (-1, 0):
#         import debugpy
        
#         # 2. 从环境变量获取调试端口
#         debugpy_port = int(os.environ.get("DEBUGPY_PORT", "5678"))
#         wait_for_client = os.environ.get("DEBUGPY_WAIT", "1") == "1"
        
#         # 3. 监听端口
#         # 0.0.0.0 允许从外部/容器外连接
#         debugpy.listen(("0.0.0.0", debugpy_port))
        
#         print(f"🐛 [Rank {local_rank}] debugpy 已启动，监听端口: {debugpy_port}")
#         print(f"👉 CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
        
#         if wait_for_client:
#             print(f"⏸️  等待 VS Code 调试器连接...")
#             print(f"💡 提示: 请确保 VS Code 打开的是真实路径 (非软链接)！")
#             # 4. 程序在此暂停，直到调试器挂载
#             debugpy.wait_for_client()
#             print(f"✅ 调试器已连接，开始训练...")
#         else:
#             print(f"🚀 不等待调试器，继续执行...")
#     else:
#         print(f"⏭️  [Rank {local_rank}] 跳过 debugpy 初始化（仅 rank 0 启用调试）")

# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import hydra
from omegaconf import OmegaConf
import pathlib
from src.workspace.base_workspace import BaseWorkspace
import torch

torch.set_float32_matmul_precision('high')

# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'src','config')), 
    config_name="train_config"
)
def main(cfg: OmegaConf):
    # Allow raising torch.compile recompile limits via env (no native env
    # support in torch._dynamo.config). Must run before any torch.compile call.
    import torch._dynamo.config as _dynamo_cfg
    _dynamo_cfg.recompile_limit = int(32)
    _dynamo_cfg.accumulated_recompile_limit = int(1024)

    # resolve immediately so all the ${now:} resolvers
    # will use the same time.
    OmegaConf.resolve(cfg)

    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    workspace.run()

if __name__ == "__main__":
    main()