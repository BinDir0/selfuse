# import os
# import deepspeed

# os.environ["DEBUGPY_PROCESS_SPAWN"] = "0"

# # 1. 获取本地 Rank，避免所有卡都监听端口
# local_rank = int(os.environ.get("LOCAL_RANK", -1))

# if local_rank == 0:
#     import debugpy
#     debugpy.configure(python="python", subProcess=False) 
#     # 2. 监听端口，等待 VS Code 连接
#     # 0.0.0.0 允许从外部/容器外连接，5678 是常用端口
#     debugpy.listen(("0.0.0.0", 5678))

#     print(f"👻 Rank {local_rank}: 等待 VS Code 调试器连接 (端口 5678)...")
#     print(f"👉 请确保 VS Code 打开的是真实路径 (非软链接)！")

#     # 3. 程序在此暂停，直到调试器挂载
#     debugpy.wait_for_client()
#     print(f"✅ 调试器已连接，开始训练...")

import sys
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
    # resolve immediately so all the ${now:} resolvers
    # will use the same time.
    OmegaConf.resolve(cfg)

    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    workspace.run()

if __name__ == "__main__":
    main()