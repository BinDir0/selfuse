# export TORCH_LOGS="+dynamo"
# Enable graph cache and CUDA code cache for torch.compile
export TORCHINDUCTOR_FX_GRAPH_CACHE=1
export TORCHINDUCTOR_FORCE_CUDA_CODE_CACHE=1
python -m src.serving.serve_policy