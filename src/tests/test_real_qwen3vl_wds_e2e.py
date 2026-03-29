import io
import json
import os
import tarfile
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from src.dataset.qwen3_vl_batching import Qwen3VLBatchProcessor, Qwen3VLChatFormatter
from src.dataset.unified_vla_collator import UnifiedVLACollator
from src.dataset.vla_dataset import UnifiedWdsDataset, VLAWdsDataset
from src.dataset.vlm_dataset import VLMWdsDataset
from src.model.action.action_head import FourierActionEncoder, MLPProjector
from src.model.common.diffloss import DiffLoss
from src.model.common.modules import TimeEmbedding
from src.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from src.model.vlm.qwen3_vl_backbone import Qwen3VLBackboneWrapper
from src.policy.legendvla import LegendVLA, FlowConfig, RTCConfig, LossConfig, ARActionTrainConfig
from src.tests.dummy_flow_expert import DummyFlowExpert


MODEL_NAME = os.environ.get("LEGENDVLA_REAL_MODEL", "Qwen/Qwen3-VL-4B-Instruct")
RUN_REAL_MODEL = os.environ.get("RUN_REAL_QWEN3VL_E2E") == "1"


pytestmark = pytest.mark.skipif(
    not RUN_REAL_MODEL,
    reason="Set RUN_REAL_QWEN3VL_E2E=1 to run the real Qwen3-VL integration test.",
)


def add_bytes_to_tar(tar_obj: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(payload)
    tar_obj.addfile(info, io.BytesIO(payload))


def encode_npy(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, array.astype(np.float32))
    return buffer.getvalue()


def encode_jpeg(image: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="JPEG")
    return buffer.getvalue()


def make_lowdim_vector(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    wrist_state = rng.normal(size=18).astype(np.float32)
    hand_state = rng.normal(size=30).astype(np.float32)
    wrist_action = rng.normal(size=18).astype(np.float32)
    hand_action = rng.normal(size=30).astype(np.float32)
    extrinsic = np.eye(4, dtype=np.float32).reshape(-1)
    intrinsic = np.array([320.0, 320.0, 32.0, 32.0], dtype=np.float32)
    return np.concatenate(
        [wrist_state, hand_state, wrist_action, hand_action, extrinsic, intrinsic],
        axis=0,
    ).astype(np.float32)


def make_rgb_image(seed: int, size: int = 64) -> np.ndarray:
    rng = np.random.default_rng(seed)
    image = rng.integers(low=0, high=255, size=(size, size, 3), dtype=np.uint8)
    image[:, :, 0] = np.linspace(0, 255, size, dtype=np.uint8)[:, None]
    return image


def write_vla_shard(shard_path: Path) -> None:
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(shard_path, "w") as tar:
        key = "mini_vla_000000"
        meta = {
            "dataset_name": "mini_vla",
            "episode_index": 0,
            "instruction": ["pick up the red block"],
            "instruction_num": 1,
            "presence": 3,
        }
        add_bytes_to_tar(tar, f"{key}.meta.json", json.dumps(meta).encode("utf-8"))
        add_bytes_to_tar(tar, f"{key}.lowdim.npy", encode_npy(make_lowdim_vector(seed=0)))
        add_bytes_to_tar(tar, f"{key}.image.jpg", encode_jpeg(make_rgb_image(seed=1)))


def write_vlm_shard(shard_path: Path) -> None:
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(shard_path, "w") as tar:
        key = "mini_vlm_000000"
        meta = {
            "dataset_name": "mini_vlm",
            "source": "mini_vlm",
            "sample_idx": 0,
            "n_images": 1,
            "texts": [
                {
                    "user": "What is visible in the picture?",
                    "assistant": "A colorful synthetic test image.",
                }
            ],
            "formatting_ratings": [1],
            "visual_dependency_ratings": [1],
            "relevance_ratings": [1],
        }
        add_bytes_to_tar(tar, f"{key}.meta.json", json.dumps(meta).encode("utf-8"))
        add_bytes_to_tar(tar, f"{key}.image_000.jpg", encode_jpeg(make_rgb_image(seed=2)))


def make_shape_meta() -> dict:
    return {
        "obs": {
            "rgb": {"shape": [64, 64, 3], "type": "rgb", "horizon": 1, "stride": 1},
            "depth": {"shape": [64, 64], "type": "depth", "horizon": 0, "stride": 1},
            "state": {
                "wrist": {"shape": [18]},
                "hand": {"shape": [30]},
                "shape": [48],
                "type": "fingertips",
                "horizon": 1,
                "stride": 1,
            },
        },
        "action": {"shape": [48], "type": "fingertips", "horizon": 1, "stride": 1},
    }


def make_identity_normalizer() -> LinearNormalizer:
    normalizer = LinearNormalizer()
    normalizer["motions"] = SingleFieldLinearNormalizer.create_identity(dtype=torch.float32)
    for parameter in normalizer.parameters():
        parameter.requires_grad = False
    return normalizer


def build_real_model(device: torch.device, dtype: torch.dtype) -> LegendVLA:
    backbone = Qwen3VLBackboneWrapper(
        model_name_or_path=MODEL_NAME,
        trust_remote_code=False,
        freeze_backbone=False,
        torch_dtype="bfloat16",
        state_token="<state>",
        action_token="<action>",
        use_lora=True,
        lora={
            "r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "bias": "none",
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        },
        use_quantization=True,
        quantization={
            "load_in_4bit": True,
            "load_in_8bit": False,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": True,
            "bnb_4bit_compute_dtype": "bfloat16",
        },
        device_map={"": 0},
        low_cpu_mem_usage=True,
    )

    action_hidden_size = 128
    time_hidden_size = 128
    diffloss_z_channels = 128

    model = LegendVLA(
        backbone=backbone,
        state_encoder=FourierActionEncoder(
            action_dim=48,
            width=backbone.hidden_size,
            time_cond=False,
            enable_fourier_embed=False,
            mlp_depth=2,
            final_layer_norm=False,
            use_mlp_layer_norm=False,
        ),
        ar_action_encoder=FourierActionEncoder(
            action_dim=48,
            width=backbone.hidden_size,
            time_cond=False,
            enable_fourier_embed=False,
            mlp_depth=2,
            final_layer_norm=False,
            use_mlp_layer_norm=False,
        ),
        action_encoder=FourierActionEncoder(
            action_dim=48,
            width=action_hidden_size,
            time_cond=False,
            enable_fourier_embed=False,
            mlp_depth=2,
            final_layer_norm=False,
            use_mlp_layer_norm=False,
        ),
        time_embedding=TimeEmbedding(time_hidden_size),
        flow_expert=DummyFlowExpert(hidden_size=action_hidden_size, time_hidden_size=time_hidden_size),
        action_decoder=MLPProjector(
            input_dim=action_hidden_size,
            output_dim=48,
            width=action_hidden_size,
            depth=2,
            final_layer_norm=False,
            use_mlp_layer_norm=False,
        ),
        latent_condition_projector=MLPProjector(
            input_dim=backbone.hidden_size,
            output_dim=diffloss_z_channels,
            width=256,
            depth=2,
            final_layer_norm=False,
            use_mlp_layer_norm=False,
        ),
        shape_meta=make_shape_meta(),
        diffloss=DiffLoss(
            target_channels=48,
            z_channels=diffloss_z_channels,
            depth=2,
            width=256,
            num_sampling_steps="10",
            grad_checkpointing=False,
            use_ddim_sampling=True,
            use_flow_matching=False,
            flow_sig_min=0.001,
            time_min_period=0.004,
            time_max_period=4.0,
            flow_sampling="uniform",
            flow_alpha=1.5,
            flow_beta=1.0,
            num_inference_steps=2,
        ),
        action_hidden_size=action_hidden_size,
        flow_config=FlowConfig(sampling="uniform", num_inference_steps=2),
        ar_action_train_config=ARActionTrainConfig(noise_std=0.0, chunk_size=1),
        rtc_config=RTCConfig(enabled=False),
        loss_config=LossConfig(),
    )

    modules_to_move = [
        model.state_encoder,
        model.ar_action_encoder,
        model.action_encoder,
        model.time_embedding,
        model.flow_expert,
        model.action_decoder,
        model.latent_condition_projector,
        model.diffloss,
    ]
    for module in modules_to_move:
        module.to(device=device, dtype=dtype)
    return model


def preprocess_batch(batch: dict[str, torch.Tensor], dtype: torch.dtype, device: torch.device) -> dict[str, torch.Tensor]:
    inputs = {
        "input_ids": batch["input_ids"].to(device=device),
        "attention_mask": batch["attention_mask"].to(device=device),
        "pixel_values": batch["pixel_values"].to(device=device, dtype=dtype)
        if batch["pixel_values"] is not None else None,
        "image_grid_thw": batch["image_grid_thw"].to(device=device)
        if batch["image_grid_thw"] is not None else None,
        "pixel_values_videos": batch["pixel_values_videos"].to(device=device, dtype=dtype)
        if batch["pixel_values_videos"] is not None else None,
        "video_grid_thw": batch["video_grid_thw"].to(device=device)
        if batch["video_grid_thw"] is not None else None,
        "mm_token_type_ids": batch["mm_token_type_ids"].to(device=device),
        "states": batch["states"].to(device=device, dtype=dtype),
        "answer_start_idx": batch["answer_start_idx"].to(device=device),
        "is_vla_data": batch["is_vla_data"].to(device=device),
        "n_states": batch["n_states"].to(device=device),
        "n_actions": batch["n_actions"].to(device=device),
        "actions": batch["actions"].to(device=device, dtype=dtype),
        "actions_valid_mask": batch["actions_valid_mask"].to(device=device),
        "labels": batch["labels"].to(device=device),
        "t": torch.full((batch["input_ids"].shape[0],), 0.5, device=device, dtype=dtype),
    }
    return inputs


def build_real_dataloader(root: Path) -> DataLoader:
    shape_meta = make_shape_meta()
    vla_shard = root / "vla" / "shard-000000.tar"
    vlm_shard = root / "vlm" / "shard-000000.tar"
    write_vla_shard(vla_shard)
    write_vlm_shard(vlm_shard)

    vla_dataset = VLAWdsDataset(
        wds_datasets=[{"name": "mini_vla", "shard_urls": str(vla_shard)}],
        shape_meta=shape_meta,
        objective=None,
        use_relative_action=False,
        mode="val",
        shuffle_buffer=1,
        history_pad_mode="repeat",
        future_pad_mode="repeat",
    )
    vlm_dataset = VLMWdsDataset(
        wds_datasets=[{"name": "mini_vlm", "shard_urls": str(vlm_shard), "weight": 1.0}],
        mode="val",
        shuffle_buffer=1,
    )

    data_collator = UnifiedVLACollator(
        formatter=Qwen3VLChatFormatter(),
        batch_processor=Qwen3VLBatchProcessor(
            model_name_or_path=MODEL_NAME,
            processor_init_kwargs={
                "trust_remote_code": False,
                "size": {"shortest_edge": 50176, "longest_edge": 50176},
            },
            processor_call_kwargs={
                "padding": "longest",
                "return_tensors": "pt",
            },
            ignore_index=-100,
        ),
    )
    vla_dataset.set_collator(data_collator)
    vlm_dataset.set_collator(data_collator)
    vla_dataset.set_normalizer(make_identity_normalizer())

    unified_dataset = UnifiedWdsDataset(
        vla_dataset=vla_dataset,
        vlm_dataset=vlm_dataset,
        vla_ratio=0.5,
        batch_size=2,
        mode="val",
    )

    return DataLoader(
        dataset=unified_dataset,
        batch_size=2,
        num_workers=0,
        collate_fn=unified_dataset.get_collator(),
    )


@pytest.mark.cuda
@pytest.mark.real_model
def test_real_qwen3vl_wds_forward_backward(tmp_path: Path):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the real Qwen3-VL integration test.")

    torch.manual_seed(0)
    np.random.seed(0)
    torch.cuda.empty_cache()

    loader = build_real_dataloader(tmp_path)
    batch = next(iter(loader))

    assert batch["input_ids"].shape[0] == 2
    assert int(batch["is_vla_data"].sum().item()) == 1
    assert int((~batch["is_vla_data"].bool()).sum().item()) == 1

    device = torch.device("cuda")
    dtype = torch.float32
    model = build_real_model(device=device, dtype=dtype)
    model.train()

    inputs = preprocess_batch(batch, dtype=dtype, device=device)
    output = model("train", inputs)

    assert output["total_loss"].requires_grad
    assert torch.isfinite(output["total_loss"])
    assert output["ce_loss"].item() > 0
    assert output["diffusion_loss"].item() > 0
    assert output["flow_loss"].item() > 0

    output["total_loss"].backward()

    has_backbone_grad = any(
        param.grad is not None and torch.isfinite(param.grad).all() and param.grad.abs().sum() > 0
        for param in model.trainable_vlm_parameters
    )
    has_expert_grad = any(
        param.grad is not None and torch.isfinite(param.grad).all() and param.grad.abs().sum() > 0
        for param in model.action_expert_parameters
    )
    has_diffloss_grad = any(
        param.grad is not None and torch.isfinite(param.grad).all() and param.grad.abs().sum() > 0
        for param in model.diffloss_parameters
    )

    assert has_backbone_grad, "No gradient reached trainable real-backbone parameters"
    assert has_expert_grad, "No gradient reached action expert parameters"
    assert has_diffloss_grad, "No gradient reached diffloss parameters"
