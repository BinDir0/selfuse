from __future__ import annotations

from src.serving.websocket_policy_server import EnvWrapper


class _EchoPolicy:
    def infer(self, obs: dict) -> dict:
        return obs


def test_single_rgb_sets_depth_none_without_chest_keys():
    wrapper = EnvWrapper(
        policy=_EchoPolicy(),
        camera_setup_mode="single",
        image_mode="rgb",
        image_key="image",
        depth_key="depth_image",
        intrinsic_key="camera_intrinsics",
    )

    obs = {
        "image": "head_rgb",
        "depth_image": "head_depth",
        "camera_intrinsics": "head_K",
        "instruction": "pick",
        "states": "states",
    }
    mapped = wrapper.infer(obs)

    assert mapped["image"] == "head_rgb"
    assert mapped["intrinsic"] == "head_K"
    assert mapped["depth"] is None
    assert "chest_image" not in mapped
    assert "chest_intrinsic" not in mapped
    assert "chest_depth" not in mapped


def test_both_rgb_maps_chest_without_depth_keys():
    wrapper = EnvWrapper(
        policy=_EchoPolicy(),
        camera_setup_mode="both",
        image_mode="rgb",
        head_camera_name="head",
        chest_camera_name="chest",
        image_key="image",
        depth_key="depth_image",
        intrinsic_key="camera_intrinsics",
    )

    obs = {
        "image": {"head": "head_rgb", "chest": "chest_rgb"},
        "camera_intrinsics": {"head": "head_K", "chest": "chest_K"},
        "instruction": "pick",
        "states": "states",
    }
    mapped = wrapper.infer(obs)

    assert mapped["image"] == "head_rgb"
    assert mapped["intrinsic"] == "head_K"
    assert mapped["chest_image"] == "chest_rgb"
    assert mapped["chest_intrinsic"] == "chest_K"
    assert mapped["depth"] is None
    assert mapped["chest_depth"] is None
    assert "chest_image" not in mapped
    assert "chest_intrinsic" not in mapped
    assert "chest_depth" not in mapped


def test_both_rgbd_maps_head_and_chest_depth():
    wrapper = EnvWrapper(
        policy=_EchoPolicy(),
        camera_setup_mode="both",
        image_mode="rgbd",
        head_camera_name="head",
        chest_camera_name="chest",
        image_key="image",
        depth_key="depth_image",
        intrinsic_key="camera_intrinsics",
    )

    obs = {
        "image": {"head": "head_rgb", "chest": "chest_rgb"},
        "depth_image": {"head": "head_depth", "chest": "chest_depth"},
        "camera_intrinsics": {"head": "head_K", "chest": "chest_K"},
        "instruction": "pick",
        "states": "states",
    }
    mapped = wrapper.infer(obs)

    assert mapped["depth"] == "head_depth"
    assert mapped["chest_depth"] == "chest_depth"
    assert "chest_depth" not in mapped
