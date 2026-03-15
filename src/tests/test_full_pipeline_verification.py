"""
Exhaustive end-to-end verification for the full data pipeline.

Covers: geometry transforms, data_transforms, normalizer, collator,
shard allocation, sliding window, and the complete WDS pipeline.
"""

import collections
import io
import json
import tarfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

# ======================================================================
# Module 1: Geometry transforms
# ======================================================================

from src.utils.geometry import (
    rot_matrix_from_6drot,
    rot_matrix_to_6drot,
    homo_matrix_from_trans_6drot,
    homo_matrix_to_trans_6drot,
    homo_matrix_from_wrist_pose,
    transform_pose_to_target_frame,
    transform_wrist_to_target_frame,
    transform_hand_points_to_target_frame,
    transform_hand_points_to_wrist_frame,
    transform_hand_points_from_wrist_to_camera_frame,
    homo_coordinates_to_cartesian,
    homo_coordinates_from_cartesian,
)


class TestRotation6D:
    """Verify 6D rotation representation (Zhou et al.)."""

    def test_gram_schmidt_produces_valid_rotation(self):
        """rot_matrix_from_6drot must produce det=+1, R^T R = I."""
        rng = np.random.default_rng(42)
        rot_6d = torch.tensor(rng.normal(size=(10, 6)), dtype=torch.float64)
        R = rot_matrix_from_6drot(rot_6d)
        assert R.shape == (10, 3, 3)

        # Orthogonality
        eye = torch.eye(3, dtype=torch.float64).unsqueeze(0).expand(10, -1, -1)
        torch.testing.assert_close(R.transpose(-1, -2) @ R, eye, atol=1e-10, rtol=0)

        # Determinant = +1
        dets = torch.linalg.det(R)
        torch.testing.assert_close(dets, torch.ones(10, dtype=torch.float64), atol=1e-10, rtol=0)

    def test_round_trip_from_valid_rotation(self):
        """For a VALID rotation matrix, R -> 6D -> R must be lossless."""
        rng = np.random.default_rng(7)
        # Build valid rotation matrices from random axis-angle
        for _ in range(20):
            axis = rng.normal(size=3)
            axis = axis / np.linalg.norm(axis)
            angle = rng.uniform(-np.pi, np.pi)
            K = np.array([[0, -axis[2], axis[1]],
                          [axis[2], 0, -axis[0]],
                          [-axis[1], axis[0], 0]])
            R_np = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
            R = torch.tensor(R_np, dtype=torch.float64)

            rot_6d = rot_matrix_to_6drot(R)
            R_recovered = rot_matrix_from_6drot(rot_6d)
            torch.testing.assert_close(R_recovered, R, atol=1e-10, rtol=0)

    def test_6d_single_vector(self):
        """Shape [6] input must work."""
        rot_6d = torch.tensor([1.0, 0, 0, 0, 1, 0], dtype=torch.float64)
        R = rot_matrix_from_6drot(rot_6d)
        assert R.shape == (3, 3)
        torch.testing.assert_close(R, torch.eye(3, dtype=torch.float64), atol=1e-10, rtol=0)

    def test_numpy_torch_consistency(self):
        """numpy and torch inputs must produce identical results."""
        rng = np.random.default_rng(99)
        arr = rng.normal(size=(5, 6)).astype(np.float64)
        R_np = rot_matrix_from_6drot(arr)
        R_pt = rot_matrix_from_6drot(torch.from_numpy(arr))
        np.testing.assert_allclose(R_np, R_pt.numpy(), atol=1e-12)


class TestHomoMatrix:
    """Verify homogeneous matrix construction and decomposition."""

    def test_round_trip(self):
        """trans,rot6d -> homo -> trans,rot6d must be lossless for valid rotations."""
        rng = np.random.default_rng(0)
        for _ in range(20):
            axis = rng.normal(size=3)
            axis = axis / np.linalg.norm(axis)
            angle = rng.uniform(-np.pi, np.pi)
            K = np.array([[0, -axis[2], axis[1]],
                          [axis[2], 0, -axis[0]],
                          [-axis[1], axis[0], 0]])
            R_np = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
            rot_6d_orig = torch.tensor(
                np.concatenate([R_np[:, 0], R_np[:, 1]]), dtype=torch.float64
            )
            trans_orig = torch.tensor(rng.normal(size=3), dtype=torch.float64)

            H = homo_matrix_from_trans_6drot(trans_orig, rot_6d_orig)
            assert H.shape == (4, 4)
            assert H[3, 3].item() == pytest.approx(1.0)
            assert all(H[3, i].item() == pytest.approx(0.0) for i in range(3))

            trans_rec, rot_6d_rec = homo_matrix_to_trans_6drot(H)
            torch.testing.assert_close(trans_rec, trans_orig, atol=1e-10, rtol=0)
            torch.testing.assert_close(rot_6d_rec, rot_6d_orig, atol=1e-10, rtol=0)

    def test_batch_shape(self):
        """Batched inputs must work correctly."""
        trans = torch.randn(3, 5, 3, dtype=torch.float64)
        rot_6d = torch.randn(3, 5, 6, dtype=torch.float64)
        H = homo_matrix_from_trans_6drot(trans, rot_6d)
        assert H.shape == (3, 5, 4, 4)
        t_rec, r_rec = homo_matrix_to_trans_6drot(H)
        assert t_rec.shape == (3, 5, 3)
        assert r_rec.shape == (3, 5, 6)


class TestWristPoseDecomposition:
    """Verify homo_matrix_from_wrist_pose preserves the documented layout."""

    def test_layout(self):
        """wrist_pose[0:3]=trans_L, [3:6]=trans_R, [6:12]=rot_L, [12:18]=rot_R."""
        rng = np.random.default_rng(1)
        trans_L = rng.normal(size=3).astype(np.float64)
        trans_R = rng.normal(size=3).astype(np.float64)
        # Use identity rotation 6D: first two columns of I
        rot_L_6d = np.array([1, 0, 0, 0, 1, 0], dtype=np.float64)
        rot_R_6d = np.array([1, 0, 0, 0, 1, 0], dtype=np.float64)
        wrist = np.concatenate([trans_L, trans_R, rot_L_6d, rot_R_6d])
        assert wrist.shape == (18,)

        wrist_t = torch.from_numpy(wrist).unsqueeze(0)  # (1, 18)
        H_left, H_right = homo_matrix_from_wrist_pose(wrist_t)

        # Left: trans should match trans_L, rotation should be identity
        np.testing.assert_allclose(H_left[0, :3, 3].numpy(), trans_L, atol=1e-12)
        np.testing.assert_allclose(H_left[0, :3, :3].numpy(), np.eye(3), atol=1e-12)

        # Right: trans should match trans_R
        np.testing.assert_allclose(H_right[0, :3, 3].numpy(), trans_R, atol=1e-12)
        np.testing.assert_allclose(H_right[0, :3, :3].numpy(), np.eye(3), atol=1e-12)


class TestTransformPoseToTargetFrame:
    """Verify world-to-camera frame transform."""

    def test_identity_extrinsic(self):
        """Identity extrinsic should leave pose unchanged."""
        pose = torch.randn(5, 4, 4, dtype=torch.float64)
        ext = torch.eye(4, dtype=torch.float64)
        result = transform_pose_to_target_frame(pose, ext)
        torch.testing.assert_close(result, pose, atol=1e-10, rtol=0)

    def test_known_translation(self):
        """A pure-translation extrinsic should shift all poses."""
        pose = torch.eye(4, dtype=torch.float64).unsqueeze(0).expand(3, -1, -1).clone()
        ext = torch.eye(4, dtype=torch.float64)
        ext[0, 3] = 10.0  # Translate x by 10
        result = transform_pose_to_target_frame(pose, ext)
        # After applying ext @ pose, the translation column should get ext's translation
        for i in range(3):
            assert result[i, 0, 3].item() == pytest.approx(10.0)

    def test_unsqueeze_broadcast_2d_to_3d(self):
        """pose (T,4,4) + extrinsic (4,4) → broadcast via unsqueeze(-3)."""
        pose = torch.randn(7, 4, 4, dtype=torch.float64)
        ext = torch.randn(4, 4, dtype=torch.float64)
        result = transform_pose_to_target_frame(pose, ext)
        assert result.shape == (7, 4, 4)
        # Verify manually
        for t in range(7):
            expected = ext @ pose[t]
            torch.testing.assert_close(result[t], expected, atol=1e-10, rtol=0)

    def test_numpy_input(self):
        """Numpy inputs must produce numpy outputs."""
        pose = np.eye(4, dtype=np.float64).reshape(1, 4, 4)
        ext = np.eye(4, dtype=np.float64)
        result = transform_pose_to_target_frame(pose, ext)
        assert isinstance(result, np.ndarray)
        np.testing.assert_allclose(result, pose, atol=1e-12)


class TestTransformWristToTargetFrame:
    """End-to-end wrist transform verification."""

    def test_identity_extrinsic_preserves_wrist(self):
        """Identity world2cam should leave wrist_pose unchanged."""
        rng = np.random.default_rng(3)
        # Build wrist_pose with valid rotations
        axis_L = rng.normal(size=3); axis_L /= np.linalg.norm(axis_L)
        angle_L = rng.uniform(-1, 1)
        K = np.array([[0, -axis_L[2], axis_L[1]], [axis_L[2], 0, -axis_L[0]], [-axis_L[1], axis_L[0], 0]])
        R_L = np.eye(3) + np.sin(angle_L) * K + (1 - np.cos(angle_L)) * (K @ K)
        rot6d_L = np.concatenate([R_L[:, 0], R_L[:, 1]])

        axis_R = rng.normal(size=3); axis_R /= np.linalg.norm(axis_R)
        angle_R = rng.uniform(-1, 1)
        K = np.array([[0, -axis_R[2], axis_R[1]], [axis_R[2], 0, -axis_R[0]], [-axis_R[1], axis_R[0], 0]])
        R_R = np.eye(3) + np.sin(angle_R) * K + (1 - np.cos(angle_R)) * (K @ K)
        rot6d_R = np.concatenate([R_R[:, 0], R_R[:, 1]])

        trans_L = rng.normal(size=3)
        trans_R = rng.normal(size=3)
        wrist = np.concatenate([trans_L, trans_R, rot6d_L, rot6d_R]).astype(np.float64)
        wrist_2d = wrist.reshape(1, 18)
        ext = np.eye(4, dtype=np.float64)

        result = transform_wrist_to_target_frame(wrist_2d, ext)
        np.testing.assert_allclose(result, wrist_2d, atol=1e-10)


class TestHomoCoordinates:
    """Verify homogeneous <-> cartesian conversion."""

    def test_w_equals_one(self):
        """Standard case: w=1.  eps=1e-6 divides by 1+eps, so expect ~1e-6 relative error."""
        pts = torch.tensor([[1.0, 2.0, 3.0, 1.0]], dtype=torch.float64)
        cart = homo_coordinates_to_cartesian(pts)
        expected = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64)
        torch.testing.assert_close(cart, expected, atol=1e-5, rtol=1e-5)

    def test_round_trip(self):
        """cartesian -> homo -> cartesian.  eps adds ~1e-6 relative error."""
        pts = torch.randn(10, 3, dtype=torch.float64)
        homo = homo_coordinates_from_cartesian(pts)
        assert homo.shape == (10, 4)
        assert (homo[:, 3] == 1.0).all()
        recovered = homo_coordinates_to_cartesian(homo)
        torch.testing.assert_close(recovered, pts, atol=1e-5, rtol=1e-5)


class TestTransformHandPoints:
    """Verify hand point transformations."""

    def test_identity_transform(self):
        """Identity extrinsic should leave hand points unchanged."""
        pts = torch.randn(5, 30, dtype=torch.float64)  # 10 points * 3D
        ext = torch.eye(4, dtype=torch.float64)
        result = transform_hand_points_to_target_frame(pts, ext)
        torch.testing.assert_close(result, pts, atol=1e-5, rtol=0)

    def test_wrist_to_camera_round_trip(self):
        """world->wrist->camera chain should equal world->camera."""
        rng = np.random.default_rng(10)
        # Random hand points in world frame
        pts_world = torch.tensor(rng.normal(size=(3, 30)), dtype=torch.float64)

        # Random valid wrist pose in world frame
        wrist = _make_valid_wrist_pose(rng, T=3)

        # Step 1: world -> wrist
        pts_wrist = transform_hand_points_to_wrist_frame(pts_world, wrist)

        # Step 2: wrist -> camera (using wrist pose as camera transform)
        pts_cam = transform_hand_points_from_wrist_to_camera_frame(pts_wrist, wrist)

        # This should recover the original world points (because camera=world here)
        torch.testing.assert_close(pts_cam, pts_world, atol=1e-4, rtol=1e-4)


def _make_valid_wrist_pose(rng, T):
    """Helper: build a (T, 18) wrist pose with valid rotations."""
    poses = []
    for _ in range(T):
        trans_L = rng.normal(size=3)
        trans_R = rng.normal(size=3)
        R_L = _random_rotation(rng)
        R_R = _random_rotation(rng)
        rot6d_L = np.concatenate([R_L[:, 0], R_L[:, 1]])
        rot6d_R = np.concatenate([R_R[:, 0], R_R[:, 1]])
        poses.append(np.concatenate([trans_L, trans_R, rot6d_L, rot6d_R]))
    return torch.tensor(np.array(poses), dtype=torch.float64)


def _random_rotation(rng):
    """Random SO(3) matrix via axis-angle."""
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    angle = rng.uniform(-np.pi, np.pi)
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


# ======================================================================
# Module 2: Data transforms (process_state_action, get_relative_action)
# ======================================================================

from src.dataset.data_transforms import (
    process_state_action,
    get_relative_action,
    get_absolute_action,
)


class TestProcessStateAction:
    """Verify the full state/action processing pipeline."""

    def test_identity_extrinsic_passthrough(self):
        """Identity extrinsic should not change state/action values (for valid rotations)."""
        rng = np.random.default_rng(5)
        N_state, N_action = 3, 4

        wrist_state = _make_valid_wrist_pose(rng, N_state).numpy().astype(np.float32)
        wrist_action = _make_valid_wrist_pose(rng, N_action).numpy().astype(np.float32)
        hand_state = rng.normal(size=(N_state, 30)).astype(np.float32)
        hand_action = rng.normal(size=(N_action, 30)).astype(np.float32)
        extrinsic = np.eye(4, dtype=np.float32)

        state, action = process_state_action(
            wrist_state=wrist_state,
            hand_state=hand_state,
            wrist_action=wrist_action,
            hand_action=hand_action,
            extrinsic=extrinsic,
            normalizer=None,
            hand_ndim=15,
            motion_type="mano",
            use_relative_action=False,
        )

        assert state.shape == (N_state, 48)
        assert action.shape == (N_action, 48)

        # With identity extrinsic, wrist part should be unchanged
        np.testing.assert_allclose(state[:, :18], wrist_state, atol=1e-4)
        np.testing.assert_allclose(action[:, :18], wrist_action, atol=1e-4)
        # hand part should also be unchanged in mano mode
        np.testing.assert_allclose(state[:, 18:], hand_state, atol=1e-6)
        np.testing.assert_allclose(action[:, 18:], hand_action, atol=1e-6)

    def test_output_dimensions_with_hand_ndim(self):
        """Output dim = 18 (wrist) + 2 * hand_ndim."""
        rng = np.random.default_rng(20)
        hand_ndim = 10
        state, action = process_state_action(
            wrist_state=rng.normal(size=(2, 18)).astype(np.float32),
            hand_state=rng.normal(size=(2, 30)).astype(np.float32),
            wrist_action=rng.normal(size=(3, 18)).astype(np.float32),
            hand_action=rng.normal(size=(3, 30)).astype(np.float32),
            extrinsic=np.eye(4, dtype=np.float32),
            normalizer=None,
            hand_ndim=hand_ndim,
            motion_type="mano",
        )
        assert state.shape == (2, 18 + 2 * hand_ndim)
        assert action.shape == (3, 18 + 2 * hand_ndim)

    def test_hand_slicing_semantics(self):
        """hand_state[:, :hand_ndim] is left, [:, all_hand_ndim:all_hand_ndim+hand_ndim] is right."""
        rng = np.random.default_rng(0)
        hand_ndim = 5
        hand_state = np.zeros((1, 30), dtype=np.float32)
        # Set left hand dims [0:5] to 1.0, right hand dims [15:20] to 2.0
        hand_state[0, :5] = 1.0
        hand_state[0, 15:20] = 2.0

        state, _ = process_state_action(
            wrist_state=_make_valid_wrist_pose(rng, 1).numpy().astype(np.float32),
            hand_state=hand_state,
            wrist_action=_make_valid_wrist_pose(rng, 1).numpy().astype(np.float32),
            hand_action=np.zeros((1, 30), dtype=np.float32),
            extrinsic=np.eye(4, dtype=np.float32),
            normalizer=None,
            hand_ndim=hand_ndim,
            motion_type="mano",
        )
        # state shape = (1, 18 + 2*5) = (1, 28)
        # hand part starts at index 18
        np.testing.assert_allclose(state[0, 18:23], 1.0, atol=1e-6)  # left
        np.testing.assert_allclose(state[0, 23:28], 2.0, atol=1e-6)  # right


class TestRelativeAction:
    """Verify relative action computation and its inverse."""

    def test_identity_relative_action(self):
        """When action == state, relative action wrist should be identity-like."""
        rng = np.random.default_rng(0)
        state_action = _make_valid_wrist_pose(rng, 1).numpy().astype(np.float32)
        # Use same pose for state[-1] and action
        wrist_dim = 18
        hand = rng.normal(size=(1, 30)).astype(np.float32)
        combined = np.concatenate([state_action, hand], axis=-1)  # (1, 48)

        rel = get_relative_action(combined[0], combined.copy())

        # Wrist: pinv(state) @ state = I → trans=[0,0,0], rot6d=[1,0,0,0,1,0]
        # For both hands, action - state = 0
        np.testing.assert_allclose(rel[0, :3], 0.0, atol=1e-4)   # left trans
        np.testing.assert_allclose(rel[0, 3:6], 0.0, atol=1e-4)  # right trans
        np.testing.assert_allclose(rel[0, 18:], 0.0, atol=1e-5)  # hand diff

    def test_round_trip_absolute_relative(self):
        """get_relative_action then get_absolute_action should recover original."""
        rng = np.random.default_rng(8)
        state_pose = _make_valid_wrist_pose(rng, 1).numpy().astype(np.float64)
        action_pose = _make_valid_wrist_pose(rng, 5).numpy().astype(np.float64)
        hand_state = rng.normal(size=(1, 30)).astype(np.float64)
        hand_action = rng.normal(size=(5, 30)).astype(np.float64)

        state = np.concatenate([state_pose, hand_state], axis=-1)
        action = np.concatenate([action_pose, hand_action], axis=-1)

        rel = get_relative_action(state[-1], action)
        recovered = get_absolute_action(state[-1], rel)

        np.testing.assert_allclose(recovered, action, atol=1e-4, rtol=1e-4)


# ======================================================================
# Module 3: Normalizer (LinearNormalizer)
# ======================================================================

from src.model.common.normalizer import LinearNormalizer, StreamingStats


class TestStreamingStats:
    """Verify streaming statistics computation."""

    def test_exact_match_with_batch_stats(self):
        """Streaming stats should match batch computation exactly."""
        rng = np.random.default_rng(0)
        data = torch.tensor(rng.normal(size=(500, 8)), dtype=torch.float64)

        ss = StreamingStats(last_n_dims=1, reservoir_size=1000)
        # Feed in chunks
        for i in range(0, 500, 50):
            ss.update(data[i:i+50])

        stats = ss.get_stats()
        torch.testing.assert_close(stats['mean'], data.mean(dim=0), atol=1e-10, rtol=0)
        torch.testing.assert_close(stats['std'], data.std(dim=0, unbiased=True), atol=1e-10, rtol=0)
        torch.testing.assert_close(stats['min'], data.min(dim=0).values, atol=1e-10, rtol=0)
        torch.testing.assert_close(stats['max'], data.max(dim=0).values, atol=1e-10, rtol=0)

    def test_quantiles_exact_when_reservoir_fits_all(self):
        """When reservoir holds all data, quantiles must be exact."""
        rng = np.random.default_rng(1)
        data = torch.tensor(rng.normal(size=(200, 4)), dtype=torch.float64)

        ss = StreamingStats(last_n_dims=1, reservoir_size=1000)
        ss.update(data)

        stats = ss.get_stats()
        torch.testing.assert_close(
            stats['q01'], torch.quantile(data, 0.01, dim=0), atol=1e-10, rtol=0
        )
        torch.testing.assert_close(
            stats['q99'], torch.quantile(data, 0.99, dim=0), atol=1e-10, rtol=0
        )


class TestLinearNormalizerIgnoreDim:
    """Verify the ignore_dim fix prevents q01/q99 clipping."""

    def test_ignored_dims_pass_through_unchanged(self):
        """After ignore_dim, those dims must not be clipped or scaled."""
        rng = np.random.default_rng(2)
        data = torch.tensor(rng.normal(size=(100, 10)), dtype=torch.float32)

        normalizer = LinearNormalizer()
        normalizer.fit({"test": data})
        normalizer.ignore_dim(key="test", dim=slice(3, 7))

        # Create input with extreme values in ignored dims
        x = torch.zeros(1, 10)
        x[0, 5] = 100.0   # Way beyond q99 in ignored dim
        x[0, 0] = 100.0   # Way beyond q99 in non-ignored dim

        result = normalizer.normalize({"test": x})["test"]

        # Ignored dim (5): should pass through unchanged (scale=1, offset=0, no clip)
        assert result[0, 5].item() == pytest.approx(100.0, abs=1e-5)

        # Non-ignored dim (0): should be clipped to q99 then scaled
        assert result[0, 0].item() != pytest.approx(100.0, abs=1.0)

    def test_ignore_dim_sets_quantiles_to_inf(self):
        """After ignore_dim, q01 should be -inf and q99 should be +inf."""
        data = torch.randn(50, 10)
        normalizer = LinearNormalizer()
        normalizer.fit({"k": data})
        normalizer.ignore_dim(key="k", dim=slice(2, 5))

        params = normalizer.params_dict["k"]
        q01 = params["input_stats"]["q01"]
        q99 = params["input_stats"]["q99"]
        assert q01[2].item() == float('-inf')
        assert q01[4].item() == float('-inf')
        assert q99[2].item() == float('inf')
        assert q99[4].item() == float('inf')
        # Non-ignored dims should remain finite
        assert np.isfinite(q01[0].item())
        assert np.isfinite(q99[0].item())

    def test_normalize_unnormalize_round_trip(self):
        """normalize then unnormalize should recover original (within clipping range)."""
        rng = np.random.default_rng(4)
        data = torch.tensor(rng.normal(size=(200, 6)), dtype=torch.float32)
        normalizer = LinearNormalizer()
        normalizer.fit({"v": data})

        # Use values within q01/q99 range
        x = data[50:55].clone()
        normed = normalizer.normalize({"v": x})["v"]
        recovered = normalizer.unnormalize({"v": normed})["v"]
        torch.testing.assert_close(recovered, x, atol=1e-4, rtol=1e-4)


class TestStreamingVsBatchFit:
    """Verify streaming normalizer matches batch normalizer."""

    def test_streaming_matches_batch(self):
        """Streaming fit should produce identical scale/offset as batch fit."""
        rng = np.random.default_rng(10)
        data = torch.tensor(rng.normal(size=(300, 6)), dtype=torch.float32)

        # Batch fit
        norm_batch = LinearNormalizer()
        norm_batch.fit({"m": data})

        # Streaming fit
        norm_stream = LinearNormalizer()
        norm_stream.start_streaming_fit(keys=["m"], reservoir_size=10000)
        for i in range(0, 300, 30):
            norm_stream.update_streaming_fit({"m": data[i:i+30]})
        norm_stream.finish_streaming_fit()

        # Compare scale and offset
        torch.testing.assert_close(
            norm_stream.params_dict["m"]["scale"],
            norm_batch.params_dict["m"]["scale"],
            atol=1e-4, rtol=1e-4,
        )
        torch.testing.assert_close(
            norm_stream.params_dict["m"]["offset"],
            norm_batch.params_dict["m"]["offset"],
            atol=1e-4, rtol=1e-4,
        )


# ======================================================================
# Module 4: ConcatDataCollator
# ======================================================================

from src.dataset.collator import ConcatDataCollator


class TestConcatDataCollator:
    """Verify ConcatDataCollator behavior."""

    def test_concat_variable_length(self):
        """Variable-length samples should be concatenated along dim 0."""
        collator = ConcatDataCollator()
        samples = [
            {"motions": torch.randn(2, 4)},
            {"motions": torch.randn(5, 4)},
            {"motions": torch.randn(3, 4)},
        ]
        batch = collator(samples)
        assert batch["motions"].shape == (10, 4)
        assert batch["_batch_num_samples"] == 3

    def test_metadata_keys_are_not_tensors(self):
        """_batch_num_samples must stay as a Python int."""
        collator = ConcatDataCollator()
        samples = [{"motions": torch.randn(3, 4)}]
        batch = collator(samples)

        assert isinstance(batch["_batch_num_samples"], int)
        assert not isinstance(batch["_batch_num_samples"], (torch.Tensor, np.ndarray))

    def test_normalizer_key_filtering(self):
        """get_normalizer must exclude _batch_num_samples."""
        collator = ConcatDataCollator()
        samples = [
            {"motions": torch.randn(2, 4)},
            {"motions": torch.randn(3, 4)},
        ]
        batch = collator(samples)

        normalizer_keys = [
            key for key, value in batch.items()
            if not key.startswith("_") and isinstance(value, (torch.Tensor, np.ndarray))
        ]
        assert normalizer_keys == ["motions"]
        assert "_batch_num_samples" not in normalizer_keys


# ======================================================================
# Module 5: Shard allocation
# ======================================================================

from src.dataset.vla_dataset import VLALowLevelWdsDataset


def _shape_meta():
    return {
        "obs": {
            "rgb": {"shape": [4, 4, 3], "type": "rgb", "horizon": 1, "stride": 1},
            "depth": {"shape": [4, 4], "type": "depth", "horizon": 1, "stride": 1},
            "state": {
                "wrist": {"shape": [18]}, "hand": {"shape": [30]},
                "shape": [48], "type": "mano", "horizon": 1, "stride": 1,
            },
        },
        "action": {"shape": [48], "type": "mano", "horizon": 1, "stride": 1},
    }


class TestShardSelection:
    """Verify shard selection semantics after simplifying the sampler."""

    def _select_counts(self, capacities, max_total, min_per, seed=0):
        groups = [
            {
                "dataset_index": i,
                "name": f"d{i}",
                "shard_spec": f"d{i}",
                "shard_urls": [f"d{i}_shard_{j}" for j in range(cap)],
            }
            for i, cap in enumerate(capacities)
        ]
        ds = VLALowLevelWdsDataset.__new__(VLALowLevelWdsDataset)
        ds.max_total_shards = max_total
        ds.min_shards_per_dataset = min_per
        ds.seed = seed
        ds.mode = "val"
        ds.build_shard_groups = lambda: groups
        _, selected_shards = ds.select_shards()
        return [
            sum(1 for dataset_index, _ in selected_shards if dataset_index == i)
            for i in range(len(capacities))
        ]

    def test_no_budget_uses_all(self):
        assert self._select_counts([5, 3], None, 1) == [5, 3]

    def test_budget_equals_total(self):
        assert self._select_counts([5, 3], 8, 1) == [5, 3]

    def test_budget_larger_than_total(self):
        assert self._select_counts([5, 3], 100, 1) == [5, 3]

    def test_budget_preserves_floor_when_possible(self):
        counts = self._select_counts([10, 2], 6, 1, seed=7)
        assert sum(counts) == 6
        assert counts[0] >= 1
        assert counts[1] >= 1
        assert counts[0] > counts[1]

    def test_three_datasets_respect_floor(self):
        counts = self._select_counts([100, 20, 5], 10, 2, seed=11)
        assert sum(counts) == 10
        assert all(c >= 2 for c in counts)
        assert counts[0] >= counts[1]

    def test_budget_too_small_for_all_floors(self):
        ds = VLALowLevelWdsDataset.__new__(VLALowLevelWdsDataset)
        ds.max_total_shards = 5
        ds.min_shards_per_dataset = 3
        ds.seed = 0
        ds.mode = "val"
        ds.build_shard_groups = lambda: [
            {"dataset_index": i, "name": f"d{i}", "shard_spec": f"d{i}", "shard_urls": [f"d{i}_{j}" for j in range(10)]}
            for i in range(3)
        ]
        with pytest.raises(ValueError, match="max_total_shards is smaller than the required minimum shard coverage"):
            ds.select_shards()

    def test_single_dataset(self):
        assert self._select_counts([10], 5, 3) == [5]
        assert self._select_counts([3], 5, 3) == [3]
        assert self._select_counts([3], None, 3) == [3]

    def test_one_dataset_has_fewer_than_min(self):
        counts = self._select_counts([10, 1], 6, 3, seed=3)
        assert sum(counts) == 6
        assert counts[1] == 1
        assert counts[0] == 5

    def test_sum_never_exceeds_budget(self):
        """Fuzz test: shard selection should respect capacity and budget."""
        rng = np.random.default_rng(77)
        for _ in range(200):
            n_datasets = rng.integers(1, 6)
            capacities = rng.integers(1, 50, size=n_datasets).tolist()
            max_total = int(rng.integers(1, sum(capacities) + 10))
            min_per = int(rng.integers(1, 10))
            minimum_selected = sum(min(cap, min_per) for cap in capacities)
            if max_total < minimum_selected:
                continue
            counts = self._select_counts(capacities, max_total, min_per, seed=17)
            expected_total = min(max_total, sum(capacities))
            assert sum(counts) == expected_total, (
                f"capacities={capacities}, budget={max_total}, min_per={min_per}, "
                f"counts={counts}, expected_sum={expected_total}"
            )
            for i, (count, capacity) in enumerate(zip(counts, capacities)):
                assert 0 <= count <= capacity, f"count[{i}]={count} > capacity={capacity}"


# ======================================================================
# Module 6: Sliding window compose
# ======================================================================

from src.dataset.wds_dataset import (
    sliding_window_compose,
    build_sample_from_window,
    gather_history_frames,
    WindowConfig,
    LOWDIM_SLICES,
)


def _make_frame(dataset_name, episode_index, frame_index, rng):
    """Build a minimal frame dict for sliding_window_compose."""
    lowdim = rng.normal(size=116).astype(np.float32)
    meta = {
        "dataset_name": dataset_name,
        "episode_index": episode_index,
        "instruction": "test",
        "instruction_num": 1,
        "presence": 3,
    }
    return {"lowdim.npy": lowdim, "meta.json": meta, "__key__": f"{frame_index:06d}"}


class TestSlidingWindowCompose:
    """Verify sliding window episode boundary and padding behavior."""

    def test_single_episode_sample_count(self):
        """N frames in one episode should produce N samples."""
        rng = np.random.default_rng(0)
        config = WindowConfig(
            action_horizon=2, action_stride=1,
            state_horizon=1, state_stride=1,
            image_horizon=1, image_stride=1,
        )
        frames = [_make_frame("d", 0, i, rng) for i in range(5)]
        samples = list(sliding_window_compose(iter(frames), config, LOWDIM_SLICES, lowdim_only=True))
        assert len(samples) == 5

    def test_two_episodes_produce_correct_count(self):
        """Two episodes of 3 frames each → 6 samples total."""
        rng = np.random.default_rng(1)
        config = WindowConfig(
            action_horizon=2, action_stride=1,
            state_horizon=1, state_stride=1,
            image_horizon=1, image_stride=1,
        )
        frames = (
            [_make_frame("d", 0, i, rng) for i in range(3)] +
            [_make_frame("d", 1, i, rng) for i in range(3)]
        )
        samples = list(sliding_window_compose(iter(frames), config, LOWDIM_SLICES, lowdim_only=True))
        assert len(samples) == 6

    def test_repeat_padding_action_length(self):
        """In repeat mode, action chunk length == action_horizon even at episode end."""
        rng = np.random.default_rng(2)
        config = WindowConfig(
            action_horizon=4, action_stride=1,
            state_horizon=1, state_stride=1,
            image_horizon=1, image_stride=1,
            future_pad_mode="repeat",
        )
        frames = [_make_frame("d", 0, i, rng) for i in range(3)]
        samples = list(sliding_window_compose(iter(frames), config, LOWDIM_SLICES, lowdim_only=True))
        for s in samples:
            assert s["wrist_action"].shape[0] == 4

    def test_truncate_padding_action_length(self):
        """In truncate mode, action chunk can be shorter than horizon."""
        rng = np.random.default_rng(3)
        config = WindowConfig(
            action_horizon=4, action_stride=1,
            state_horizon=1, state_stride=1,
            image_horizon=1, image_stride=1,
            future_pad_mode="truncate",
        )
        frames = [_make_frame("d", 0, i, rng) for i in range(3)]
        samples = list(sliding_window_compose(iter(frames), config, LOWDIM_SLICES, lowdim_only=True))
        # Last frame has only 1 available future frame → action len = 1
        assert samples[-1]["wrist_action"].shape[0] == 1
        assert samples[-1]["valid_action_len"] == 1

    def test_valid_action_len_values(self):
        """valid_action_len must reflect actual available frames."""
        rng = np.random.default_rng(4)
        config = WindowConfig(
            action_horizon=10, action_stride=1,
            state_horizon=1, state_stride=1,
            image_horizon=1, image_stride=1,
            future_pad_mode="repeat",
        )
        frames = [_make_frame("d", 0, i, rng) for i in range(5)]
        samples = list(sliding_window_compose(iter(frames), config, LOWDIM_SLICES, lowdim_only=True))
        # Frame 0: 5 avail, frame 1: 4 avail, ..., frame 4: 1 avail
        expected_lens = [5, 4, 3, 2, 1]
        actual_lens = [s["valid_action_len"] for s in samples]
        assert actual_lens == expected_lens


class TestGatherHistoryFrames:
    """Verify history frame gathering with padding."""

    def test_repeat_mode_full_history(self):
        """With enough past frames, all history should be real."""
        config = WindowConfig(
            action_horizon=1, action_stride=1,
            state_horizon=3, state_stride=1,
            image_horizon=1, image_stride=1,
        )
        past = collections.deque([{"id": 0}, {"id": 1}, {"id": 2}], maxlen=config.past_size)
        buf = collections.deque([{"id": 3}])
        frames = gather_history_frames(past, buf, horizon=3, stride=1, pad_mode="repeat")
        assert len(frames) == 3
        assert [f["id"] for f in frames] == [1, 2, 3]

    def test_repeat_mode_pads_with_earliest(self):
        """With insufficient history, repeat mode pads with earliest available."""
        past = collections.deque(maxlen=10)  # empty
        buf = collections.deque([{"id": 0}])
        frames = gather_history_frames(past, buf, horizon=3, stride=1, pad_mode="repeat")
        assert len(frames) == 3
        assert all(f["id"] == 0 for f in frames)

    def test_truncate_mode_shorter_output(self):
        """With insufficient history, truncate mode produces fewer frames."""
        past = collections.deque(maxlen=10)  # empty
        buf = collections.deque([{"id": 0}])
        frames = gather_history_frames(past, buf, horizon=3, stride=1, pad_mode="truncate")
        assert len(frames) == 1
        assert frames[0]["id"] == 0


# ======================================================================
# Module 7: End-to-end WDS pipeline
# ======================================================================


def _add_bytes_to_tar(tar_obj, name, data):
    info = tarfile.TarInfo(name=name)
    info.size = len(data)
    tar_obj.addfile(info, io.BytesIO(data))


def _encode_npy(array):
    buf = io.BytesIO()
    np.save(buf, array.astype(np.float32))
    return buf.getvalue()


def _make_lowdim_vector_deterministic(rng):
    ws = rng.normal(size=18).astype(np.float32)
    hs = rng.normal(size=30).astype(np.float32)
    wa = rng.normal(size=18).astype(np.float32)
    ha = rng.normal(size=30).astype(np.float32)
    ext = np.eye(4, dtype=np.float32).reshape(-1)
    intr = rng.normal(size=4).astype(np.float32)
    return np.concatenate([ws, hs, wa, ha, ext, intr]).astype(np.float32)


def _write_shard(shard_path, dataset_name, ep_idx, n_frames, seed):
    rng = np.random.default_rng(seed)
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(shard_path, "w") as tar:
        for f in range(n_frames):
            key = f"ep{ep_idx:04d}_{f:06d}"
            meta = {"dataset_name": dataset_name, "episode_index": ep_idx,
                    "instruction": "test", "instruction_num": 1, "presence": 3}
            _add_bytes_to_tar(tar, f"{key}.meta.json", json.dumps(meta).encode())
            _add_bytes_to_tar(tar, f"{key}.lowdim.npy", _encode_npy(_make_lowdim_vector_deterministic(rng)))


class TestEndToEndWDSPipeline:
    """Full pipeline: WDS → VLALowLevelWdsDataset → normalizer."""

    def test_deterministic_iteration(self, tmp_path):
        """Two iterations over the same dataset must yield identical results."""
        ds_dir = tmp_path / "ds"
        for i in range(2):
            _write_shard(ds_dir / f"shard-{i:06d}.tar", "ds", i, 5, seed=i * 100)
        wds_cfg = [{"name": "ds", "shard_urls": str(ds_dir / "shard-*.tar"), "weight": 1.0}]

        dataset = VLALowLevelWdsDataset(
            wds_datasets=wds_cfg, shape_meta=_shape_meta(), mode="val", seed=0,
        )
        rows_1 = [s["motions"].clone() for s in dataset]
        rows_2 = [s["motions"].clone() for s in dataset]
        assert len(rows_1) == len(rows_2)
        for a, b in zip(rows_1, rows_2):
            torch.testing.assert_close(a, b)

    def test_normalizer_matches_direct_computation(self, tmp_path):
        """Normalizer from get_normalizer must match direct stats on same data."""
        from src.dataset.normalizer_utils import get_normalizer

        ds_dir = tmp_path / "ds"
        for i in range(3):
            _write_shard(ds_dir / f"shard-{i:06d}.tar", "ds", i, 4, seed=i * 100)
        wds_cfg = [{"name": "ds", "shard_urls": str(ds_dir / "shard-*.tar"), "weight": 1.0}]

        dataset = VLALowLevelWdsDataset(
            wds_datasets=wds_cfg, shape_meta=_shape_meta(), mode="val",
            max_total_shards=2, min_shards_per_dataset=1, seed=42,
        )

        direct_rows = torch.cat([s["motions"] for s in dataset], dim=0)
        normalizer = get_normalizer({"batch_size": 4, "num_workers": 0}, dataset)

        stats = normalizer.params_dict["motions"]["input_stats"]
        direct_f64 = direct_rows.to(torch.float64)
        torch.testing.assert_close(
            stats["mean"].to(torch.float64), direct_f64.mean(dim=0), atol=1e-4, rtol=1e-4
        )
        torch.testing.assert_close(
            stats["min"].to(torch.float64), direct_f64.min(dim=0).values, atol=1e-4, rtol=1e-4
        )
        torch.testing.assert_close(
            stats["max"].to(torch.float64), direct_f64.max(dim=0).values, atol=1e-4, rtol=1e-4
        )

    def test_shard_budget_limits_data(self, tmp_path):
        """max_total_shards should limit how many shards are scanned."""
        ds_dir = tmp_path / "ds"
        for i in range(10):
            _write_shard(ds_dir / f"shard-{i:06d}.tar", "ds", i, 3, seed=i)
        wds_cfg = [{"name": "ds", "shard_urls": str(ds_dir / "shard-*.tar"), "weight": 1.0}]

        dataset_all = VLALowLevelWdsDataset(
            wds_datasets=wds_cfg, shape_meta=_shape_meta(), mode="val",
        )
        dataset_limited = VLALowLevelWdsDataset(
            wds_datasets=wds_cfg, shape_meta=_shape_meta(), mode="val",
            max_total_shards=3, min_shards_per_dataset=1,
        )
        count_all = sum(1 for _ in dataset_all)
        count_limited = sum(1 for _ in dataset_limited)
        assert count_limited < count_all
        # 3 shards × 3 frames per shard = 9 samples
        assert count_limited == 9
