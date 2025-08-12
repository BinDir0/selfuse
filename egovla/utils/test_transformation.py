import unittest
import torch
import torch.nn.functional as F
import numpy as np
import sys
import os

# Add the parent directory to the path to import the transformation module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformation import rot_matrix_from_6drot, rot_matrix_to_6drot, transform_to_target_frame, transform_wrist_to_target_frame


def generate_orthogonal_6d_rotation(batch_shape=None):
    """
    Generate orthogonal 6D rotation vectors.
    
    Args:
        batch_shape: Optional tuple for batch dimensions (e.g., (5,) for batch_size=5)
    
    Returns:
        rot_6d: torch.Tensor with shape [batch_shape..., 6] or [6] if batch_shape is None
    """
    if batch_shape is None:
        # Generate single 6D rotation
        # Create two random 3D vectors
        a = torch.randn(3)
        b = torch.randn(3)
        
        # Normalize the first vector
        a_norm = F.normalize(a, p=2, dim=0)
        
        # Make the second vector orthogonal to the first
        b_proj = torch.sum(a_norm * b) * a_norm
        b_ortho = b - b_proj
        
        # Normalize the second vector
        b_norm = F.normalize(b_ortho, p=2, dim=0)
        
        # Concatenate to form 6D rotation
        rot_6d = torch.cat([a_norm, b_norm])
    else:
        # Generate batch of 6D rotations
        total_elements = np.prod(batch_shape)
        
        # Create random 3D vectors for the batch
        a = torch.randn(total_elements, 3)
        b = torch.randn(total_elements, 3)
        
        # Normalize the first vectors
        a_norm = F.normalize(a, p=2, dim=1)
        
        # Make the second vectors orthogonal to the first
        b_proj = torch.sum(a_norm * b, dim=1, keepdim=True) * a_norm
        b_ortho = b - b_proj
        
        # Normalize the second vectors
        b_norm = F.normalize(b_ortho, p=2, dim=1)
        
        # Concatenate to form 6D rotations
        rot_6d = torch.cat([a_norm, b_norm], dim=1)
        
        # Reshape to batch shape
        rot_6d = rot_6d.reshape(batch_shape + (6,))
    
    return rot_6d


class TestTransformation(unittest.TestCase):
    """Test cases for transformation functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Set random seed for reproducible tests
        torch.manual_seed(42)
        np.random.seed(42)
        
    def test_rot_matrix_from_6drot_single_vector(self):
        """Test 6D rotation to matrix conversion with single vector."""
        # Test single vector [6] - use orthogonal 6D rotation
        rot_6d = generate_orthogonal_6d_rotation()
        rot_matrix = rot_matrix_from_6drot(rot_6d)
        
        self.assertEqual(rot_matrix.shape, (3, 3))
        self.assertIsInstance(rot_matrix, torch.Tensor)
        
        # Test orthogonality
        identity = torch.eye(3)
        orthogonality_error = torch.norm(torch.matmul(rot_matrix, rot_matrix.T) - identity)
        self.assertLess(orthogonality_error.item(), 1e-6)
        
    def test_rot_matrix_from_6drot_batch(self):
        """Test 6D rotation to matrix conversion with batch."""
        # Test batch of vectors [B, 6] - use orthogonal 6D rotations
        batch_size = 5
        rot_6d = generate_orthogonal_6d_rotation((batch_size,))
        rot_matrix = rot_matrix_from_6drot(rot_6d)
        
        self.assertEqual(rot_matrix.shape, (batch_size, 3, 3))
        self.assertIsInstance(rot_matrix, torch.Tensor)
        
        # Test orthogonality for each matrix in batch
        identity = torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1)
        orthogonality_error = torch.norm(torch.matmul(rot_matrix, rot_matrix.transpose(1, 2)) - identity)
        self.assertLess(orthogonality_error.item(), 1e-6)
        
    def test_rot_matrix_from_6drot_3d_tensor(self):
        """Test 6D rotation to matrix conversion with 3D tensor."""
        # Test 3D tensor [B, T, 6] - use orthogonal 6D rotations
        rot_6d = generate_orthogonal_6d_rotation((2, 3))
        rot_matrix = rot_matrix_from_6drot(rot_6d)
        
        self.assertEqual(rot_matrix.shape, (2, 3, 3, 3))
        self.assertIsInstance(rot_matrix, torch.Tensor)
        
    def test_rot_matrix_from_6drot_4d_tensor(self):
        """Test 6D rotation to matrix conversion with 4D tensor."""
        # Test 4D tensor [B, T, H, 6] - use orthogonal 6D rotations
        rot_6d = generate_orthogonal_6d_rotation((2, 3, 4))
        rot_matrix = rot_matrix_from_6drot(rot_6d)
        
        self.assertEqual(rot_matrix.shape, (2, 3, 4, 3, 3))
        self.assertIsInstance(rot_matrix, torch.Tensor)
        
    def test_rot_matrix_from_6drot_numpy(self):
        """Test 6D rotation to matrix conversion with numpy input."""
        # Test numpy array - use orthogonal 6D rotations
        rot_6d = generate_orthogonal_6d_rotation((3,)).numpy()
        rot_matrix = rot_matrix_from_6drot(rot_6d)
        
        self.assertEqual(rot_matrix.shape, (3, 3, 3))
        self.assertIsInstance(rot_matrix, np.ndarray)
        
    def test_rot_matrix_to_6drot_single_matrix(self):
        """Test matrix to 6D rotation conversion with single matrix."""
        # Test single matrix [3, 3]
        rot_matrix = torch.randn(3, 3)
        # Make it orthogonal
        u, _, v = torch.svd(rot_matrix)
        rot_matrix = torch.matmul(u, v.T)
        
        rot_6d = rot_matrix_to_6drot(rot_matrix)
        
        self.assertEqual(rot_6d.shape, (6,))
        self.assertIsInstance(rot_6d, torch.Tensor)
        
    def test_rot_matrix_to_6drot_batch(self):
        """Test matrix to 6D rotation conversion with batch."""
        # Test batch of matrices [B, 3, 3]
        batch_size = 4
        rot_matrix = torch.randn(batch_size, 3, 3)
        # Make them orthogonal
        for i in range(batch_size):
            u, _, v = torch.svd(rot_matrix[i])
            rot_matrix[i] = torch.matmul(u, v.T)
        
        rot_6d = rot_matrix_to_6drot(rot_matrix)
        
        self.assertEqual(rot_6d.shape, (batch_size, 6))
        self.assertIsInstance(rot_6d, torch.Tensor)
        
    def test_rot_matrix_to_6drot_3d_tensor(self):
        """Test matrix to 6D rotation conversion with 3D tensor."""
        # Test 3D tensor [B, T, 3, 3]
        rot_matrix = torch.randn(2, 3, 3, 3)
        # Make them orthogonal
        for i in range(2):
            for j in range(3):
                u, _, v = torch.svd(rot_matrix[i, j])
                rot_matrix[i, j] = torch.matmul(u, v.T)
        
        rot_6d = rot_matrix_to_6drot(rot_matrix)
        
        self.assertEqual(rot_6d.shape, (2, 3, 6))
        self.assertIsInstance(rot_6d, torch.Tensor)
        
    def test_rot_matrix_to_6drot_numpy(self):
        """Test matrix to 6D rotation conversion with numpy input."""
        # Test numpy array
        rot_matrix = np.random.randn(3, 3, 3)
        # Make them orthogonal
        for i in range(3):
            u, _, v = np.linalg.svd(rot_matrix[i])
            rot_matrix[i] = np.matmul(u, v)
        
        rot_6d = rot_matrix_to_6drot(rot_matrix)
        
        self.assertEqual(rot_6d.shape, (3, 6))
        self.assertIsInstance(rot_6d, np.ndarray)
        
    def test_round_trip_conversion(self):
        """Test round-trip conversion: 6D -> matrix -> 6D."""
        # Test with batch - use orthogonal 6D rotations
        original_6d = generate_orthogonal_6d_rotation((5,))
        matrix = rot_matrix_from_6drot(original_6d)
        converted_6d = rot_matrix_to_6drot(matrix)
        
        # Check if the conversion is close (within numerical precision)
        error = torch.norm(original_6d - converted_6d)
        self.assertLess(error.item(), 1e-6)
        
    def test_transform_to_target_frame_single(self):
        """Test transform_to_target_frame with single pose."""
        # Create test data
        pose = torch.eye(4).unsqueeze(0)  # [1, 4, 4]
        target_extrinsic = torch.eye(4)
        
        transformed_pose = transform_to_target_frame(pose, target_extrinsic)
        
        self.assertEqual(transformed_pose.shape, (1, 4, 4))
        self.assertIsInstance(transformed_pose, torch.Tensor)
        
        # Test with non-identity transformation
        target_extrinsic = torch.eye(4)
        target_extrinsic[:3, 3] = torch.tensor([1.0, 2.0, 3.0])  # Translation
        
        transformed_pose = transform_to_target_frame(pose, target_extrinsic)
        
        # The transformed pose should have the inverse translation
        expected_translation = -torch.tensor([1.0, 2.0, 3.0])
        self.assertTrue(torch.allclose(transformed_pose[0, :3, 3], expected_translation, atol=1e-6))
        
    def test_transform_to_target_frame_batch(self):
        """Test transform_to_target_frame with batch."""
        # Create test data
        batch_size = 3
        time_steps = 4
        pose = torch.eye(4).unsqueeze(0).unsqueeze(0).repeat(batch_size, time_steps, 1, 1)
        target_extrinsic = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1)
        
        transformed_pose = transform_to_target_frame(pose, target_extrinsic)
        
        self.assertEqual(transformed_pose.shape, (batch_size, time_steps, 4, 4))
        self.assertIsInstance(transformed_pose, torch.Tensor)
        
    def test_transform_to_target_frame_numpy(self):
        """Test transform_to_target_frame with numpy input."""
        # Create test data
        pose = np.eye(4).reshape(1, 4, 4).astype(np.float32)
        target_extrinsic = np.eye(4).astype(np.float32)
        
        transformed_pose = transform_to_target_frame(pose, target_extrinsic)
        
        self.assertEqual(transformed_pose.shape, (1, 4, 4))
        self.assertIsInstance(transformed_pose, np.ndarray)
        
    def test_transform_wrist_to_target_frame_single(self):
        """Test transform_wrist_to_target_frame with single action."""
        # Create test data: [T, 18] where 18 = 6 (left pos) + 6 (right pos) + 6 (left rot) + 6 (right rot)
        time_steps = 5
        wrist_action = torch.randn(time_steps, 18)
        target_extrinsic = torch.eye(4)
        
        transformed_action = transform_wrist_to_target_frame(wrist_action, target_extrinsic)
        
        self.assertEqual(transformed_action.shape, (time_steps, 18))
        self.assertIsInstance(transformed_action, torch.Tensor)
        
    def test_transform_wrist_to_target_frame_batch(self):
        """Test transform_wrist_to_target_frame with batch."""
        # Create test data: [B, T, 18]
        batch_size = 2
        time_steps = 3
        wrist_action = torch.randn(batch_size, time_steps, 18)
        target_extrinsic = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1)
        
        transformed_action = transform_wrist_to_target_frame(wrist_action, target_extrinsic)
        
        self.assertEqual(transformed_action.shape, (batch_size, time_steps, 18))
        self.assertIsInstance(transformed_action, torch.Tensor)
        
    def test_transform_wrist_to_target_frame_numpy(self):
        """Test transform_wrist_to_target_frame with numpy input."""
        # Create test data
        time_steps = 4
        wrist_action = np.random.randn(time_steps, 18).astype(np.float32)
        target_extrinsic = np.eye(4).astype(np.float32)
        
        transformed_action = transform_wrist_to_target_frame(wrist_action, target_extrinsic)
        
        self.assertEqual(transformed_action.shape, (time_steps, 18))
        self.assertIsInstance(transformed_action, np.ndarray)
        
    def test_edge_cases(self):
        """Test edge cases and error conditions."""
        # Test with zero vectors (should still work)
        rot_6d_zero = torch.zeros(6)
        rot_matrix_zero = rot_matrix_from_6drot(rot_6d_zero)
        self.assertEqual(rot_matrix_zero.shape, (3, 3))
        
        # Test with very small values
        rot_6d_small = torch.tensor([1e-8, 1e-8, 1e-8, 1e-8, 1e-8, 1e-8])
        rot_matrix_small = rot_matrix_from_6drot(rot_6d_small)
        self.assertEqual(rot_matrix_small.shape, (3, 3))
        
    def test_orthogonality_property(self):
        """Test that rotation matrices are orthogonal."""
        # Test with multiple random orthogonal 6D rotations
        num_tests = 10
        for _ in range(num_tests):
            rot_6d = generate_orthogonal_6d_rotation()
            rot_matrix = rot_matrix_from_6drot(rot_6d)
            
            # Check orthogonality: R * R^T = I
            identity = torch.eye(3)
            orthogonality_error = torch.norm(torch.matmul(rot_matrix, rot_matrix.T) - identity)
            self.assertLess(orthogonality_error.item(), 1e-6)
            
            # Check determinant is 1 (proper rotation)
            det = torch.det(rot_matrix)
            self.assertAlmostEqual(det.item(), 1.0, places=6)
            
    def test_arbitrary_dimensions(self):
        """Test functions with various arbitrary dimensions."""
        # Test 5D tensor - use orthogonal 6D rotations
        rot_6d_5d = generate_orthogonal_6d_rotation((2, 3, 4, 5))
        rot_matrix_5d = rot_matrix_from_6drot(rot_6d_5d)
        self.assertEqual(rot_matrix_5d.shape, (2, 3, 4, 5, 3, 3))
        
        # Test 6D tensor - use orthogonal 6D rotations
        rot_6d_6d = generate_orthogonal_6d_rotation((2, 3, 4, 5, 6))
        rot_matrix_6d = rot_matrix_from_6drot(rot_6d_6d)
        self.assertEqual(rot_matrix_6d.shape, (2, 3, 4, 5, 6, 3, 3))
        
        # Test corresponding matrix to 6D conversion
        rot_6d_back = rot_matrix_to_6drot(rot_matrix_5d)
        self.assertEqual(rot_6d_back.shape, (2, 3, 4, 5, 6))
        
    def test_numerical_stability(self):
        """Test numerical stability with extreme values."""
        # Test with very large values - use orthogonal 6D rotations
        rot_6d_large = generate_orthogonal_6d_rotation() * 1e6
        rot_matrix_large = rot_matrix_from_6drot(rot_6d_large)
        self.assertEqual(rot_matrix_large.shape, (3, 3))
        
        # Test with very small values - use orthogonal 6D rotations
        rot_6d_small = generate_orthogonal_6d_rotation() * 1e-6
        rot_matrix_small = rot_matrix_from_6drot(rot_6d_small)
        self.assertEqual(rot_matrix_small.shape, (3, 3))

    def test_specific_rotation_examples(self):
        """Test specific rotation examples to verify correctness."""
        # Test 1: Identity rotation (no rotation)
        identity_6d = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])  # Identity rotation
        identity_matrix = rot_matrix_from_6drot(identity_6d)
        expected_identity = torch.eye(3)
        self.assertTrue(torch.allclose(identity_matrix, expected_identity, atol=1e-6))
        
        # Test 2: 90-degree rotation around Z-axis
        # For 90-degree rotation around Z: cos(90°) = 0, sin(90°) = 1
        rot_90_z_6d = torch.tensor([0.0, 1.0, 0.0, -1.0, 0.0, 0.0])
        rot_90_z_matrix = rot_matrix_from_6drot(rot_90_z_6d)
        expected_90_z = torch.tensor([[0.0, -1.0, 0.0],
                                     [1.0,  0.0, 0.0],
                                     [0.0,  0.0, 1.0]])
        self.assertTrue(torch.allclose(rot_90_z_matrix, expected_90_z, atol=1e-6))
        
        # Test 3: 180-degree rotation around X-axis
        rot_180_x_6d = torch.tensor([1.0, 0.0, 0.0, 0.0, -1.0, 0.0])
        rot_180_x_matrix = rot_matrix_from_6drot(rot_180_x_6d)
        expected_180_x = torch.tensor([[1.0,  0.0,  0.0],
                                      [0.0, -1.0,  0.0],
                                      [0.0,  0.0, -1.0]])
        self.assertTrue(torch.allclose(rot_180_x_matrix, expected_180_x, atol=1e-6))

    def test_rotation_matrix_properties(self):
        """Test that rotation matrices have the correct properties."""
        # Generate a random orthogonal 6D rotation
        rot_6d = generate_orthogonal_6d_rotation()
        rot_matrix = rot_matrix_from_6drot(rot_6d)
        
        # Property 1: R * R^T = I (orthogonality)
        identity = torch.eye(3)
        orthogonality_error = torch.norm(torch.matmul(rot_matrix, rot_matrix.T) - identity)
        self.assertLess(orthogonality_error.item(), 1e-6)
        
        # Property 2: det(R) = 1 (proper rotation)
        det = torch.det(rot_matrix)
        self.assertAlmostEqual(det.item(), 1.0, places=6)
        
        # Property 3: R^T * R = I (orthogonality from the other side)
        orthogonality_error_2 = torch.norm(torch.matmul(rot_matrix.T, rot_matrix) - identity)
        self.assertLess(orthogonality_error_2.item(), 1e-6)

    def test_transform_to_target_frame_specific_examples(self):
        """Test transform_to_target_frame with specific examples."""
        # Test 1: Identity transformation (should not change the pose)
        pose = torch.eye(4).unsqueeze(0)  # [1, 4, 4]
        target_extrinsic = torch.eye(4)
        
        transformed_pose = transform_to_target_frame(pose, target_extrinsic)
        self.assertTrue(torch.allclose(transformed_pose, pose, atol=1e-6))
        
        # Test 2: Pure translation transformation
        pose = torch.eye(4).unsqueeze(0)  # [1, 4, 4]
        target_extrinsic = torch.eye(4)
        target_extrinsic[:3, 3] = torch.tensor([1.0, 2.0, 3.0])  # Translation by [1, 2, 3]
        
        transformed_pose = transform_to_target_frame(pose, target_extrinsic)
        expected_translation = -torch.tensor([1.0, 2.0, 3.0])
        self.assertTrue(torch.allclose(transformed_pose[0, :3, 3], expected_translation, atol=1e-6))
        
        # Test 3: Rotation transformation
        pose = torch.eye(4).unsqueeze(0)  # [1, 4, 4]
        # Create a 90-degree rotation around Z-axis
        target_extrinsic = torch.eye(4)
        target_extrinsic[:3, :3] = torch.tensor([[0.0, -1.0, 0.0],
                                                 [1.0,  0.0, 0.0],
                                                 [0.0,  0.0, 1.0]])
        
        transformed_pose = transform_to_target_frame(pose, target_extrinsic)
        # The transformed pose should have the inverse rotation
        expected_rotation = torch.tensor([[0.0,  1.0, 0.0],
                                         [-1.0, 0.0, 0.0],
                                         [0.0,  0.0, 1.0]])
        self.assertTrue(torch.allclose(transformed_pose[0, :3, :3], expected_rotation, atol=1e-6))

    def test_transform_wrist_to_target_frame_specific_examples(self):
        """Test transform_wrist_to_target_frame with specific examples."""
        # Test 1: Identity transformation (should not change the wrist action)
        time_steps = 3
        wrist_action = torch.randn(time_steps, 18)
        target_extrinsic = torch.eye(4)
        
        transformed_action = transform_wrist_to_target_frame(wrist_action, target_extrinsic)
        self.assertTrue(torch.allclose(transformed_action, wrist_action, atol=1e-6))
        
        # Test 2: Pure translation transformation
        time_steps = 2
        wrist_action = torch.zeros(time_steps, 18)
        # Set some specific positions and rotations
        wrist_action[:, :3] = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # Left wrist positions
        wrist_action[:, 3:6] = torch.tensor([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])  # Right wrist positions
        wrist_action[:, 6:12] = torch.tensor([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]])  # Left wrist rotations
        wrist_action[:, 12:18] = torch.tensor([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]])  # Right wrist rotations
        
        target_extrinsic = torch.eye(4)
        target_extrinsic[:3, 3] = torch.tensor([1.0, 2.0, 3.0])  # Translation by [1, 2, 3]
        
        transformed_action = transform_wrist_to_target_frame(wrist_action, target_extrinsic)
        
        # Check that positions are transformed correctly
        expected_left_pos_1 = torch.tensor([0.0, 0.0, 0.0])  # [1,2,3] - [1,2,3] = [0,0,0]
        expected_left_pos_2 = torch.tensor([3.0, 3.0, 3.0])  # [4,5,6] - [1,2,3] = [3,3,3]
        expected_right_pos_1 = torch.tensor([6.0, 6.0, 6.0])  # [7,8,9] - [1,2,3] = [6,6,6]
        expected_right_pos_2 = torch.tensor([9.0, 9.0, 9.0])  # [10,11,12] - [1,2,3] = [9,9,9]
        
        self.assertTrue(torch.allclose(transformed_action[0, :3], expected_left_pos_1, atol=1e-6))
        self.assertTrue(torch.allclose(transformed_action[1, :3], expected_left_pos_2, atol=1e-6))
        self.assertTrue(torch.allclose(transformed_action[0, 3:6], expected_right_pos_1, atol=1e-6))
        self.assertTrue(torch.allclose(transformed_action[1, 3:6], expected_right_pos_2, atol=1e-6))

    def test_round_trip_consistency(self):
        """Test that round-trip conversions are consistent."""
        # Test 1: 6D -> matrix -> 6D should be consistent
        original_6d = generate_orthogonal_6d_rotation()
        matrix = rot_matrix_from_6drot(original_6d)
        converted_6d = rot_matrix_to_6drot(matrix)
        
        # Check if the conversion is close (within numerical precision)
        error = torch.norm(original_6d - converted_6d)
        self.assertLess(error.item(), 1e-6)
        
        # Test 2: matrix -> 6D -> matrix should be consistent
        original_matrix = torch.eye(3)
        rot_6d = rot_matrix_to_6drot(original_matrix)
        converted_matrix = rot_matrix_from_6drot(rot_6d)
        
        error = torch.norm(original_matrix - converted_matrix)
        self.assertLess(error.item(), 1e-6)

    def test_numerical_precision(self):
        """Test numerical precision of the transformations."""
        # Test 1: High precision rotation
        rot_6d = generate_orthogonal_6d_rotation()
        rot_matrix = rot_matrix_from_6drot(rot_6d)
        
        # Check that the rotation matrix is truly orthogonal
        identity = torch.eye(3)
        orthogonality_error = torch.norm(torch.matmul(rot_matrix, rot_matrix.T) - identity)
        self.assertLess(orthogonality_error.item(), 1e-6)  # Higher precision requirement
        
        # Test 2: Determinant should be exactly 1
        det = torch.det(rot_matrix)
        self.assertAlmostEqual(det.item(), 1.0, places=6)  # Higher precision requirement
        
        # Test 3: Round-trip conversion should be very precise
        converted_6d = rot_matrix_to_6drot(rot_matrix)
        error = torch.norm(rot_6d - converted_6d)
        self.assertLess(error.item(), 1e-6)  # Higher precision requirement

if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)
