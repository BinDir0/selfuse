import torchvision.transforms as tvf
from uniception.models.encoders.image_normalizations import IMAGE_NORMALIZATION_DICT
from any4d.utils.cropping import crop_resize_if_necessary
from PIL.ImageOps import exif_transpose
import torchvision.transforms as tvf
import torch
import numpy as np
from PIL import Image
from any4d.utils.inference import preprocess_input_views_for_inference
from scipy.spatial.transform import Rotation as R


# Fixed resolution mappings with precomputed aspect ratios as keys
RESOLUTION_MAPPINGS = {
    518: {
        1.000: (518, 518),  # 1:1
        1.321: (518, 392),  # 4:3
        1.542: (518, 336),  # 3:2
        1.762: (518, 294),  # 16:9
        2.056: (518, 252),  # 2:1
        3.083: (518, 168),  # 3.2:1
        0.757: (392, 518),  # 3:4
        0.649: (336, 518),  # 2:3
        0.567: (294, 518),  # 9:16
        0.486: (252, 518),  # 1:2
    },
    512: {
        1.000: (512, 512),  # 1:1
        1.333: (512, 384),  # 4:3
        1.524: (512, 336),  # 3:2
        1.778: (512, 288),  # 16:9
        2.000: (512, 256),  # 2:1
        3.200: (512, 160),  # 3.2:1
        0.750: (384, 512),  # 3:4
        0.656: (336, 512),  # 2:3
        0.562: (288, 512),  # 9:16
        0.500: (256, 512),  # 1:2
    },
}

# Precomputed sorted aspect ratio keys for efficient lookup
ASPECT_RATIO_KEYS = {
    518: sorted(RESOLUTION_MAPPINGS[518].keys()),
    512: sorted(RESOLUTION_MAPPINGS[512].keys()),
}

def find_closest_aspect_ratio(aspect_ratio, resolution_set):
    """
    Find the closest aspect ratio from the resolution mappings using efficient key lookup.

    Args:
        aspect_ratio (float): Target aspect ratio
        resolution_set (int): Resolution set to use (518 or 512)

    Returns:
        tuple: (target_width, target_height) from the resolution mapping
    """
    aspect_keys = ASPECT_RATIO_KEYS[resolution_set]

    # Find the closest aspect ratio key using binary search approach
    closest_key = min(aspect_keys, key=lambda x: abs(x - aspect_ratio))

    return RESOLUTION_MAPPINGS[resolution_set][closest_key]

def resize_intrinsics(K, old_size, new_size):
    """
    调整相机内参以适应新的图像尺寸。
    
    Args:
        K (np.array): 原始内参矩阵 3x3
        old_size (tuple): (Height, Width) 原始尺寸
        new_size (tuple): (Height, Width) 新尺寸
        
    Returns:
        np.array: 新的内参矩阵
    """
    old_h, old_w = old_size
    new_h, new_w = new_size
    
    # 1. 计算宽和高的缩放比例
    scale_x = new_w / old_w
    scale_y = new_h / old_h
    
    # 2. 复制一份 K，防止修改原数据
    K_new = K.copy()
    
    # 3. 更新焦距 (fx, fy)
    K_new[0, 0] *= scale_x  # fx
    K_new[1, 1] *= scale_y  # fy
    
    # 4. 更新光心 (cx, cy)
    K_new[0, 2] *= scale_x  # cx
    K_new[1, 2] *= scale_y  # cy
    
    return K_new

def load_images(video_images, # (T, H, W, 3)
    extrinsics, # (T, 4, 4)
    intrinsics, # (3, 3)
    resize_mode="fixed_mapping",
    norm_type="dinov2",
    resolution_set=518):

    H, W, _ = video_images[0].shape
    aspect_ratio = W / H
    # Determine target size for all images based on resize mode
    if resize_mode == "fixed_mapping":
        # Resolution mappings are already compatible with their respective patch sizes
        # 518 mappings are divisible by 14, 512 mappings are divisible by 16
        target_width, target_height = find_closest_aspect_ratio(
            aspect_ratio, resolution_set
        )
        target_size = (target_width, target_height)

    # Get the image normalization function based on the norm_type
    if norm_type in IMAGE_NORMALIZATION_DICT.keys():
        img_norm = IMAGE_NORMALIZATION_DICT[norm_type]
        ImgNorm = tvf.Compose(
            [tvf.ToTensor(), tvf.Normalize(mean=img_norm.mean, std=img_norm.std)]
        )
    else:
        raise ValueError(
            f"Unknown image normalization type: {norm_type}. Available options: {list(IMAGE_NORMALIZATION_DICT.keys())}"
        )

    imgs = []
    for id, image in enumerate(video_images):
        # binary_mask = np.ones((H, W), dtype=np.float32)
        # transform = tvf.Compose([tvf.ToTensor()])
        # input_moge_img = transform(image).unsqueeze(0)  # (1, 3, H, W)
        # moge_output = run_moge_inference(moge_model, input_moge_img, device="cuda")
        # non_ambiguous_mask = moge_output["mask"].squeeze(0).cpu().numpy().astype(np.float32)  # (H, W)
        # additional_quantities = [non_ambiguous_mask, binary_mask]
        # # img, additional_quantities = crop_resize_if_necessary(img, resolution=size, additional_quantities=additional_quantities)
        # img = exif_transpose(Image.fromarray(image)).convert("RGB")
        # img, additional_quantities = crop_resize_if_necessary(img, resolution=target_size, additional_quantities=additional_quantities)
        # non_ambiguous_mask = torch.tensor(additional_quantities[0]).bool()
        # binary_mask = torch.tensor(additional_quantities[1]).bool()

        additional_quantities = None
        img = exif_transpose(Image.fromarray(image)).convert("RGB")
        # img = crop_resize_if_necessary(img, resolution=size)[0]
        img = crop_resize_if_necessary(img, resolution=target_size)[0]
        non_ambiguous_mask = torch.tensor(np.ones_like(img)).bool()  # Default mask, all pixels are valid
        binary_mask = torch.tensor(np.ones_like(img)).bool()
        imgs.append(
            dict(
                img=ImgNorm(img)[None],
                intrinsics=torch.from_numpy(intrinsics)[None].float(),
                camera_poses=torch.from_numpy(np.linalg.inv(extrinsics[id]))[None].float(),
                true_shape=np.int32([img.size[::-1]]),
                idx=len(imgs),
                instance=str(len(imgs)),
                data_norm_type=[norm_type],
                non_ambiguous_mask=non_ambiguous_mask,
                binary_mask=binary_mask
            )
        )
    views = preprocess_input_views_for_inference(imgs)

    return views

def load_images_vitra(
    video_images, # (T, H, W, 3)
    resize_mode="fixed_mapping",
    norm_type="dinov2",
    resolution_set=518
):
    """
    VITRA 专用加载函数：无需内外参，手动堆叠 Batch
    """
    H, W, _ = video_images[0].shape
    aspect_ratio = W / H
    
    # 1. 计算目标分辨率
    if resize_mode == "fixed_mapping":
        target_width, target_height = find_closest_aspect_ratio(
            aspect_ratio, resolution_set
        )
        target_size = (target_width, target_height)

    # 2. 准备 Normalize
    if norm_type in IMAGE_NORMALIZATION_DICT.keys():
        img_norm = IMAGE_NORMALIZATION_DICT[norm_type]
        ImgNorm = tvf.Compose(
            [tvf.ToTensor(), tvf.Normalize(mean=img_norm.mean, std=img_norm.std)]
        )
    else:
        raise ValueError(f"Unknown norm type: {norm_type}")

    imgs = []
    # 3. 逐帧处理
    for id, image in enumerate(video_images):
        img = exif_transpose(Image.fromarray(image)).convert("RGB")
        
        # ==================== 核心修改 ====================
        # 原写法报错: img, _ = crop_resize_if_necessary(...) 
        # 改为直接取 [0]
        img = crop_resize_if_necessary(img, resolution=target_size)[0]
        # =================================================
        
        non_ambiguous_mask = torch.tensor(np.ones_like(img)).bool()  # Default mask, all pixels are valid
        binary_mask = torch.tensor(np.ones_like(img)).bool()
        
        imgs.append(
            dict(
                img=ImgNorm(img)[None],
                true_shape=np.int32([img.size[::-1]]),
                idx=len(imgs),
                instance=str(len(imgs)),
                data_norm_type=[norm_type],
                non_ambiguous_mask=non_ambiguous_mask,
                binary_mask=binary_mask
            )
        )
    
    return imgs

def load_images_holoassist(video_images, # (T, H, W, 3)
    intrinsics, # (3, 3)
    resize_mode="fixed_mapping",
    norm_type="dinov2",
    resolution_set=518):

    H, W, _ = video_images[0].shape
    aspect_ratio = W / H
    # Determine target size for all images based on resize mode
    if resize_mode == "fixed_mapping":
        # Resolution mappings are already compatible with their respective patch sizes
        # 518 mappings are divisible by 14, 512 mappings are divisible by 16
        target_width, target_height = find_closest_aspect_ratio(
            aspect_ratio, resolution_set
        )
        target_size = (target_width, target_height)

    # Get the image normalization function based on the norm_type
    if norm_type in IMAGE_NORMALIZATION_DICT.keys():
        img_norm = IMAGE_NORMALIZATION_DICT[norm_type]
        ImgNorm = tvf.Compose(
            [tvf.ToTensor(), tvf.Normalize(mean=img_norm.mean, std=img_norm.std)]
        )
    else:
        raise ValueError(
            f"Unknown image normalization type: {norm_type}. Available options: {list(IMAGE_NORMALIZATION_DICT.keys())}"
        )

    imgs = []
    for id, image in enumerate(video_images):
        img = exif_transpose(Image.fromarray(image)).convert("RGB")
        # img = crop_resize_if_necessary(img, resolution=size)[0]
        img = crop_resize_if_necessary(img, resolution=target_size)[0]
        non_ambiguous_mask = torch.tensor(np.ones_like(img)).bool()  # Default mask, all pixels are valid
        binary_mask = torch.tensor(np.ones_like(img)).bool()
        imgs.append(
            dict(
                img=ImgNorm(img)[None],
                intrinsics=torch.from_numpy(intrinsics)[None].float(),
                true_shape=np.int32([img.size[::-1]]),
                idx=len(imgs),
                instance=str(len(imgs)),
                data_norm_type=[norm_type],
                non_ambiguous_mask=non_ambiguous_mask,
                binary_mask=binary_mask
            )
        )
    views = preprocess_input_views_for_inference(imgs)

    return views

def quaternion_to_rotation_matrix(quats):
    """
    Args:
        quats: (N, 4) tensor, [x, y, z, w] or [w, x, y, z] depending on library.
        Any4D/PyTorch3D usually assumes [w, x, y, z] or [x, y, z, w].
        Let's assume standard [x, y, z, w] for scipy, or check model spec.
        Most MOGE/Any4D models output [x, y, z, w].
    """
    # Scipy expects [x, y, z, w]
    r = R.from_quat(quats.cpu().numpy())
    return r.as_matrix() # (N, 3, 3)

def compute_alignment_transform(pose_ref, pose_src):
    """
    计算将 pose_src 对齐到 pose_ref 的变换矩阵 T_align。
    使得 T_align @ pose_src \approx pose_ref
    
    Args:
        pose_ref: (N, 4, 4) numpy array, 基准 Chunk 的重叠部分 Pose (C2W)
        pose_src: (N, 4, 4) numpy array, 待对齐 Chunk 的重叠部分 Pose (C2W)
    Returns:
        T_align: (4, 4) 变换矩阵
    """
    assert pose_ref.shape == pose_src.shape
    num_frames = pose_ref.shape[0]
    
    # 关系: Pose_ref = T_align * Pose_src
    # => T_align = Pose_ref * inv(Pose_src)
    # 我们对每一帧都算一个 T_align，然后取中位数
    
    rel_translations = []
    rel_rotations = [] # 存旋转向量 (Rotation Vector) 以便计算中位数
    
    for i in range(num_frames):
        P_ref = pose_ref[i]
        P_src = pose_src[i]
        
        # T_i = P_ref @ inv(P_src)
        T_i = P_ref @ np.linalg.inv(P_src)
        
        rel_translations.append(T_i[:3, 3])
        
        # 提取旋转并转为旋转向量 (3,)
        rot_mat = T_i[:3, :3]
        r = R.from_matrix(rot_mat)
        rel_rotations.append(r.as_rotvec())
        
    rel_translations = np.array(rel_translations) # (N, 3)
    rel_rotations = np.array(rel_rotations)       # (N, 3)
    
    # 取中位数 (Median)
    # 这里的 axis=0 表示对 N 个样本取中值，得到 (3,)
    med_trans = np.median(rel_translations, axis=0)
    med_rotvec = np.median(rel_rotations, axis=0)
    
    # 构建最终变换矩阵
    T_align = np.eye(4)
    T_align[:3, 3] = med_trans
    T_align[:3, :3] = R.from_rotvec(med_rotvec).as_matrix()
    
    return T_align