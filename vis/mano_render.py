"""Render MANO hands on frame. Code extracted from zarr-web-viewer."""
import inspect
import os
import sys
import collections

import numpy as np
import cv2
import torch


# MANO 相关导入
MANO_AVAILABLE = False
MANO_LAYER = None
ManoLayer = None
_MANO_LAYERS_CACHE = None
_KNOWN_MANOPTH_PATHS = (
    "/home/guantianrui/manopth",
    "/home/guantianrui",
    "/share_data/lijiayi/EgoYOLO/manopth"
)
MANO_ROOT = os.environ.get("MANO_ROOT", "/share_data/lijiayi/EgoYOLO/manopth/mano/models")

LEFT_MANO_SLICE = slice(0, 45)
RIGHT_MANO_SLICE = slice(45, 90)
LEFT_TRANSLATION_SLICE = slice(0, 3)
RIGHT_TRANSLATION_SLICE = slice(3, 6)
LEFT_ROTATION_SLICE = slice(6, 12)
RIGHT_ROTATION_SLICE = slice(12, 18)
LEFT_SHAPE_SLICE = slice(0, 10)
RIGHT_SHAPE_SLICE = slice(10, 20)
LEFT_FINGERTIPS_SLICE = slice(0, 15)
RIGHT_FINGERTIPS_SLICE = slice(15, 30)

# 注入 inspect.getargspec 和 inspect.formatargspec 以兼容 Python 3.11+，因为chumpy过于老旧
# 注入 numpy的类型别名以兼容MANO
def inject_monkey_patch():
    """Inject inspect.getargspec/formatargspec for legacy deps on Python 3.11+."""
    if hasattr(inspect, "getargspec") and hasattr(inspect, "formatargspec"):
        return

    arg_spec_type = collections.namedtuple("ArgSpec", ["args", "varargs", "keywords", "defaults"])

    if not hasattr(inspect, "getargspec"):
        def getargspec(func):
            full_spec = inspect.getfullargspec(func)
            return arg_spec_type(
                args=full_spec.args,
                varargs=full_spec.varargs,
                keywords=full_spec.varkw,
                defaults=full_spec.defaults,
            )

        setattr(inspect, "getargspec", getargspec)

    if not hasattr(inspect, "formatargspec"):
        def formatargspec(
            args,
            varargs=None,
            varkw=None,
            defaults=None,
            kwonlyargs=(),
            kwonlydefaults=None,
            annotations=None,
            formatarg=str,
            formatvarargs=lambda name: "*" + name,
            formatvarkw=lambda name: "**" + name,
            formatvalue=lambda value: "=" + repr(value),
            formatreturns=lambda text: " -> " + text,
            formatannotation=None,
        ):
            del kwonlyargs, kwonlydefaults, annotations, formatreturns, formatannotation
            defaults = defaults or ()
            specs = []
            first_default = len(args) - len(defaults) if defaults else -1

            for idx, arg in enumerate(args):
                spec = formatarg(arg)
                if first_default >= 0 and idx >= first_default:
                    spec += formatvalue(defaults[idx - first_default])
                specs.append(spec)

            if varargs is not None:
                specs.append(formatvarargs(varargs))
            if varkw is not None:
                specs.append(formatvarkw(varkw))

            return "(" + ", ".join(specs) + ")"

        setattr(inspect, "formatargspec", formatargspec)

    if not hasattr(np, "int"):
        np.int = int
    
    if not hasattr(np, "float"):
        np.float = float
    
    if not hasattr(np, "bool"):
        np.bool = bool

    if not hasattr(np, "object"):
        np.object = object
    
    if not hasattr(np, "str"):
        np.str = str
    
    if not hasattr(np, "complex"):
        np.complex = complex

    if not hasattr(np, "unicode"):
        np.unicode = str

    if not hasattr(np, "long"):
        np.long = int

inject_monkey_patch()


def _append_existing_paths_to_sys_path(paths):
    for raw_path in paths:
        if not raw_path:
            continue
        path = str(raw_path)
        if path not in sys.path and os.path.exists(path):
            sys.path.insert(0, path)



try:
    _append_existing_paths_to_sys_path(_KNOWN_MANOPTH_PATHS)
    from manopth.manolayer import ManoLayer  # type: ignore

    MANO_AVAILABLE = True
    print("✓ MANO 模型已成功加载")
except ImportError:
    print("⚠️  MANO 未安装，将使用简化的手部渲染")
    MANO_AVAILABLE = False



def _instantiate_mano_layers(layer_cls, **layer_kwargs):
    left_layer = layer_cls(side="left", **layer_kwargs)
    right_layer = layer_cls(side="right", **layer_kwargs)
    return {
        "left": left_layer,
        "right": right_layer,
    }


def get_cached_mano_layers():
    """Lazily initialize and cache left/right MANO layers."""
    global _MANO_LAYERS_CACHE

    if _MANO_LAYERS_CACHE is not None:
        # print("✓ Using cached MANO layers")
        return _MANO_LAYERS_CACHE
    if not MANO_AVAILABLE or torch is None:
        print("Error: MANO is not available, cannot initialize layers")
        return None

    layer_cls = ManoLayer
    if layer_cls is None:
        print("Error: ManoLayer import is missing")
        return None

    mano_root = MANO_ROOT
    if not os.path.exists(mano_root):
        print(f"Error: MANO root does not exist: {mano_root}")
        return None

    try:
        _MANO_LAYERS_CACHE = {
            "left": layer_cls(
                mano_root=mano_root,
                use_pca=True,
                ncomps=45,
                flat_hand_mean=True,
                side="left",
                center_idx=0,
            ),
            "right": layer_cls(
                mano_root=mano_root,
                use_pca=True,
                ncomps=45,
                flat_hand_mean=True,
                side="right",
                center_idx=0,
            ),
        }
        print("MANO layers initialized")
        return _MANO_LAYERS_CACHE
    except Exception as e:
        print(f"Error occurred while initializing MANO layers: {e}")
        return None


def split_mano_params(mano_row):
    mano_row = np.asarray(mano_row, dtype=np.float32).reshape(-1)
    if mano_row.size < RIGHT_MANO_SLICE.stop:
        raise ValueError(f"Invalid mano row size: expected >=90 values, got {mano_row.size}")
    return {
        "left": mano_row[LEFT_MANO_SLICE],
        "right": mano_row[RIGHT_MANO_SLICE],
    }


def split_wrist_params(wrist_row):
    wrist_row = np.asarray(wrist_row, dtype=np.float32).reshape(-1)
    if wrist_row.size < RIGHT_ROTATION_SLICE.stop:
        raise ValueError(f"Invalid wrist row size: expected >=18 values, got {wrist_row.size}")
    return {
        "left_translation": wrist_row[LEFT_TRANSLATION_SLICE],
        "right_translation": wrist_row[RIGHT_TRANSLATION_SLICE],
        "left_rotation": wrist_row[LEFT_ROTATION_SLICE],
        "right_rotation": wrist_row[RIGHT_ROTATION_SLICE],
    }


def split_shape_params(shape_row):
    shape_row = np.asarray(shape_row, dtype=np.float32).reshape(-1)
    if shape_row.size < RIGHT_SHAPE_SLICE.stop:
        raise ValueError(f"Invalid shape row size: expected >=20 values, got {shape_row.size}")
    return {
        "left": shape_row[LEFT_SHAPE_SLICE],
        "right": shape_row[RIGHT_SHAPE_SLICE],
    }


def split_fingertips_params(fingertips_row):
    fingertips_row = np.asarray(fingertips_row, dtype=np.float32).reshape(-1)
    if fingertips_row.size < RIGHT_FINGERTIPS_SLICE.stop:
        raise ValueError(f"Invalid fingertips row size: expected >=30 values, got {fingertips_row.size}")
    return {
        "left": fingertips_row[LEFT_FINGERTIPS_SLICE],
        "right": fingertips_row[RIGHT_FINGERTIPS_SLICE],
    }

def render_hand_on_frame(frame, mano_params=None, wrist_params=None, extrinsic=None, 
                        intrinsic=None, presence=None, shape_params=None, mano_layers=None,
                        fingertips=None):
    """在帧上渲染手部动作（参考 visualize_episode_video.py 的实现）
    
    Args:
        frame: 原始图像帧（RGB 格式）
        mano_params: MANO参数 (dict with 'left' and 'right')
        wrist_params: 手腕参数 (dict with translations and rotations)
        extrinsic: 相机外参 (4x4矩阵)
        intrinsic: 相机内参 (3x3矩阵或4维向量)
        presence: 手的可见性 (0=无, 1=左手, 2=右手, 3=双手)
        shape_params: 形状参数 (dict with 'left' and 'right', 可选)
        mano_layers: MANO模型层 (dict with 'left' and 'right', 可选)
        fingertips: 指尖位置 (dict with 'left' and 'right', 每个是15维的3D坐标)
    
    Returns:
        渲染后的图像帧（RGB 格式，用于视频生成）
    """
    # 确保图像格式正确
    if frame.dtype == np.float32 or frame.dtype == np.float64:
        frame = (frame * 255).astype(np.uint8)
    elif frame.dtype != np.uint8:
        frame = frame.astype(np.uint8)
    
    # 转换为BGR格式（OpenCV绘图函数需要 BGR）
    if len(frame.shape) == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.shape[2] == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    elif frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
    
    H, W = frame.shape[:2]
    
    # 检查手的可见性
    L_present = True
    R_present = True
    if presence is not None:
        L_present = presence in [1, 3]
        R_present = presence in [2, 3]
    
    # 解析相机内参
    fx, fy, cx, cy = 384.0, 384.0, 192.0, 192.0
    if intrinsic is not None:
        parsed = parse_intrinsics(intrinsic)
        if parsed:
            fx, fy, cx, cy = parsed
    
    # 解析相机外参
    if extrinsic is not None:
        if extrinsic.shape == (16,):
            extr_mat = extrinsic.reshape(4, 4)
        else:
            extr_mat = extrinsic
        R_wc = extr_mat[:3, :3]
        t_wc = extr_mat[:3, 3]
    else:
        R_wc = np.eye(3, dtype=np.float32)
        t_wc = np.zeros(3, dtype=np.float32)
    
    # ========== 预计算：检测手部标注是否超出画面范围 ==========
    all_projected_points = []
    cached_mano_data = {}  # 缓存MANO计算结果以避免重复计算
    
    # 如果有MANO模型，收集所有需要渲染的点
    if MANO_AVAILABLE and mano_layers is not None and mano_params is not None and wrist_params is not None:
        # 计算左手的投影点
        if L_present and 'left' in mano_params and 'left_translation' in wrist_params:
            try:
                L_pose = mano_params['left']
                Lt3 = wrist_params['left_translation']
                L_rot6 = wrist_params['left_rotation']
                L_rotmat = rot6_to_rotmat(L_rot6)
                L_aa = rotmat_to_axisangle(L_rotmat)
                L_shape = shape_params.get('left') if shape_params else None
                
                L_verts_world, L_joints_world = generate_mano_mesh(
                    mano_layers['left'], L_pose, L_aa, Lt3, L_shape
                )
                
                if L_verts_world is not None:
                    # 缓存计算结果
                    cached_mano_data['left'] = {
                        'verts_world': L_verts_world,
                        'joints_world': L_joints_world
                    }
                    
                    verts_cam = (R_wc @ L_verts_world.T).T + t_wc
                    uv_verts = project_points(verts_cam, fx, fy, cx, cy)
                    # 只收集有效深度的点
                    valid_mask = verts_cam[:, 2] > 1e-6
                    all_projected_points.extend(uv_verts[valid_mask].tolist())
            except Exception as e:
                pass  # 忽略错误，继续处理
        
        # 计算右手的投影点
        if R_present and 'right' in mano_params and 'right_translation' in wrist_params:
            try:
                R_pose = mano_params['right']
                Rt3 = wrist_params['right_translation']
                R_rot6 = wrist_params['right_rotation']
                R_rotmat = rot6_to_rotmat(R_rot6)
                R_aa = rotmat_to_axisangle(R_rotmat)
                R_shape = shape_params.get('right') if shape_params else None
                
                R_verts_world, R_joints_world = generate_mano_mesh(
                    mano_layers['right'], R_pose, R_aa, Rt3, R_shape
                )
                
                if R_verts_world is not None:
                    # 缓存计算结果
                    cached_mano_data['right'] = {
                        'verts_world': R_verts_world,
                        'joints_world': R_joints_world
                    }
                    
                    verts_cam = (R_wc @ R_verts_world.T).T + t_wc
                    uv_verts = project_points(verts_cam, fx, fy, cx, cy)
                    valid_mask = verts_cam[:, 2] > 1e-6
                    all_projected_points.extend(uv_verts[valid_mask].tolist())
            except Exception as e:
                pass
    
    # 计算是否需要缩放画面
    scale_factor = 1.0
    offset_x = 0
    offset_y = 0
    new_W = W
    new_H = H
    img_overlay = frame.copy()
    
    if len(all_projected_points) > 0:
        pts = np.array(all_projected_points)
        min_x, min_y = pts.min(axis=0)
        max_x, max_y = pts.max(axis=0)
        
        # 检查是否超出边界（留10像素边距）
        margin = 10
        out_of_bounds = (min_x < margin or min_y < margin or 
                        max_x > W - margin or max_y > H - margin)
        
        if out_of_bounds:
            # 计算需要的画布尺寸
            required_w = max(max_x - min_x + 2 * margin, W)
            required_h = max(max_y - min_y + 2 * margin, H)
            
            # 计算缩放比例（保持原始画面宽高比）
            if np.isfinite(required_w) and np.isfinite(required_h) and required_w > 0 and required_h > 0:
                scale_w = W / required_w
                scale_h = H / required_h
                scale_factor = min(scale_w, scale_h, 0.85)  # 最多缩小到85%

                if np.isfinite(scale_factor) and scale_factor > 0:
                    # 目标尺寸至少为 1，避免 OpenCV resize 因 0 尺寸报错
                    scaled_w = max(1, int(round(W * scale_factor)))
                    scaled_h = max(1, int(round(H * scale_factor)))
                    new_W = W
                    new_H = H
                    offset_x = max(0, (new_W - scaled_w) // 2)
                    offset_y = max(0, (new_H - scaled_h) // 2)

                    # 缩放原始图像并放在新画布中央（使用深灰色背景以突出显示缩放效果）
                    scaled_frame = cv2.resize(frame, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)
                    # 创建深灰色背景 (40, 40, 40)
                    img_overlay = np.full((new_H, new_W, 3), 40, dtype=np.uint8)
                    img_overlay[offset_y:offset_y+scaled_h, offset_x:offset_x+scaled_w] = scaled_frame

                    # 调整相机内参以匹配缩放
                    fx = fx * scale_factor
                    fy = fy * scale_factor
                    cx = cx * scale_factor + offset_x
                    cy = cy * scale_factor + offset_y

                    # 更新画面尺寸
                    H, W = new_H, new_W
    
    # 如果有MANO模型且有必要参数，使用完整渲染
    if MANO_AVAILABLE and mano_layers is not None and mano_params is not None and wrist_params is not None:
        # 定义MANO关节树（手指连接关系）
        joint_tree = [
            [(0, 1), (1, 2), (2, 3), (3, 4)],      # 大拇指
            [(0, 5), (5, 6), (6, 7), (7, 8)],      # 食指
            [(0, 9), (9, 10), (10, 11), (11, 12)], # 中指
            [(0, 13), (13, 14), (14, 15), (15, 16)], # 无名指
            [(0, 17), (17, 18), (18, 19), (19, 20)]  # 小指
        ]
        
        # 创建半透明层用于点云混合
        overlay_alpha = img_overlay.copy()
        
        # 渲染左手
        if L_present and 'left' in mano_params and 'left_translation' in wrist_params:
            # 使用缓存的MANO数据（如果有），否则重新计算
            if 'left' in cached_mano_data:
                L_verts_world = cached_mano_data['left']['verts_world']
                L_joints_world = cached_mano_data['left']['joints_world']
            else:
                # 提取参数
                L_pose = mano_params['left']
                Lt3 = wrist_params['left_translation']
                L_rot6 = wrist_params['left_rotation']
                
                # 转换旋转
                L_rotmat = rot6_to_rotmat(L_rot6)
                L_aa = rotmat_to_axisangle(L_rotmat)
                
                # 获取形状参数
                L_shape = shape_params.get('left') if shape_params else None
                
                # 生成MANO网格和关节
                L_verts_world, L_joints_world = generate_mano_mesh(
                    mano_layers['left'], L_pose, L_aa, Lt3, L_shape
                )
            
            if L_verts_world is not None and L_joints_world is not None:
                # 转换到相机坐标系
                verts_cam = (R_wc @ L_verts_world.T).T + t_wc
                joints_cam = (R_wc @ L_joints_world.T).T + t_wc
                
                # 投影到图像平面
                uv_verts = project_points(verts_cam, fx, fy, cx, cy)
                uv_joints = project_points(joints_cam, fx, fy, cx, cy).astype(np.int32)
                
                # 绘制点云（小点，半透明）- 青色
                mask_verts = (verts_cam[:, 2] > 1e-6) & \
                            (uv_verts[:, 0] >= 0) & (uv_verts[:, 0] < W) & \
                            (uv_verts[:, 1] >= 0) & (uv_verts[:, 1] < H)
                for pt in uv_verts[mask_verts].astype(np.int32):
                    cv2.circle(overlay_alpha, tuple(pt), 1, (255, 255, 0), -1)
                
                # 应用半透明混合
                cv2.addWeighted(overlay_alpha, 0.6, img_overlay, 0.4, 0, img_overlay)
                
                # 绘制关节连线（骨架）- 亮蓝色
                mask_joints = (joints_cam[:, 2] > 1e-6) & \
                             (uv_joints[:, 0] >= 0) & (uv_joints[:, 0] < W) & \
                             (uv_joints[:, 1] >= 0) & (uv_joints[:, 1] < H)
                for finger_chain in joint_tree:
                    for (j1, j2) in finger_chain:
                        if mask_joints[j1] and mask_joints[j2]:
                            cv2.line(img_overlay, tuple(uv_joints[j1]), tuple(uv_joints[j2]), 
                                   (255, 100, 0), 2)
                
                # 绘制关节点 - 深蓝色
                for i in range(21):
                    if mask_joints[i]:
                        cv2.circle(img_overlay, tuple(uv_joints[i]), 2, (255, 0, 0), -1)
                        cv2.circle(img_overlay, tuple(uv_joints[i]), 3, (255, 255, 255), 1)
        
        # 渲染右手
        if R_present and 'right' in mano_params and 'right_translation' in wrist_params:
            # 使用缓存的MANO数据（如果有），否则重新计算
            if 'right' in cached_mano_data:
                R_verts_world = cached_mano_data['right']['verts_world']
                R_joints_world = cached_mano_data['right']['joints_world']
            else:
                # 提取参数
                R_pose = mano_params['right']
                Rt3 = wrist_params['right_translation']
                R_rot6 = wrist_params['right_rotation']
                
                # 转换旋转
                R_rotmat = rot6_to_rotmat(R_rot6)
                R_aa = rotmat_to_axisangle(R_rotmat)
                
                # 获取形状参数
                R_shape = shape_params.get('right') if shape_params else None
                
                # 生成MANO网格和关节
                R_verts_world, R_joints_world = generate_mano_mesh(
                    mano_layers['right'], R_pose, R_aa, Rt3, R_shape
                )
            
            if R_verts_world is not None and R_joints_world is not None:
                # 转换到相机坐标系
                verts_cam = (R_wc @ R_verts_world.T).T + t_wc
                joints_cam = (R_wc @ R_joints_world.T).T + t_wc
                
                # 投影到图像平面
                uv_verts = project_points(verts_cam, fx, fy, cx, cy)
                uv_joints = project_points(joints_cam, fx, fy, cx, cy).astype(np.int32)
                
                # 绘制点云（小点，半透明）- 黄色
                mask_verts = (verts_cam[:, 2] > 1e-6) & \
                            (uv_verts[:, 0] >= 0) & (uv_verts[:, 0] < W) & \
                            (uv_verts[:, 1] >= 0) & (uv_verts[:, 1] < H)
                overlay_alpha2 = img_overlay.copy()
                for pt in uv_verts[mask_verts].astype(np.int32):
                    cv2.circle(overlay_alpha2, tuple(pt), 1, (0, 255, 255), -1)
                
                # 应用半透明混合
                cv2.addWeighted(overlay_alpha2, 0.6, img_overlay, 0.4, 0, img_overlay)
                
                # 绘制关节连线（骨架）- 亮红色
                mask_joints = (joints_cam[:, 2] > 1e-6) & \
                             (uv_joints[:, 0] >= 0) & (uv_joints[:, 0] < W) & \
                             (uv_joints[:, 1] >= 0) & (uv_joints[:, 1] < H)
                for finger_chain in joint_tree:
                    for (j1, j2) in finger_chain:
                        if mask_joints[j1] and mask_joints[j2]:
                            cv2.line(img_overlay, tuple(uv_joints[j1]), tuple(uv_joints[j2]), 
                                   (0, 100, 255), 2)
                
                # 绘制关节点 - 深红色
                for i in range(21):
                    if mask_joints[i]:
                        cv2.circle(img_overlay, tuple(uv_joints[i]), 2, (0, 0, 255), -1)
                        cv2.circle(img_overlay, tuple(uv_joints[i]), 3, (255, 255, 255), 1)
    
    # === 额外渲染：数据集中的 fingertips（无论是否有MANO） ===
    if fingertips is not None:
        finger_names = ['拇指', '食指', '中指', '无名指', '小指']
        
        # 渲染左手指尖（使用不同的颜色和样式以区分）
        if L_present and 'left' in fingertips:
            left_tips = np.array(fingertips['left']).reshape(5, 3)
            tips_cam = (R_wc @ left_tips.T).T + t_wc
            uv_tips = project_points(tips_cam, fx, fy, cx, cy).astype(np.int32)
            mask_tips = (tips_cam[:, 2] > 1e-6) & \
                       (uv_tips[:, 0] >= 0) & (uv_tips[:, 0] < W) & \
                       (uv_tips[:, 1] >= 0) & (uv_tips[:, 1] < H)
            
            # 绘制指尖点 - 使用特殊样式（带边框的圆）
            for i in range(5):
                if mask_tips[i]:
                    pt = tuple(uv_tips[i])
                    # 缩小点的大小：外圈从7改为3，内圈从4改为2
                    cv2.circle(img_overlay, pt, 3, (0, 255, 0), 1)  # 外圈：绿色，线宽1
                    cv2.circle(img_overlay, pt, 2, (255, 255, 0), -1)  # 内圈：青色填充
                    # 标签：显示指尖名称（字体稍小）
                    cv2.putText(img_overlay, f"L{i}", (pt[0]+5, pt[1]-5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
        
        # 渲染右手指尖
        if R_present and 'right' in fingertips:
            right_tips = np.array(fingertips['right']).reshape(5, 3)
            tips_cam = (R_wc @ right_tips.T).T + t_wc
            uv_tips = project_points(tips_cam, fx, fy, cx, cy).astype(np.int32)
            mask_tips = (tips_cam[:, 2] > 1e-6) & \
                       (uv_tips[:, 0] >= 0) & (uv_tips[:, 0] < W) & \
                       (uv_tips[:, 1] >= 0) & (uv_tips[:, 1] < H)
            
            # 绘制指尖点 - 使用特殊样式（带边框的圆）
            for i in range(5):
                if mask_tips[i]:
                    pt = tuple(uv_tips[i])
                    # 缩小点的大小：外圈从7改为3，内圈从4改为2
                    cv2.circle(img_overlay, pt, 3, (255, 0, 255), 1)  # 外圈：洋红色，线宽1
                    cv2.circle(img_overlay, pt, 2, (0, 255, 255), -1)  # 内圈：黄色填充
                    # 标签：显示指尖名称（字体稍小）
                    cv2.putText(img_overlay, f"R{i}", (pt[0]+5, pt[1]-5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 255), 1)
    
    else:
        # Fallback: 简化渲染（只显示手腕位置）
        if wrist_params is not None and intrinsic is not None:
            # 绘制左手腕
            if L_present and 'left_translation' in wrist_params:
                wrist_pos = wrist_params['left_translation']
                if wrist_pos[2] > 0:
                    # 世界坐标转相机坐标
                    wrist_cam = R_wc @ wrist_pos + t_wc
                    if wrist_cam[2] > 0:
                        x = int(fx * wrist_cam[0] / wrist_cam[2] + cx)
                        y = int(fy * wrist_cam[1] / wrist_cam[2] + cy)
                        if 0 <= x < W and 0 <= y < H:
                            cv2.circle(img_overlay, (x, y), 8, (255, 255, 0), -1)
                            cv2.putText(img_overlay, "L", (x+10, y), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            
            # 绘制右手腕
            if R_present and 'right_translation' in wrist_params:
                wrist_pos = wrist_params['right_translation']
                if wrist_pos[2] > 0:
                    # 世界坐标转相机坐标
                    wrist_cam = R_wc @ wrist_pos + t_wc
                    if wrist_cam[2] > 0:
                        x = int(fx * wrist_cam[0] / wrist_cam[2] + cx)
                        y = int(fy * wrist_cam[1] / wrist_cam[2] + cy)
                        if 0 <= x < W and 0 <= y < H:
                            cv2.circle(img_overlay, (x, y), 8, (0, 255, 255), -1)
                            cv2.putText(img_overlay, "R", (x+10, y), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    
    # 转回 RGB 格式（用于视频生成）
    img_overlay = cv2.cvtColor(img_overlay, cv2.COLOR_BGR2RGB)
    return img_overlay




# ========== MANO 和渲染相关的辅助函数 ==========

def rot6_to_rotmat(r6: np.ndarray) -> np.ndarray:
    """将6D旋转表示转换为3x3旋转矩阵"""
    a1 = r6[:3]
    a2 = r6[3:6]
    b1 = a1 / (np.linalg.norm(a1) + 1e-8)
    a2 = a2 - np.dot(b1, a2) * b1
    b2 = a2 / (np.linalg.norm(a2) + 1e-8)
    b3 = np.cross(b1, b2)
    R = np.stack([b1, b2, b3], axis=1)
    return R.astype(np.float32)


def rotmat_to_axisangle(R: np.ndarray) -> np.ndarray:
    """将旋转矩阵转换为轴角表示"""
    R = R.astype(np.float32)
    cos_theta = (np.trace(R) - 1.0) / 2.0
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    theta = np.arccos(cos_theta)
    if theta < 1e-8:
        return np.zeros((3,), dtype=np.float32)
    rx = R[2, 1] - R[1, 2]
    ry = R[0, 2] - R[2, 0]
    rz = R[1, 0] - R[0, 1]
    axis = np.array([rx, ry, rz], dtype=np.float32)
    axis = axis / (np.linalg.norm(axis) + 1e-8)
    return (axis * theta).astype(np.float32)


def project_points(pts_cam: np.ndarray, fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """将3D点投影到2D图像平面"""
    zs = pts_cam[:, 2] + 1e-8
    us = fx * (pts_cam[:, 0] / zs) + cx
    vs = fy * (pts_cam[:, 1] / zs) + cy
    return np.stack([us, vs], axis=1)


def parse_intrinsics(intrinsics):
    """解析相机内参"""
    if intrinsics is None:
        return None
    intrinsics = np.asarray(intrinsics, dtype=np.float32)
    if intrinsics.shape == (3, 3):
        return intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]
    if intrinsics.shape == (9,):
        intrinsics = intrinsics.reshape(3, 3)
        return intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]
    if intrinsics.shape == (4,):
        return intrinsics[0], intrinsics[1], intrinsics[2], intrinsics[3]
    print(f"Warning: Unrecognized intrinsics shape {intrinsics.shape}, using default values")
    return None


def generate_mano_mesh(mano_layer, pose_params, global_r_aa, trans, shape_params=None):
    """使用MANO模型生成手部网格和关节
    
    Args:
        mano_layer: MANO layer instance
        pose_params: 手指姿态参数 (15 or 45 维)
        global_r_aa: 全局旋转（轴角表示）
        trans: 平移向量
        shape_params: 手部形状参数（10维，可选）
    
    Returns:
        verts_np: 顶点坐标 (778, 3)
        joints_np: 关节坐标 (21, 3)
    """
    if not MANO_AVAILABLE or mano_layer is None:
        print(mano_layer)
        return None, None
    
    # 构造完整的姿态参数
    if pose_params.shape[-1] == 15:
        theta = np.concatenate([global_r_aa.reshape(1, 3), pose_params.reshape(1, 15)], axis=1)
    else:
        theta = np.concatenate([global_r_aa.reshape(1, 3), pose_params.reshape(1, 45)], axis=1)
    theta_t = torch.from_numpy(theta).float()
    
    # 形状参数
    if shape_params is not None:
        beta_t = torch.from_numpy(shape_params.reshape(1, 10)).float()
    else:
        beta_t = torch.zeros((1, 10), dtype=torch.float32)
    
    # MANO 推理
    with torch.no_grad():
        verts, joints = mano_layer(theta_t, beta_t)
        # 转换单位并添加平移
        verts_np = verts.detach().cpu().numpy()[0] / 1000.0 + trans.reshape(1, 3)
        joints_np = joints.detach().cpu().numpy()[0] / 1000.0 + trans.reshape(1, 3)
    
    return verts_np.astype(np.float32), joints_np.astype(np.float32)
