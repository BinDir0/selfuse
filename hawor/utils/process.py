import torch
from lib.models.mano_wrapper import MANO
from hawor.utils.geometry import aa_to_rotmat
import numpy as np
import sys
import os
from pathlib import Path

def block_print():
    sys.stdout = open(os.devnull, 'w')

def enable_print():
    sys.stdout = sys.__stdout__


def _shared_mano_dir_candidates():
    share_data_root = Path("/share_data")
    if not share_data_root.exists():
        return []
    return sorted(share_data_root.glob("*/manopth/mano/models"))


def resolve_mano_model_dir(is_right=True):
    filename = "MANO_RIGHT.pkl" if is_right else "MANO_LEFT.pkl"
    legacy_dir = Path("_DATA/data/mano") if is_right else Path("_DATA/data_left/mano_left")

    candidates = []
    env_dir = os.environ.get("HAWOR_MANO_DIR")
    if env_dir:
        candidates.append(Path(env_dir).expanduser())
    candidates.extend(_shared_mano_dir_candidates())
    candidates.append(legacy_dir)

    seen = set()
    for candidate in candidates:
        candidate = candidate.resolve() if candidate.exists() else candidate
        candidate_key = str(candidate)
        if candidate_key in seen:
            continue
        seen.add(candidate_key)
        if (candidate / filename).exists():
            return candidate

    tried = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        f"Could not locate {filename}. Set HAWOR_MANO_DIR or place MANO files in one of: {tried}"
    )


def get_mano_cfg(is_right=True):
    model_dir = resolve_mano_model_dir(is_right=is_right)
    cfg = {
        "data_dir": str(model_dir),
        "model_path": str(model_dir),
        "gender": "neutral",
        "num_hand_joints": 15,
        "create_body_pose": False,
    }
    if not is_right:
        cfg["is_rhand"] = False
    return cfg

def get_mano_faces():
    block_print()
    mano_cfg = get_mano_cfg(is_right=True)
    mano = MANO(**mano_cfg)
    enable_print()
    return mano.faces


def run_mano(trans, root_orient, hand_pose, is_right=None, betas=None, use_cuda=True, mano_model=None):
    """
    Forward pass of the SMPL model and populates pred_data accordingly with
    joints3d, verts3d, points3d.

    trans : B x T x 3
    root_orient : B x T x 3
    body_pose : B x T x J*3
    betas : (optional) B x D
    mano_model : (optional) Pre-created MANO model to reuse
    """
    block_print()

    if mano_model is None:
        # Create new model if not provided (backward compatibility)
        mano_cfg = get_mano_cfg(is_right=True)
        mano = MANO(**mano_cfg)
        if use_cuda:
            mano = mano.cuda()
    else:
        # Reuse provided model
        mano = mano_model

    B, T, _ = root_orient.shape
    NUM_JOINTS = 15
    mano_params = {
        'global_orient': root_orient.reshape(B*T, -1),
        'hand_pose': hand_pose.reshape(B*T*NUM_JOINTS, 3),
        'betas': betas.reshape(B*T, -1),
    }
    rotmat_mano_params = mano_params
    rotmat_mano_params['global_orient'] = aa_to_rotmat(mano_params['global_orient']).view(B*T, 1, 3, 3)
    rotmat_mano_params['hand_pose'] = aa_to_rotmat(mano_params['hand_pose']).view(B*T, NUM_JOINTS, 3, 3)
    rotmat_mano_params['transl'] = trans.reshape(B*T, 3)

    if use_cuda:
        mano_output = mano(**{k: v.float().cuda() for k,v in rotmat_mano_params.items()}, pose2rot=False)
    else:
        mano_output = mano(**{k: v.float() for k,v in rotmat_mano_params.items()}, pose2rot=False)

    faces_right = mano.faces
    faces_new = np.array([[92, 38, 234],
                        [234, 38, 239],
                        [38, 122, 239],
                        [239, 122, 279],
                        [122, 118, 279],
                        [279, 118, 215],
                        [118, 117, 215],
                        [215, 117, 214],
                        [117, 119, 214],
                        [214, 119, 121],
                        [119, 120, 121],
                        [121, 120, 78],
                        [120, 108, 78],
                        [78, 108, 79]])
    faces_right = np.concatenate([faces_right, faces_new], axis=0)
    faces_n = len(faces_right)
    faces_left = faces_right[:,[0,2,1]]
    
    outputs = {
        "joints": mano_output.joints.reshape(B, T, -1, 3),
        "vertices": mano_output.vertices.reshape(B, T, -1, 3),
    }

    if not is_right is None:
        # outputs["vertices"][..., 0] = (2*is_right-1)*outputs["vertices"][..., 0]
        # outputs["joints"][..., 0] = (2*is_right-1)*outputs["joints"][..., 0]
        is_right = (is_right[:, :, 0].cpu().numpy() > 0)
        faces_result = np.zeros((B, T, faces_n, 3))
        faces_right_expanded = np.expand_dims(np.expand_dims(faces_right, axis=0), axis=0) 
        faces_left_expanded = np.expand_dims(np.expand_dims(faces_left, axis=0), axis=0) 
        faces_result = np.where(is_right[..., np.newaxis, np.newaxis], faces_right_expanded, faces_left_expanded)
        outputs["faces"] = torch.from_numpy(faces_result.astype(np.int32))


    enable_print()
    return outputs

def run_mano_left(trans, root_orient, hand_pose, is_right=None, betas=None, use_cuda=True, fix_shapedirs=True, mano_model=None):
    """
    Forward pass of the SMPL model and populates pred_data accordingly with
    joints3d, verts3d, points3d.

    trans : B x T x 3
    root_orient : B x T x 3
    body_pose : B x T x J*3
    betas : (optional) B x D
    mano_model : (optional) Pre-created MANO model to reuse
    """
    block_print()

    if mano_model is None:
        # Create new model if not provided (backward compatibility)
        mano_cfg = get_mano_cfg(is_right=False)
        mano = MANO(**mano_cfg)
        if use_cuda:
            mano = mano.cuda()

        # fix MANO shapedirs of the left hand bug (https://github.com/vchoutas/smplx/issues/48)
        if fix_shapedirs:
            mano.shapedirs[:, 0, :] *= -1
    else:
        # Reuse provided model
        mano = mano_model

    B, T, _ = root_orient.shape
    NUM_JOINTS = 15
    mano_params = {
        'global_orient': root_orient.reshape(B*T, -1),
        'hand_pose': hand_pose.reshape(B*T*NUM_JOINTS, 3),
        'betas': betas.reshape(B*T, -1),
    }
    rotmat_mano_params = mano_params
    rotmat_mano_params['global_orient'] = aa_to_rotmat(mano_params['global_orient']).view(B*T, 1, 3, 3)
    rotmat_mano_params['hand_pose'] = aa_to_rotmat(mano_params['hand_pose']).view(B*T, NUM_JOINTS, 3, 3)
    rotmat_mano_params['transl'] = trans.reshape(B*T, 3)

    if use_cuda:
        mano_output = mano(**{k: v.float().cuda() for k,v in rotmat_mano_params.items()}, pose2rot=False)
    else:
        mano_output = mano(**{k: v.float() for k,v in rotmat_mano_params.items()}, pose2rot=False)

    faces_right = mano.faces
    faces_new = np.array([[92, 38, 234],
                        [234, 38, 239],
                        [38, 122, 239],
                        [239, 122, 279],
                        [122, 118, 279],
                        [279, 118, 215],
                        [118, 117, 215],
                        [215, 117, 214],
                        [117, 119, 214],
                        [214, 119, 121],
                        [119, 120, 121],
                        [121, 120, 78],
                        [120, 108, 78],
                        [78, 108, 79]])
    faces_right = np.concatenate([faces_right, faces_new], axis=0)
    faces_n = len(faces_right)
    faces_left = faces_right[:,[0,2,1]]
    
    outputs = {
        "joints": mano_output.joints.reshape(B, T, -1, 3),
        "vertices": mano_output.vertices.reshape(B, T, -1, 3),
    }

    if not is_right is None:
        # outputs["vertices"][..., 0] = (2*is_right-1)*outputs["vertices"][..., 0]
        # outputs["joints"][..., 0] = (2*is_right-1)*outputs["joints"][..., 0]
        is_right = (is_right[:, :, 0].cpu().numpy() > 0)
        faces_result = np.zeros((B, T, faces_n, 3))
        faces_right_expanded = np.expand_dims(np.expand_dims(faces_right, axis=0), axis=0) 
        faces_left_expanded = np.expand_dims(np.expand_dims(faces_left, axis=0), axis=0) 
        faces_result = np.where(is_right[..., np.newaxis, np.newaxis], faces_right_expanded, faces_left_expanded)
        outputs["faces"] = torch.from_numpy(faces_result.astype(np.int32))


    enable_print()
    return outputs

def run_mano_twohands(init_trans, init_rot, init_hand_pose, is_right, init_betas, use_cuda=True, fix_shapedirs=True):
    outputs_left = run_mano_left(init_trans[0:1], init_rot[0:1], init_hand_pose[0:1], None, init_betas[0:1], use_cuda=use_cuda, fix_shapedirs=fix_shapedirs)
    outputs_right = run_mano(init_trans[1:2], init_rot[1:2], init_hand_pose[1:2], None, init_betas[1:2], use_cuda=use_cuda)
    outputs_two = {
        "vertices": torch.cat((outputs_left["vertices"], outputs_right["vertices"]), dim=0),
        "joints": torch.cat((outputs_left["joints"], outputs_right["joints"]), dim=0)

    }
    return outputs_two
