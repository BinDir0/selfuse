"""
Mostly copied from transforms3d library

"""

import math

import numpy as np
import torch
import torch.nn.functional as F

_FLOAT_EPS = np.finfo(np.float64).eps

# axis sequences for Euler angles
_NEXT_AXIS = [1, 2, 0, 1]

# map axes strings to/from tuples of inner axis, parity, repetition, frame
_AXES2TUPLE = {
    "sxyz": (0, 0, 0, 0),
    "sxyx": (0, 0, 1, 0),
    "sxzy": (0, 1, 0, 0),
    "sxzx": (0, 1, 1, 0),
    "syzx": (1, 0, 0, 0),
    "syzy": (1, 0, 1, 0),
    "syxz": (1, 1, 0, 0),
    "syxy": (1, 1, 1, 0),
    "szxy": (2, 0, 0, 0),
    "szxz": (2, 0, 1, 0),
    "szyx": (2, 1, 0, 0),
    "szyz": (2, 1, 1, 0),
    "rzyx": (0, 0, 0, 1),
    "rxyx": (0, 0, 1, 1),
    "ryzx": (0, 1, 0, 1),
    "rxzx": (0, 1, 1, 1),
    "rxzy": (1, 0, 0, 1),
    "ryzy": (1, 0, 1, 1),
    "rzxy": (1, 1, 0, 1),
    "ryxy": (1, 1, 1, 1),
    "ryxz": (2, 0, 0, 1),
    "rzxz": (2, 0, 1, 1),
    "rxyz": (2, 1, 0, 1),
    "rzyz": (2, 1, 1, 1),
}

_TUPLE2AXES = dict((v, k) for k, v in _AXES2TUPLE.items())

# For testing whether a number is close to zero
_EPS4 = np.finfo(float).eps * 4.0


def mat2euler(mat, axes="sxyz"):
    """Return Euler angles from rotation matrix for specified axis sequence.

    Note that many Euler angle triplets can describe one matrix.

    Parameters
    ----------
    mat : array-like shape (3, 3) or (4, 4)
        Rotation matrix or affine.
    axes : str, optional
        Axis specification; one of 24 axis sequences as string or encoded
        tuple - e.g. ``sxyz`` (the default).

    Returns
    -------
    ai : float
        First rotation angle (according to `axes`).
    aj : float
        Second rotation angle (according to `axes`).
    ak : float
        Third rotation angle (according to `axes`).

    Examples
    --------
    >>> R0 = euler2mat(1, 2, 3, 'syxz')
    >>> al, be, ga = mat2euler(R0, 'syxz')
    >>> R1 = euler2mat(al, be, ga, 'syxz')
    >>> np.allclose(R0, R1)
    True
    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
    except (AttributeError, KeyError):
        _TUPLE2AXES[axes]  # validation
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i + parity]
    k = _NEXT_AXIS[i - parity + 1]

    M = np.array(mat, dtype=np.float64, copy=False)[:3, :3]
    if repetition:
        sy = math.sqrt(M[i, j] * M[i, j] + M[i, k] * M[i, k])
        if sy > _EPS4:
            ax = math.atan2(M[i, j], M[i, k])
            ay = math.atan2(sy, M[i, i])
            az = math.atan2(M[j, i], -M[k, i])
        else:
            ax = math.atan2(-M[j, k], M[j, j])
            ay = math.atan2(sy, M[i, i])
            az = 0.0
    else:
        cy = math.sqrt(M[i, i] * M[i, i] + M[j, i] * M[j, i])
        if cy > _EPS4:
            ax = math.atan2(M[k, j], M[k, k])
            ay = math.atan2(-M[k, i], cy)
            az = math.atan2(M[j, i], M[i, i])
        else:
            ax = math.atan2(-M[j, k], M[j, j])
            ay = math.atan2(-M[k, i], cy)
            az = 0.0

    if parity:
        ax, ay, az = -ax, -ay, -az
    if frame:
        ax, az = az, ax
    return ax, ay, az


def quat2mat(q):
    """Calculate rotation matrix corresponding to quaternion

    Parameters
    ----------
    q : 4 element array-like

    Returns
    -------
    M : (3,3) array
      Rotation matrix corresponding to input quaternion *q*

    Notes
    -----
    Rotation matrix applies to column vectors, and is applied to the
    left of coordinate vectors.  The algorithm here allows quaternions that
    have not been normalized.

    References
    ----------
    Algorithm from http://en.wikipedia.org/wiki/Rotation_matrix#Quaternion

    Examples
    --------
    >>> import numpy as np
    >>> M = quat2mat([1, 0, 0, 0]) # Identity quaternion
    >>> np.allclose(M, np.eye(3))
    True
    >>> M = quat2mat([0, 1, 0, 0]) # 180 degree rotn around axis 0
    >>> np.allclose(M, np.diag([1, -1, -1]))
    True
    """
    w, x, y, z = q
    Nq = w * w + x * x + y * y + z * z
    if Nq < _FLOAT_EPS:
        return np.eye(3)
    s = 2.0 / Nq
    X = x * s
    Y = y * s
    Z = z * s
    wX = w * X
    wY = w * Y
    wZ = w * Z
    xX = x * X
    xY = x * Y
    xZ = x * Z
    yY = y * Y
    yZ = y * Z
    zZ = z * Z
    return np.array(
        [
            [1.0 - (yY + zZ), xY - wZ, xZ + wY],
            [xY + wZ, 1.0 - (xX + zZ), yZ - wX],
            [xZ - wY, yZ + wX, 1.0 - (xX + yY)],
        ]
    )


# Checks if a matrix is a valid rotation matrix.
def isrotation(
    R: np.ndarray,
    thresh=1e-6,
) -> bool:
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    iden = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(iden - shouldBeIdentity)
    return n < thresh


def euler2mat(ai, aj, ak, axes="sxyz"):
    """Return rotation matrix from Euler angles and axis sequence.

    Parameters
    ----------
    ai : float
        First rotation angle (according to `axes`).
    aj : float
        Second rotation angle (according to `axes`).
    ak : float
        Third rotation angle (according to `axes`).
    axes : str, optional
        Axis specification; one of 24 axis sequences as string or encoded
        tuple - e.g. ``sxyz`` (the default).

    Returns
    -------
    mat : array (3, 3)
        Rotation matrix or affine.

    Examples
    --------
    >>> R = euler2mat(1, 2, 3, 'syxz')
    >>> np.allclose(np.sum(R[0]), -1.34786452)
    True
    >>> R = euler2mat(1, 2, 3, (0, 1, 0, 1))
    >>> np.allclose(np.sum(R[0]), -0.383436184)
    True
    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes]
    except (AttributeError, KeyError):
        _TUPLE2AXES[axes]  # validation
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i + parity]
    k = _NEXT_AXIS[i - parity + 1]

    if frame:
        ai, ak = ak, ai
    if parity:
        ai, aj, ak = -ai, -aj, -ak

    si, sj, sk = math.sin(ai), math.sin(aj), math.sin(ak)
    ci, cj, ck = math.cos(ai), math.cos(aj), math.cos(ak)
    cc, cs = ci * ck, ci * sk
    sc, ss = si * ck, si * sk

    M = np.eye(3)
    if repetition:
        M[i, i] = cj
        M[i, j] = sj * si
        M[i, k] = sj * ci
        M[j, i] = sj * sk
        M[j, j] = -cj * ss + cc
        M[j, k] = -cj * cs - sc
        M[k, i] = -sj * ck
        M[k, j] = cj * sc + cs
        M[k, k] = cj * cc - ss
    else:
        M[i, i] = cj * ck
        M[i, j] = sj * sc - cs
        M[i, k] = sj * cc + ss
        M[j, i] = cj * sk
        M[j, j] = sj * ss + cc
        M[j, k] = sj * cs - sc
        M[k, i] = -sj
        M[k, j] = cj * si
        M[k, k] = cj * ci
    return M


def euler2axangle(ai, aj, ak, axes="sxyz"):
    """Return angle, axis corresponding to Euler angles, axis specification

    Parameters
    ----------
    ai : float
        First rotation angle (according to `axes`).
    aj : float
        Second rotation angle (according to `axes`).
    ak : float
        Third rotation angle (according to `axes`).
    axes : str, optional
        Axis specification; one of 24 axis sequences as string or encoded
        tuple - e.g. ``sxyz`` (the default).

    Returns
    -------
    vector : array shape (3,)
       axis around which rotation occurs
    theta : scalar
       angle of rotation

    Examples
    --------
    >>> vec, theta = euler2axangle(0, 1.5, 0, 'szyx')
    >>> np.allclose(vec, [0, 1, 0])
    True
    >>> theta
    1.5
    """
    return quat2axangle(euler2quat(ai, aj, ak, axes))


def euler2quat(ai, aj, ak, axes="sxyz"):
    """Return `quaternion` from Euler angles and axis sequence `axes`

    Parameters
    ----------
    ai : float
        First rotation angle (according to `axes`).
    aj : float
        Second rotation angle (according to `axes`).
    ak : float
        Third rotation angle (according to `axes`).
    axes : str, optional
        Axis specification; one of 24 axis sequences as string or encoded
        tuple - e.g. ``sxyz`` (the default).

    Returns
    -------
    quat : array shape (4,)
       Quaternion in w, x, y z (real, then vector) format

    Examples
    --------
    >>> q = euler2quat(1, 2, 3, 'ryxz')
    >>> np.allclose(q, [0.435953, 0.310622, -0.718287, 0.444435])
    True
    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
    except (AttributeError, KeyError):
        _TUPLE2AXES[axes]  # validation
        firstaxis, parity, repetition, frame = axes

    i = firstaxis + 1
    j = _NEXT_AXIS[i + parity - 1] + 1
    k = _NEXT_AXIS[i - parity] + 1

    if frame:
        ai, ak = ak, ai
    if parity:
        aj = -aj

    ai = ai / 2.0
    aj = aj / 2.0
    ak = ak / 2.0
    ci = math.cos(ai)
    si = math.sin(ai)
    cj = math.cos(aj)
    sj = math.sin(aj)
    ck = math.cos(ak)
    sk = math.sin(ak)
    cc = ci * ck
    cs = ci * sk
    sc = si * ck
    ss = si * sk

    q = np.empty((4,))
    if repetition:
        q[0] = cj * (cc - ss)
        q[i] = cj * (cs + sc)
        q[j] = sj * (cc + ss)
        q[k] = sj * (cs - sc)
    else:
        q[0] = cj * cc + sj * ss
        q[i] = cj * sc - sj * cs
        q[j] = cj * ss + sj * cc
        q[k] = cj * cs - sj * sc
    if parity:
        q[j] *= -1.0

    return q


def quat2axangle(quat, identity_thresh=None):
    """Convert quaternion to rotation of angle around axis

    Parameters
    ----------
    quat : 4 element sequence
       w, x, y, z forming quaternion.
    identity_thresh : None or scalar, optional
       Threshold below which the norm of the vector part of the quaternion (x,
       y, z) is deemed to be 0, leading to the identity rotation.  None (the
       default) leads to a threshold estimated based on the precision of the
       input.

    Returns
    -------
    theta : scalar
       angle of rotation.
    vector : array shape (3,)
       axis around which rotation occurs.

    Examples
    --------
    >>> vec, theta = quat2axangle([0, 1, 0, 0])
    >>> vec
    array([1., 0., 0.])
    >>> np.allclose(theta, np.pi)
    True

    If this is an identity rotation, we return a zero angle and an arbitrary
    vector:

    >>> quat2axangle([1, 0, 0, 0])
    (array([1., 0., 0.]), 0.0)

    If any of the quaternion values are not finite, we return a NaN in the
    angle, and an arbitrary vector:

    >>> quat2axangle([1, np.inf, 0, 0])
    (array([1., 0., 0.]), nan)

    Notes
    -----
    A quaternion for which x, y, z are all equal to 0, is an identity rotation.
    In this case we return a 0 angle and an arbitrary vector, here [1, 0, 0].

    The algorithm allows for quaternions that have not been normalized.
    """
    quat = np.asarray(quat)
    Nq = np.sum(quat**2)
    if not np.isfinite(Nq):
        return np.array([1.0, 0, 0]), float("nan")
    if identity_thresh is None:
        try:
            identity_thresh = np.finfo(Nq.type).eps * 3
        except (AttributeError, ValueError):  # Not a numpy type or not float
            identity_thresh = _FLOAT_EPS * 3
    if Nq < _FLOAT_EPS**2:  # Results unreliable after normalization
        return np.array([1.0, 0, 0]), 0.0
    if Nq != 1:  # Normalize if not normalized
        s = math.sqrt(Nq)
        quat = quat / s
    xyz = quat[1:]
    len2 = np.sum(xyz**2)
    if len2 < identity_thresh**2:
        # if vec is nearly 0,0,0, this is an identity rotation
        return np.array([1.0, 0, 0]), 0.0
    # Make sure w is not slightly above 1 or below -1
    theta = 2 * math.acos(max(min(quat[0], 1), -1))
    return xyz / math.sqrt(len2), theta


def quat2euler(quaternion, axes="sxyz"):
    """Euler angles from `quaternion` for specified axis sequence `axes`

    Parameters
    ----------
    q : 4 element sequence
       w, x, y, z of quaternion
    axes : str, optional
        Axis specification; one of 24 axis sequences as string or encoded
        tuple - e.g. ``sxyz`` (the default).

    Returns
    -------
    ai : float
        First rotation angle (according to `axes`).
    aj : float
        Second rotation angle (according to `axes`).
    ak : float
        Third rotation angle (according to `axes`).

    Examples
    --------
    >>> angles = quat2euler([0.99810947, 0.06146124, 0, 0])
    >>> np.allclose(angles, [0.123, 0, 0])
    True
    """
    return mat2euler(quat2mat(quaternion), axes)


def rot_matrix_from_6drot(rot):
    '''
    Convert 6D rotation representation to 3x3 rotation matrix.
    
    The 6D representation uses two 3D vectors a and b, where:
    - The first vector a represents the first column of the rotation matrix
    - The second vector b represents the second column of the rotation matrix
    - The third column is computed as the cross product of a and b, then normalized
    
    Args:
        rot: torch.Tensor or np.ndarray, shape: [..., 6] or [6], where the first 3 elements are vector a,
             and the last 3 elements are vector b. Supports arbitrary dimensions.
    Returns:
        rot_matrix: torch.Tensor or np.ndarray, shape: [..., 3, 3] or [3, 3]
    '''
    if isinstance(rot, np.ndarray):
        is_numpy = True
        rot = torch.from_numpy(rot)
    else:
        is_numpy = False
    
    # Store original shape for later restoration
    original_shape = rot.shape
    
    # Handle single vector case (shape: [6])
    # Reshape to 2D for easier processing: [..., 6] -> [N, 6]
    rot = rot.reshape(-1, 6)
    
    # Extract the two 3D vectors
    a = rot[..., :3]  # First 3 elements: [N, 3]
    b = rot[..., 3:]  # Last 3 elements: [N, 3]
    
    # Schmidth orthogonalization
    a = F.normalize(a, dim=-1)
    b = b - torch.sum(a * b, dim=-1, keepdim=True) * a
    b = F.normalize(b, dim=-1)
    c = torch.cross(a, b, dim=-1)
    
    # Stack to form the rotation matrix
    rot_matrix = torch.stack([a, b, c], dim=-1)  # [N, 3, 3]
    
    # Reshape back to original dimensions if needed
    if len(original_shape) > 1:
        # Remove the last dimension (6) and add [3, 3] at the end
        new_shape = list(original_shape[:-1]) + [3, 3]
        rot_matrix = rot_matrix.reshape(*new_shape)
    else:
        rot_matrix = rot_matrix.squeeze(0)

    if is_numpy:
        rot_matrix = rot_matrix.numpy()
    
    return rot_matrix

def rot_matrix_to_6drot(rot_matrix):
    '''
    Convert 3x3 rotation matrix to 6D rotation representation.
    
    Args:
        rot_matrix: torch.Tensor or np.ndarray, shape: [..., 3, 3] or [3, 3]. Supports arbitrary dimensions.
    Returns:
        rot_6d: torch.Tensor or np.ndarray, shape: [..., 6] or [6]
    '''
    if isinstance(rot_matrix, np.ndarray):
        is_numpy = True
        rot_matrix = torch.from_numpy(rot_matrix)
    else:
        is_numpy = False
    
    # Store original shape for later restoration
    original_shape = rot_matrix.shape
    
    # Handle single rotation matrix case (shape: [3, 3])
    # Reshape to 3D for easier processing: [..., 3, 3] -> [N, 3, 3]
    rot_matrix = rot_matrix.reshape(-1, 3, 3)
    
    # Extract the first two columns
    a = rot_matrix[..., :, 0]  # First column: [N, 3]
    b = rot_matrix[..., :, 1]  # Second column: [N, 3]
    
    # Concatenate to form 6D representation
    rot_6d = torch.cat([a, b], dim=-1)  # [N, 6]
    
    # Reshape back to original dimensions if needed
    if len(original_shape) > 2:
        # Remove the last two dimensions (3, 3) and add [6] at the end
        new_shape = list(original_shape[:-2]) + [6]
        rot_6d = rot_6d.reshape(*new_shape)
    else:
        rot_6d = rot_6d.squeeze(0)
    
    if is_numpy:
        rot_6d = rot_6d.numpy()
    
    return rot_6d

def transform_to_target_frame(pose, target_extrinsic):
    '''
    Transform the pose to the target frame.
    Args:
        pose: torch.Tensor or np.ndarray, shape: [T, 4, 4] or [B, T, 4, 4] in world frame
        target_extrinsic: torch.Tensor or np.ndarray, shape: [4, 4] or [B, 4, 4]
        we assume the target_extrinsic is world2cam, and we want to transform the pose in the world frame to the camera frame
    Returns:
        pose: torch.Tensor, shape: [T, 4, 4] or [B, T, 4, 4] in target frame
    '''
    assert pose.dtype == target_extrinsic.dtype, "pose and target_extrinsic must have the same dtype"
    # print(f"pose.shape: {pose.shape}, target_extrinsic.shape: {target_extrinsic.shape}")
    if isinstance(pose, np.ndarray):
        is_numpy = True
        pose = torch.from_numpy(pose)
        target_extrinsic = torch.from_numpy(target_extrinsic)
    else:
        is_numpy = False
    target_extrinsic = target_extrinsic.unsqueeze(-3)

    '''
    # use pseudo-inverse to avoid NaN
    # for cam2world, we need to use the inverse of the target_extrinsic
    target_extrinsic_inv = torch.linalg.pinv(target_extrinsic)
    '''

    pose = torch.matmul(target_extrinsic, pose)
    
    # check if the result contains NaN and handle it
    if torch.isnan(pose).any():
        print(f"Warning: NaN detected in pose after transformation")
        pose = torch.where(torch.isnan(pose), torch.zeros_like(pose), pose)
    
    if is_numpy:
        pose = pose.numpy()
    return pose

# TODO: check whether the target_extrinsic is cam2world or world2cam
def transform_wrist_to_target_frame(wrist_action, target_extrinsic):
    '''
    Transform the wrist action to the target frame.
    Args:
        wrist_action: torch.Tensor or np.ndarray, shape: [T, 18] or [B, T, 18]
        target_extrinsic: torch.Tensor or np.ndarray, shape: [4, 4] or [B, 4, 4]
        we assume the target_extrinsic is world2cam, and we want to transform the wrist action in the world frame to the camera frame
    Returns:
        wrist_action: torch.Tensor, shape: [T, 18] or [B, T, 18]
    '''
    assert wrist_action.dtype == target_extrinsic.dtype, "wrist_action and target_extrinsic must have the same dtype"
    if isinstance(wrist_action, np.ndarray):
        is_numpy = True
        wrist_action = torch.from_numpy(wrist_action)
        target_extrinsic = torch.from_numpy(target_extrinsic)
    else:
        is_numpy = False

    T = wrist_action.shape[-2]

    # left wrist rotation is the first 6 elements, right wrist rotation is the last 6 elements
    wrist_rot_6d = torch.cat([wrist_action[..., 6:12], wrist_action[..., 12:18]], dim=-2)
    wrist_pose = torch.zeros(wrist_rot_6d.shape[:-1] + (4, 4))
    wrist_pose[..., :3, 3] = torch.cat([wrist_action[..., :3], wrist_action[..., 3:6]], dim=-2)
    wrist_pose[..., :3, :3] = rot_matrix_from_6drot(wrist_rot_6d)
    wrist_pose[..., 3, 3] = 1

    wrist_pose = transform_to_target_frame(wrist_pose, target_extrinsic)

    wrist_rot_6d = rot_matrix_to_6drot(wrist_pose[..., :3, :3])
    # print(wrist_action[..., :3].shape, wrist_pose[..., 0:T, :3, 3].shape, wrist_pose.shape)
    wrist_action[..., :3] = wrist_pose[..., 0:T, :3, 3]
    wrist_action[..., 3:6] = wrist_pose[..., T:2*T, :3, 3]
    wrist_action[..., 6:12] = wrist_rot_6d[..., 0:T, :]
    wrist_action[..., 12:18] = wrist_rot_6d[..., T:2*T, :]

    if is_numpy:
        wrist_action = wrist_action.numpy()
    
    return wrist_action
