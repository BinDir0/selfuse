import numpy as np
import cv2
import torch
from torchmin import minimize


def est_scale_iterative(slam_depth, pred_depth, iters=10, msk=None):
    """ Simple depth-align by iterative median and thresholding """
    s = pred_depth / slam_depth
    
    if msk is None:
        msk = np.zeros_like(pred_depth)
    else:
        msk = cv2.resize(msk, (pred_depth.shape[1], pred_depth.shape[0]))

    robust = (msk<0.5) * (0<pred_depth) * (pred_depth<10)
    s_est = s[robust]
    scale = np.median(s_est)
    scales_ = [scale]

    for _ in range(iters):
        slam_depth_0 = slam_depth * scale
        robust = (msk<0.5) * (0<slam_depth_0) * (slam_depth_0<10) * (0<pred_depth) * (pred_depth<10)
        s_est = s[robust]
        scale = np.median(s_est)
        scales_.append(scale)

    return scale


def est_scale_gmof(slam_depth, pred_depth, lr=1, sigma=0.5, iters=500, msk=None):
    """ Simple depth-align by robust least-square """
    if msk is None:
        msk = np.zeros_like(pred_depth)
    else:
        msk = cv2.resize(msk, (pred_depth.shape[1], pred_depth.shape[0]))

    robust = (msk<0.5) * (0<pred_depth) * (pred_depth<10)
    pm = torch.from_numpy(pred_depth[robust])
    sm = torch.from_numpy(slam_depth[robust])

    scale = torch.tensor([1.], requires_grad=True)
    optim = torch.optim.Adam([scale], lr=lr)
    losses = []
    for i in range(iters):
        loss = sm * scale - pm
        loss = gmof(loss, sigma=sigma).mean()

        optim.zero_grad()
        loss.backward()
        optim.step()
        losses.append(loss.item())

    scale = scale.detach().cpu().item()

    return scale

def est_offset(pred_depth, hand_depth, sigma=0.5, msk=None, 
                     far_thresh=10):
    """ Depth-align by iterative + robust least-square """
    if msk is None:
        msk = np.zeros_like(pred_depth)
    else:
        msk = cv2.resize(msk, (pred_depth.shape[1], pred_depth.shape[0]))

    # Stage 1: Iterative steps
    s = pred_depth - hand_depth

    robust = (msk<0.5) * (0<pred_depth) * (pred_depth<far_thresh)
    s_est = s[robust]
    offset = np.median(s_est)
    return offset

def est_scale_hybrid(slam_depth, pred_depth, sigma=0.5, msk=None, near_thresh=0,
                     far_thresh=10):
    """ Depth-align by iterative + robust least-square """
    if msk is None:
        msk = np.zeros_like(pred_depth)
    else:
        msk = cv2.resize(msk, (pred_depth.shape[1], pred_depth.shape[0]))

    # Stage 1: Iterative steps
    s = pred_depth / slam_depth

    robust = (msk<0.5) * (near_thresh<pred_depth) * (pred_depth<far_thresh)
    s_est = s[robust]
    scale = np.median(s_est)

    for _ in range(10):
        slam_depth_0 = slam_depth * scale
        robust = (msk<0.5) * (0<slam_depth_0) * (slam_depth_0<far_thresh) * (near_thresh<pred_depth) * (pred_depth<far_thresh)
        s_est = s[robust]
        scale = np.median(s_est)


    # Stage 2: Robust optimization on GPU
    robust = (msk<0.5) * (0<slam_depth_0) * (slam_depth_0<far_thresh) * (near_thresh<pred_depth) * (pred_depth<far_thresh)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pm = torch.from_numpy(pred_depth[robust]).to(device)
    sm = torch.from_numpy(slam_depth[robust]).to(device)

    def f(x):
        loss = sm * x - pm
        loss = gmof(loss, sigma=sigma).mean()
        return loss

    x0 = torch.tensor([scale], device=device)
    result = minimize(f, x0,  method='bfgs')
    scale = result.x.detach().cpu().item()

    return scale


def est_scale_wo_mask(slam_depth, pred_depth, sigma=0.5):
    """ Depth-align by iterative + robust least-square """
    msk=None
    near_thresh=0
    far_thresh=10000

    if msk is None:
        msk = np.zeros_like(pred_depth)
    else:
        msk = cv2.resize(msk, (pred_depth.shape[1], pred_depth.shape[0]))

    # Stage 1: Iterative steps
    s = pred_depth / slam_depth

    robust = (msk<0.5) * (near_thresh<pred_depth) * (pred_depth<far_thresh)
    s_est = s[robust]
    scale = np.median(s_est)

    for _ in range(10):
        slam_depth_0 = slam_depth * scale
        robust = (msk<0.5) * (0<slam_depth_0) * (slam_depth_0<far_thresh) * (near_thresh<pred_depth) * (pred_depth<far_thresh)
        s_est = s[robust]
        scale = np.median(s_est)


    # Stage 2: Robust optimization
    robust = (msk<0.5) * (0<slam_depth_0) * (slam_depth_0<far_thresh) * (near_thresh<pred_depth) * (pred_depth<far_thresh)
    pm = torch.from_numpy(pred_depth[robust])
    sm = torch.from_numpy(slam_depth[robust])

    def f(x):
        loss = sm * x - pm
        loss = gmof(loss, sigma=sigma).mean()
        return loss

    x0 = torch.tensor([scale])
    result = minimize(f, x0,  method='bfgs')
    scale = result.x.detach().cpu().item()

    return scale

def scale_shift_align(smpl_depth, pred_depth, sigma=0.5):
    """ Align pred_depth to smpl depth """
    smpl = torch.from_numpy(smpl_depth)
    pred = torch.from_numpy(pred_depth)

    def f(x):
        loss = smpl - (pred * x[0] + x[1])
        loss = gmof(loss, sigma=sigma).mean()
        return loss

    x0 = torch.tensor([1., 0.])
    result = minimize(f, x0,  method='bfgs')
    scale_shift = result.x.detach().cpu().numpy()

    return scale_shift


def shift_align(smpl_depth, pred_depth, sigma=0.5):
    """ Align pred_depth to smpl depth by only shift """
    smpl = torch.from_numpy(smpl_depth)
    pred = torch.from_numpy(pred_depth)

    def f(x):
        loss = smpl - (pred + x)
        loss = gmof(loss, sigma=sigma).mean()
        return loss

    x0 = torch.tensor([0.])
    result = minimize(f, x0,  method='bfgs')
    scale_shift = result.x.detach().cpu().numpy()

    return scale_shift


def est_scale_hybrid_batch(slam_depths, pred_depths, sigma=0.5, masks=None,
                           near_thresh=0.4, far_thresh=0.7):
    """Batch scale estimation: all keyframes in one vectorized GPU BFGS call.

    Same algorithm as est_scale_hybrid but avoids N separate minimize() calls.
    Stage 1 (iterative median) runs per-keyframe on CPU.
    Stage 2 (BFGS) runs once on GPU with all keyframes packed into padded tensors.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    N = len(slam_depths)

    # Stage 1: iterative median per keyframe (CPU, fast)
    init_scales = []
    sm_arrays = []
    pm_arrays = []
    for i in range(N):
        slam_depth = slam_depths[i]
        pred_depth = pred_depths[i]
        if masks is not None:
            msk = cv2.resize(masks[i], (pred_depth.shape[1], pred_depth.shape[0]))
        else:
            msk = np.zeros_like(pred_depth)

        s = pred_depth / slam_depth
        nt, ft = near_thresh, far_thresh
        robust = (msk < 0.5) * (nt < pred_depth) * (pred_depth < ft)
        s_est = s[robust]
        if s_est.size == 0:
            init_scales.append(float('nan'))
            sm_arrays.append(None)
            pm_arrays.append(None)
            continue
        scale = np.median(s_est)

        for _ in range(10):
            slam_depth_0 = slam_depth * scale
            robust = (msk < 0.5) * (0 < slam_depth_0) * (slam_depth_0 < ft) * (nt < pred_depth) * (pred_depth < ft)
            s_est = s[robust]
            if s_est.size == 0:
                break
            scale = np.median(s_est)

        robust = (msk < 0.5) * (0 < slam_depth_0) * (slam_depth_0 < ft) * (nt < pred_depth) * (pred_depth < ft)
        sm_filtered = slam_depth[robust]
        pm_filtered = pred_depth[robust]

        if sm_filtered.size == 0:
            init_scales.append(float('nan'))
            sm_arrays.append(None)
            pm_arrays.append(None)
        else:
            init_scales.append(scale)
            sm_arrays.append(sm_filtered)
            pm_arrays.append(pm_filtered)

    # Separate valid vs NaN keyframes
    valid_idx = [i for i in range(N) if not np.isnan(init_scales[i])]
    if len(valid_idx) == 0:
        return [float('nan')] * N

    # Stage 2: pack into padded tensors and run one BFGS
    max_len = max(sm_arrays[i].size for i in valid_idx)
    M = len(valid_idx)
    sm_padded = torch.zeros(M, max_len, device=device)
    pm_padded = torch.zeros(M, max_len, device=device)
    valid_mask = torch.zeros(M, max_len, device=device)
    counts = torch.zeros(M, device=device)

    for j, i in enumerate(valid_idx):
        L = sm_arrays[i].size
        sm_padded[j, :L] = torch.from_numpy(sm_arrays[i].ravel()).to(device)
        pm_padded[j, :L] = torch.from_numpy(pm_arrays[i].ravel()).to(device)
        valid_mask[j, :L] = 1.0
        counts[j] = L

    x0 = torch.tensor([init_scales[i] for i in valid_idx], device=device, dtype=torch.float32)

    sigma_sq = sigma ** 2

    def f_batched(x):
        residuals = sm_padded * x[:, None] - pm_padded  # (M, max_len)
        r_sq = residuals ** 2
        losses = (sigma_sq * r_sq) / (sigma_sq + r_sq)  # gmof inlined
        losses = (losses * valid_mask).sum(dim=1) / counts  # mean per keyframe
        return losses.sum()

    result = minimize(f_batched, x0, method='bfgs')
    opt_scales = result.x.detach().cpu().numpy()

    # Assemble final results
    scales = [float('nan')] * N
    for j, i in enumerate(valid_idx):
        scales[i] = float(opt_scales[j])

    return scales


def gmof(x, sigma=100):
    """
    Geman-McClure error function
    """
    x_squared =  x ** 2
    sigma_squared = sigma ** 2
    return (sigma_squared * x_squared) / (sigma_squared + x_squared)

