import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
import math
from collections import Counter

log = logging.getLogger(__name__)

class GradientSolver:
    def __init__(self, dim, device='cuda'):
        self.dim = dim
        self.device = torch.device(device)
    
    def find_candidates(self, current_spheres, num_candidates=512, steps=200):
        m, d = current_spheres.shape
        candidates = torch.randn(num_candidates, d, device=self.device)
        candidates = candidates / torch.norm(candidates, dim=1, keepdim=True)
        candidates.requires_grad_(True)
        
        optimizer = optim.Adam([candidates], lr=0.05)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
        current_spheres_frozen = current_spheres.detach()

        for _ in range(steps):
            optimizer.zero_grad()
            cand_norm = candidates / (torch.norm(candidates, dim=1, keepdim=True) + 1e-8)
            cos_mat = torch.mm(cand_norm, current_spheres_frozen.T)
            
            # Loss 1: Overlap (Hard)
            overlap_violation = torch.relu(cos_mat - 0.5)
            loss_overlap = torch.sum(overlap_violation ** 2, dim=1)
            
            # Loss 2: Gravity (Weak)
            if m < self.dim * 2:
                topk_cos, _ = torch.topk(cos_mat, k=1, dim=1)
                loss_gravity = -torch.mean(topk_cos, dim=1) * 0.1 
            else:
                loss_gravity = 0.0
            
            # Combine
            loss_per_candidate = (loss_overlap * 100.0) + loss_gravity
            
            # === 修复点：无论如何都要取平均，变成标量 ===
            loss = loss_per_candidate.mean()
                
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            with torch.no_grad():
                candidates.data = candidates.data / torch.norm(candidates.data, dim=1, keepdim=True)

        with torch.no_grad():
            final_cands = candidates.detach()
            cos_final = torch.mm(final_cands, current_spheres_frozen.T)
            max_cos = torch.max(cos_final, dim=1)[0]
            valid_mask = max_cos <= 0.5 + 1e-4
            # 接触检查
            touching_mask = max_cos >= 0.5 - 0.1 
            final_mask = valid_mask & touching_mask
            return final_cands[final_mask]

# ... (PackingSimulator, rational_snap_strict, clean_cosine_set_smart, snap_matrix, find_max_volume_basis, get_cosine_set_for_dim 保持之前最终版不变) ...
# 为了方便，这里把后面的关键函数再贴一下，确保你复制完整

class PackingSimulator:
    def __init__(self, dim, max_spheres=200):
        self.dim = dim
        self.max_spheres = max_spheres
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.solver = GradientSolver(dim, device=self.device)
        s0 = torch.zeros(dim, dtype=torch.float32, device=self.device)
        s0[0] = 1.0
        s1 = torch.zeros(dim, dtype=torch.float32, device=self.device)
        s1[0] = 0.5
        s1[1] = math.sqrt(0.75)
        self.initial_spheres = torch.stack([s0, s1])
    def run_simulation(self, seed=None):
        if seed is not None: torch.manual_seed(seed)
        current_spheres = self.initial_spheres.clone()
        no_progress_count = 0
        while current_spheres.shape[0] < self.max_spheres:
            candidates = self.solver.find_candidates(current_spheres, num_candidates=1024, steps=150)
            if len(candidates) == 0:
                no_progress_count += 1
                if no_progress_count > 5: break
                continue
            no_progress_count = 0
            perm = torch.randperm(candidates.size(0))
            candidates = candidates[perm]
            temp_spheres = current_spheres
            added = 0
            for i in range(len(candidates)):
                cand = candidates[i].unsqueeze(0)
                cos_check = torch.mm(cand, temp_spheres.T)
                if torch.max(cos_check) <= 0.5 + 1e-4:
                    temp_spheres = torch.cat([temp_spheres, cand], dim=0)
                    added += 1
                    if added >= 10: break
            current_spheres = temp_spheres
        return current_spheres

def rational_snap_strict(val, tolerance=2e-2):
    if val > 0.5 + tolerance: return None 
    if abs(val - 0.5) < tolerance: return 0.5
    if abs(val + 0.5) < tolerance: return -0.5
    if abs(val) < tolerance: return 0.0
    if abs(val + 1.0) < tolerance: return -1.0
    known_constants = {
        -1/math.sqrt(2): -0.7071, 1/math.sqrt(2): 0.7071,
        -math.sqrt(3)/2: -0.8660, math.sqrt(3)/2: 0.8660
    }
    for k, v in known_constants.items():
        if abs(val - v) < tolerance:
            if k <= 0.5: return float(k)
    denominators = [2, 3, 4] 
    best_diff = float('inf'); best_frac = None
    for d in denominators:
        n = round(val * d)
        candidate = n / d
        if candidate > 0.5 + 1e-9: continue
        if candidate < -1.0 - 1e-9: continue
        diff = abs(val - candidate)
        if diff < tolerance and diff < best_diff:
            best_diff = diff; best_frac = candidate
    if best_frac is not None: return best_frac
    return None

def clean_cosine_set_smart(all_raw_cosines, total_spheres_accumulated):
    if isinstance(all_raw_cosines, list):
        if len(all_raw_cosines)>0 and isinstance(all_raw_cosines[0], torch.Tensor): data=torch.cat(all_raw_cosines).cpu().numpy()
        else: data=np.array(all_raw_cosines)
    elif isinstance(all_raw_cosines, torch.Tensor): data=all_raw_cosines.cpu().numpy()
    else: data=np.array(all_raw_cosines)
    rounded = np.round(data, decimals=3)
    counts = Counter(rounded)
    threshold = max(15, int(total_spheres_accumulated * 0.02)) 
    kept_values = set()
    for val, count in counts.most_common():
        if val > 0.51: continue
        if count >= threshold:
            snapped = rational_snap_strict(val, tolerance=3e-2)
            if snapped is not None: kept_values.add(snapped)
    kept_values.add(0.5); kept_values.add(0.0); kept_values.add(-1.0)
    return sorted(list(kept_values))

def snap_matrix(gram_mat):
    cleaned = gram_mat.clone()
    rows, cols = gram_mat.shape
    for i in range(rows):
        for j in range(cols):
            if i == j: 
                cleaned[i, j] = 1.0
                continue
            val = float(gram_mat[i, j])
            snapped = rational_snap_strict(val, tolerance=3e-2)
            if snapped is not None:
                cleaned[i, j] = snapped
            else:
                if abs(val) < 0.1: cleaned[i, j] = 0.0
                elif abs(val - 0.5) < 0.1: cleaned[i, j] = 0.5
                elif abs(val + 0.5) < 0.1: cleaned[i, j] = -0.5
    return cleaned

def find_max_volume_basis(spheres, dim):
    n = spheres.shape[0]
    if n < dim: return spheres @ spheres.T
    best_log_det = -float('inf')
    best_subset = None
    
    for _ in range(5000):
        indices = torch.randperm(n)[:dim]
        subset = spheres[indices]
        gram = subset @ subset.T
        try:
            sign, logdet = torch.linalg.slogdet(gram + torch.eye(dim, device=gram.device)*1e-6)
            if sign > 0:
                if logdet > best_log_det:
                    best_log_det = logdet
                    best_subset = subset
        except: continue
            
    if best_subset is not None:
        log.info(f"[Seed Search] Found Max-Volume Basis (logdet={best_log_det:.4f})")
        return best_subset @ best_subset.T
    else:
        return spheres[:dim] @ spheres[:dim].T

def get_cosine_set_for_dim(dim):
    NUM_SIMULATIONS = 10 
    MAX_SPHERES_PER_SIM = max(dim * 15, 120) 
    
    print(f"[Simulation] Starting DIVERSE Multi-Round Simulation for Dim {dim}...")
    all_raw_cosines_list = []
    total_spheres_count = 0
    all_spheres_pool = []
    
    for i in range(NUM_SIMULATIONS):
        sim = PackingSimulator(dim, max_spheres=MAX_SPHERES_PER_SIM)
        spheres = sim.run_simulation(seed=i*999 + 1)
        n = spheres.shape[0]
        total_spheres_count += n
        gram = spheres @ spheres.T
        triu_idx = torch.triu_indices(n, n, offset=1)
        raw_values = gram[triu_idx[0], triu_idx[1]].detach()
        all_raw_cosines_list.append(raw_values)
        all_spheres_pool.append(spheres)
        print(f"[Simulation] Round {i+1}: Found {n} spheres.")
        
    print(f"[Simulation] Filtering Cosine Set...")
    if not all_raw_cosines_list: return [-1.0, 0.0, 0.5], None
    all_raw_tensor = torch.cat(all_raw_cosines_list)
    final_set = clean_cosine_set_smart(all_raw_tensor, total_spheres_count)
    print(f"[Simulation] Final Cosine Set: {final_set}")

    print("[Simulation] Mining for MAX VOLUME Seed Basis...")
    best_seed_gram = None
    best_vol_score = -float('inf')
    for spheres in all_spheres_pool:
        seed_gram = find_max_volume_basis(spheres, dim)
        try:
            sign, logdet = torch.linalg.slogdet(seed_gram.double() + torch.eye(dim, device=seed_gram.device)*1e-6)
            if sign > 0 and logdet > best_vol_score:
                best_vol_score = logdet
                best_seed_gram = seed_gram.clone()
        except: pass
    
    if best_seed_gram is not None:
        best_seed_gram = snap_matrix(best_seed_gram)
            
    print(f"[Simulation] Best Seed Selected.\n")
    return final_set, best_seed_gram