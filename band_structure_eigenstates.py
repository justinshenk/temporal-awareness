"""
Rigorous band structure: feed individual eigenstates through model,
extract eigenvalues via Rayleigh quotient at each k-point.

For eigenstate input ψ_n: H|ψ_n⟩ = ε_n|ψ_n⟩
So ε_n = Re(<ψ_n|H|ψ_n⟩) / <ψ_n|ψ_n⟩

Run 8 times (one per band) to get all eigenvalues.
"""
import torch, sys, numpy as np, h5py
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, '.')
from pwdft_fno.model_v2 import CrystalModelV2
from pwdft_fno.data import StateTensor
from pwdft_fno.util.shared import construct_grid
from torch.fft import fftn

device = 'cuda'
two_pi_i = 2 * torch.pi * 1j

ckpt = torch.load('checkpoints/online_phase_best.pt', map_location=device, weights_only=False)
model = CrystalModelV2(ckpt['config']['hyperparams']).to(device)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
print(f"Model: imp {(1-ckpt['val_mse']/ckpt['baseline_mse'])*100:.1f}%")

# High-symmetry path
HIGH_SYM = {
    'Gamma': np.array([0.0, 0.0, 0.0]),
    'X': np.array([0.5, 0.0, 0.5]),
    'W': np.array([0.5, 0.25, 0.75]),
    'L': np.array([0.5, 0.5, 0.5]),
    'K': np.array([0.375, 0.375, 0.75]),
}
for key in HIGH_SYM:
    HIGH_SYM[key] = HIGH_SYM[key] - np.floor(HIGH_SYM[key] + 0.5)

PATH = ['Gamma', 'X', 'W', 'L', 'Gamma', 'K']
N_PER_SEG = 100

def build_path(path_labels, n_per_seg):
    k_points, distances, labels_pos = [], [], []
    total_dist = 0
    for i in range(len(path_labels) - 1):
        start = HIGH_SYM[path_labels[i]]
        end = HIGH_SYM[path_labels[i+1]]
        seg_dist = np.linalg.norm(end - start)
        if i == 0: labels_pos.append(0.0)
        for j in range(n_per_seg):
            t = j / n_per_seg
            k_points.append(start + t * (end - start))
            distances.append(total_dist + t * seg_dist)
        total_dist += seg_dist
        labels_pos.append(total_dist)
    k_points.append(HIGH_SYM[path_labels[-1]])
    distances.append(total_dist)
    return np.array(k_points), np.array(distances), labels_pos

k_path, k_dist, label_positions = build_path(PATH, N_PER_SEG)
print(f"Path: {len(k_path)} k-points")

# Load eigenstates
print("Loading eigenstates...")
kpt_8 = 8; kpt_32 = 32; nbnd = 8; step = kpt_32 // kpt_8
N = 4  # for DFT R-grid

R_coords = torch.arange(-N//2, N//2, dtype=torch.float32, device=device)
Rx, Ry, Rz = torch.meshgrid(R_coords, R_coords, R_coords, indexing='ij')
R_grid = torch.stack([Rx.flatten(), Ry.flatten(), Rz.flatten()], dim=-1)

# Load 8^3 eigenstates for input, 32^3 for ground truth
eig_8 = np.zeros((nbnd, kpt_8, kpt_8, kpt_8, 25, 25, 25, 2), dtype=np.float32)
energy_8 = np.zeros((nbnd, kpt_8, kpt_8, kpt_8), dtype=np.float32)

with h5py.File('/work/nvme/bdiw/tdahiya/data/wfcr_data_8_phase.h5', 'r') as hdf:
    for i in range(kpt_8):
        for j in range(kpt_8):
            for k in range(kpt_8):
                idx = i * kpt_8 * kpt_8 + j * kpt_8 + k + 1
                eig_8[:, i, j, k] = hdf[f'point_{idx}'][:].astype(np.float32)
                energy_8[:, i, j, k] = hdf[f'energy_{idx}'][:].astype(np.float32)

# Ground truth eigenvalues from 32^3
print("Reading 32^3 ground truth...")
energies_gt = np.zeros((len(k_path), nbnd))
with h5py.File('/work/nvme/bdiw/tdahiya/data/wfcr_data_32_phase.h5', 'r') as hdf:
    for q, kf in enumerate(k_path):
        idx = np.round((kf + 0.5) * kpt_32).astype(int) % kpt_32
        point_idx = idx[0]*kpt_32*kpt_32 + idx[1]*kpt_32 + idx[2] + 1
        energies_gt[q] = hdf[f'energy_{point_idx}'][:]
for q in range(len(k_path)):
    energies_gt[q] = np.sort(energies_gt[q])

# 8^3 ground truth (snapped)
energies_8 = np.zeros((len(k_path), nbnd))
with h5py.File('/work/nvme/bdiw/tdahiya/data/wfcr_data_32_phase.h5', 'r') as hdf:
    for q, kf in enumerate(k_path):
        idx_8 = np.round((kf + 0.5) * kpt_8).astype(int) % kpt_8
        idx_32 = idx_8 * step
        point_idx = idx_32[0]*kpt_32*kpt_32 + idx_32[1]*kpt_32 + idx_32[2] + 1
        energies_8[q] = hdf[f'energy_{point_idx}'][:]
for q in range(len(k_path)):
    energies_8[q] = np.sort(energies_8[q])

# For each band, feed eigenstate as input and extract eigenvalues
# Input: set coefs so only band n is active (c_n=1, others=0) at all k-points
# ψ(k,r) = ψ_n(k,r), then H|ψ⟩ = ε_n(k) ψ_n(k,r)
k4 = construct_grid(4).unsqueeze(-2).unsqueeze(-2).unsqueeze(-2).to(device)
r_grid_np = np.zeros((25, 25, 25, 3), dtype=np.float32)
step_r = 1.0 / 25
for i in range(25):
    for j in range(25):
        for k in range(25):
            r_grid_np[i, j, k] = [i*step_r, j*step_r, k*step_r]
r_grid = torch.tensor(r_grid_np).unsqueeze(-5).unsqueeze(-5).unsqueeze(-5).to(device)

predicted_eigenvalues = np.zeros((len(k_path), nbnd))

for band in range(nbnd):
    print(f"\nBand {band}: feeding eigenstate through model...")
    
    # Create input: only band n active
    coefs = np.zeros((nbnd, kpt_8, kpt_8, kpt_8), dtype=np.float32)
    coefs[band] = 1.0  # c_n = 1 at all k-points
    # No need to normalize per k-point since only one coef is nonzero
    
    # Input wavefunction: ψ = ψ_n (just the eigenstate for this band)
    wf_ld = np.einsum('abcd,abcdefgh->bcdefgh', coefs, eig_8, optimize=True)
    
    # Target: H|ψ_n⟩ = ε_n ψ_n
    ham_target = np.einsum('abcdefgh,abcd,abcd->abcdefgh', eig_8, coefs, energy_8, optimize=True)
    
    wf_c = torch.complex(torch.tensor(wf_ld[...,0]), torch.tensor(wf_ld[...,1])).unsqueeze(0).unsqueeze(1).to(device)
    ham_c = torch.complex(torch.tensor(ham_target[...,0]), torch.tensor(ham_target[...,1])).to(device)
    
    inp = StateTensor(wf_c, k4, r_grid, has_phase=True)
    
    with torch.no_grad():
        x = model.fno(inp)
        x.resolve_operations()
        
        # FFT to Wannier space
        wannier = fftn(x.state, dim=(-6,-5,-4), norm="ortho")
        W_shifted = torch.fft.fftshift(wannier, dim=(-6,-5,-4))
        b, c, _, _, _, r1, r2, r3 = W_shifted.shape
        W = W_shifted.reshape(b, c, N**3, r1*r2*r3)
        
        # DFT at each path k-point
        batch_size = 20
        for start in range(0, len(k_path), batch_size):
            end = min(start + batch_size, len(k_path))
            k_batch = torch.tensor(k_path[start:end], device=device, dtype=torch.float32)
            
            phases = torch.exp(two_pi_i * ((k_batch + 0.5) @ R_grid.T)).to(W.dtype)
            result = torch.einsum('qr,bcrf->bcqf', phases, W) / (N**1.5)
            nq = end - start
            result = result.reshape(b, c, nq, 1, 1, r1, r2, r3)
            
            k_exp = k_batch.reshape(nq, 1, 1, 3).unsqueeze(-2).unsqueeze(-2).unsqueeze(-2)
            gauge_inp = StateTensor(result, k_exp, r_grid, has_phase=True)
            out = model.gauge_fno(gauge_inp)
            out.resolve_operations()
            pred = out.state.squeeze(0)  # (8, nq, 1, 1, 25, 25, 25)
            
            # Also get eigenstate at these k-points for Rayleigh quotient
            # For each query k, find nearest 32^3 eigenstate
            for qi in range(nq):
                kf = k_path[start + qi]
                idx_32 = np.round((kf + 0.5) * kpt_32).astype(int) % kpt_32
                
                # Get eigenstate ψ_n at this k-point from 32^3
                with h5py.File('/work/nvme/bdiw/tdahiya/data/wfcr_data_32_phase.h5', 'r') as hdf:
                    pidx = idx_32[0]*kpt_32*kpt_32 + idx_32[1]*kpt_32 + idx_32[2] + 1
                    eig_k = hdf[f'point_{pidx}'][:].astype(np.float32)  # (8, 25, 25, 25, 2)
                
                psi_n = torch.complex(
                    torch.tensor(eig_k[band, ..., 0]),
                    torch.tensor(eig_k[band, ..., 1])
                ).to(device)  # (25, 25, 25)
                
                # Model output for this k-point: pred[:, qi, 0, 0, :, :, :]  (8 bands, 25, 25, 25)
                H_psi_pred = pred[:, qi, 0, 0]  # (8, 25, 25, 25)
                
                # Sum over output bands to get total H|ψ⟩
                H_psi_total = H_psi_pred.sum(dim=0)  # (25, 25, 25)
                
                # Rayleigh quotient: ε = Re(<ψ_n|H|ψ_n⟩) / <ψ_n|ψ_n⟩
                N_r = 25**3
                numerator = (psi_n.conj() * H_psi_total).sum().real.item() / N_r
                denominator = (psi_n.conj() * psi_n).sum().real.item() / N_r
                eps_pred = numerator / denominator if denominator > 1e-10 else 0.0
                
                predicted_eigenvalues[start + qi, band] = eps_pred
        
        if band % 2 == 0:
            print(f"  Sample eigenvalues at Gamma: predicted={predicted_eigenvalues[0, band]:.4f}, "
                  f"ground truth={energies_gt[0, band]:.4f}")

# Sort predicted eigenvalues for clean plotting
for q in range(len(k_path)):
    predicted_eigenvalues[q] = np.sort(predicted_eigenvalues[q])

# Compute eigenvalue MAE
mae = np.abs(predicted_eigenvalues - energies_gt).mean()
max_err = np.abs(predicted_eigenvalues - energies_gt).max()
print(f"\nEigenvalue MAE: {mae:.4f} eV")
print(f"Eigenvalue max error: {max_err:.4f} eV")
print(f"Energy range: {energies_gt.min():.2f} to {energies_gt.max():.2f} eV")

# Plot
fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True)
path_labels_display = [r'$\Gamma$', 'X', 'W', 'L', r'$\Gamma$', 'K']

for ax, data, title, color in [
    (axes[0], energies_8, '8³ Ground Truth (coarse)', 'blue'),
    (axes[1], predicted_eigenvalues, f'Model Prediction (MAE={mae:.2f} eV)', 'green'),
    (axes[2], energies_gt, '32³ Ground Truth (reference)', 'red'),
]:
    for band_idx in range(nbnd):
        ax.plot(k_dist, data[:, band_idx], f'{color[0]}-', linewidth=1.0)
    for pos in label_positions:
        ax.axvline(x=pos, color='gray', linestyle='--', alpha=0.5)
    ax.set_xticks(label_positions)
    ax.set_xticklabels(path_labels_display)
    ax.set_title(title)
    ax.set_xlim(k_dist[0], k_dist[-1])

axes[0].set_ylabel('Energy (eV)')
plt.suptitle('Diamond Band Structure: 8³ → Model → 32³', fontsize=14)
plt.tight_layout()
plt.savefig('diamond_bandstructure_eigenvalues.png', dpi=150)
print("Saved: diamond_bandstructure_eigenvalues.png")

# Also save data
np.savez('diamond_bandstructure_eigenvalues.npz',
         k_dist=k_dist, predicted=predicted_eigenvalues,
         ground_truth_32=energies_gt, ground_truth_8=energies_8,
         label_positions=label_positions, mae=mae, max_err=max_err)
print("Saved: diamond_bandstructure_eigenvalues.npz")
