#!/usr/bin/env python3
"""
experiment_physics_consistency.py — Experiment 3 (Physical Consistency of Predicted Uncertainty)

Measures whether predicted uncertainty localizes where the prediction violates the
governing PDE, rather than merely correlating with error in aggregate.

For each test sample we compute
  - the pointwise predictive std  sigma_ij = sqrt(Var[u_ij])  from the NIG parameters, and
  - the pointwise PDE residual    r_ij     of the *predicted* field,
then report three statistics over all valid interior grid points:

  Pearson r          linear association between sigma and r
  Spearman rho       rank association (robust to the heavy-tailed residual distribution)
  Conc. ratio        mean(r | sigma in top 5%) / mean(r)
                     1.0 => uncertainty says nothing about physics violation
                     >1  => the model flags exactly where it breaks the equations

Outputs physics_consistency_results.csv, which populates Table V of the manuscript.

Usage
-----
  python experiment_physics_consistency.py \
      --data data/darcy.pt --pde-type darcy --epochs 20 --out exp_physics_darcy

  python experiment_physics_consistency.py \
      --data data/ns.pt --pde-type navier_stokes --dt 1.0 --nu 1e-3 \
      --epochs 20 --out exp_physics_ns

IMPORTANT (--dt): for Navier-Stokes the residual contains a time-derivative term
(omega_pred - omega_in)/dt. `dt` MUST match the snapshot spacing of the dataset or the
residual is mis-scaled and the correlation is meaningless. Verify it against the data
generator before trusting any Navier-Stokes number here.

NOTE: no trained checkpoints are saved by the golden_exp runs (save_every=50 > epochs=20),
so this script retrains each scheme from scratch using the same configuration as
main.py:train_evidential_method.

Verification of the residual operators (manufactured solutions):
  darcy_residual          u = sin(pi x) sin(pi y), constant and variable k
                          -> clean 2nd-order convergence, mean|res| 6.8e-3 (n=33)
                             -> 1.0e-4 (n=257), i.e. 4x reduction per grid doubling.
  navier_stokes_residual  steady Euler solution omega = sin(2pi x) sin(2pi y)
                          -> advection term u.grad(omega) = 6.7e-15 (machine precision);
                             recovered velocity div-free to 1.1e-14; curl(u) = +omega.

CAVEAT: the benchmark data are 128x128 (Li et al. FNO datasets: nsforcing_train_128.pt,
darcy_train_128.pt), NOT 64x64. The Darcy coefficient is stored as a binary {0,1} mask, so
the residual operator must map it back to the two true coefficient values before use, and the
Dirichlet boundary and constant forcing f=1 must match the generator; these are NOT yet
calibrated to give ~0 residual on ground truth. The NS input->output pair spans a fixed time
HORIZON (input vorticity vs. vorticity a fixed interval later), not a single small dt, so the
one-step transient residual here is only an approximation. Absolute residual magnitudes should
not be over-interpreted until this calibration is done.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from config import Config, Factory, get_evidential_methods_configs
from datautils import load_pde_data, PDEDataset
from fno import EvidentialFNO2d, EvidentialFNOTrainer
from main import get_regularization_function

from torch.utils.data import DataLoader


# Manuscript scheme names -> get_regularization_function keys
SCHEMES = {
    'Linear-Evidence': 'standard',
    'Log-Barrier': 'improved',
    'Inverse-Variance': 'uncertainty_aware',
    'Exp-Error': 'adaptive',
    'L2-Evidence': 'l2_evidence',
    'KL-Divergence': 'kl_divergence',
}


# --------------------------------------------------------------------------
# Data: replicate main.py's split exactly, but keep the raw arrays and the
# normalization statistics so predictions can be mapped back to physical units.
# --------------------------------------------------------------------------

def build_splits(data_path, pde_type, max_samples, batch_size,
                 train_ratio=0.85, val_ratio=0.05, seed=42):
    """Mirror of datautils.create_dataloaders_no_leakage, additionally returning
    the raw test arrays and the train-set normalization statistics."""
    X, U = load_pde_data(data_path, pde_type, max_samples=max_samples, return_raw=True)

    n_total = len(X)
    n_train = int(train_ratio * n_total)
    n_val = int(val_ratio * n_total)

    idx = np.random.RandomState(seed=seed).permutation(n_total)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]

    X_tr, U_tr = X[train_idx].copy(), U[train_idx].copy()
    X_va, U_va = X[val_idx].copy(), U[val_idx].copy()
    X_te, U_te = X[test_idx].copy(), U[test_idx].copy()

    # Train-only statistics (identical to create_dataloaders_no_leakage)
    x_min, x_max = X_tr[..., 0].min(), X_tr[..., 0].max()
    y_min, y_max = X_tr[..., 1].min(), X_tr[..., 1].max()
    fields_mean, fields_std = [], []
    for ch in range(2, X_tr.shape[-1]):
        fields_mean.append(X_tr[..., ch].mean())
        fields_std.append(X_tr[..., ch].std())
    u_mean, u_std = U_tr.mean(), U_tr.std()

    common = dict(normalize_output=True, normalize_input=True, normalize_coords=True,
                  precomputed_u_mean=u_mean, precomputed_u_std=u_std,
                  precomputed_x_min=x_min, precomputed_x_max=x_max,
                  precomputed_y_min=y_min, precomputed_y_max=y_max,
                  precomputed_x_fields_mean=fields_mean or None,
                  precomputed_x_fields_std=fields_std or None)

    train_ds = PDEDataset(X_tr, U_tr, **common)
    val_ds = PDEDataset(X_va, U_va, **common)
    test_ds = PDEDataset(X_te, U_te, **common)

    loaders = (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0),
    )
    stats = dict(u_mean=float(u_mean), u_std=float(u_std),
                 field_mean=float(fields_mean[0]), field_std=float(fields_std[0]))
    return loaders, (X_te, U_te), stats


# --------------------------------------------------------------------------
# PDE residuals — computed on the *physical* (denormalized) predicted field
# --------------------------------------------------------------------------

def darcy_residual(u, k, f=1.0):
    """|div(k grad u) + f| for -div(k grad u) = f on the unit square.

    u, k: (B, H, W) physical units, uniform grid from linspace(0,1,n).
    Returns (B, H, W) with a 1-cell NaN border (residual undefined on the boundary).
    """
    B, H, W = u.shape
    h = 1.0 / (H - 1)

    # Harmonic-free face averages: k_{i+1/2,j} = (k_i + k_{i+1}) / 2
    kx_p = 0.5 * (k[:, 1:, :] + k[:, :-1, :])      # (B, H-1, W) faces in x
    ky_p = 0.5 * (k[:, :, 1:] + k[:, :, :-1])      # (B, H, W-1) faces in y

    flux_x = kx_p * (u[:, 1:, :] - u[:, :-1, :]) / h     # (B, H-1, W)
    flux_y = ky_p * (u[:, :, 1:] - u[:, :, :-1]) / h     # (B, H, W-1)

    # Accumulate both directions; only interior cells receive both contributions,
    # and the 1-cell border is masked out below.
    div = np.zeros((B, H, W), dtype=np.float64)
    div[:, 1:-1, :] += (flux_x[:, 1:, :] - flux_x[:, :-1, :]) / h
    div[:, :, 1:-1] += (flux_y[:, :, 1:] - flux_y[:, :, :-1]) / h

    res = np.abs(div + f)
    res[:, 0, :] = res[:, -1, :] = np.nan
    res[:, :, 0] = res[:, :, -1] = np.nan
    return res


def _spectral_grads(field, n):
    """Return (d/dx, d/dy) of a periodic field via FFT. field: (B, n, n)."""
    kx = 2.0 * np.pi * np.fft.fftfreq(n, d=1.0 / n)
    KX, KY = np.meshgrid(kx, kx, indexing='ij')
    fh = np.fft.fft2(field, axes=(-2, -1))
    dx = np.real(np.fft.ifft2(1j * KX * fh, axes=(-2, -1)))
    dy = np.real(np.fft.ifft2(1j * KY * fh, axes=(-2, -1)))
    return dx, dy


def navier_stokes_residual(w_in, w_pred, dt, nu, f_amp=0.1):
    """|dw/dt + u.grad(w) - nu*lap(w) - f| for 2D vorticity transport on the torus.

    Velocity is recovered from the streamfunction (lap(psi) = -w) spectrally.
    Forcing follows the standard FNO benchmark: f = f_amp(sin(2pi(x+y)) + cos(2pi(x+y))).
    Set f_amp=0 for unforced (decaying) turbulence, e.g. the exp4 trajectory data.
    w_in, w_pred: (B, n, n) physical vorticity at t and t+dt.
    """
    B, n, _ = w_in.shape
    kx = 2.0 * np.pi * np.fft.fftfreq(n, d=1.0 / n)
    KX, KY = np.meshgrid(kx, kx, indexing='ij')
    K2 = KX ** 2 + KY ** 2
    K2_inv = K2.copy()
    K2_inv[0, 0] = 1.0        # guard the division only; K2 itself stays exact
                              # so the Laplacian keeps a zero (0,0) mode

    # Evaluate the transport terms at the midpoint (Crank-Nicolson style),
    # which is second-order accurate in dt rather than first.
    w_mid = 0.5 * (w_in + w_pred)

    wh = np.fft.fft2(w_mid, axes=(-2, -1))
    psih = wh / K2_inv
    psih[:, 0, 0] = 0.0                      # zero-mean streamfunction

    u = np.real(np.fft.ifft2(1j * KY * psih, axes=(-2, -1)))    #  d(psi)/dy
    v = np.real(np.fft.ifft2(-1j * KX * psih, axes=(-2, -1)))   # -d(psi)/dx

    dwdx, dwdy = _spectral_grads(w_mid, n)
    lap_w = np.real(np.fft.ifft2(-K2 * wh, axes=(-2, -1)))

    xs = np.arange(n) / n
    XX, YY = np.meshgrid(xs, xs, indexing='ij')
    f = f_amp * (np.sin(2 * np.pi * (XX + YY)) + np.cos(2 * np.pi * (XX + YY)))

    dwdt = (w_pred - w_in) / dt
    return np.abs(dwdt + u * dwdx + v * dwdy - nu * lap_w - f[None, ...])


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def physics_consistency(sigma, residual, top_frac=0.05, max_points=2_000_000, seed=0):
    """Pearson, Spearman and residual-concentration ratio between sigma and residual."""
    s = sigma.ravel()
    r = residual.ravel()
    ok = np.isfinite(s) & np.isfinite(r)
    s, r = s[ok], r[ok]
    if s.size == 0:
        return dict(pearson=np.nan, spearman=np.nan, conc_ratio=np.nan, n_points=0)

    pearson = float(np.corrcoef(s, r)[0, 1])

    rng = np.random.RandomState(seed)
    if s.size > max_points:
        sel = rng.choice(s.size, max_points, replace=False)
        spearman = float(spearmanr(s[sel], r[sel]).statistic)
    else:
        spearman = float(spearmanr(s, r).statistic)

    thresh = np.quantile(s, 1.0 - top_frac)
    top = r[s >= thresh]
    denom = r.mean()
    conc = float(top.mean() / denom) if denom > 0 else np.nan

    return dict(pearson=pearson, spearman=spearman, conc_ratio=conc, n_points=int(s.size))


# --------------------------------------------------------------------------
# Train + evaluate one scheme
# --------------------------------------------------------------------------

def run_scheme(scheme_label, reg_key, loaders, raw_test, stats, args):
    train_loader, val_loader, test_loader = loaders
    X_te_raw, _ = raw_test
    device = args.device

    mcfg = get_evidential_methods_configs()['der_nig']

    model = EvidentialFNO2d(
        modes1=12, modes2=12, width=32, n_layers=4, in_channels=3,
        nu_min=mcfg['nu_min'], alpha_min=mcfg['alpha_min'], beta_min=mcfg['beta_min'],
    ).to(device)

    optimizer = Factory.create_optimizer(
        {'type': 'adam', 'lr': args.lr, 'weight_decay': 0.0}, model.parameters())
    scheduler = Factory.create_scheduler({'type': 'exponential_lr', 'gamma': 0.93}, optimizer)

    trainer_config = Config({
        'device': device, 'lr': args.lr, 'weight_decay': 0.0,
        'enable_tracking': False, 'log_dir': str(Path(args.out) / 'logs'),
        'experiment_name': f'physics_{reg_key}',
        'checkpoint_dir': str(Path(args.out) / 'checkpoints' / reg_key),
        'save_every': 10 ** 6, 'use_tensorboard': False,
    })

    trainer = EvidentialFNOTrainer(
        model=model, config=trainer_config, method_name='der_nig',
        optimizer=optimizer, scheduler=scheduler,
        method_config=mcfg, reg_fn=get_regularization_function(reg_key),
    )
    trainer.train(train_loader, val_loader, epochs=args.epochs)

    # ---- collect predictions on the test set (order preserved: shuffle=False)
    model.eval()
    gammas, sigmas = [], []
    with torch.no_grad():
        for x, _ in test_loader:
            g, nu, al, be = model(x.to(device))
            var = (be / (al - 1.0)) * ((nu + 1.0) / nu)     # Eq. (9)
            gammas.append(g.squeeze(-1).cpu().numpy())
            sigmas.append(torch.sqrt(var).squeeze(-1).cpu().numpy())
    gamma = np.concatenate(gammas, axis=0).astype(np.float64)
    sigma = np.concatenate(sigmas, axis=0).astype(np.float64)

    # ---- back to physical units
    u_pred = gamma * stats['u_std'] + stats['u_mean']
    sigma_phys = sigma * stats['u_std']                     # scale only
    in_field = X_te_raw[..., 2].astype(np.float64)          # raw, never normalized

    if args.pde_type == 'darcy':
        residual = darcy_residual(u_pred, in_field, f=args.forcing)
    elif args.pde_type == 'navier_stokes':
        residual = navier_stokes_residual(in_field, u_pred, dt=args.dt, nu=args.nu,
                                          f_amp=args.ns_forcing)
    else:
        raise ValueError(f'no residual defined for pde_type={args.pde_type}')

    m = physics_consistency(sigma_phys, residual, top_frac=args.top_frac)
    m.update(scheme=scheme_label, reg_key=reg_key, pde=args.pde_type)
    print(f"  {scheme_label:18s}  pearson={m['pearson']:+.3f}  "
          f"spearman={m['spearman']:+.3f}  conc={m['conc_ratio']:.2f}")
    return m


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--pde-type', required=True, choices=['darcy', 'navier_stokes'])
    p.add_argument('--out', default='exp_physics')
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--batch-size', type=int, default=20)
    p.add_argument('--max-samples', type=int, default=8000)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--top-frac', type=float, default=0.05)
    p.add_argument('--forcing', type=float, default=1.0,
                   help='constant f for Darcy (standard benchmark uses f=1)')
    p.add_argument('--dt', type=float, default=1.0,
                   help='NS snapshot spacing — MUST match the dataset generator')
    p.add_argument('--nu', type=float, default=1e-3, help='NS viscosity')
    p.add_argument('--ns-forcing', type=float, default=0.0,
                   help='NS forcing amplitude f_amp; 0 for unforced exp4 trajectory data, '
                        '0.1 for the standard forced-turbulence benchmark')
    p.add_argument('--schemes', nargs='*', default=list(SCHEMES.keys()))
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    loaders, raw_test, stats = build_splits(
        args.data, args.pde_type, args.max_samples, args.batch_size, seed=args.seed)
    print(f"normalization stats: {stats}")

    rows = []
    for label in args.schemes:
        print(f"\n=== {label} ===")
        rows.append(run_scheme(label, SCHEMES[label], loaders, raw_test, stats, args))

    import pandas as pd
    df = pd.DataFrame(rows)[
        ['scheme', 'pde', 'pearson', 'spearman', 'conc_ratio', 'n_points']]
    csv_path = out / 'physics_consistency_results.csv'
    df.to_csv(csv_path, index=False)
    with open(out / 'physics_consistency_results.json', 'w') as fh:
        json.dump(rows, fh, indent=2)

    print(f"\n{df.to_string(index=False)}")
    print(f"\nwrote {csv_path}")


if __name__ == '__main__':
    main()
