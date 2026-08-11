"""
Experiment (harder benchmark): does predicted uncertainty grow with autoregressive
rollout error?

Data: PDEArena NavierStokes-2D shards (HDF5), velocity fields (vx, vy) on 128x128,
14-step trajectories. We convert to vorticity omega = d(vy)/dx - d(vx)/dy and learn the
one-step map omega(t) -> omega(t+1) with an evidential FNO. At test time we roll the model
out autoregressively from omega_0 and, at each step k, record the relative L2 error against
the true trajectory and the mean predicted uncertainty. A well-calibrated model should have
uncertainty that *grows with* the accumulating rollout error.

Reported per scheme:
  final_err     : mean relative L2 error at the last rollout step
  err_growth    : error(last)/error(first)  -- how much error accumulates
  unc_growth    : uncertainty(last)/uncertainty(first)  -- does uncertainty accumulate too
  step_corr     : Pearson corr between the per-step error curve and uncertainty curve
                  (does uncertainty track error across the rollout)
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np
import torch
import h5py
from torch.utils.data import DataLoader

from config import Config, Factory, get_evidential_methods_configs
from datautils import PDEDataset
from fno import EvidentialFNO2d, EvidentialFNOTrainer
from main import get_regularization_function
from experiment_physics_consistency import SCHEMES


def _curl(vx, vy, n):
    """omega = d vy/dx - d vx/dy, spectral, on (..., n, n)."""
    kx = 2 * np.pi * np.fft.fftfreq(n, d=1.0 / n)
    KX, KY = np.meshgrid(kx, kx, indexing='ij')
    dvydx = np.real(np.fft.ifft2(1j * KX * np.fft.fft2(vy, axes=(-2, -1)), axes=(-2, -1)))
    dvxdy = np.real(np.fft.ifft2(1j * KY * np.fft.fft2(vx, axes=(-2, -1)), axes=(-2, -1)))
    return dvydx - dvxdy


def _shard_vorticity(path, group, max_traj=None):
    """Return vorticity trajectories (n_traj, T, n, n) from one PDEArena shard."""
    with h5py.File(path, 'r') as h:
        g = h[group]
        vx = np.asarray(g['vx'][:max_traj]); vy = np.asarray(g['vy'][:max_traj])
    n = vx.shape[-1]
    return _curl(vx, vy, n).astype(np.float32)


def build_data(shard_dir, batch_size, n_test_traj=50, seed=42):
    """Training pairs (omega_t -> omega_{t+1}) from train shards + full test trajectories."""
    tr_files = sorted(glob.glob(f'{shard_dir}/NavierStokes2D_train_*.h5'))
    te_files = sorted(glob.glob(f'{shard_dir}/NavierStokes2D_test_*.h5'))
    if not tr_files or not te_files:
        raise FileNotFoundError(f'no PDEArena shards under {shard_dir}')

    Xs, Us = [], []
    for f in tr_files:
        w = _shard_vorticity(f, 'train')          # (ntraj, T, n, n)
        ntraj, T, n, _ = w.shape
        xs = np.linspace(0, 1, n, endpoint=False, dtype=np.float32)
        GX, GY = np.meshgrid(xs, xs, indexing='ij')
        for t in range(T - 1):
            X = np.zeros((ntraj, n, n, 3), dtype=np.float32)
            X[..., 0] = GX; X[..., 1] = GY; X[..., 2] = w[:, t]
            Xs.append(X); Us.append(w[:, t + 1][..., None])
    X = np.concatenate(Xs); U = np.concatenate(Us)
    del Xs, Us

    # single vorticity statistic -> consistent autoregressive round-trip
    wmean, wstd = float(X[..., 2].mean()), float(X[..., 2].std())
    idx = np.random.RandomState(seed).permutation(len(X))
    n_tr = int(0.9 * len(X))
    common = dict(normalize_output=True, normalize_input=True, normalize_coords=True,
                  precomputed_u_mean=wmean, precomputed_u_std=wstd,
                  precomputed_x_min=0.0, precomputed_x_max=1.0,
                  precomputed_y_min=0.0, precomputed_y_max=1.0,
                  precomputed_x_fields_mean=[wmean], precomputed_x_fields_std=[wstd])
    tr_loader = DataLoader(PDEDataset(X[idx[:n_tr]], U[idx[:n_tr]], **common),
                           batch_size=batch_size, shuffle=True, num_workers=0)
    va_loader = DataLoader(PDEDataset(X[idx[n_tr:]], U[idx[n_tr:]], **common),
                           batch_size=batch_size, shuffle=False, num_workers=0)

    w_te = _shard_vorticity(te_files[0], 'test', max_traj=n_test_traj)   # (n_test, T, n, n)
    return (tr_loader, va_loader), w_te, dict(wmean=wmean, wstd=wstd)


def rollout(model, w_te, stats, device):
    """Autoregressive rollout from omega_0; per-step relative error and mean uncertainty."""
    wmean, wstd = stats['wmean'], stats['wstd']
    B, T, n, _ = w_te.shape
    xs = np.linspace(0, 1, n, endpoint=False, dtype=np.float32)
    GX, GY = np.meshgrid(xs, xs, indexing='ij')
    cx = torch.from_numpy((2 * GX - 1).astype(np.float32)).to(device)
    cy = torch.from_numpy((2 * GY - 1).astype(np.float32)).to(device)
    w_hat = torch.from_numpy(w_te[:, 0].copy()).to(device)              # (B,n,n) physical

    # radial-wavenumber mask for the high-frequency energy fraction (|k| > n/4)
    kk = torch.fft.fftfreq(n, d=1.0 / n).to(device)
    KX2, KY2 = torch.meshgrid(kk, kk, indexing='ij')
    highk_mask = (torch.sqrt(KX2 ** 2 + KY2 ** 2) > n / 4.0)

    def roughness(field):
        gx = field[:, 1:, :] - field[:, :-1, :]
        gy = field[:, :, 1:] - field[:, :, :-1]
        return 0.5 * (gx.abs().mean() + gy.abs().mean())

    def highk_frac(field):
        P = (torch.fft.fft2(field).abs() ** 2)
        return (P[:, highk_mask].sum() / (P.sum() + 1e-12))

    errs, uncs, roughs, highks = [], [], [], []
    model.eval()
    with torch.no_grad():
        for k in range(1, T):
            # w_hat is the INPUT field at this step; sigma reflects it, so measure its
            # roughness/spectrum here (before the update) to align with sigma.
            roughs.append(roughness(w_hat).item())
            highks.append(highk_frac(w_hat).item())
            inp = torch.stack([cx.expand(B, n, n),
                               cy.expand(B, n, n),
                               (w_hat - wmean) / wstd], dim=-1)          # (B,n,n,3)
            g, nu, al, be = model(inp)
            var = (be / (al - 1.0)) * ((nu + 1.0) / nu)
            w_hat = g.squeeze(-1) * wstd + wmean
            sigma = torch.sqrt(var).squeeze(-1) * wstd
            w_true = torch.from_numpy(w_te[:, k].copy()).to(device)
            num = torch.linalg.norm((w_hat - w_true).reshape(B, -1), dim=1)
            den = torch.linalg.norm(w_true.reshape(B, -1), dim=1)
            errs.append((num / den).mean().item())
            uncs.append(sigma.mean().item())
    return (np.array(errs), np.array(uncs), np.array(roughs), np.array(highks))


def run_scheme(label, reg_key, loaders, w_te, stats, args):
    tr_loader, va_loader = loaders
    device = args.device
    mcfg = get_evidential_methods_configs()['der_nig']
    model = EvidentialFNO2d(modes1=12, modes2=12, width=32, n_layers=4, in_channels=3,
                            nu_min=mcfg['nu_min'], alpha_min=mcfg['alpha_min'],
                            beta_min=mcfg['beta_min']).to(device)
    opt = Factory.create_optimizer({'type': 'adam', 'lr': args.lr, 'weight_decay': 0.0}, model.parameters())
    sched = Factory.create_scheduler({'type': 'exponential_lr', 'gamma': 0.93}, opt)
    tcfg = Config({'device': device, 'lr': args.lr, 'weight_decay': 0.0, 'enable_tracking': False,
                   'experiment_name': f'rollout_{reg_key}', 'use_tensorboard': False,
                   'checkpoint_dir': str(Path(args.out) / 'ckpt' / reg_key), 'save_every': 10 ** 6})
    trainer = EvidentialFNOTrainer(model=model, config=tcfg, method_name='der_nig', optimizer=opt,
                                   scheduler=sched, method_config=mcfg,
                                   reg_fn=get_regularization_function(reg_key))
    trainer.train(tr_loader, va_loader, epochs=args.epochs)

    errs, uncs, roughs, highks = rollout(model, w_te, stats, device)
    step_corr = float(np.corrcoef(errs, uncs)[0, 1])
    corr_sig_rough = float(np.corrcoef(uncs, roughs)[0, 1])    # sigma tracks input roughness?
    corr_rough_err = float(np.corrcoef(roughs, errs)[0, 1])    # field smooths as error grows?
    m = dict(scheme=label, reg_key=reg_key, pde='ns_rollout',
             final_err=float(errs[-1]), err_growth=float(errs[-1] / (errs[0] + 1e-9)),
             unc_growth=float(uncs[-1] / (uncs[0] + 1e-9)), step_corr=step_corr,
             corr_sig_rough=corr_sig_rough, corr_rough_err=corr_rough_err,
             rough_growth=float(roughs[-1] / (roughs[0] + 1e-9)),
             highk_growth=float(highks[-1] / (highks[0] + 1e-9)),
             err_curve=errs.tolist(), unc_curve=uncs.tolist(),
             rough_curve=roughs.tolist(), highk_curve=highks.tolist())
    print(f"  {label:18s}  step_corr={step_corr:+.3f}  "
          f"corr(sig,rough)={corr_sig_rough:+.3f}  corr(rough,err)={corr_rough_err:+.3f}  "
          f"rough_growth={m['rough_growth']:.2f}x", flush=True)
    return m


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--shard-dir', default='/media/arifadh/CRUCIAL2T/pdehard/pdearena')
    p.add_argument('--out', default='exp_rollout')
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--n-test-traj', type=int, default=50)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--schemes', nargs='*', default=list(SCHEMES.keys()))
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    loaders, w_te, stats = build_data(args.shard_dir, args.batch_size, args.n_test_traj, args.seed)
    print(f"train pairs loaded; test trajectories {w_te.shape}; stats {stats}", flush=True)

    rows = []
    for label in args.schemes:
        print(f"\n=== {label} ===", flush=True)
        rows.append(run_scheme(label, SCHEMES[label], loaders, w_te, stats, args))

    import pandas as pd
    df = pd.DataFrame(rows)[['scheme', 'pde', 'final_err', 'err_growth', 'unc_growth', 'step_corr',
                             'corr_sig_rough', 'corr_rough_err', 'rough_growth']]
    df.to_csv(out / 'rollout_drift_results.csv', index=False)
    with open(out / 'rollout_drift_results.json', 'w') as fh:
        json.dump(rows, fh, indent=2)
    print(f"\n{df.to_string(index=False)}\nwrote {out/'rollout_drift_results.csv'}", flush=True)


if __name__ == '__main__':
    main()
