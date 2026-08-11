"""
Build the two datasets for the physical-consistency experiment (Experiment 3) in the
{'x','y'} layout that datautils.load_pt_2d already reads.

Why these and not the Table III/IV data:
  - The Table IV Darcy set uses a *piecewise-constant* coefficient, so the pointwise
    strong-form residual -div(k grad u) - f is singular at coefficient interfaces
    (verified: residual ~17x higher on interface cells, ~46% of mass on ~5% of cells).
  - The Table III NS set (nsforcing) maps an initial condition to a field a long time
    horizon later (||dw||/||w|| ~ 3.85, enstrophy x21), so a one-step transient residual
    does not apply (a fitted dt comes out negative).
Both substitutes below give a well-posed pointwise residual:
  - smooth-coefficient Darcy: ground-truth residual ~1e-13 (machine zero) at f=-1.
  - exp4 NS trajectories at a single Reynolds number: consecutive snapshots dt=0.05 apart,
    unforced; the residual recovers the true dt to ~4%.
"""
import argparse
from pathlib import Path
import numpy as np
import torch


def build_darcy_smooth(out_path, variant='medium_clean'):
    """Smooth-coefficient Darcy: x = permeability k, y = solution u. Satisfies
    -div(k grad u) = f with f = -1 (that sign is what the reference solutions obey)."""
    data_dir = Path(__file__).resolve().parent.parent / 'data'
    xs, ys = [], []
    for split in ('train', 'val'):
        z = np.load(data_dir / f'darcy_{variant}_res64_{split}.npz')
        X, U = z['X'], z['U']              # (N,64,64,3), (N,64,64,1)
        xs.append(X[..., 2].astype(np.float32))   # permeability k
        ys.append(U[..., 0].astype(np.float32))   # solution u
    x = np.concatenate(xs); y = np.concatenate(ys)
    torch.save({'x': torch.from_numpy(x), 'y': torch.from_numpy(y),
                'meta': {'pde': 'darcy', 'coeff': 'smooth', 'variant': variant,
                         'forcing': -1.0, 'note': 'x=permeability, y=solution'}}, out_path)
    print(f"[darcy] wrote {out_path}  x{tuple(x.shape)} k in [{x.min():.3f},{x.max():.3f}]  "
          f"u in [{y.min():.3f},{y.max():.3f}]")


def _spectral_curl(u, v, n):
    kx = 2 * np.pi * np.fft.fftfreq(n, d=1.0 / n)
    KX, KY = np.meshgrid(kx, kx, indexing='ij')
    dvdx = np.real(np.fft.ifft2(1j * KX * np.fft.fft2(v)))
    dudy = np.real(np.fft.ifft2(1j * KY * np.fft.fft2(u)))
    return dvdx - dudy


def build_ns_traj(out_path, re=1000.0, n=64, max_pairs=5000, seed=42):
    """exp4 trajectories at one Reynolds number -> consecutive vorticity pairs
    x = w(t), y = w(t+dt), dt = 0.05, single fixed nu."""
    src = Path(__file__).resolve().parent / 'data' / 'exp4' / 'ns_train_id.pt'
    d = torch.load(src, map_location='cpu', weights_only=False)
    xs, ys = [], []
    nu = None
    for s in d['data']:
        if float(s['reynolds']) != re:
            continue
        nu = float(s['viscosity'])
        vel = np.asarray(s['velocity'])[:, :, :n, :n]        # (S,2,n,n), crop 65->64
        w = np.stack([_spectral_curl(vel[t, 0], vel[t, 1], n) for t in range(vel.shape[0])])
        for t in range(w.shape[0] - 1):
            xs.append(w[t]); ys.append(w[t + 1])
    x = np.asarray(xs, dtype=np.float32); y = np.asarray(ys, dtype=np.float32)
    if len(x) > max_pairs:
        idx = np.random.RandomState(seed).choice(len(x), max_pairs, replace=False)
        x, y = x[idx], y[idx]
    dt = float(d['data'][0]['dt']) * 10                       # save_every=10
    torch.save({'x': torch.from_numpy(x), 'y': torch.from_numpy(y),
                'meta': {'pde': 'navier_stokes', 'reynolds': re, 'nu': nu, 'dt': dt,
                         'forcing': 0.0, 'note': 'x=vorticity(t), y=vorticity(t+dt)'}}, out_path)
    print(f"[ns] wrote {out_path}  x{tuple(x.shape)}  Re={re} nu={nu:.6g} dt={dt}  "
          f"w in [{x.min():.3f},{x.max():.3f}]")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out-dir', default='../data/physics')
    p.add_argument('--darcy-variant', default='medium_clean')
    p.add_argument('--ns-re', type=float, default=1000.0)
    p.add_argument('--ns-max-pairs', type=int, default=5000)
    args = p.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    build_darcy_smooth(out / 'darcy_smooth_res64.pt', args.darcy_variant)
    build_ns_traj(out / 'ns_traj_re1000_res64.pt', args.ns_re, max_pairs=args.ns_max_pairs)
