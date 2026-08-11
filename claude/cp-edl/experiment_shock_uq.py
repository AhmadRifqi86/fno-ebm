"""
Experiment (harder benchmark): does predicted uncertainty concentrate at shocks?

Data: locally generated 2D compressible-Euler Riemann problems (datagen/euler2d.py),
{'x','y'} with primitive channels [rho,u,v,p] at t=0 (input) and t=T (output, shocks formed).

Task: the evidential FNO maps the full initial state -> density field at time T. We then
test whether the pointwise predictive uncertainty is large exactly at the shocks, using the
gradient magnitude |grad rho| of the true solution as the shock indicator. We report the
Pearson and Spearman correlation between uncertainty and |grad rho|, and the shock
concentration ratio (mean uncertainty in the top-5% highest-gradient cells over the field
mean). A ratio > 1 means the model flags discontinuities as uncertain.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import Config, Factory, get_evidential_methods_configs
from datautils import PDEDataset
from fno import EvidentialFNO2d, EvidentialFNOTrainer
from main import get_regularization_function
from experiment_physics_consistency import physics_consistency, SCHEMES


def grad_mag(field):
    """|grad rho| via central differences on the unit grid (B,H,W)."""
    gx = np.zeros_like(field); gy = np.zeros_like(field)
    gx[:, 1:-1, :] = 0.5 * (field[:, 2:, :] - field[:, :-2, :])
    gy[:, :, 1:-1] = 0.5 * (field[:, :, 2:] - field[:, :, :-2])
    return np.sqrt(gx ** 2 + gy ** 2)


def build_splits(data_path, batch_size, train_ratio=0.85, val_ratio=0.05, seed=42):
    """Input = full IC state [rho,u,v,p] + coords (6 channels); output = density at T."""
    d = torch.load(data_path, map_location='cpu', weights_only=False)
    x = d['x'].numpy().astype(np.float32)   # (N,H,W,4) [rho,u,v,p] at t=0
    y = d['y'].numpy().astype(np.float32)   # (N,H,W,4) at t=T
    N, H, W, _ = x.shape

    xs = np.linspace(0, 1, H, endpoint=False, dtype=np.float32)
    ys = np.linspace(0, 1, W, endpoint=False, dtype=np.float32)
    GX, GY = np.meshgrid(xs, ys, indexing='ij')
    X = np.zeros((N, H, W, 6), dtype=np.float32)
    X[..., 0] = GX; X[..., 1] = GY
    X[..., 2:] = x                                   # rho,u,v,p at t=0
    U = y[..., 0:1]                                  # density at T

    idx = np.random.RandomState(seed).permutation(N)
    n_tr = int(train_ratio * N); n_va = int(val_ratio * N)
    tr, va, te = idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:]
    Xtr, Utr = X[tr], U[tr]
    Xte, Ute = X[te], U[te]

    fields_mean = [Xtr[..., c].mean() for c in range(2, 6)]
    fields_std = [Xtr[..., c].std() for c in range(2, 6)]
    u_mean, u_std = float(Utr.mean()), float(Utr.std())
    common = dict(normalize_output=True, normalize_input=True, normalize_coords=True,
                  precomputed_u_mean=u_mean, precomputed_u_std=u_std,
                  precomputed_x_min=0.0, precomputed_x_max=1.0,
                  precomputed_y_min=0.0, precomputed_y_max=1.0,
                  precomputed_x_fields_mean=fields_mean,
                  precomputed_x_fields_std=fields_std)
    loaders = (
        DataLoader(PDEDataset(Xtr, Utr, **common), batch_size=batch_size, shuffle=True, num_workers=0),
        DataLoader(PDEDataset(X[va], U[va], **common), batch_size=batch_size, shuffle=False, num_workers=0),
        DataLoader(PDEDataset(Xte, Ute, **common), batch_size=batch_size, shuffle=False, num_workers=0),
    )
    return loaders, (Xte, Ute), dict(u_mean=u_mean, u_std=u_std)


def run_scheme(label, reg_key, loaders, raw_test, stats, args):
    train_loader, val_loader, test_loader = loaders
    _, Ute = raw_test
    device = args.device
    mcfg = get_evidential_methods_configs()['der_nig']

    model = EvidentialFNO2d(modes1=12, modes2=12, width=32, n_layers=4, in_channels=6,
                            nu_min=mcfg['nu_min'], alpha_min=mcfg['alpha_min'],
                            beta_min=mcfg['beta_min']).to(device)
    opt = Factory.create_optimizer({'type': 'adam', 'lr': args.lr, 'weight_decay': 0.0}, model.parameters())
    sched = Factory.create_scheduler({'type': 'exponential_lr', 'gamma': 0.93}, opt)
    tcfg = Config({'device': device, 'lr': args.lr, 'weight_decay': 0.0, 'enable_tracking': False,
                   'experiment_name': f'shock_{reg_key}', 'use_tensorboard': False,
                   'checkpoint_dir': str(Path(args.out) / 'ckpt' / reg_key), 'save_every': 10 ** 6})
    trainer = EvidentialFNOTrainer(model=model, config=tcfg, method_name='der_nig', optimizer=opt,
                                   scheduler=sched, method_config=mcfg,
                                   reg_fn=get_regularization_function(reg_key))
    trainer.train(train_loader, val_loader, epochs=args.epochs)

    model.eval()
    sigmas = []
    with torch.no_grad():
        for x, _ in test_loader:
            g, nu, al, be = model(x.to(device))
            var = (be / (al - 1.0)) * ((nu + 1.0) / nu)
            sigmas.append(torch.sqrt(var).squeeze(-1).cpu().numpy())
    sigma = np.concatenate(sigmas, axis=0).astype(np.float64)
    sigma_phys = sigma * stats['u_std']

    rho_true = Ute[..., 0].astype(np.float64)     # true density at T
    shock = grad_mag(rho_true)                    # |grad rho| shock indicator
    m = physics_consistency(sigma_phys, shock, top_frac=args.top_frac)
    m.update(scheme=label, reg_key=reg_key, pde='euler_shock')
    print(f"  {label:18s}  pearson={m['pearson']:+.3f}  spearman={m['spearman']:+.3f}  "
          f"shock_conc={m['conc_ratio']:.2f}", flush=True)
    return m


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='../data/physics/euler2d_shock_res64.pt')
    p.add_argument('--out', default='exp_shock')
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--top-frac', type=float, default=0.05)
    p.add_argument('--schemes', nargs='*', default=list(SCHEMES.keys()))
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    loaders, raw_test, stats = build_splits(args.data, args.batch_size, seed=args.seed)
    print(f"stats: {stats}", flush=True)

    rows = []
    for label in args.schemes:
        print(f"\n=== {label} ===", flush=True)
        rows.append(run_scheme(label, SCHEMES[label], loaders, raw_test, stats, args))

    import pandas as pd
    df = pd.DataFrame(rows)[['scheme', 'pde', 'pearson', 'spearman', 'conc_ratio']]
    df.to_csv(out / 'shock_uq_results.csv', index=False)
    with open(out / 'shock_uq_results.json', 'w') as fh:
        json.dump(rows, fh, indent=2)
    print(f"\n{df.to_string(index=False)}\nwrote {out/'shock_uq_results.csv'}", flush=True)


if __name__ == '__main__':
    main()
