"""
Baseline uncertainty methods (MC Dropout, Deep Ensemble) on the two harder-regime
experiments, to show the shock/rollout behaviour is a property of pointwise, locally
calibrated uncertainty in general -- NOT an artefact of the evidential parameterisation.

For each task the baseline predictive uncertainty is the disagreement std (across MC-Dropout
samples, or across ensemble members). It is fed through the SAME metric code the evidential
sweeps use, so the numbers are directly comparable to Tables (shock) / (rollout):

  shock   : physics_consistency(sigma, |grad rho|) -> pearson, spearman, conc_ratio
  rollout : autoregressive rollout -> final_err, unc_growth, step_corr, corr(sig,rough)

The evidential result files are left untouched; baseline rows go to separate CSV/JSON.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from fno import MCDropoutFNO2d, FNO2d, FNOEnsemble

# reuse the exact data builders + metric from the validated evidential scripts
from experiment_shock_uq import build_splits, grad_mag
from experiment_rollout_drift import build_data
from experiment_physics_consistency import physics_consistency


# --------------------------------------------------------------------------------------
# Training (mirrors the canonical baseline loops in main.py: forward_single for dropout,
# bootstrap-resampled independent members for the ensemble, plain Adam lr=1e-3 MSE)
# --------------------------------------------------------------------------------------
def train_mc_dropout(in_channels, tr_loader, va_loader, args):
    device = args.device
    model = MCDropoutFNO2d(modes1=12, modes2=12, width=32, n_layers=4, in_channels=in_channels,
                           dropout_rate=args.dropout_rate, n_samples=args.mc_samples).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    for epoch in range(args.epochs):
        model.train()
        for x, y in tr_loader:
            x, y = x.to(device), y.to(device)
            loss = F.mse_loss(model.forward_single(x), y)   # single pass while training
            opt.zero_grad(); loss.backward(); opt.step()
        if (epoch + 1) % 5 == 0 or epoch == 0:
            model.eval(); vl = 0.0; nb = 0
            with torch.no_grad():
                for x, y in va_loader:
                    vl += F.mse_loss(model.forward_single(x.to(device)), y.to(device)).item(); nb += 1
            print(f"    mc_dropout epoch {epoch+1}/{args.epochs}  val_mse={vl/max(nb,1):.5f}", flush=True)
    return model


def train_ensemble(in_channels, tr_loader, va_loader, args):
    device = args.device
    train_dataset = tr_loader.dataset
    n_train = len(train_dataset)
    models = []
    for i in range(args.n_models):
        rng = np.random.RandomState(seed=args.seed + i)
        boot = rng.choice(n_train, size=n_train, replace=True).tolist()
        loader_i = DataLoader(Subset(train_dataset, boot), batch_size=args.batch_size, shuffle=True)
        m = FNO2d(modes1=12, modes2=12, width=32, n_layers=4, in_channels=in_channels).to(device)
        opt = torch.optim.Adam(m.parameters(), lr=args.lr)
        for epoch in range(args.epochs):
            m.train()
            for x, y in loader_i:
                x, y = x.to(device), y.to(device)
                loss = F.mse_loss(m(x), y)
                opt.zero_grad(); loss.backward(); opt.step()
        m.eval(); vl = 0.0; nb = 0
        with torch.no_grad():
            for x, y in va_loader:
                vl += F.mse_loss(m(x.to(device)), y.to(device)).item(); nb += 1
        print(f"    ensemble member {i+1}/{args.n_models}  val_mse={vl/max(nb,1):.5f}", flush=True)
        models.append(m)
    return FNOEnsemble(models=models, device=device)


def _sigma_field(pred_out):
    """A baseline forward returns (mean, std); take the std, drop the trailing channel."""
    _, std = pred_out
    return std.squeeze(-1)


# --------------------------------------------------------------------------------------
# Shock experiment
# --------------------------------------------------------------------------------------
def run_shock(method, args):
    loaders, raw_test, stats = build_splits(args.shock_data, args.batch_size, seed=args.seed)
    tr_loader, va_loader, te_loader = loaders
    _, Ute = raw_test
    device = args.device

    if method == 'mc_dropout':
        model = train_mc_dropout(6, tr_loader, va_loader, args)
        predict = lambda x: model(x.to(device), return_uncertainty=True)
    else:
        ens = train_ensemble(6, tr_loader, va_loader, args)
        predict = lambda x: ens.predict(x.to(device), return_uncertainty=True)

    sigmas = []
    with torch.no_grad():
        for x, _ in te_loader:
            sigmas.append(_sigma_field(predict(x)).cpu().numpy())
    sigma = np.concatenate(sigmas, axis=0).astype(np.float64) * stats['u_std']

    rho_true = Ute[..., 0].astype(np.float64)
    m = physics_consistency(sigma, grad_mag(rho_true), top_frac=args.top_frac)
    m.update(scheme=method, method=method, pde='euler_shock')
    print(f"  [shock] {method:12s}  pearson={m['pearson']:+.3f}  spearman={m['spearman']:+.3f}  "
          f"shock_conc={m['conc_ratio']:.2f}", flush=True)
    return m


# --------------------------------------------------------------------------------------
# Rollout experiment (generic body; mirrors rollout() in experiment_rollout_drift.py but
# takes a predict-fn returning normalized (mean, sigma) fields)
# --------------------------------------------------------------------------------------
def rollout_baseline(predict, w_te, stats, device):
    wmean, wstd = stats['wmean'], stats['wstd']
    B, T, n, _ = w_te.shape
    xs = np.linspace(0, 1, n, endpoint=False, dtype=np.float32)
    GX, GY = np.meshgrid(xs, xs, indexing='ij')
    cx = torch.from_numpy((2 * GX - 1).astype(np.float32)).to(device)
    cy = torch.from_numpy((2 * GY - 1).astype(np.float32)).to(device)
    w_hat = torch.from_numpy(w_te[:, 0].copy()).to(device)

    def roughness(field):
        gx = field[:, 1:, :] - field[:, :-1, :]
        gy = field[:, :, 1:] - field[:, :, :-1]
        return 0.5 * (gx.abs().mean() + gy.abs().mean())

    errs, uncs, roughs = [], [], []
    with torch.no_grad():
        for k in range(1, T):
            roughs.append(roughness(w_hat).item())
            inp = torch.stack([cx.expand(B, n, n), cy.expand(B, n, n),
                               (w_hat - wmean) / wstd], dim=-1)          # (B,n,n,3) normalized
            mean_n, sig_n = predict(inp)                                 # normalized fields
            w_hat = mean_n.squeeze(-1) * wstd + wmean
            sigma = sig_n.squeeze(-1) * wstd
            w_true = torch.from_numpy(w_te[:, k].copy()).to(device)
            num = torch.linalg.norm((w_hat - w_true).reshape(B, -1), dim=1)
            den = torch.linalg.norm(w_true.reshape(B, -1), dim=1)
            errs.append((num / den).mean().item())
            uncs.append(sigma.mean().item())
    return np.array(errs), np.array(uncs), np.array(roughs)


def run_rollout(method, args):
    loaders, w_te, stats = build_data(args.shard_dir, args.batch_size, args.n_test_traj, args.seed)
    tr_loader, va_loader = loaders
    device = args.device

    if method == 'mc_dropout':
        model = train_mc_dropout(3, tr_loader, va_loader, args)
        predict = lambda inp: model(inp.to(device), return_uncertainty=True)
    else:
        ens = train_ensemble(3, tr_loader, va_loader, args)
        predict = lambda inp: ens.predict(inp.to(device), return_uncertainty=True)

    errs, uncs, roughs = rollout_baseline(predict, w_te, stats, device)
    step_corr = float(np.corrcoef(errs, uncs)[0, 1])
    corr_sig_rough = float(np.corrcoef(uncs, roughs)[0, 1])
    corr_rough_err = float(np.corrcoef(roughs, errs)[0, 1])
    m = dict(scheme=method, method=method, pde='ns_rollout',
             final_err=float(errs[-1]), err_growth=float(errs[-1] / (errs[0] + 1e-9)),
             unc_growth=float(uncs[-1] / (uncs[0] + 1e-9)), step_corr=step_corr,
             corr_sig_rough=corr_sig_rough, corr_rough_err=corr_rough_err,
             rough_growth=float(roughs[-1] / (roughs[0] + 1e-9)),
             err_curve=errs.tolist(), unc_curve=uncs.tolist(), rough_curve=roughs.tolist())
    print(f"  [rollout] {method:12s}  step_corr={step_corr:+.3f}  "
          f"unc_growth={m['unc_growth']:.2f}x  corr(sig,rough)={corr_sig_rough:+.3f}", flush=True)
    return m


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tasks', nargs='*', default=['shock', 'rollout'], choices=['shock', 'rollout'])
    p.add_argument('--methods', nargs='*', default=['mc_dropout', 'ensemble'],
                   choices=['mc_dropout', 'ensemble'])
    p.add_argument('--shock-data', default='../data/physics/euler2d_shock_res64.pt')
    p.add_argument('--shard-dir', default='../data/pdearena')
    p.add_argument('--shock-out', default='exp_shock')
    p.add_argument('--rollout-out', default='exp_rollout')
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--top-frac', type=float, default=0.05)
    p.add_argument('--dropout-rate', type=float, default=0.1)
    p.add_argument('--mc-samples', type=int, default=30)
    p.add_argument('--n-models', type=int, default=5)
    p.add_argument('--n-test-traj', type=int, default=50)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    import pandas as pd

    if 'shock' in args.tasks:
        print("\n########## SHOCK baselines ##########", flush=True)
        rows = [run_shock(m, args) for m in args.methods]
        out = Path(args.shock_out); out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows)[['method', 'pde', 'pearson', 'spearman', 'conc_ratio']].to_csv(
            out / 'baselines_shock.csv', index=False)
        json.dump(rows, open(out / 'baselines_shock.json', 'w'), indent=2)
        print(f"wrote {out/'baselines_shock.csv'}", flush=True)

    if 'rollout' in args.tasks:
        print("\n########## ROLLOUT baselines ##########", flush=True)
        rows = [run_rollout(m, args) for m in args.methods]
        out = Path(args.rollout_out); out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows)[['method', 'pde', 'final_err', 'err_growth', 'unc_growth',
                            'step_corr', 'corr_sig_rough', 'corr_rough_err', 'rough_growth']].to_csv(
            out / 'baselines_rollout.csv', index=False)
        json.dump(rows, open(out / 'baselines_rollout.json', 'w'), indent=2)
        print(f"wrote {out/'baselines_rollout.csv'}", flush=True)


if __name__ == '__main__':
    main()
