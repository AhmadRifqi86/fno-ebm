"""
2D compressible Euler solver (finite-volume, Rusanov/local-Lax-Friedrichs flux with
MUSCL minmod reconstruction) for generating shock-capturing benchmark data locally.

State (conservative): U = [rho, rho*u, rho*v, E],  E = p/(gamma-1) + 0.5 rho (u^2+v^2).
Periodic boundaries on the unit square (matches PDEBench 2D_CFD_*_periodic).

Randomized four-quadrant Riemann initial conditions develop interacting shocks,
contacts and rarefactions -- the standard 2D compressible test family. Each sample
stores the initial state and the state at a fixed final time T, by which shocks
have formed.
"""
import numpy as np

GAMMA = 1.4


def _primitive(U):
    rho = np.maximum(U[0], 1e-8)
    u = U[1] / rho
    v = U[2] / rho
    p = (GAMMA - 1.0) * (U[3] - 0.5 * rho * (u * u + v * v))
    p = np.maximum(p, 1e-8)
    return rho, u, v, p


def _flux_x(U):
    rho, u, v, p = _primitive(U)
    return np.stack([rho * u,
                     rho * u * u + p,
                     rho * u * v,
                     u * (U[3] + p)])


def _flux_y(U):
    rho, u, v, p = _primitive(U)
    return np.stack([rho * v,
                     rho * u * v,
                     rho * v * v + p,
                     v * (U[3] + p)])


def _sound_speed(U):
    rho, u, v, p = _primitive(U)
    return np.sqrt(GAMMA * p / rho), u, v


def _minmod(a, b):
    return np.where(a * b > 0, np.sign(a) * np.minimum(np.abs(a), np.abs(b)), 0.0)


def _muscl_faces(U, axis):
    """Left/right reconstructed states at the face to the 'right' (+) of each cell,
    along `axis` (2=x, 3=y), periodic. Returns (UL, UR) at N faces."""
    Um = np.roll(U, 1, axis=axis)     # i-1
    Up = np.roll(U, -1, axis=axis)    # i+1
    slope = _minmod(U - Um, Up - U)   # limited slope in cell i
    UL = U + 0.5 * slope              # value at i's right face, from cell i
    UR_cell = np.roll(U, -1, axis=axis)
    slope_R = np.roll(slope, -1, axis=axis)
    UR = UR_cell - 0.5 * slope_R      # value at i's right face, from cell i+1
    return UL, UR


def _rusanov(UL, UR, axis):
    FL = _flux_x(UL) if axis == 2 else _flux_y(UL)
    FR = _flux_x(UR) if axis == 2 else _flux_y(UR)
    cL, uL, vL = _sound_speed(UL)
    cR, uR, vR = _sound_speed(UR)
    velL = uL if axis == 2 else vL
    velR = uR if axis == 2 else vR
    smax = np.maximum(np.abs(velL) + cL, np.abs(velR) + cR)
    return 0.5 * (FL + FR) - 0.5 * smax * (UR - UL)


def _rhs(U, dx, dy):
    # x-direction faces
    UL, UR = _muscl_faces(U, axis=2)
    Fx = _rusanov(UL, UR, axis=2)          # flux at right face of each cell i
    dFx = (Fx - np.roll(Fx, 1, axis=2)) / dx
    # y-direction
    UL, UR = _muscl_faces(U, axis=3)
    Fy = _rusanov(UL, UR, axis=3)
    dFy = (Fy - np.roll(Fy, 1, axis=3)) / dy
    return -(dFx + dFy)


def _max_speed(U, dx, dy):
    c, u, v = _sound_speed(U)
    return np.max((np.abs(u) + c) / dx + (np.abs(v) + c) / dy)


def evolve(U0, T, cfl=0.4, dx=None, dy=None, max_steps=5000):
    """SSP-RK2 time integration to final time T. U0: (4, B, H, W)."""
    n = U0.shape[-1]
    dx = dx or 1.0 / n
    dy = dy or 1.0 / n
    U = U0.copy()
    t = 0.0
    for _ in range(max_steps):
        smax = _max_speed(U, dx, dy)
        dt = cfl / max(smax, 1e-8)
        if t + dt > T:
            dt = T - t
        U1 = U + dt * _rhs(U, dx, dy)                 # Euler predictor
        U = 0.5 * U + 0.5 * (U1 + dt * _rhs(U1, dx, dy))  # SSP-RK2 corrector
        t += dt
        if t >= T - 1e-12:
            break
    return U


def _quadrant_ic(B, n, rng):
    """Random four-quadrant primitive states -> conservative U0 (4,B,n,n)."""
    xs = np.linspace(0, 1, n, endpoint=False) + 0.5 / n
    X, Y = np.meshgrid(xs, xs, indexing='ij')
    q_x = X > 0.5
    q_y = Y > 0.5
    rho = np.empty((B, n, n)); u = np.empty((B, n, n))
    v = np.empty((B, n, n)); p = np.empty((B, n, n))
    for b in range(B):
        for qx in (False, True):
            for qy in (False, True):
                m = (q_x == qx) & (q_y == qy)
                rho[b][m] = rng.uniform(0.5, 3.0)
                p[b][m] = rng.uniform(0.5, 3.0)
                u[b][m] = rng.uniform(-0.6, 0.6)
                v[b][m] = rng.uniform(-0.6, 0.6)
    E = p / (GAMMA - 1.0) + 0.5 * rho * (u * u + v * v)
    return np.stack([rho, rho * u, rho * v, E])


def generate(n_samples=500, n=64, T=0.15, seed=0, batch=25, out_path=None):
    """Generate (initial, final) compressible-Euler pairs with shocks.

    Saves {'x': (N, n, n, 4), 'y': (N, n, n, 4)} in primitive-ish channels
    [rho, u, v, p] for both input (t=0) and output (t=T)."""
    rng = np.random.default_rng(seed)
    xs_in, xs_out = [], []
    done = 0
    while done < n_samples:
        b = min(batch, n_samples - done)
        U0 = _quadrant_ic(b, n, rng)
        UT = evolve(U0, T)
        for U, store in [(U0, xs_in), (UT, xs_out)]:
            rho, u, v, p = _primitive(U)
            store.append(np.stack([rho, u, v, p], axis=-1).astype(np.float32))  # (b,n,n,4)
        done += b
        print(f"  generated {done}/{n_samples}", flush=True)
    x = np.concatenate(xs_in); y = np.concatenate(xs_out)
    if out_path:
        import torch
        torch.save({'x': torch.from_numpy(x), 'y': torch.from_numpy(y),
                    'meta': {'pde': 'euler2d', 'gamma': GAMMA, 'T': T, 'n': n,
                             'channels': ['rho', 'u', 'v', 'p']}}, out_path)
        print(f"wrote {out_path}  x{x.shape} y{y.shape}")
    return x, y


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-samples', type=int, default=800)
    ap.add_argument('--n', type=int, default=64)
    ap.add_argument('--T', type=float, default=0.15)
    ap.add_argument('--out', default='../data/physics/euler2d_shock_res64.pt')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--batch', type=int, default=100)
    args = ap.parse_args()
    generate(args.n_samples, args.n, args.T, args.seed, batch=args.batch, out_path=args.out)
