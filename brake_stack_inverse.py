"""
INVERSE PROBLEM: identifying physical parameters from the sensor.

Principle: the parametric PINN (brake_stack_pinn_param.pt, FROZEN, no
retraining) is a differentiable function T(z, t ; p). The inversion is
therefore a simple gradient descent ON ITS INPUTS p: we look for the p that
best reproduces the sensor time series. Cost: a few seconds.

Protocol against the "inverse crime": measurements do NOT come from the PINN but from
the FD solver (fd_solve, copied from brake_stack_pinn_param.py), with 3 K noise and
30 s sampling (flight-data recorder rate). The inversion model and
the data generator are therefore two independent codes.

Unknowns: E_brake, h_lat, h_end   (tb and T_init assumed known)
Identifiability study: the same inversion is repeated with increasing
observation windows (5 min, 20 min, 2 h) and 8 random starts
each: the spread of the solutions tells what the sensor constrains or not.
Physically expected: E from the peak; h_lat with the slope of the tail;
h_end poorly constrained (correlated with h_lat, its effect acts through the ends
that the central sensor barely sees).
"""

import os
import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Font sizes chosen for the report: LaTeX scales the figure to ~6.1 inches
# wide, so the effective font size is
# fontsize x 6.1 / figsize_width. Target ~8.5 pt once printed.
plt.rcParams.update({'font.size': 18, 'axes.titlesize': 18, 'axes.labelsize': 18,
                     'xtick.labelsize': 16, 'ytick.labelsize': 16,
                     'legend.fontsize': 13, 'lines.linewidth': 1.8})

torch.manual_seed(0)
np.random.seed(0)
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DTYPE = torch.float32

# ================================================================ definitions
# IDENTICAL to brake_stack_pinn_param.py (required to reload the .pt)
N_DISC, E_DISC = 9, 0.025
L = N_DISC * E_DISC
R_EXT, R_INT = 0.20, 0.12
A_SEC = np.pi * (R_EXT**2 - R_INT**2)
PERIM = 2 * np.pi * (R_EXT + R_INT)
RHO, K0, CP0 = 1750.0, 15.0, 1000.0
T_AMB = 293.15
SIGMA_Q = 2.0e-3
N_INTERFACE = N_DISC - 1
Z_INT = np.arange(1, N_DISC) * E_DISC
T_WINDOW = 7200.0
P_NAMES = ['E_brake_MJ', 't_brake_s', 'T_init_C', 'h_lat', 'h_end']
P_LO = np.array([4.0, 20.0, 20.0, 5.0, 5.0])
P_HI = np.array([25.0, 60.0, 300.0, 40.0, 40.0])
ALPHA_D = K0 / (RHO * CP0)
DT_REF = P_HI[0] * 1e6 / (RHO * A_SEC * L * CP0)
FO = ALPHA_D * T_WINDOW / L**2
SIG_HAT = SIGMA_Q / L
ZI_HAT = torch.tensor(Z_INT / L, dtype=DTYPE, device=DEV)

def unpack(p):
    E = p[:, 0:1] * 1e6
    tb = p[:, 1:2] / T_WINDOW
    theta0 = (p[:, 2:3] + 273.15 - T_AMB) / DT_REF
    s_hat = p[:, 3:4] * PERIM / A_SEC * T_WINDOW / (RHO * CP0)
    bi = p[:, 4:5] * L / K0
    q0 = (T_WINDOW / (RHO * CP0 * DT_REF)) * 2 * E / p[:, 1:2] / (L * A_SEC)
    return tb, theta0, s_hat, bi, q0

N_MODES = 240
n_ = torch.arange(N_MODES, dtype=DTYPE, device=DEV)
LAM_DIFF = (n_ * np.pi)**2 * FO
G_N = (torch.cos(np.pi * n_[None, :] * ZI_HAT[:, None]).mean(dim=0)
       * torch.exp(-0.5 * (np.pi * n_ * SIG_HAT)**2))
G_N = torch.where(n_ == 0, torch.ones_like(G_N), 2 * G_N)

def theta_particular(zh, th, tb, theta0, s_hat, q0):
    lam = LAM_DIFF[None, :] + s_hat
    u = torch.minimum(th, tb)
    E_u = torch.exp(-lam * (th - u))
    E_t = torch.exp(-lam * th)
    I1 = (E_u - E_t) / lam
    I2 = u * E_u / lam - I1 / lam
    A = q0 * (I1 - I2 / tb)
    series = (A * G_N[None, :] * torch.cos(np.pi * n_[None, :] * zh)).sum(dim=1, keepdim=True)
    return theta0 * torch.exp(-s_hat * th) + series

TAU_0 = 0.001
TAU_MAX = float(np.log1p(1.0 / TAU_0))
def tau_of(th):
    return torch.log1p(th / TAU_0) / TAU_MAX

RAMP_HAT = 0.01
N_FF, FF_SCALE = 32, (3.0, 2.0)
WIDTH, DEPTH = 96, 5
P_LO_T = torch.tensor(P_LO, dtype=DTYPE, device=DEV)
P_HI_T = torch.tensor(P_HI, dtype=DTYPE, device=DEV)

class ParamPINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('B', torch.randn(2, N_FF) * torch.tensor(FF_SCALE)[:, None])
        layers, n = [], 2 + 2 * N_FF + 5
        for _ in range(DEPTH):
            layers += [nn.Linear(n, WIDTH), nn.Tanh()]
            n = WIDTH
        layers += [nn.Linear(n, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, zh, th, p):
        tb, theta0, s_hat, bi, q0 = unpack(p)
        tau = tau_of(th)
        pn = 2 * (p - P_LO_T) / (P_HI_T - P_LO_T) - 1
        f = 2 * np.pi * torch.cat([zh, tau], dim=1) @ self.B
        x = torch.cat([2 * zh - 1, 2 * tau - 1, torch.sin(f), torch.cos(f), pn], dim=1)
        ramp = 1 - torch.exp(-th / RAMP_HAT)
        return theta_particular(zh, th, tb, theta0, s_hat, q0) + ramp * self.net(x)

import os
CKPT = 'brake_stack_pinn_param.pt'
try:
    from google.colab import drive
    drive.mount('/content/drive')
    CKPT = '/content/drive/MyDrive/brake_stack_pinn_param.pt'
except Exception as e:
    print("Drive not mounted (", e, "): looking for the .pt locally")
if not os.path.exists(CKPT):
    CKPT = 'brake_stack_pinn_param.pt'   # fallback: file uploaded to the session
model = ParamPINN().to(DEV)
model.load_state_dict(torch.load(CKPT, map_location=DEV))
model.eval()
for w in model.parameters():
    w.requires_grad_(False)                    # the model is FROZEN: only p is optimized
print("Model loaded from:", CKPT)

# ================================================================ FD ground truth
def fd_solve(p_phys, nz=450):
    E, tb, T_init_C, h_lat, h_end = p_phys
    E *= 1e6
    dz = L / nz
    z = (np.arange(nz) + 0.5) * dz
    shape = np.zeros(nz)
    for zi in Z_INT:
        g = np.exp(-0.5 * ((z - zi) / SIGMA_Q)**2)
        shape += g / (g.sum() * dz)
    shape /= N_INTERFACE
    s_lat = h_lat * PERIM / A_SEC
    kf = K0 / dz**2 * np.ones(nz - 1)
    main = np.zeros(nz)
    main[:-1] -= kf; main[1:] -= kf
    main -= s_lat
    main[0] -= h_end / dz; main[-1] -= h_end / dz
    A = sp.diags([kf, main, kf], [-1, 0, 1], format='csc')
    b = s_lat * T_AMB * np.ones(nz)
    b[0] += h_end * T_AMB / dz; b[-1] += h_end * T_AMB / dz
    M = sp.diags(RHO * CP0 * np.ones(nz))
    t = np.concatenate([np.arange(0.0, 90.0, 0.1), np.arange(90.0, T_WINDOW + 1e-9, 5.0)])
    P = lambda tt: np.where((tt >= 0) & (tt < tb), 2 * E / tb * (1 - tt / tb), 0.0)
    T = np.full(nz, T_init_C + 273.15)
    out = np.empty((len(t), nz)); out[0] = T
    lu = {}
    for n in range(1, len(t)):
        dt = round(t[n] - t[n - 1], 6)
        if dt not in lu:
            lu[dt] = spla.splu((M - 0.5 * dt * A).tocsc())
        q = 0.5 * (P(t[n - 1]) + P(t[n])) * shape / A_SEC
        T = lu[dt].solve((M + 0.5 * dt * A) @ T + dt * (b + q))
        out[n] = T
    return z, t, out - 273.15

P_TRUE = [12.0, 40.0, 20.0, 15.0, 20.0]        # ground truth
TB_KNOWN, T0_KNOWN = P_TRUE[1], P_TRUE[2]      # assumed known for the inversion
NOISE_C, DT_SENSOR = 3.0, 30.0

print("Generating the ground truth (FD)...")
z_fd, t_fd, T_fd = fd_solve(P_TRUE)
t_s = np.arange(DT_SENSOR, T_WINDOW + 1e-9, DT_SENSOR)

def make_data(source, z_s, seed=42):
    """Noisy sensor series. source='fd': measurements independent of the inversion
    model. source='pinn': DELIBERATE inverse crime, used as a control to
    attribute the bias of long windows to the error of the direct model."""
    rng = np.random.default_rng(seed)
    if source == 'fd':
        i_z = np.argmin(np.abs(z_fd - z_s))
        T_clean = np.interp(t_s, t_fd, T_fd[:, i_z])
    else:
        zz = torch.full((len(t_s), 1), z_s / L, dtype=DTYPE, device=DEV)
        tt = torch.tensor(t_s / T_WINDOW, dtype=DTYPE, device=DEV)[:, None]
        pp = torch.tensor(P_TRUE, dtype=DTYPE, device=DEV).expand(len(t_s), 5)
        with torch.no_grad():
            T_clean = (T_AMB + DT_REF * model(zz, tt, pp).cpu().numpy()[:, 0]) - 273.15
    return T_clean + rng.normal(0, NOISE_C, len(t_s))

# ================================================================ inversion
FREE = [0, 3, 4]                               # indices of the unknowns: E, h_lat, h_end

def invert(z_s, t_data, T_data, seed, n_iter=400):
    """Adam descent on u (free parameters in logit space, box bounds
    enforced by a sigmoid). Returns the estimated p."""
    g = torch.Generator(device='cpu').manual_seed(seed)
    u = torch.randn(len(FREE), generator=g).to(DEV).requires_grad_(True)
    zz = torch.full((len(t_data), 1), z_s / L, dtype=DTYPE, device=DEV)
    tt = torch.tensor(t_data / T_WINDOW, dtype=DTYPE, device=DEV)[:, None]
    yy = torch.tensor((T_data + 273.15 - T_AMB) / DT_REF, dtype=DTYPE, device=DEV)[:, None]
    fixed = torch.tensor([0.0, TB_KNOWN, T0_KNOWN, 0.0, 0.0], dtype=DTYPE, device=DEV)
    opt = torch.optim.Adam([u], lr=0.05)
    for it in range(n_iter):
        opt.zero_grad()
        p = fixed.clone().expand(len(t_data), 5).contiguous()
        vals = P_LO_T[FREE] + (P_HI_T[FREE] - P_LO_T[FREE]) * torch.sigmoid(u)
        p[:, FREE] = vals
        loss = ((model(zz, tt, p) - yy)**2).mean()
        loss.backward()
        opt.step()
    return vals.detach().cpu().numpy()

WINDOWS_MIN = [5, 20, 120]
N_START = 8
Z_CENTER, Z_EDGE = 4.5 * E_DISC, 0.5 * E_DISC

# Three experiments: reference, inverse-crime control, moved sensor
EXPS = [('reference: FD, central sensor', 'fd', Z_CENTER),
        ('inverse-crime control: PINN data, central sensor', 'pinn', Z_CENTER),
        ('sensor at the stack END: FD, z = 0.5 e_disc', 'fd', Z_EDGE)]

all_results = {}
t0_ = time.time()
for label, source, z_s in EXPS:
    T_data = make_data(source, z_s)
    res = {}
    print(f"\n=== {label} ===")
    for w in WINDOWS_MIN:
        m = t_s <= w * 60
        sols = np.array([invert(z_s, t_s[m], T_data[m], seed) for seed in range(N_START)])
        res[w] = sols
        med, lo, hi = np.median(sols, 0), sols.min(0), sols.max(0)
        print(f"  window {w:3d} min ({time.time() - t0_:.0f} s):")
        for j, idx in enumerate(FREE):
            print(f"    {P_NAMES[idx]:>10s}: true {P_TRUE[idx]:6.1f}   estimated {med[j]:6.1f}"
                  f"   [min {lo[j]:6.1f}, max {hi[j]:6.1f}]")
    all_results[label] = res
results = all_results[EXPS[0][0]]              # the reference feeds the diagnostics below
T_s = make_data('fd', Z_CENTER)
Z_S = Z_CENTER

# The pair (h_lat, h_end) is correlated: the central sensor constrains the GLOBAL
# cooling rate, not its split between the flank and the ends. We
# check it by computing, for each solution, the rate of the slowest
# eigenmode lambda(h_lat, h_end): it must be much tighter than the
# two h taken separately.
def robin_mu0(bi):
    lo, hi = 1e-6, np.pi - 1e-6
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        (lo, hi) = (mid, hi) if mid * np.tan(0.5 * mid) < bi else (lo, mid)
    return 0.5 * (lo + hi)

def lam_of(h_lat, h_end):
    return (ALPHA_D * robin_mu0(h_end * L / K0)**2 / L**2
            + h_lat * PERIM / A_SEC / (RHO * CP0))

lam_true = lam_of(P_TRUE[3], P_TRUE[4])
lams = np.array([lam_of(hl, he) for _, hl, he in results[120]])
print(f"\nGlobal cooling rate lambda (2 h window):")
print(f"  true {lam_true * 1e4:.3f} e-4/s   estimated {np.median(lams) * 1e4:.3f}"
      f"   [min {lams.min() * 1e4:.3f}, max {lams.max() * 1e4:.3f}] "
      f"-> the COMBINATION is identified, not the split")

# ================================================================ figures
fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
p_best = np.array([0.0, TB_KNOWN, T0_KNOWN, 0.0, 0.0])
p_best[FREE] = np.median(results[120], 0)
zz = torch.full((len(t_s), 1), Z_S / L, dtype=DTYPE, device=DEV)
tt = torch.tensor(t_s / T_WINDOW, dtype=DTYPE, device=DEV)[:, None]
pp = torch.tensor(p_best, dtype=DTYPE, device=DEV).expand(len(t_s), 5)
with torch.no_grad():
    T_fit = (T_AMB + DT_REF * model(zz, tt, pp).cpu().numpy()[:, 0]) - 273.15
ax[0].plot(t_s / 60, T_s, '.', ms=3, alpha=0.5, label='noisy sensor (FD)')
ax[0].plot(t_s / 60, T_fit, 'r', lw=1.5, label='PINN at the identified p')
ax[0].set(xlabel='t (min)', ylabel='T (C)', title='Fit (2 h window)')
ax[0].legend()

# panel 2: E at 120 min for the three experiments (model bias)
# panel 3: h_end at 120 min for the three experiments (sensor placement)
short_labels = ['reference\n(FD, center)', 'inverse crime\n(PINN, center)', 'end sensor\n(FD, z=e/2)']
for a, j, idx, ttl in [(ax[1], 0, 0, 'Identified E, 2 h window (dashed = true)'),
                       (ax[2], 2, 4, 'Identified h_end, 2 h window (dashed = true)')]:
    for e, (label, _, _) in enumerate(EXPS):
        y = all_results[label][120][:, j]
        a.plot(np.full_like(y, e), y, 'o', ms=5, alpha=0.6, color=f'C{e}')
    a.axhline(P_TRUE[idx], color='k', ls='--', lw=1)
    a.set(xticks=range(3), xticklabels=short_labels, title=ttl)
fig.tight_layout()
plt.show()
