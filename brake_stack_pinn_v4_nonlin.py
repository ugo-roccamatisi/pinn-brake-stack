"""
PINN 1D NON LINEAIRE : proprietes k(T) et cp(T) du carbone-carbone.

    rho cp(T) dT/dt = d/dz( k(T) dT/dz ) + q(z, t) - s_lat (T - T_amb)
    k(T)  = K0  (1 + ALPHA_K  (T - T_REF))     ALPHA_K  = -4e-4 /K
    cp(T) = CP0 (1 + ALPHA_CP (T - T_REF))     ALPHA_CP = +8e-4 /K
    (a 800 C : k -30 %, cp +60 % : les pics chauds sont nettement plus bas
     et plus lents a refroidir que dans le modele lineaire)

C'est le premier cas ou la serie de cosinus n'est PAS la solution : elle
resout le probleme lineaire tangent et sert de point de depart, et le reseau
porte TOUT l'ecart non lineaire plus la convection aux extremites. Pas de
concurrent analytique ici : c'est le vrai terrain du PINN.

Le script est autonome : FD non lineaire (proprietes evaluees au pas
precedent) et FD lineaire de comparaison inclus. Trace en temps LINEAIRE.
Scenario de reference : E = 12 MJ, tb = 40 s, T0 = 20 C, h_lat = 15, h_end = 20.
"""

import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Tailles de police dimensionnees pour l'insertion dans le rapport : LaTeX
# ramene la figure a ~6.1 pouces de large, donc la police effective vaut
# fontsize x 6.1 / figsize_largeur. Objectif ~8.5 pt une fois imprime.
plt.rcParams.update({'font.size': 15, 'axes.titlesize': 15, 'axes.labelsize': 15,
                     'xtick.labelsize': 13, 'ytick.labelsize': 13,
                     'legend.fontsize': 10, 'lines.linewidth': 1.8})

torch.manual_seed(0)
np.random.seed(0)
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DTYPE = torch.float32

# ---------------------------------------------------------------- physique
N_DISC, E_DISC = 9, 0.025
L = N_DISC * E_DISC
R_EXT, R_INT = 0.20, 0.12
A_SEC = np.pi * (R_EXT**2 - R_INT**2)
PERIM = 2 * np.pi * (R_EXT + R_INT)
RHO, K0, CP0 = 1750.0, 15.0, 1000.0
T_AMB = T_REF = 293.15
ALPHA_K, ALPHA_CP = -4e-4, +8e-4
H_END, H_LAT = 20.0, 15.0
T_INIT_C = 20.0
E_BRAKE, T_BRAKE = 12.0e6, 40.0
SIGMA_Q = 2.0e-3
N_INTERFACE = N_DISC - 1
Z_INT = np.arange(1, N_DISC) * E_DISC
T_WINDOW = 7200.0

# ---------------------------------------------------------------- FD (reference)
def fd_solve(nonlinear=True, nz=450):
    dz = L / nz
    z = (np.arange(nz) + 0.5) * dz
    shape = np.zeros(nz)
    for zi in Z_INT:
        g = np.exp(-0.5 * ((z - zi) / SIGMA_Q)**2)
        shape += g / (g.sum() * dz)
    shape /= N_INTERFACE
    s_lat = H_LAT * PERIM / A_SEC
    t = np.concatenate([np.arange(0.0, 90.0, 0.05), np.arange(90.0, T_WINDOW + 1e-9, 2.0)])
    P = lambda tt: np.where((tt >= 0) & (tt < T_BRAKE), 2 * E_BRAKE / T_BRAKE * (1 - tt / T_BRAKE), 0.0)
    T = np.full(nz, T_INIT_C + 273.15)
    out = np.empty((len(t), nz), dtype=np.float32)
    out[0] = T
    ones = np.ones(nz)
    for n in range(1, len(t)):
        dt = t[n] - t[n - 1]
        kT = K0 * (1 + ALPHA_K * (T - T_REF)) if nonlinear else K0 * ones
        cpT = CP0 * (1 + ALPHA_CP * (T - T_REF)) if nonlinear else CP0 * ones
        kf = 0.5 * (kT[1:] + kT[:-1]) / dz**2
        main = np.zeros(nz)
        main[:-1] -= kf; main[1:] -= kf
        main -= s_lat
        main[0] -= H_END / dz; main[-1] -= H_END / dz
        A = sp.diags([kf, main, kf], [-1, 0, 1], format='csc')
        b = s_lat * T_AMB * ones.copy()
        b[0] += H_END * T_AMB / dz; b[-1] += H_END * T_AMB / dz
        M = sp.diags(RHO * cpT)
        q = 0.5 * (P(t[n - 1]) + P(t[n])) * shape / A_SEC
        T = spla.spsolve((M - 0.5 * dt * A).tocsc(), (M + 0.5 * dt * A) @ T + dt * (b + q))
        out[n] = T
    return z, t, out - 273.15

# ---------------------------------------------------------------- adimensionnement
ALPHA_D = K0 / (RHO * CP0)
DT_REF = E_BRAKE / (RHO * A_SEC * L * CP0)
FO = ALPHA_D * T_WINDOW / L**2
BI = H_END * L / K0
S_HAT = H_LAT * PERIM / A_SEC * T_WINDOW / (RHO * CP0)
Q_SCALE = T_WINDOW / (RHO * CP0 * DT_REF)
TB_HAT = T_BRAKE / T_WINDOW
SIG_HAT = SIGMA_Q / L
ZI_HAT = torch.tensor(Z_INT / L, dtype=DTYPE, device=DEV)
THETA_0 = (T_INIT_C + 273.15 - T_AMB) / DT_REF
Q0_MAX = Q_SCALE * 2 * E_BRAKE / T_BRAKE / (L * A_SEC)
AK = ALPHA_K * DT_REF        # pentes de k^ et cp^ par unite de theta (T_REF = T_amb)
ACP = ALPHA_CP * DT_REF
print(f"FO = {FO:.3f}  BI = {BI:.3f}  S^ = {S_HAT:.3f}  dT_ref = {DT_REF:.0f} K  "
      f"k^/cp^ a theta=1 : {1 + AK:.2f} / {1 + ACP:.2f}")

def k_hat(theta):
    return 1 + AK * theta

def cp_hat(theta):
    return 1 + ACP * theta

def power_hat(th):
    return torch.where((th >= 0) & (th < TB_HAT),
                       Q0_MAX * (1 - th / TB_HAT), torch.zeros_like(th))

def source_hat(zh, th):
    g = torch.exp(-0.5 * ((zh - ZI_HAT) / SIG_HAT)**2).sum(dim=1, keepdim=True)
    g = g / (SIG_HAT * np.sqrt(2 * np.pi)) / N_INTERFACE
    return power_hat(th) * g

# ---------------------------------------------------------------- theta_p : serie (probleme lineaire tangent)
N_MODES = 240
n_ = torch.arange(N_MODES, dtype=DTYPE, device=DEV)
LAM = (n_ * np.pi)**2 * FO + S_HAT
G_N = (torch.cos(np.pi * n_[None, :] * ZI_HAT[:, None]).mean(dim=0)
       * torch.exp(-0.5 * (np.pi * n_ * SIG_HAT)**2))
G_N = torch.where(n_ == 0, torch.ones_like(G_N), 2 * G_N)

def theta_particular(zh, th):
    u = torch.clamp(th, max=TB_HAT)
    E_u = torch.exp(-LAM[None, :] * (th - u))
    E_t = torch.exp(-LAM[None, :] * th)
    I1 = (E_u - E_t) / LAM[None, :]
    I2 = u * E_u / LAM[None, :] - I1 / LAM[None, :]
    A = Q0_MAX * (I1 - I2 / TB_HAT)
    series = (A * G_N[None, :] * torch.cos(np.pi * n_[None, :] * zh)).sum(dim=1, keepdim=True)
    return THETA_0 * torch.exp(-S_HAT * th) + series

# ---------------------------------------------------------------- log-temps + reseau
TAU_0 = 0.001
TAU_MAX = float(np.log1p(1.0 / TAU_0))
def tau_of(th):
    return torch.log1p(th / TAU_0) / TAU_MAX
def t_of_tau(tau):
    return TAU_0 * torch.expm1(tau * TAU_MAX)

RAMP_HAT = 0.5 * TB_HAT      # l'ecart non lineaire nait PENDANT le freinage
N_FF, FF_SCALE = 48, (10.0, 2.0)   # z^ doit suivre les pics d'interface de la correction
WIDTH, DEPTH = 96, 5

class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('B', torch.randn(2, N_FF) * torch.tensor(FF_SCALE)[:, None])
        layers, n = [], 2 + 2 * N_FF
        for _ in range(DEPTH):
            layers += [nn.Linear(n, WIDTH), nn.Tanh()]
            n = WIDTH
        layers += [nn.Linear(n, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, zh, th):
        tau = tau_of(th)
        f = 2 * np.pi * torch.cat([zh, tau], dim=1) @ self.B
        x = torch.cat([2 * zh - 1, 2 * tau - 1, torch.sin(f), torch.cos(f)], dim=1)
        ramp = 1 - torch.exp(-th / RAMP_HAT)
        return theta_particular(zh, th) + ramp * self.net(x)

def derivs(model, zh, th):
    zh = zh.requires_grad_(True)
    th = th.requires_grad_(True)
    theta = model(zh, th)
    theta_z, theta_t = torch.autograd.grad(theta, (zh, th), torch.ones_like(theta), create_graph=True)
    theta_zz = torch.autograd.grad(theta_z, zh, torch.ones_like(theta_z), create_graph=True)[0]
    return theta, theta_z, theta_t, theta_zz

# ---------------------------------------------------------------- pertes (residu NON LINEAIRE)
N_PDE, N_BC, RESAMPLE_EVERY = 6000, 500, 200
T_SPLIT = 3 * TB_HAT                 # frontiere freinage / queue de refroidissement

def sample_pde(n):
    """70 % uniforme en tau (resout le freinage), 30 % uniforme en t^ (peuple
    la queue de refroidissement, sinon quasi vide en log-temps)."""
    n_lin = int(0.3 * n)
    th = torch.cat([t_of_tau(torch.rand(n - n_lin, 1)), torch.rand(n_lin, 1)])
    zh = torch.rand(n, 1)
    brake = (th < 2 * TB_HAT).squeeze()
    nb = int(brake.sum())
    if nb:
        idx = torch.randint(0, N_INTERFACE, (nb, 1))
        z_i = (ZI_HAT.cpu()[idx] + 4 * SIG_HAT * torch.randn(nb, 1)).clamp(0, 1)
        half = torch.rand(nb, 1) < 0.5
        zh[brake] = torch.where(half, z_i, zh[brake])
    return zh.to(DEV), th.to(DEV)

def sample_bc(n):
    return t_of_tau(torch.rand(n, 1)).to(DEV)

def losses(model, zh, th, th_bc):
    theta, theta_z, theta_t, theta_zz = derivs(model, zh, th)
    # forme conservative : d/dz(k^ theta_z) = k^ theta_zz + AK theta_z^2
    src = source_hat(zh, th)
    res = (cp_hat(theta) * theta_t - FO * (k_hat(theta) * theta_zz + AK * theta_z**2)
           - src + S_HAT * theta)
    # Deux regimes, deux echelles de residu : pendant et juste apres le
    # freinage (t < T_SPLIT) le residu vaut jusqu'a ~84 et un residu absolu
    # de 0.1 est excellent ; dans la queue de refroidissement les termes de
    # l'equation valent ~0.1 et un residu de 0.05 laisse deriver T de 20 K en
    # 2 h (observe : pic exact, queue a -20 K, quelle que soit la
    # normalisation testee). On separe donc la MSE en deux moyennes, la queue
    # etant pesee W_TAIL fois : c'est une exigence de precision RELATIVE par
    # regime, pas un compromis global.
    tail = (th > T_SPLIT).squeeze()
    l_brake = (res[~tail]**2).mean()
    l_tail = (res[tail]**2).mean()
    z0, z1 = torch.zeros_like(th_bc), torch.ones_like(th_bc)
    th0, thz0, _, _ = derivs(model, z0, th_bc)
    th1, thz1, _, _ = derivs(model, z1, th_bc)
    l_bc = (((k_hat(th0) * thz0 - BI * th0)**2).mean()
            + ((k_hat(th1) * thz1 + BI * th1)**2).mean())
    return l_brake, l_tail, l_bc

# ---------------------------------------------------------------- entrainement
# Ponderation AUTO-EQUILIBREE : trois termes (freinage, queue, CL) d'echelles
# naturelles separees par 3 ordres de grandeur (~84, ~0.1, ~0.1 avec en plus
# des vitesses de convergence differentes). Les poids fixes essayes (W_BC=100,
# W_TAIL=200...) deplacaient le probleme d'un terme a l'autre a chaque run :
# pic ou queue ou CL, jamais les trois. Ici chaque terme est divise par sa
# moyenne mobile exponentielle (detachee) : chacun converge en RELATIF a la
# meme vitesse, quel que soit son ordre de grandeur. Pour L-BFGS (objectif
# fixe requis par la recherche lineaire), les poids sont geles a leur valeur
# de fin d'Adam.
model = PINN().to(DEV)
N_ADAM, N_LBFGS = 1000, 6000
EMA_BETA = 0.95
hist = []
ema = None
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
t0_ = time.time()
for it in range(N_ADAM):
    if it % RESAMPLE_EVERY == 0:
        zh, th = sample_pde(N_PDE)
        th_bc = sample_bc(N_BC)
    opt.zero_grad()
    terms = losses(model, zh, th, th_bc)
    vals = [t_.item() for t_ in terms]
    ema = vals if ema is None else [EMA_BETA * e + (1 - EMA_BETA) * v for e, v in zip(ema, vals)]
    loss = sum(t_ / e for t_, e in zip(terms, ema))
    loss.backward()
    opt.step()
    hist.append(vals)
    if it % 100 == 0:
        print(f"Adam {it:5d}  freinage {vals[0]:.2e}  queue {vals[1]:.2e}  bc {vals[2]:.2e}  ({time.time() - t0_:.0f} s)")

W_FIX = [1.0 / e for e in ema]
print(f"poids geles pour L-BFGS : {W_FIX[0]:.2e} / {W_FIX[1]:.2e} / {W_FIX[2]:.2e}")
zh, th = sample_pde(2 * N_PDE)
th_bc = sample_bc(2 * N_BC)
opt = torch.optim.LBFGS(model.parameters(), max_iter=N_LBFGS, history_size=50,
                        tolerance_grad=1e-9, tolerance_change=1e-12, line_search_fn='strong_wolfe')

def closure():
    opt.zero_grad()
    terms = losses(model, zh, th, th_bc)
    loss = sum(w * t_ for w, t_ in zip(W_FIX, terms))
    loss.backward()
    hist.append([t_.item() for t_ in terms])
    return loss

opt.step(closure)
print(f"L-BFGS termine : freinage {hist[-1][0]:.2e}  queue {hist[-1][1]:.2e}  bc {hist[-1][2]:.2e}  ({time.time() - t0_:.0f} s)")

# ---------------------------------------------------------------- references FD
print("\nFD non lineaire...")
z_fd, t_all, T_nl = fd_solve(nonlinear=True)
print("FD lineaire (comparaison)...")
_, _, T_lin = fd_solve(nonlinear=False)
sel = np.arange(0, len(t_all), 4)
t_fd = t_all[sel]
T_nl, T_lin = T_nl[sel].astype(np.float64), T_lin[sel].astype(np.float64)

ZZ, TT = np.meshgrid(z_fd / L, t_fd / T_WINDOW)
zz = torch.tensor(ZZ.ravel()[:, None], dtype=DTYPE, device=DEV)
tt = torch.tensor(TT.ravel()[:, None], dtype=DTYPE, device=DEV)
with torch.no_grad():
    theta = torch.cat([model(zz[i:i + 20000], tt[i:i + 20000]) for i in range(0, len(zz), 20000)])
T_pinn = (T_AMB + DT_REF * theta.cpu().numpy().reshape(ZZ.shape)) - 273.15
err = T_pinn - T_nl
print(f"\nEffet non lineaire : pic FD lineaire {T_lin.max():.0f} C -> non lineaire {T_nl.max():.0f} C")
print(f"RMSE PINN vs FD non lineaire 0..2 h : {np.sqrt((err**2).mean()):.2f} K   max |err| : {np.abs(err).max():.1f} K")
i_z = np.argmin(np.abs(z_fd - 4.5 * E_DISC))
for tt_ in [10, 40, 120, 600, 1800, 7200]:
    i = min(np.searchsorted(t_fd, tt_), len(t_fd) - 1)
    print(f"  t = {tt_:5d} s : RMSE {np.sqrt((err[i]**2).mean()):6.2f} K   "
          f"capteur FD_nl / PINN / FD_lin : {T_nl[i, i_z]:6.1f} / {T_pinn[i, i_z]:6.1f} / {T_lin[i, i_z]:6.1f} C")

# ---------------------------------------------------------------- figures (temps lineaire)
hist = np.array(hist)
fig, ax = plt.subplots(2, 2, figsize=(12, 8))
ax[0, 0].semilogy(hist[:, 0], label='EDP freinage')
ax[0, 0].semilogy(hist[:, 1], label='EDP queue')
ax[0, 0].semilogy(hist[:, 2], label='CL')
ax[0, 0].axvline(N_ADAM, color='k', lw=0.5)
ax[0, 0].set(xlabel='iteration', ylabel='perte', title='Convergence')
ax[0, 0].legend()

for tt_, c in zip([10, 40, 120, 600, 1800, 7200], plt.cm.viridis(np.linspace(0, 1, 6))):
    i = min(np.searchsorted(t_fd, tt_), len(t_fd) - 1)
    ax[0, 1].plot(z_fd * 1e3, T_nl[i], color=c, label=f'FD nl {tt_} s')
    ax[0, 1].plot(z_fd * 1e3, T_pinn[i], '--', color=c)
ax[0, 1].set(xlabel='z (mm)', ylabel='T (C)', title='Profils : FD non lineaire (trait) vs PINN (tirets)')
ax[0, 1].legend(fontsize=10, ncol=2, framealpha=0.85)

ax[1, 0].plot(t_fd / 60, T_nl[:, i_z], label='FD non lineaire, capteur')
ax[1, 0].plot(t_fd / 60, T_pinn[:, i_z], '--', label='PINN, capteur')
ax[1, 0].plot(t_fd / 60, T_lin[:, i_z], ':', label='FD lineaire (pour reference)')
ax[1, 0].set(xlabel='t (min)', ylabel='T (C)', title='Historique au capteur')
ax[1, 0].legend(fontsize=10)
axins = ax[1, 0].inset_axes([0.45, 0.45, 0.5, 0.5])
m5 = t_fd <= 300
axins.plot(t_fd[m5] / 60, T_nl[m5, i_z])
axins.plot(t_fd[m5] / 60, T_pinn[m5, i_z], '--')
axins.plot(t_fd[m5] / 60, T_lin[m5, i_z], ':')
axins.set(title='5 premieres minutes', xlabel='t (min)')

emax = np.abs(err).max()
pc = ax[1, 1].pcolormesh(t_fd / 60, z_fd * 1e3, err.T, shading='auto', cmap='RdBu_r',
                         vmin=-emax, vmax=emax)
ax[1, 1].set(xlabel='t (min)', ylabel='z (mm)', title='PINN - FD non lineaire (K)')
fig.colorbar(pc, ax=ax[1, 1])
fig.tight_layout()
plt.show()
