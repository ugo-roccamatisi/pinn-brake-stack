"""
PINN 1D v3 : fenetre longue (0..2 h), freinage + courbe de refroidissement.

Meme physique que brake_stack_pinn_v2.py (lit brake_stack_data.npz), mais :

1. theta_p n'est plus une quadrature de gaussiennes + images : c'est la serie
   de cosinus exacte du probleme a flux nul aux extremites, avec puits lateral,
       theta_p(z^, t^) = sum_n g_n cos(n pi z^) A_n(t^)
       A_n(t^) = int_0^{min(t^,TB)} q0(t') exp(-lambda_n (t^ - t')) dt'   (forme close)
       lambda_n = (n pi)^2 FO + S^
   Chaque mode verifie l'EDP exactement (residu nul), pas de quadrature, cout
   N_MODES cosinus par point. Les gaussiennes d'interface sont resolues avec
   ~150 modes (facteur exp(-(n pi sigma^)^2 / 2)). Sur 2 h, les images auraient
   demande 4 a 5 ordres de reflexion : la serie est plus simple et exacte.

2. Le reseau apprend UNIQUEMENT la convection aux extremites (Robin, Bi = 0.3),
   absente de theta_p : ~1/3 du refroidissement total. C'est une correction
   lisse en z, lente en t, negative. Ansatz :
       theta = theta_p + (1 - exp(-t^ / TB)) * N(z^, tau)
   avec tau = log-temps, car t^ doit couvrir 40 s (t^ = 0.0056) et 2 h (t^ = 1).

3. Collocation echantillonnee uniformement en tau (log-temps), points
   d'interface seulement pendant le freinage.
"""

import time
import numpy as np
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

# ---------------------------------------------------------------- donnees FD
d = np.load('brake_stack_data.npz')
L, RHO, K0, CP0 = float(d['L']), float(d['rho']), float(d['k0']), float(d['cp0'])
H_END, S_LAT, T_AMB, T_INIT = (float(d['h_end']), float(d['s_lat']),
                               float(d['t_amb']), float(d['t_init']))
E_BRAKE, T_BRAKE, SIGMA_Q, A_SEC = (float(d['e_brake']), float(d['t_brake']),
                                    float(d['sigma_q']), float(d['a_sec']))
Z_INT = d['z_int']
N_INTERFACE = len(Z_INT)

# ---------------------------------------------------------------- adimensionnement
T_WINDOW = float(d['t'][-1])                   # toute la duree simulee (7200 s)
ALPHA = K0 / (RHO * CP0)
DT_REF = E_BRAKE / (RHO * A_SEC * L * CP0)
FO = ALPHA * T_WINDOW / L**2
BI = H_END * L / K0
S_HAT = S_LAT * T_WINDOW / (RHO * CP0)
Q_SCALE = T_WINDOW / (RHO * CP0 * DT_REF)
TB_HAT = T_BRAKE / T_WINDOW
SIG_HAT = SIGMA_Q / L
ZI_HAT = torch.tensor(Z_INT / L, dtype=DTYPE, device=DEV)
THETA_0 = (T_INIT - T_AMB) / DT_REF
Q0_MAX = Q_SCALE * 2 * E_BRAKE / T_BRAKE / (L * A_SEC)     # amplitude de q0 a t = 0
print(f"FO = {FO:.3f}   BI = {BI:.3f}   S^ = {S_HAT:.3f}   TB^ = {TB_HAT:.4f}   dT_ref = {DT_REF:.0f} K")

def power_hat(th):
    return torch.where((th >= 0) & (th < TB_HAT),
                       2 * E_BRAKE / T_BRAKE * (1 - th / TB_HAT), torch.zeros_like(th))

def source_hat(zh, th):
    g = torch.exp(-0.5 * ((zh - ZI_HAT) / SIG_HAT)**2).sum(dim=1, keepdim=True)
    g = g / (SIG_HAT * np.sqrt(2 * np.pi)) / N_INTERFACE
    return Q_SCALE * power_hat(th) / (L * A_SEC) * g

# ---------------------------------------------------------------- theta_p : serie de cosinus
N_MODES = 240
n_ = torch.arange(N_MODES, dtype=DTYPE, device=DEV)                    # 0..N-1
LAM = (n_ * np.pi)**2 * FO + S_HAT                                      # (K,)
# coefficients g_n de la forme spatiale (moyenne des 8 gaussiennes, integrale 1)
G_N = (torch.cos(np.pi * n_[None, :] * ZI_HAT[:, None]).mean(dim=0)
       * torch.exp(-0.5 * (np.pi * n_ * SIG_HAT)**2))
G_N = torch.where(n_ == 0, torch.ones_like(G_N), 2 * G_N)               # g_0 = 1, g_n = 2 <g cos>

def theta_particular(zh, th):
    """Serie de cosinus (flux nul aux extremites + puits lateral), forme close
    de A_n pour la puissance triangulaire q0(t') = Q0_MAX (1 - t'/TB)."""
    u = torch.clamp(th, max=TB_HAT)                                     # (N, 1)
    E_u = torch.exp(-LAM[None, :] * (th - u))                           # (N, K)
    E_t = torch.exp(-LAM[None, :] * th)
    I1 = (E_u - E_t) / LAM[None, :]
    I2 = u * E_u / LAM[None, :] - I1 / LAM[None, :]
    A = Q0_MAX * (I1 - I2 / TB_HAT)                                     # (N, K)
    return (A * G_N[None, :] * torch.cos(np.pi * n_[None, :] * zh)).sum(dim=1, keepdim=True)

# ---------------------------------------------------------------- log-temps
TAU_0 = 0.2 * TB_HAT        # echelle du log : lineaire en dessous, logarithmique au-dessus
TAU_MAX = float(np.log1p(1.0 / TAU_0))

def tau_of(th):
    return torch.log1p(th / TAU_0) / TAU_MAX                            # [0, 1]

def t_of_tau(tau):
    return TAU_0 * torch.expm1(tau * TAU_MAX)

# ---------------------------------------------------------------- reseau
RAMP_HAT = 3 * TB_HAT      # montee de la correction (sa derivee 1/RAMP_HAT entre dans le residu)
FOURIER = True
N_FF = 32
FF_SCALE = (3.0, 2.0)       # (z^, tau) : correction lisse (la couche limite fine du freinage est sacrifiee)
WIDTH, DEPTH = 64, 4

class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        n_in = 2 + (2 * N_FF if FOURIER else 0)
        if FOURIER:
            self.register_buffer('B', torch.randn(2, N_FF) * torch.tensor(FF_SCALE)[:, None])
        layers, n = [], n_in
        for _ in range(DEPTH):
            layers += [nn.Linear(n, WIDTH), nn.Tanh()]
            n = WIDTH
        layers += [nn.Linear(n, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, zh, th):
        tau = tau_of(th)
        x = torch.cat([2 * zh - 1, 2 * tau - 1], dim=1)
        if FOURIER:
            p = 2 * np.pi * torch.cat([zh, tau], dim=1) @ self.B
            x = torch.cat([x, torch.sin(p), torch.cos(p)], dim=1)
        ramp = 1 - torch.exp(-th / RAMP_HAT)
        return THETA_0 + theta_particular(zh, th) + ramp * self.net(x)

def derivs(model, zh, th):
    zh = zh.requires_grad_(True)
    th = th.requires_grad_(True)
    theta = model(zh, th)
    theta_z, theta_t = torch.autograd.grad(theta, (zh, th), torch.ones_like(theta), create_graph=True)
    theta_zz = torch.autograd.grad(theta_z, zh, torch.ones_like(theta_z), create_graph=True)[0]
    return theta, theta_z, theta_t, theta_zz

# ---------------------------------------------------------------- echantillonnage
N_PDE, N_BC, RESAMPLE_EVERY = 4000, 400, 200
W_PDE, W_BC, W_DATA = 1.0, 100.0, 1.0   # la CL de Robin est la seule information que le reseau doit porter : poids fort

def sample_pde(n):
    """Temps uniforme en tau (log-temps) ; z uniforme, plus points d'interface
    pour les instants de freinage."""
    th = t_of_tau(torch.rand(n, 1))
    zh = torch.rand(n, 1)
    brake = (th < 1.5 * TB_HAT).squeeze()
    nb = int(brake.sum())
    if nb:
        idx = torch.randint(0, N_INTERFACE, (nb, 1))
        z_i = (ZI_HAT.cpu()[idx] + 4 * SIG_HAT * torch.randn(nb, 1)).clamp(0, 1)
        half = torch.rand(nb, 1) < 0.5
        zh[brake] = torch.where(half, z_i, zh[brake])
    return zh.to(DEV), th.to(DEV)

def sample_bc(n):
    return t_of_tau(torch.rand(n, 1)).to(DEV)

USE_SENSOR = False
z_sens = torch.full((len(d['t_sens']), 1), float(d['z_sensor']) / L, dtype=DTYPE, device=DEV)
t_sens = torch.tensor(d['t_sens'] / T_WINDOW, dtype=DTYPE, device=DEV)[:, None]
th_sens = torch.tensor((d['T_sens'] - T_AMB) / DT_REF, dtype=DTYPE, device=DEV)[:, None]

# ---------------------------------------------------------------- pertes
def losses(model, zh, th, th_bc):
    theta, _, theta_t, theta_zz = derivs(model, zh, th)
    res = theta_t - FO * theta_zz - source_hat(zh, th) + S_HAT * theta
    l_pde = (res**2).mean()
    z0, z1 = torch.zeros_like(th_bc), torch.ones_like(th_bc)
    th0, thz0, _, _ = derivs(model, z0, th_bc)
    th1, thz1, _, _ = derivs(model, z1, th_bc)
    l_bc = ((thz0 - BI * th0)**2).mean() + ((thz1 + BI * th1)**2).mean()
    l_data = ((model(z_sens, t_sens) - th_sens)**2).mean() if USE_SENSOR else torch.tensor(0.0)
    return l_pde, l_bc, l_data

# ---------------------------------------------------------------- entrainement
model = PINN().to(DEV)
N_ADAM, N_LBFGS = 500, 3000
hist = []
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
t0_ = time.time()
for it in range(N_ADAM):
    if it % RESAMPLE_EVERY == 0:
        zh, th = sample_pde(N_PDE)
        th_bc = sample_bc(N_BC)
    opt.zero_grad()
    l_pde, l_bc, l_data = losses(model, zh, th, th_bc)
    loss = W_PDE * l_pde + W_BC * l_bc + W_DATA * l_data
    loss.backward()
    opt.step()
    hist.append([l_pde.item(), l_bc.item(), l_data.item()])
    if it % 100 == 0:
        print(f"Adam {it:5d}  pde {l_pde.item():.2e}  bc {l_bc.item():.2e}  ({time.time() - t0_:.0f} s)")

zh, th = sample_pde(2 * N_PDE)
th_bc = sample_bc(2 * N_BC)
opt = torch.optim.LBFGS(model.parameters(), max_iter=N_LBFGS, history_size=50,
                        tolerance_grad=1e-9, tolerance_change=1e-12, line_search_fn='strong_wolfe')

def closure():
    opt.zero_grad()
    l_pde, l_bc, l_data = losses(model, zh, th, th_bc)
    loss = W_PDE * l_pde + W_BC * l_bc + W_DATA * l_data
    loss.backward()
    hist.append([l_pde.item(), l_bc.item(), l_data.item()])
    return loss

opt.step(closure)
print(f"L-BFGS termine : pde {hist[-1][0]:.2e}  bc {hist[-1][1]:.2e}  ({time.time() - t0_:.0f} s)")

# ---------------------------------------------------------------- evaluation
z_fd = d['z']
sel = np.arange(0, len(d['t']), 4)
t_fd = d['t'][sel]
T_fd = d['T'][sel].astype(np.float64)
ZZ, TT = np.meshgrid(z_fd / L, t_fd / T_WINDOW)
zz = torch.tensor(ZZ.ravel()[:, None], dtype=DTYPE, device=DEV)
tt = torch.tensor(TT.ravel()[:, None], dtype=DTYPE, device=DEV)
with torch.no_grad():
    theta = torch.cat([model(zz[i:i + 20000], tt[i:i + 20000]) for i in range(0, len(zz), 20000)])
    theta_p_only = torch.cat([theta_particular(zz[i:i + 20000], tt[i:i + 20000])
                              for i in range(0, len(zz), 20000)])
T_pinn = T_AMB + DT_REF * theta.cpu().numpy().reshape(ZZ.shape)
T_p = T_AMB + DT_REF * theta_p_only.cpu().numpy().reshape(ZZ.shape)
err = T_pinn - T_fd
print(f"\nRMSE global 0..2 h : {np.sqrt((err**2).mean()):.2f} K   max |err| : {np.abs(err).max():.1f} K")
print("  t (s)   RMSE PINN   RMSE theta_p seule   T_moy FD / PINN / theta_p (C)")
for tt_ in [10, 40, 120, 600, 1800, 3600, 7200]:
    i = min(np.searchsorted(t_fd, tt_), len(t_fd) - 1)
    print(f"  {tt_:5d}   {np.sqrt((err[i]**2).mean()):8.2f}   {np.sqrt(((T_p[i] - T_fd[i])**2).mean()):8.2f}"
          f"         {T_fd[i].mean() - 273.15:6.1f} / {T_pinn[i].mean() - 273.15:6.1f} / {T_p[i].mean() - 273.15:6.1f}")

# ---------------------------------------------------------------- figures
hist = np.array(hist)
fig, ax = plt.subplots(2, 2, figsize=(12, 8))
ax[0, 0].semilogy(hist[:, 0], label='EDP')
ax[0, 0].semilogy(hist[:, 1], label='CL')
ax[0, 0].axvline(N_ADAM, color='k', lw=0.5)
ax[0, 0].set(xlabel='iteration', ylabel='perte', title='Convergence')
ax[0, 0].legend()

for tt_, c in zip([10, 40, 120, 600, 1800, 7200], plt.cm.viridis(np.linspace(0, 1, 6))):
    i = min(np.searchsorted(t_fd, tt_), len(t_fd) - 1)
    ax[0, 1].plot(z_fd * 1e3, T_fd[i] - 273.15, color=c, label=f'FD {tt_} s')
    ax[0, 1].plot(z_fd * 1e3, T_pinn[i] - 273.15, '--', color=c)
ax[0, 1].set(xlabel='z (mm)', ylabel='T (C)', title='Profils : FD (trait) vs PINN (tirets)')
ax[0, 1].legend(fontsize=10, ncol=2, framealpha=0.85)

i_z = np.argmin(np.abs(z_fd - float(d['z_sensor'])))
ax[1, 0].plot(t_fd / 60, T_fd[:, i_z] - 273.15, label='FD capteur')
ax[1, 0].plot(t_fd / 60, T_pinn[:, i_z] - 273.15, '--', label='PINN capteur')
ax[1, 0].plot(t_fd / 60, T_p[:, i_z] - 273.15, ':', label='theta_p seule (sans convection aux bouts)')
ax[1, 0].plot(t_fd / 60, T_fd[:, 0] - 273.15, label='FD bord z = 0')
ax[1, 0].plot(t_fd / 60, T_pinn[:, 0] - 273.15, '--', label='PINN bord z = 0')
ax[1, 0].set(xlabel='t (min)', ylabel='T (C)', title='Historiques', xscale='log', xlim=(0.05, 120))
ax[1, 0].legend(fontsize=10)

emax = np.abs(err).max()
pc = ax[1, 1].pcolormesh(t_fd / 60, z_fd * 1e3, err.T, shading='auto', cmap='RdBu_r', vmin=-emax, vmax=emax)
ax[1, 1].set(xlabel='t (min)', ylabel='z (mm)', title='PINN - FD (K)', xscale='log', xlim=(0.05, 120))
fig.colorbar(pc, ax=ax[1, 1])
fig.tight_layout()
plt.show()
