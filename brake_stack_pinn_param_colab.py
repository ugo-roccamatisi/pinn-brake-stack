"""
PINN PARAMETRIQUE 1D : un reseau pour une famille de scenarios de freinage.

Entrees du reseau : (z^, tau, p) avec p = 5 parametres de scenario :
    E_brake  [MJ]    energie dissipee par frein          4 .. 25
    t_brake  [s]     duree du roulement freine           20 .. 60
    T_init   [C]     temperature initiale uniforme       20 .. 300  (vols successifs)
    h_lat    [W/m2K] convection laterale                 5 .. 40    (ventilateurs)
    h_end    [W/m2K] convection aux extremites           5 .. 40
Les points de collocation sont tires dans (z, t, p) : le reseau apprend la
solution pour TOUT p de la boite en un seul entrainement, sans aucune donnee.
Ensuite predict(p, z, t) donne T(z, t) en quelques microsecondes par point.

Structure identique a brake_stack_pinn_v3.py : theta = theta_p + rampe * N,
theta_p = serie de cosinus (exacte sauf convection aux extremites), qui depend
analytiquement de E, t_brake, T_init, h_lat ; le reseau porte l'effet de h_end.

Validation : le script contient le solveur FD 1D (fonction fd_solve) et
compare le PINN a des scenarios tires au hasard, jamais vus a l'entrainement
(ils ne peuvent pas l'etre : l'entrainement ne voit pas de scenarios, il voit
des points).
"""

import os
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
plt.rcParams.update({'font.size': 20, 'axes.titlesize': 20, 'axes.labelsize': 20,
                     'xtick.labelsize': 18, 'ytick.labelsize': 18,
                     'legend.fontsize': 15, 'lines.linewidth': 1.8})

torch.manual_seed(0)
np.random.seed(0)
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DTYPE = torch.float32
print("Device :", DEV)

# ---------------------------------------------------------------- sauvegarde Drive
# Sur Google Colab, /content est efface a la fin de la session : on monte le
# Drive et on sauvegarde dessus (checkpoint apres Adam + modele final).
# Hors Colab, le script retombe sur le dossier courant.
SAVE_DIR = '.'
try:
    from google.colab import drive
    drive.mount('/content/drive')
    SAVE_DIR = '/content/drive/MyDrive'
except Exception as e:
    print("Drive non monte (", e, ") : sauvegarde dans le dossier courant,")
    print("pensez a copier le .pt ailleurs avant la fin de la session Colab.")
CKPT = os.path.join(SAVE_DIR, 'brake_stack_pinn_param.pt')
print("Le modele sera sauvegarde dans :", CKPT)

# ---------------------------------------------------------------- constantes fixes
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

# ---------------------------------------------------------------- boite de scenarios
P_NAMES = ['E_brake_MJ', 't_brake_s', 'T_init_C', 'h_lat', 'h_end']
P_LO = np.array([4.0, 20.0, 20.0, 5.0, 5.0])
P_HI = np.array([25.0, 60.0, 300.0, 40.0, 40.0])

# ---------------------------------------------------------------- adimensionnement (echelles FIXES)
ALPHA = K0 / (RHO * CP0)
DT_REF = P_HI[0] * 1e6 / (RHO * A_SEC * L * CP0)     # elevation adiabatique du cas le plus energetique
FO = ALPHA * T_WINDOW / L**2
SIG_HAT = SIGMA_Q / L
ZI_HAT = torch.tensor(Z_INT / L, dtype=DTYPE, device=DEV)
print(f"FO = {FO:.3f}   dT_ref = {DT_REF:.0f} K   fenetre {T_WINDOW:.0f} s")

def unpack(p):
    """p : (N, 5) tenseur physique -> grandeurs adimensionnees, chacune (N, 1)."""
    E = p[:, 0:1] * 1e6
    tb = p[:, 1:2] / T_WINDOW
    theta0 = (p[:, 2:3] + 273.15 - T_AMB) / DT_REF
    s_hat = p[:, 3:4] * PERIM / A_SEC * T_WINDOW / (RHO * CP0)
    bi = p[:, 4:5] * L / K0
    q0 = (T_WINDOW / (RHO * CP0 * DT_REF)) * 2 * E / p[:, 1:2] / (L * A_SEC)   # amplitude de q0 a t = 0
    return tb, theta0, s_hat, bi, q0

def power_hat(th, tb, q0):
    return torch.where((th >= 0) & (th < tb), q0 * (1 - th / tb), torch.zeros_like(th))

def source_hat(zh, th, tb, q0):
    g = torch.exp(-0.5 * ((zh - ZI_HAT) / SIG_HAT)**2).sum(dim=1, keepdim=True)
    g = g / (SIG_HAT * np.sqrt(2 * np.pi)) / N_INTERFACE
    return power_hat(th, tb, q0) * g

# ---------------------------------------------------------------- theta_p : serie de cosinus parametrique
N_MODES = 240
n_ = torch.arange(N_MODES, dtype=DTYPE, device=DEV)
LAM_DIFF = (n_ * np.pi)**2 * FO                                           # (K,) ; + s_hat par scenario
G_N = (torch.cos(np.pi * n_[None, :] * ZI_HAT[:, None]).mean(dim=0)
       * torch.exp(-0.5 * (np.pi * n_ * SIG_HAT)**2))
G_N = torch.where(n_ == 0, torch.ones_like(G_N), 2 * G_N)

def theta_particular(zh, th, tb, theta0, s_hat, q0):
    lam = LAM_DIFF[None, :] + s_hat                                        # (N, K)
    u = torch.minimum(th, tb)
    E_u = torch.exp(-lam * (th - u))
    E_t = torch.exp(-lam * th)
    I1 = (E_u - E_t) / lam
    I2 = u * E_u / lam - I1 / lam
    A = q0 * (I1 - I2 / tb)
    series = (A * G_N[None, :] * torch.cos(np.pi * n_[None, :] * zh)).sum(dim=1, keepdim=True)
    return theta0 * torch.exp(-s_hat * th) + series      # CI uniforme : mode 0 seul, decroissance exp(-S^ t^)

# ---------------------------------------------------------------- log-temps
TAU_0 = 0.001
TAU_MAX = float(np.log1p(1.0 / TAU_0))
def tau_of(th):
    return torch.log1p(th / TAU_0) / TAU_MAX
def t_of_tau(tau):
    return TAU_0 * torch.expm1(tau * TAU_MAX)

# ---------------------------------------------------------------- reseau
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

def derivs(model, zh, th, p):
    zh = zh.requires_grad_(True)
    th = th.requires_grad_(True)
    theta = model(zh, th, p)
    theta_z, theta_t = torch.autograd.grad(theta, (zh, th), torch.ones_like(theta), create_graph=True)
    theta_zz = torch.autograd.grad(theta_z, zh, torch.ones_like(theta_z), create_graph=True)[0]
    return theta, theta_z, theta_t, theta_zz

# ---------------------------------------------------------------- echantillonnage dans (z, t, p)
N_PDE, N_BC, RESAMPLE_EVERY = 6000, 600, 200
W_PDE, W_BC = 1.0, 100.0

def sample_params(n):
    return (P_LO_T + (P_HI_T - P_LO_T) * torch.rand(n, 5, device=DEV))

def sample_pde(n):
    p = sample_params(n)
    th = t_of_tau(torch.rand(n, 1, device=DEV))
    zh = torch.rand(n, 1, device=DEV)
    tb = p[:, 1:2] / T_WINDOW
    brake = (th < 1.5 * tb).squeeze()
    nb = int(brake.sum())
    if nb:
        idx = torch.randint(0, N_INTERFACE, (nb, 1), device=DEV)
        z_i = (ZI_HAT[idx] + 4 * SIG_HAT * torch.randn(nb, 1, device=DEV)).clamp(0, 1)
        half = torch.rand(nb, 1, device=DEV) < 0.5
        zh[brake] = torch.where(half, z_i, zh[brake])
    return zh, th, p

def sample_bc(n):
    return t_of_tau(torch.rand(n, 1, device=DEV)), sample_params(n)

def losses(model, zh, th, p, th_bc, p_bc):
    tb, _, s_hat, _, q0 = unpack(p)
    theta, _, theta_t, theta_zz = derivs(model, zh, th, p)
    res = theta_t - FO * theta_zz - source_hat(zh, th, tb, q0) + s_hat * theta
    l_pde = (res**2).mean()
    _, _, _, bi, _ = unpack(p_bc)
    z0, z1 = torch.zeros_like(th_bc), torch.ones_like(th_bc)
    th0, thz0, _, _ = derivs(model, z0, th_bc, p_bc)
    th1, thz1, _, _ = derivs(model, z1, th_bc, p_bc)
    l_bc = ((thz0 - bi * th0)**2).mean() + ((thz1 + bi * th1)**2).mean()
    return l_pde, l_bc

# ---------------------------------------------------------------- entrainement
model = ParamPINN().to(DEV)
N_ADAM, N_LBFGS = 1000, 4000
hist = []
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
t0_ = time.time()
for it in range(N_ADAM):
    if it % RESAMPLE_EVERY == 0:
        zh, th, p = sample_pde(N_PDE)
        th_bc, p_bc = sample_bc(N_BC)
    opt.zero_grad()
    l_pde, l_bc = losses(model, zh, th, p, th_bc, p_bc)
    loss = W_PDE * l_pde + W_BC * l_bc
    loss.backward()
    opt.step()
    hist.append([l_pde.item(), l_bc.item()])
    if it % 100 == 0:
        print(f"Adam {it:5d}  pde {l_pde.item():.2e}  bc {l_bc.item():.2e}  ({time.time() - t0_:.0f} s)")

torch.save(model.state_dict(), CKPT)          # checkpoint : Adam termine
print("Checkpoint (fin d'Adam) sauvegarde.")

zh, th, p = sample_pde(3 * N_PDE)
th_bc, p_bc = sample_bc(3 * N_BC)
opt = torch.optim.LBFGS(model.parameters(), max_iter=N_LBFGS, history_size=50,
                        tolerance_grad=1e-9, tolerance_change=1e-12, line_search_fn='strong_wolfe')

def closure():
    opt.zero_grad()
    l_pde, l_bc = losses(model, zh, th, p, th_bc, p_bc)
    loss = W_PDE * l_pde + W_BC * l_bc
    loss.backward()
    hist.append([l_pde.item(), l_bc.item()])
    return loss

opt.step(closure)
print(f"L-BFGS termine : pde {hist[-1][0]:.2e}  bc {hist[-1][1]:.2e}  ({time.time() - t0_:.0f} s)")
torch.save(model.state_dict(), CKPT)
print("Modele final sauvegarde dans :", CKPT)

# ---------------------------------------------------------------- API de prediction
def predict(p_phys, z, t):
    """p_phys : 5 valeurs physiques ; z (m), t (s) : tableaux 1D -> T (t, z) en C."""
    ZZ, TT = np.meshgrid(z / L, t / T_WINDOW)
    zz = torch.tensor(ZZ.ravel()[:, None], dtype=DTYPE, device=DEV)
    tt = torch.tensor(TT.ravel()[:, None], dtype=DTYPE, device=DEV)
    pp = torch.tensor(np.asarray(p_phys, dtype=np.float32), device=DEV).expand(len(zz), 5)
    with torch.no_grad():
        th = torch.cat([model(zz[i:i + 20000], tt[i:i + 20000], pp[i:i + 20000])
                        for i in range(0, len(zz), 20000)])
    return (T_AMB + DT_REF * th.cpu().numpy().reshape(ZZ.shape)) - 273.15

# ---------------------------------------------------------------- solveur FD (validation)
def fd_solve(p_phys, nz=450):
    """Reference volumes finis / Crank-Nicolson pour un scenario. Retourne z (m), t (s), T (t, z) en C."""
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

# ---------------------------------------------------------------- validation sur scenarios tires au hasard
rng = np.random.default_rng(1)
N_TEST = 4
tests = P_LO + (P_HI - P_LO) * rng.random((N_TEST, 5))
tests[0] = [12.0, 40.0, 20.0, 15.0, 20.0]        # le scenario de reference des scripts precedents
print("\nValidation contre le FD (scenarios non vus) :")
print("  " + "  ".join(f"{n:>10s}" for n in P_NAMES) + "    RMSE 0-2h   max|err|   RMSE capteur   pic FD / PINN")
results = []
for p_test in tests:
    z, t, T_fd = fd_solve(p_test)
    sel = np.arange(0, len(t), 3)
    T_pinn = predict(p_test, z, t[sel])
    err = T_pinn - T_fd[sel]
    i_z = np.argmin(np.abs(z - 4.5 * E_DISC))
    results.append((p_test, z, t[sel], T_fd[sel], T_pinn))
    print("  " + "  ".join(f"{v:10.1f}" for v in p_test)
          + f"    {np.sqrt((err**2).mean()):7.2f}    {np.abs(err).max():7.1f}    "
          f"{np.sqrt((err[:, i_z]**2).mean()):7.2f}       {T_fd.max():5.0f} / {T_pinn.max():5.0f}")

# ---------------------------------------------------------------- figures
hist = np.array(hist)
fig, ax = plt.subplots(2, 3, figsize=(16, 9))
ax[0, 0].semilogy(hist[:, 0], label='EDP'); ax[0, 0].semilogy(hist[:, 1], label='CL')
ax[0, 0].axvline(N_ADAM, color='k', lw=0.5)
ax[0, 0].set(xlabel='iteration', ylabel='perte', title='Convergence'); ax[0, 0].legend()
for k, (p_test, z, t, T_fd, T_pinn) in enumerate(results):
    a = ax.ravel()[k + 1]
    i_z = np.argmin(np.abs(z - 4.5 * E_DISC))
    a.plot(t / 60, T_fd[:, i_z], label='FD capteur'); a.plot(t / 60, T_pinn[:, i_z], '--', label='PINN capteur')
    a.plot(t / 60, T_fd[:, 0], label='FD bord'); a.plot(t / 60, T_pinn[:, 0], '--', label='PINN bord')
    a.set(xscale='log', xlim=(0.05, 120), xlabel='t (min)', ylabel='T (C)',
          title=f"E={p_test[0]:.0f} MJ, tb={p_test[1]:.0f} s, T0={p_test[2]:.0f} C, h_lat={p_test[3]:.0f}, h_end={p_test[4]:.0f}")
    a.legend(fontsize=15)
# balayage : pic de temperature capteur en fonction de E, pour 3 T_init (PINN seul, instantane)
a = ax[1, 2]
E_grid = np.linspace(P_LO[0], P_HI[0], 30)
z1 = np.array([4.5 * E_DISC]); t1 = np.linspace(0, 300, 301)
for T0 in [20, 150, 300]:
    peaks = [predict([E, 40.0, T0, 15.0, 20.0], z1, t1).max() for E in E_grid]
    a.plot(E_grid, peaks, label=f'T_init = {T0} C')
a.set(xlabel='E_brake (MJ)', ylabel='pic capteur (C)', title='Balayage PINN : pic vs energie (tb = 40 s)')
a.legend(fontsize=15)
fig.tight_layout()
plt.show()
