"""
PINN 1D pour la pile de freins : probleme DIRECT sur la fenetre 0..T_WINDOW.

Lit brake_stack_data.npz (produit par brake_stack_fd.py) pour recuperer les
constantes physiques (memes valeurs, meme lissage des sources) et la verite
terrain T(z,t) qui sert uniquement a NOTER le PINN, pas a l'entrainer
(sauf si USE_SENSOR = True, premier pas vers le probleme inverse).

Adimensionnement :
    z^ = z / L,   t^ = t / T_WINDOW,   theta = (T - T_amb) / dT_ref
    dT_ref = E_brake / (m_pile cp)   (elevation adiabatique)

EDP adimensionnee :
    d theta/dt^ = FO d2theta/dz^2 + Q^(z^,t^) - S^ theta
    z^ = 0 :  d theta/dz^ - BI theta = 0
    z^ = 1 :  d theta/dz^ + BI theta = 0
Condition initiale encodee en dur : theta = theta_0 + t^ * N(z^, t^).

v2, trois ameliorations par rapport au squelette :
  1. residu EDP normalise par max Q^ (perte d'ordre 1)
  2. curriculum temporel : t^ echantillonne dans [0, f] avec f qui monte de
     T_FRAC_0 a 1 pendant la premiere moitie d'Adam
  3. ANSATZ = 'green' : theta = theta_0 + theta_p + t^ * N, ou theta_p est la
     solution particuliere exacte (milieu infini, fonction de Green de la
     chaleur + puits lateral) calculee par quadrature en temps. Le reseau
     n'apprend que la correction due aux extremites (Robin), qui est lisse.
     ANSATZ = 'plain' retrouve le squelette v1.
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
if float(d['alpha_k']) != 0.0 or float(d['alpha_cp']) != 0.0:
    print("ATTENTION : donnees FD non lineaires, ce squelette suppose k, cp constants")

# ---------------------------------------------------------------- adimensionnement
T_WINDOW = 300.0                                   # s, fenetre etudiee
ALPHA = K0 / (RHO * CP0)
DT_REF = E_BRAKE / (RHO * A_SEC * L * CP0)
FO = ALPHA * T_WINDOW / L**2                       # nombre de Fourier
BI = H_END * L / K0                                # Biot aux extremites
S_HAT = S_LAT * T_WINDOW / (RHO * CP0)             # puits lateral
Q_SCALE = T_WINDOW / (RHO * CP0 * DT_REF)          # q_vol -> Q^
TB_HAT = T_BRAKE / T_WINDOW
SIG_HAT = SIGMA_Q / L
ZI_HAT = torch.tensor(Z_INT / L, dtype=DTYPE, device=DEV)
THETA_0 = (T_INIT - T_AMB) / DT_REF
print(f"FO = {FO:.4f}   BI = {BI:.3f}   S^ = {S_HAT:.4f}   "
      f"dT_ref = {DT_REF:.0f} K   sigma^ = {SIG_HAT:.4f}")

def power_hat(th):
    """P(t^) en W."""
    return torch.where((th >= 0) & (th < TB_HAT),
                       2 * E_BRAKE / T_BRAKE * (1 - th / TB_HAT),
                       torch.zeros_like(th))

def source_hat(zh, th):
    """Q^(z^, t^) : terme source adimensionne, memes gaussiennes que le FD."""
    g = torch.exp(-0.5 * ((zh - ZI_HAT) / SIG_HAT)**2).sum(dim=1, keepdim=True)
    g = g / (SIG_HAT * np.sqrt(2 * np.pi)) / N_INTERFACE     # integrale = 1 en z^
    q_vol = power_hat(th) * g / (L * A_SEC)                   # W/m3
    return Q_SCALE * q_vol

ANSATZ = 'green'            # 'green' ou 'plain'
IMAGES = True               # sources images (Neumann aux extremites) dans theta_p
N_QUAD = 48                 # noeuds de Gauss-Legendre (en log w) pour theta_p

def q0_hat(th):
    """Amplitude de la source : Q^ = q0_hat(t^) * g(z^)."""
    return Q_SCALE * power_hat(th) / (L * A_SEC)

# Sources vues par theta_p : les 8 interfaces, plus leurs images miroir par
# rapport a z^=0 et z^=1 (methode des images). Avec les images, theta_p verifie
# exactement d theta/dz = 0 aux deux extremites (pas de fuite vers l'infini) :
# le reseau n'a plus a reconstruire la chaleur reflechie (~150 K aux bords),
# seulement la petite perte convective de Robin (Bi = 0.3). Les images d'ordre
# superieur (2 + z^i, -2 + z^i...) sont a plus de 1.8 de la pile, negligeables
# devant la longueur de diffusion sqrt(2 FO) ~ 0.32 sur la fenetre.
Z_SRC = torch.cat([ZI_HAT, -ZI_HAT, 2 - ZI_HAT]) if IMAGES else ZI_HAT

_GL_X, _GL_W = np.polynomial.legendre.leggauss(N_QUAD)
GL_V = torch.tensor(0.5 * (_GL_X + 1), dtype=DTYPE, device=DEV)      # noeuds sur [0,1]
GL_W = torch.tensor(0.5 * _GL_W, dtype=DTYPE, device=DEV)            # poids sur [0,1]

def theta_particular(zh, th):
    """Solution exacte de d theta/dt = FO theta_zz + Q^ - S^ theta en milieu infini
    (gaussiennes qui diffusent). Integrale sur s = th - t' avec le changement de
    variable w = sqrt(sig^2 + 2 FO s) (largeur de la gaussienne diffusee), qui
    supprime la singularite 1/w. Deux precautions indispensables :
      - la borne basse est fixee a t' = min(th, TB) : la source est nulle apres
        le freinage, integrer au-dela mettait le point anguleux de P(t) au
        milieu de la quadrature (residu irreductible ~1e-2, plateau observe) ;
      - noeuds de Gauss-Legendre en log(w), pour resoudre la gaussienne quand
        elle est encore fine (w ~ sig) tout en couvrant w_max ~ 0.3.
    Residu EDP absolu de theta_p seule avec N_QUAD = 48 : ~1e-8."""
    w_max = torch.sqrt(SIG_HAT**2 + 2 * FO * th)                              # (N, 1)
    w_lo = torch.sqrt(SIG_HAT**2 + 2 * FO * (th - TB_HAT).clamp(min=0))       # (N, 1)
    lr = torch.log(w_max / w_lo)                                              # (N, 1)
    w = w_lo * torch.exp(lr * GL_V[None, :])                                  # (N, M)
    s_ = (w**2 - SIG_HAT**2) / (2 * FO)
    tp = th - s_
    dz2 = (zh[:, None, :] - Z_SRC[None, None, :])**2                          # (N, 1, I)
    g = torch.exp(-0.5 * dz2 / w[:, :, None]**2).sum(dim=2)                   # (N, M)
    integrand = q0_hat(tp) * torch.exp(-S_HAT * s_) * g / (np.sqrt(2 * np.pi) * N_INTERFACE * FO)
    return ((integrand * w) @ GL_W)[:, None] * lr                             # dw = w lr dv

Q_MAX = float(q0_hat(torch.zeros(1, 1, device=DEV))) / (SIG_HAT * np.sqrt(2 * np.pi)) / N_INTERFACE
# Echelle de normalisation du residu EDP : Q^_max pour 'plain' (le reseau doit
# produire la source entiere) ; O(1) pour 'green' (theta_p absorbe la source,
# le residu restant est celui de la correction, d'ordre theta_t ~ 1). Normaliser
# par Q^_max en 'green' sous-pondere l'EDP face aux CL et laisse un residu de
# plusieurs unites : c'etait la cause du centre trop chaud / bords trop froids.
RES_SCALE = Q_MAX if ANSATZ == 'plain' else 1.0
print(f"Q^_max = {Q_MAX:.1f}   echelle du residu = {RES_SCALE:.1f}")

# ---------------------------------------------------------------- reseau
FOURIER = True
N_FF = 64
FF_SCALE = (6.0, 2.0)       # frequences (z^, t^) ; avec l'ansatz Green, N est lisse
WIDTH, DEPTH = 64, 4

class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        if FOURIER:
            B = torch.randn(2, N_FF) * torch.tensor(FF_SCALE)[:, None]
            self.register_buffer('B', B)
            n_in = 2 + 2 * N_FF
        else:
            n_in = 2
        layers, n = [], n_in
        for _ in range(DEPTH):
            layers += [nn.Linear(n, WIDTH), nn.Tanh()]
            n = WIDTH
        layers += [nn.Linear(n, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, zh, th):
        x = torch.cat([2 * zh - 1, 2 * th - 1], dim=1)
        if FOURIER:
            p = 2 * np.pi * torch.cat([zh, th], dim=1) @ self.B
            x = torch.cat([x, torch.sin(p), torch.cos(p)], dim=1)
        out = THETA_0 + th * self.net(x)       # CI exacte : theta(t^=0) = theta_0
        if ANSATZ == 'green':
            out = out + theta_particular(zh, th)
        return out

def derivs(model, zh, th):
    zh = zh.requires_grad_(True)
    th = th.requires_grad_(True)
    theta = model(zh, th)
    g = torch.autograd.grad(theta, (zh, th), torch.ones_like(theta), create_graph=True)
    theta_z, theta_t = g
    theta_zz = torch.autograd.grad(theta_z, zh, torch.ones_like(theta_z),
                                   create_graph=True)[0]
    return theta, theta_z, theta_t, theta_zz

# ---------------------------------------------------------------- echantillonnage
N_PDE, N_BC, RESAMPLE_EVERY = 4000, 400, 200
CURRICULUM, T_FRAC_0 = True, 0.15
T_FRAC = 1.0                # borne courante de t^ pour l'echantillonnage
W_PDE, W_BC, W_DATA = 1.0, 1.0, 1.0

def sample_pde(n):
    """Moitie uniforme, moitie concentree pres des interfaces et du freinage."""
    n2 = n // 2
    z_u = torch.rand(n2, 1)
    idx = torch.randint(0, N_INTERFACE, (n - n2, 1))
    z_i = (ZI_HAT.cpu()[idx] + 4 * SIG_HAT * torch.randn(n - n2, 1)).clamp(0, 1)
    t_u = torch.rand(n2, 1) * T_FRAC
    t_b = torch.rand(n - n2, 1) * min(1.5 * TB_HAT, T_FRAC)
    zh = torch.cat([z_u, z_i]).to(DEV)
    th = torch.cat([t_u, t_b]).to(DEV)
    return zh, th

def sample_bc(n):
    return torch.rand(n, 1, device=DEV) * T_FRAC

# capteur (desactive par defaut : probleme direct pur)
USE_SENSOR = False
m_s = d['t_sens'] <= T_WINDOW
z_sens = torch.full((m_s.sum(), 1), float(d['z_sensor']) / L, dtype=DTYPE, device=DEV)
t_sens = torch.tensor(d['t_sens'][m_s] / T_WINDOW, dtype=DTYPE, device=DEV)[:, None]
th_sens = torch.tensor((d['T_sens'][m_s] - T_AMB) / DT_REF, dtype=DTYPE, device=DEV)[:, None]

# ---------------------------------------------------------------- pertes
def losses(model, zh, th, th_bc):
    theta, _, theta_t, theta_zz = derivs(model, zh, th)
    res = theta_t - FO * theta_zz - source_hat(zh, th) + S_HAT * theta
    l_pde = ((res / RES_SCALE)**2).mean()

    z0 = torch.zeros_like(th_bc)
    z1 = torch.ones_like(th_bc)
    th0, thz0, _, _ = derivs(model, z0, th_bc)
    th1, thz1, _, _ = derivs(model, z1, th_bc)
    l_bc = ((thz0 - BI * th0)**2).mean() + ((thz1 + BI * th1)**2).mean()

    l_data = ((model(z_sens, t_sens) - th_sens)**2).mean() if USE_SENSOR else torch.tensor(0.0)
    return l_pde, l_bc, l_data

# ---------------------------------------------------------------- entrainement
model = PINN().to(DEV)
N_ADAM, N_LBFGS = 1500, 4000
hist = []

opt = torch.optim.Adam(model.parameters(), lr=1e-3)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, N_ADAM, eta_min=1e-4)
t0 = time.time()
for it in range(N_ADAM):
    if it % RESAMPLE_EVERY == 0:
        if CURRICULUM:
            T_FRAC = min(1.0, T_FRAC_0 + (1 - T_FRAC_0) * it / (0.5 * N_ADAM))
        zh, th = sample_pde(N_PDE)
        th_bc = sample_bc(N_BC)
    opt.zero_grad()
    l_pde, l_bc, l_data = losses(model, zh, th, th_bc)
    loss = W_PDE * l_pde + W_BC * l_bc + W_DATA * l_data
    loss.backward()
    opt.step()
    sched.step()
    hist.append([l_pde.item(), l_bc.item(), l_data.item()])
    if it % 500 == 0:
        print(f"Adam {it:5d}  pde {l_pde.item():.2e}  bc {l_bc.item():.2e}  "
              f"data {l_data.item():.2e}  ({time.time() - t0:.0f} s)")

T_FRAC = 1.0
zh, th = sample_pde(2 * N_PDE)
th_bc = sample_bc(2 * N_BC)
opt = torch.optim.LBFGS(model.parameters(), max_iter=N_LBFGS, history_size=50,
                        tolerance_grad=1e-9, tolerance_change=1e-12,
                        line_search_fn='strong_wolfe')

def closure():
    opt.zero_grad()
    l_pde, l_bc, l_data = losses(model, zh, th, th_bc)
    loss = W_PDE * l_pde + W_BC * l_bc + W_DATA * l_data
    loss.backward()
    hist.append([l_pde.item(), l_bc.item(), l_data.item()])
    return loss

opt.step(closure)
print(f"L-BFGS termine : pde {hist[-1][0]:.2e}  bc {hist[-1][1]:.2e}  "
      f"({time.time() - t0:.0f} s)")

# ---------------------------------------------------------------- evaluation
z_fd = d['z']
m_t = d['t'] <= T_WINDOW
t_fd = d['t'][m_t]
T_fd = d['T'][m_t].astype(np.float64)

ZZ, TT = np.meshgrid(z_fd / L, t_fd / T_WINDOW)
zz = torch.tensor(ZZ.ravel()[:, None], dtype=DTYPE, device=DEV)
tt = torch.tensor(TT.ravel()[:, None], dtype=DTYPE, device=DEV)
with torch.no_grad():                       # par lots : theta_p coute N x N_QUAD x 8
    theta = torch.cat([model(zz[i:i + 20000], tt[i:i + 20000])
                       for i in range(0, len(zz), 20000)])
T_pinn = T_AMB + DT_REF * theta.cpu().numpy().reshape(ZZ.shape)
err = T_pinn - T_fd
print(f"\nRMSE global 0..{T_WINDOW:.0f} s : {np.sqrt((err**2).mean()):.2f} K   "
      f"max |err| : {np.abs(err).max():.1f} K")
for tt in [10, 40, 60, 120, 300]:
    i = np.searchsorted(t_fd, tt)
    print(f"  t = {tt:4d} s : RMSE {np.sqrt((err[i]**2).mean()):6.2f} K   "
          f"T_max FD {T_fd[i].max() - 273.15:6.1f} C   PINN {T_pinn[i].max() - 273.15:6.1f} C")

# ---------------------------------------------------------------- figures
fig, ax = plt.subplots(2, 2, figsize=(12, 8))
hist = np.array(hist)
ax[0, 0].semilogy(hist[:, 0], label='EDP')
ax[0, 0].semilogy(hist[:, 1], label='CL')
if USE_SENSOR:
    ax[0, 0].semilogy(hist[:, 2], label='capteur')
ax[0, 0].axvline(N_ADAM, color='k', lw=0.5)
ax[0, 0].set(xlabel='iteration', ylabel='perte', title='Convergence')
ax[0, 0].legend()

for tt, c in zip([10, 40, 60, 120, 300], plt.cm.viridis(np.linspace(0, 1, 5))):
    i = np.searchsorted(t_fd, tt)
    ax[0, 1].plot(z_fd * 1e3, T_fd[i] - 273.15, color=c, label=f'FD {tt} s')
    ax[0, 1].plot(z_fd * 1e3, T_pinn[i] - 273.15, '--', color=c)
ax[0, 1].set(xlabel='z (mm)', ylabel='T (C)', title='Profils : FD (trait) vs PINN (tirets)')
ax[0, 1].legend(fontsize=10)

i_z = np.argmin(np.abs(z_fd - float(d['z_sensor'])))
i_i = np.argmin(np.abs(z_fd - Z_INT[3]))
ax[1, 0].plot(t_fd, T_fd[:, i_z] - 273.15, label='FD capteur')
ax[1, 0].plot(t_fd, T_pinn[:, i_z] - 273.15, '--', label='PINN capteur')
ax[1, 0].plot(t_fd, T_fd[:, i_i] - 273.15, label='FD interface 4')
ax[1, 0].plot(t_fd, T_pinn[:, i_i] - 273.15, '--', label='PINN interface 4')
if USE_SENSOR:
    ax[1, 0].plot(t_sens.cpu().numpy() * T_WINDOW, th_sens.cpu().numpy() * DT_REF + T_AMB - 273.15,
                  '.', label='points capteur')
ax[1, 0].set(xlabel='t (s)', ylabel='T (C)', title='Historiques')
ax[1, 0].legend(fontsize=10)

pc = ax[1, 1].pcolormesh(t_fd, z_fd * 1e3, err.T, shading='auto', cmap='RdBu_r',
                         vmin=-np.abs(err).max(), vmax=np.abs(err).max())
ax[1, 1].set(xlabel='t (s)', ylabel='z (mm)', title='PINN - FD (K)')
fig.colorbar(pc, ax=ax[1, 1])

fig.tight_layout()
plt.show()
