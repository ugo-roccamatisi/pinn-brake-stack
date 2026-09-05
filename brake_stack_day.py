"""
JOURNEE COMPLETE : enchainement du PINN parametrique sur plusieurs rotations.

Recycle brake_stack_pinn_param.pt (entraine par brake_stack_pinn_param.py,
place dans le meme dossier) SANS aucun reentrainement. Chaque rotation est
decoupee selon l'echelle physique dominante :

  1. atterrissage + taxi-in (8 min)  : PINN parametrique (pics, gradients)
  2. impulsions de freinage de taxi  : increments adiabatiques uniformes
                                       dT = E_taxi / (m cp) (trop lents pour
                                       creer un gradient axial)
  3. parking porte (h_sol) et vol (h_vol) : mode propre de Robin le plus lent,
     T - T_amb ~ exp(-lambda t) avec lambda = alpha mu0^2/L^2 + s_lat/(rho cp)
     et mu0 tan(mu0/2) = Bi. Exact pour un champ uniforme en physique lineaire,
     ce qui est le cas 5 min apres un freinage (profils plats verifies).

La temperature de fin de segment devient le T_init du segment suivant : c'est
exactement le role du parametre T_init de la boite d'entrainement.
Verification operationnelle : temperature capteur a chaque decollage comparee
a la limite A320 de 150 C (freins trop chauds = decollage interdit).
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Tailles de police dimensionnees pour l'insertion dans le rapport : LaTeX
# ramene la figure a ~6.1 pouces de large, donc la police effective vaut
# fontsize x 6.1 / figsize_largeur. Objectif ~8.5 pt une fois imprime.
plt.rcParams.update({'font.size': 16, 'axes.titlesize': 16, 'axes.labelsize': 16,
                     'xtick.labelsize': 14, 'ytick.labelsize': 14,
                     'legend.fontsize': 11, 'lines.linewidth': 1.8})

torch.manual_seed(0)
np.random.seed(0)
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DTYPE = torch.float32

# ================================================================ definitions
# IDENTIQUES a brake_stack_pinn_param.py (necessaires pour recharger le .pt)
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
P_LO = np.array([4.0, 20.0, 20.0, 5.0, 5.0])
P_HI = np.array([25.0, 60.0, 300.0, 40.0, 40.0])
ALPHA_D = K0 / (RHO * CP0)
DT_REF = P_HI[0] * 1e6 / (RHO * A_SEC * L * CP0)
FO = ALPHA_D * T_WINDOW / L**2
SIG_HAT = SIGMA_Q / L
ZI_HAT = torch.tensor(Z_INT / L, dtype=DTYPE, device=DEV)
M_STACK = RHO * A_SEC * L                      # kg (masse de la pile)

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
    print("Drive non monte (", e, ") : recherche du .pt en local")
if not os.path.exists(CKPT):
    CKPT = 'brake_stack_pinn_param.pt'   # repli : fichier televerse dans la session
model = ParamPINN().to(DEV)
model.load_state_dict(torch.load(CKPT, map_location=DEV))
model.eval()
print("Modele charge depuis :", CKPT)

def predict(p_phys, z, t):
    """T (t, z) en C pour un scenario de la boite."""
    ZZ, TT = np.meshgrid(z / L, np.asarray(t) / T_WINDOW)
    zz = torch.tensor(ZZ.ravel()[:, None], dtype=DTYPE, device=DEV)
    tt = torch.tensor(TT.ravel()[:, None], dtype=DTYPE, device=DEV)
    pp = torch.tensor(np.asarray(p_phys, dtype=np.float32), device=DEV).expand(len(zz), 5)
    with torch.no_grad():
        th = torch.cat([model(zz[i:i + 20000], tt[i:i + 20000], pp[i:i + 20000])
                        for i in range(0, len(zz), 20000)])
    return (T_AMB + DT_REF * th.cpu().numpy().reshape(ZZ.shape)) - 273.15

# ================================================================ refroidissement pur
def robin_mu0(bi):
    """Plus petite racine de mu tan(mu/2) = Bi (mode propre de Robin symetrique)."""
    lo, hi = 1e-6, np.pi - 1e-6
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if mid * np.tan(0.5 * mid) < bi:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)

def cool(T0_C, dur_s, h_lat, h_end, n_pts=80):
    """Segment sans freinage : decroissance du mode propre le plus lent."""
    mu0 = robin_mu0(h_end * L / K0)
    lam = ALPHA_D * mu0**2 / L**2 + h_lat * PERIM / A_SEC / (RHO * CP0)
    t = np.linspace(0, dur_s, n_pts)
    return t, (T_AMB - 273.15) + (T0_C - (T_AMB - 273.15)) * np.exp(-lam * t)

def taxi(T0_C, dur_s, dT_K, h_lat, h_end, n_pts=60):
    """Phase de taxi : les freinages intermittents deposent dT_K au total,
    repartis uniformement sur la duree (puissance moyenne dT/dur), en
    competition avec le refroidissement. Forme close du 1er ordre :
    T = T_amb + (T0 - T_amb) e^{-lam t} + (dT/dur) (1 - e^{-lam t}) / lam."""
    mu0 = robin_mu0(h_end * L / K0)
    lam = ALPHA_D * mu0**2 / L**2 + h_lat * PERIM / A_SEC / (RHO * CP0)
    t = np.linspace(0, dur_s, n_pts)
    amb = T_AMB - 273.15
    return t, (amb + (T0_C - amb) * np.exp(-lam * t)
               + dT_K / dur_s * (1 - np.exp(-lam * t)) / lam)

# ================================================================ journee
Z_S = np.array([4.5 * E_DISC])                 # capteur au centre
Z_FULL = np.linspace(0.5, L / 0.0005 - 0.5, 200) * 0.0005   # pour la moyenne de fin de segment
Z_FULL = np.linspace(0, L, 200)
H_SOL, H_VOL, H_END = 15.0, 35.0, 20.0         # h_vol : convection forcee train sorti / baie (equivalent)
E_TAXI_MJ = 1.5                                # energie de freinage par frein et par phase de taxi
T_BRAKE_SEG = 5 * 60.0                         # roulement d'atterrissage + degagement piste (PINN)
T_TAXI_IN = 5 * 60.0                           # taxi-in : rechauffement progressif (freinages intermittents)
T_TAXI_OUT = 10 * 60.0                         # taxi-out : idem, avant decollage
T_GATE = 20 * 60.0                             # escale courte : rotation serree
T_VOL_DEFAUT = 75 * 60.0                       # duree de vol par defaut si non precisee
LIMITE_DECOLLAGE_C = 150.0                     # limite A320 freins avant decollage

# Deux scenarios traces l'un sous l'autre :
#  A. journee REALISTE : energies d'atterrissage variees d'un vol a l'autre.
#     L'accumulation thermique existe mais est masquee par la variabilite des
#     pics (+/- 100 K de pic contre ~70 K d'accumulation).
#  B. rotations IDENTIQUES : experience controlee qui isole l'histoire
#     thermique ; toute difference entre rotations vient du vol precedent, et
#     la convergence geometrique vers le cycle limite devient lisible.
# Chaque rotation precise sa duree de vol (>= 1 h, variee pour la journee
# realiste ; fixe pour le scenario controle, ou tout doit etre identique).
SCENARIOS = [
    ("A. journee realiste : energies et durees de vol variees",
     [dict(E=12.0, tb=40.0, vol=75), dict(E=14.5, tb=35.0, vol=95),
      dict(E=11.0, tb=45.0, vol=60), dict(E=16.0, tb=30.0, vol=120),
      dict(E=10.5, tb=42.0, vol=65), dict(E=13.0, tb=38.0, vol=90)]),
    ("B. rotations identiques (E = 13,5 MJ, vol 75 min) : accumulation isolee",
     [dict(E=13.5, tb=40.0, vol=75)] * 6),
]

dT_taxi = E_TAXI_MJ * 1e6 / (M_STACK * CP0)
print(f"Increment adiabatique par phase de taxi : +{dT_taxi:.1f} K")

def simulate_day(rotations):
    """Enchaine les rotations ; retourne trace, marques et lignes de tableau."""
    t_all, T_all, marks, rows = [], [], [], []
    T_cur, t0 = 20.0, 0.0
    for k, rot in enumerate(rotations):
        # 1. taxi-out : rechauffement progressif par freinages intermittents
        t_seg, T_seg = taxi(T_cur, T_TAXI_OUT, dT_taxi, H_SOL, H_END)
        t_all.append(t0 + t_seg); T_all.append(T_seg)
        T_cur = T_seg[-1]
        t0 += T_TAXI_OUT
        T_deco = T_cur
        marks.append(('decollage', t0))

        # 2. vol (duree propre a la rotation, en minutes)
        dur_vol = rot.get('vol', T_VOL_DEFAUT / 60) * 60.0
        t_seg, T_seg = cool(T_cur, dur_vol, H_VOL, H_END)
        t_all.append(t0 + t_seg); T_all.append(T_seg)
        T_cur = T_seg[-1]
        t0 += dur_vol

        # 3. atterrissage + degagement : PINN
        T_avant = T_cur
        p = [rot['E'], rot['tb'], T_cur, H_SOL, H_END]
        t_seg = np.linspace(0, T_BRAKE_SEG, 300)
        T_sens = predict(p, Z_S, t_seg)[:, 0]
        t_all.append(t0 + t_seg); T_all.append(T_sens)
        marks.append(('atterrissage', t0, T_avant))
        pic = T_sens.max()
        T_cur = predict(p, Z_FULL, [T_BRAKE_SEG])[0].mean()
        t0 += T_BRAKE_SEG

        # 4. taxi-in : rechauffement residuel en plein refroidissement
        t_seg, T_seg = taxi(T_cur, T_TAXI_IN, dT_taxi, H_SOL, H_END)
        t_all.append(t0 + t_seg); T_all.append(T_seg)
        T_cur = T_seg[-1]
        t0 += T_TAXI_IN

        # 5. escale a la porte
        t_seg, T_seg = cool(T_cur, T_GATE, H_SOL, H_END)
        t_all.append(t0 + t_seg); T_all.append(T_seg)
        T_cur = T_seg[-1]
        t0 += T_GATE
        rows.append((k + 1, rot['E'], rot.get('vol', T_VOL_DEFAUT / 60), T_deco, T_avant, pic, T_cur))
    return np.concatenate(t_all), np.concatenate(T_all), marks, rows

days = []
for titre, rotations in SCENARIOS:
    t_all, T_all, marks, rows = simulate_day(rotations)
    days.append((titre, t_all, T_all, marks))
    print(f"\n{titre}")
    print(f"{'rot':>3s} {'E [MJ]':>7s} {'vol [min]':>9s} {'T decollage':>12s} {'T avant atterr.':>16s} {'pic':>8s} {'T fin escale':>13s}")
    for k, E, vol, T_deco, T_avant, pic, T_fin in rows:
        avert = "  <-- DEPASSE LA LIMITE" if T_deco > LIMITE_DECOLLAGE_C else ""
        print(f"{k:3d} {E:7.1f} {vol:9.0f} {T_deco:10.1f} C {T_avant:14.1f} C {pic:6.0f} C {T_fin:11.1f} C{avert}")

# ================================================================ figure
fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
for ax, (titre, t_all, T_all, marks) in zip(axes, days):
    ax.plot(t_all / 3600, T_all, lw=1.5)
    ax.axhline(LIMITE_DECOLLAGE_C, color='r', ls='--', lw=1,
               label=f'limite decollage {LIMITE_DECOLLAGE_C:.0f} C')
    for m in marks:
        ax.axvline(m[1] / 3600, color='k', lw=0.4, alpha=0.4)
        if m[0] == 'atterrissage':      # annoter la T heritee du vol precedent
            ax.annotate(f'{m[2]:.0f}', (m[1] / 3600, m[2]), textcoords='offset points',
                        xytext=(-4, -14), ha='right', fontsize=11, color='C3')
            ax.plot(m[1] / 3600, m[2], 'o', ms=4, color='C3')
    ax.set(ylabel='T capteur (C)', title=titre)
axes[0].legend(loc='upper left', fontsize=11)
axes[1].set(xlabel='temps (h)')
axes[1].text(0.99, 0.95, 'points rouges : T avant atterrissage (accumulation -> cycle limite)',
             transform=axes[1].transAxes, ha='right', va='top', fontsize=11, color='C3')
fig.tight_layout()
plt.show()
