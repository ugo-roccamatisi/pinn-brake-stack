"""
PROBLEME INVERSE : identifier des parametres physiques a partir du capteur.

Principe : le PINN parametrique (brake_stack_pinn_param.pt, GELE, aucun
reentrainement) est une fonction differentiable T(z, t ; p). L'inversion est
donc une simple descente de gradient SUR SES ENTREES p : on cherche le p qui
reproduit au mieux la serie temporelle du capteur. Cout : quelques secondes.

Protocole anti "crime inverse" : les mesures ne viennent PAS du PINN mais du
solveur FD (fd_solve, copie de brake_stack_pinn_param.py), avec bruit 3 K et
echantillonnage 30 s (cadence enregistreur de vol). Le modele d'inversion et
le generateur de donnees sont donc deux codes independants.

Inconnues : E_brake, h_lat, h_end   (tb et T_init supposes connus)
Etude d'identifiabilite : la meme inversion est repetee avec des fenetres
d'observation croissantes (5 min, 20 min, 2 h) et 8 departs aleatoires
chacune : la dispersion des solutions dit ce que le capteur contraint ou non.
Attendu physiquement : E des le pic ; h_lat avec la pente de la queue ;
h_end mal contraint (correle a h_lat, son effet passe par les extremites
que le capteur central ne voit presque pas).
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
plt.rcParams.update({'font.size': 18, 'axes.titlesize': 18, 'axes.labelsize': 18,
                     'xtick.labelsize': 16, 'ytick.labelsize': 16,
                     'legend.fontsize': 13, 'lines.linewidth': 1.8})

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
    print("Drive non monte (", e, ") : recherche du .pt en local")
if not os.path.exists(CKPT):
    CKPT = 'brake_stack_pinn_param.pt'   # repli : fichier televerse dans la session
model = ParamPINN().to(DEV)
model.load_state_dict(torch.load(CKPT, map_location=DEV))
model.eval()
for w in model.parameters():
    w.requires_grad_(False)                    # le modele est GELE : on n'optimise que p
print("Modele charge depuis :", CKPT)

# ================================================================ verite terrain FD
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

P_TRUE = [12.0, 40.0, 20.0, 15.0, 20.0]        # verite terrain
TB_KNOWN, T0_KNOWN = P_TRUE[1], P_TRUE[2]      # supposes connus a l'inversion
NOISE_C, DT_SENSOR = 3.0, 30.0

print("Generation de la verite terrain (FD)...")
z_fd, t_fd, T_fd = fd_solve(P_TRUE)
t_s = np.arange(DT_SENSOR, T_WINDOW + 1e-9, DT_SENSOR)

def make_data(source, z_s, seed=42):
    """Serie capteur bruitee. source='fd' : mesures independantes du modele
    d'inversion. source='pinn' : crime inverse ASSUME, sert de controle pour
    attribuer le biais des fenetres longues a l'erreur du modele direct."""
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
FREE = [0, 3, 4]                               # indices des inconnues : E, h_lat, h_end

def invert(z_s, t_data, T_data, seed, n_iter=400):
    """Descente Adam sur u (parametres libres en logit, bornes de la boite
    imposees par sigmoide). Retourne p estime."""
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
Z_CENTRE, Z_BORD = 4.5 * E_DISC, 0.5 * E_DISC

# Trois experiences : reference, controle du crime inverse, capteur deplace
EXPS = [('reference : FD, capteur centre', 'fd', Z_CENTRE),
        ('controle crime inverse : donnees PINN, capteur centre', 'pinn', Z_CENTRE),
        ('capteur en BORD de pile : FD, z = 0.5 e_disc', 'fd', Z_BORD)]

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
        print(f"  fenetre {w:3d} min ({time.time() - t0_:.0f} s) :")
        for j, idx in enumerate(FREE):
            print(f"    {P_NAMES[idx]:>10s} : vrai {P_TRUE[idx]:6.1f}   estime {med[j]:6.1f}"
                  f"   [min {lo[j]:6.1f}, max {hi[j]:6.1f}]")
    all_results[label] = res
results = all_results[EXPS[0][0]]              # la reference alimente les diagnostics ci-dessous
T_s = make_data('fd', Z_CENTRE)
Z_S = Z_CENTRE

# Le couple (h_lat, h_end) est correle : le capteur central contraint le taux
# de refroidissement GLOBAL, pas sa repartition entre flanc et extremites. On
# le verifie en calculant, pour chaque solution, le taux du mode propre le
# plus lent lambda(h_lat, h_end) : il doit etre bien mieux ressere que les
# deux h pris separement.
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
print(f"\nTaux de refroidissement global lambda (fenetre 2 h) :")
print(f"  vrai {lam_true * 1e4:.3f} e-4/s   estime {np.median(lams) * 1e4:.3f}"
      f"   [min {lams.min() * 1e4:.3f}, max {lams.max() * 1e4:.3f}] "
      f"-> la COMBINAISON est identifiee, pas la repartition")

# ================================================================ figures
fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
p_best = np.array([0.0, TB_KNOWN, T0_KNOWN, 0.0, 0.0])
p_best[FREE] = np.median(results[120], 0)
zz = torch.full((len(t_s), 1), Z_S / L, dtype=DTYPE, device=DEV)
tt = torch.tensor(t_s / T_WINDOW, dtype=DTYPE, device=DEV)[:, None]
pp = torch.tensor(p_best, dtype=DTYPE, device=DEV).expand(len(t_s), 5)
with torch.no_grad():
    T_fit = (T_AMB + DT_REF * model(zz, tt, pp).cpu().numpy()[:, 0]) - 273.15
ax[0].plot(t_s / 60, T_s, '.', ms=3, alpha=0.5, label='capteur bruite (FD)')
ax[0].plot(t_s / 60, T_fit, 'r', lw=1.5, label='PINN au p identifie')
ax[0].set(xlabel='t (min)', ylabel='T (C)', title='Ajustement (fenetre 2 h)')
ax[0].legend()

# panneau 2 : E a 120 min pour les trois experiences (biais de modele)
# panneau 3 : h_end a 120 min pour les trois experiences (placement du capteur)
labels_courts = ['reference\n(FD, centre)', 'crime inverse\n(PINN, centre)', 'capteur bord\n(FD, z=e/2)']
for a, j, idx, ttl in [(ax[1], 0, 0, 'E identifie, fenetre 2 h (tirets = vrai)'),
                       (ax[2], 2, 4, 'h_end identifie, fenetre 2 h (tirets = vrai)')]:
    for e, (label, _, _) in enumerate(EXPS):
        y = all_results[label][120][:, j]
        a.plot(np.full_like(y, e), y, 'o', ms=5, alpha=0.6, color=f'C{e}')
    a.axhline(P_TRUE[idx], color='k', ls='--', lw=1)
    a.set(xticks=range(3), xticklabels=labels_courts, title=ttl)
fig.tight_layout()
plt.show()
