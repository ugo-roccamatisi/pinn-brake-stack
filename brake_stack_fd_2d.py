"""
Solveur de reference axisymetrique 2D (r, z) pour la pile de freins.

    rho cp dT/dt = k [ (1/r) d/dr( r dT/dr ) + d2T/dz2 ] + q(r, z, t)
    Robin partout : -k dT/dn = h (T - T_amb)
        h_end aux extremites z = 0, z = L ; h_lat en r = R_int, r = R_ext

Source de friction : les 8 interfaces (gaussiennes de largeur SIGMA_Q en z)
avec un profil radial phi(r) = r (flux ~ p * omega * r, pression uniforme)
ou uniforme, normalise pour que l'integrale sur la couronne vaille P(t).

Schema : volumes finis en (r, z), Crank-Nicolson, proprietes constantes donc
matrice factorisee une fois par pas de temps distinct (splu). Exporte
T(t, z, r) sous-echantillonne en temps + capteur BTMS bruite dans
brake_stack_data_2d.npz. Superpose le 1D (brake_stack_data.npz) s'il existe.
"""

import os
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

# Tailles de police dimensionnees pour l'insertion dans le rapport : LaTeX
# ramene la figure a ~6.1 pouces de large, donc la police effective vaut
# fontsize x 6.1 / figsize_largeur. Objectif ~8.5 pt une fois imprime.
plt.rcParams.update({'font.size': 16, 'axes.titlesize': 16, 'axes.labelsize': 16,
                     'xtick.labelsize': 14, 'ytick.labelsize': 14,
                     'legend.fontsize': 11, 'lines.linewidth': 1.8})

# ---------------------------------------------------------------- geometrie
N_DISC = 9
E_DISC = 0.025
L = N_DISC * E_DISC
R_EXT, R_INT = 0.20, 0.12
A_SEC = np.pi * (R_EXT**2 - R_INT**2)

# ---------------------------------------------------------------- materiau
RHO, K0, CP0 = 1750.0, 15.0, 1000.0            # proprietes constantes

# ---------------------------------------------------------------- echanges
T_AMB, T_INIT = 293.15, 293.15
H_END = 20.0                                   # W/m2K, z = 0 et z = L
H_LAT = 15.0                                   # W/m2K, r = R_int et r = R_ext

# ---------------------------------------------------------------- freinage
E_BRAKE, T_BRAKE = 12.0e6, 40.0
N_INTERFACE = N_DISC - 1
SIGMA_Q = 2.0e-3
Q_RADIAL = 'linear'                            # 'linear' (q ~ r) ou 'uniform'

def power(t):
    return np.where((t >= 0) & (t < T_BRAKE),
                    2 * E_BRAKE / T_BRAKE * (1 - t / T_BRAKE), 0.0)

# ---------------------------------------------------------------- maillage
NZ, NR = 300, 32
dz = L / NZ
dr = (R_EXT - R_INT) / NR
z = (np.arange(NZ) + 0.5) * dz
r = R_INT + (np.arange(NR) + 0.5) * dr
r_face = R_INT + np.arange(NR + 1) * dr        # faces radiales
z_int = np.arange(1, N_DISC) * E_DISC
N = NZ * NR
idx = lambda i, j: i * NR + j                  # (z, r) -> indice plat

# forme axiale normalisee (integrale = 1 sur les 8 interfaces)
g_z = np.zeros(NZ)
for zi in z_int:
    g = np.exp(-0.5 * ((z - zi) / SIGMA_Q)**2)
    g_z += g / (g.sum() * dz)
g_z /= N_INTERFACE

# forme radiale normalisee : integrale de phi(r) * 2 pi r dr = 1
phi_r = r.copy() if Q_RADIAL == 'linear' else np.ones(NR)
phi_r /= (phi_r * 2 * np.pi * r * dr).sum()

shape = np.outer(g_z, phi_r).ravel()           # W/m3 par W injecte

def q_vol(t):
    return power(t) * shape

# ---------------------------------------------------------------- assemblage
# Volumes finis par radian : V = r dr dz ; faces axiales r dr ; faces radiales r_face dz
V = np.outer(np.ones(NZ), r * dr * dz).ravel()
rows, cols, vals = [], [], []
diag = np.zeros(N)
b_const = np.zeros(N)

def link(a, b, G):
    """Conductance G (W/K par radian) entre les cellules a et b."""
    rows.extend([a, b]); cols.extend([b, a]); vals.extend([G, G])
    diag[a] -= G
    diag[b] -= G

for i in range(NZ):
    for j in range(NR):
        a = idx(i, j)
        if i + 1 < NZ:                                     # voisin axial
            link(a, idx(i + 1, j), K0 * r[j] * dr / dz)
        if j + 1 < NR:                                     # voisin radial
            link(a, idx(i, j + 1), K0 * r_face[j + 1] * dz / dr)
        # Robin (demi-maille negligee : h applique a la cellule de bord)
        if i == 0 or i == NZ - 1:
            G = H_END * r[j] * dr
            diag[a] -= G; b_const[a] += G * T_AMB
        if j == 0:
            G = H_LAT * r_face[0] * dz
            diag[a] -= G; b_const[a] += G * T_AMB
        if j == NR - 1:
            G = H_LAT * r_face[NR] * dz
            diag[a] -= G; b_const[a] += G * T_AMB

A = sp.coo_matrix((vals, (rows, cols)), shape=(N, N)).tocsc() + sp.diags(diag)
M = sp.diags(RHO * CP0 * V)

# ---------------------------------------------------------------- temps
T_END = 7200.0
t = np.concatenate([np.arange(0.0, 60.0, 0.1),
                    np.arange(60.0, T_END + 1e-9, 2.0)])
NT = len(t)
# instants stockes : tout jusqu'a 60 s, puis 2 s jusqu'a 300 s, puis 20 s
keep = (t <= 60.0) | ((t <= 300.0) & (np.round(t) % 2 == 0)) | (np.round(t) % 20 == 0)
t_keep = t[keep]
T_hist = np.empty((len(t_keep), NZ, NR), dtype=np.float32)

solvers = {}
def solver_for(dt):
    if dt not in solvers:
        solvers[dt] = spla.splu((M - 0.5 * dt * A).tocsc())
    return solvers[dt]

T = np.full(N, T_INIT)
T_hist[0] = T.reshape(NZ, NR)
q_prev = q_vol(t[0])
k_keep = 1
for n in range(1, NT):
    dt = round(t[n] - t[n - 1], 6)
    q_new = q_vol(t[n])
    rhs = (M + 0.5 * dt * A) @ T + dt * (b_const + 0.5 * V * (q_prev + q_new))
    T = solver_for(dt).solve(rhs)
    q_prev = q_new
    if keep[n]:
        T_hist[k_keep] = T.reshape(NZ, NR)
        k_keep += 1
    if n % 500 == 0:
        print(f"t = {t[n]:7.1f} s   T_max = {T.max() - 273.15:6.1f} C")

# ---------------------------------------------------------------- bilan
E_stored = (RHO * CP0 * V * (T - T_INIT)).sum() * 2 * np.pi
print(f"\nEnergie injectee      : {E_BRAKE / 1e6:.2f} MJ")
print(f"Energie stockee a 2 h : {E_stored / 1e6:.2f} MJ (le reste est parti en convection)")
print(f"Pic de temperature    : {T_hist.max() - 273.15:.0f} C")

# ---------------------------------------------------------------- capteur
Z_SENSOR = 4.5 * E_DISC
R_SENSOR = 0.5 * (R_INT + R_EXT)
NOISE_C, DT_SENSOR = 3.0, 30.0
rng = np.random.default_rng(0)
i_z = np.argmin(np.abs(z - Z_SENSOR))
j_r = np.argmin(np.abs(r - R_SENSOR))
t_sens = np.arange(0.0, T_END + 1e-9, DT_SENSOR)
i_t = np.searchsorted(t_keep, t_sens)
T_sens = T_hist[i_t, i_z, j_r] + rng.normal(0, NOISE_C, len(t_sens))

np.savez('brake_stack_data_2d.npz',
         z=z, r=r, t=t_keep, T=T_hist, P=power(t_keep),
         t_sens=t_sens, T_sens=T_sens, z_sensor=z[i_z], r_sensor=r[j_r],
         L=L, e_disc=E_DISC, r_int=R_INT, r_ext=R_EXT, rho=RHO, k0=K0, cp0=CP0,
         h_end=H_END, h_lat=H_LAT, t_amb=T_AMB, t_init=T_INIT,
         e_brake=E_BRAKE, t_brake=T_BRAKE, sigma_q=SIGMA_Q, q_radial=Q_RADIAL,
         a_sec=A_SEC, z_int=z_int)
print("\nDonnees ecrites dans brake_stack_data_2d.npz")

# ---------------------------------------------------------------- figures
TC = T_hist - 273.15
w_area = r / r.sum()                               # poids de moyenne en aire (dr constant)
T_avg = (TC * w_area).sum(axis=2)                  # moyenne radiale (t, z)

fig, ax = plt.subplots(2, 2, figsize=(13, 9))

i40 = np.searchsorted(t_keep, 40.0)
pc = ax[0, 0].pcolormesh(z * 1e3, r * 1e3, TC[i40].T, shading='auto', cmap='inferno')
ax[0, 0].set(xlabel='z (mm)', ylabel='r (mm)', title='T(r, z) a t = 40 s')
fig.colorbar(pc, ax=ax[0, 0], label='T (C)')

for tt in [10, 40, 60, 120, 300, 900]:
    i = np.searchsorted(t_keep, tt)
    ax[0, 1].plot(r * 1e3, TC[i, i_z], label=f't = {tt} s')
ax[0, 1].set(xlabel='r (mm)', ylabel='T (C)', title='Profils radiaux au centre de la pile')
ax[0, 1].legend(fontsize=11)

for j, lab in [(0, 'R_int'), (j_r, 'r milieu'), (NR - 1, 'R_ext')]:
    ax[1, 0].plot(t_keep / 60, TC[:, i_z, j], label=f'2D, {lab}')
ax[1, 0].plot(t_keep / 60, T_avg[:, i_z], 'k--', label='2D, moyenne radiale')
if os.path.exists('brake_stack_data.npz'):
    d1 = np.load('brake_stack_data.npz')
    i_z1 = np.argmin(np.abs(d1['z'] - Z_SENSOR))
    ax[1, 0].plot(d1['t'] / 60, d1['T'][:, i_z1] - 273.15, 'r:', lw=2, label='1D')
ax[1, 0].plot(t_sens / 60, T_sens - 273.15, '.', ms=3, alpha=0.5, label='capteur bruite')
ax[1, 0].set(xlabel='t (min)', ylabel='T (C)', title='Historiques au centre de la pile', xlim=(0, 30))
ax[1, 0].legend(fontsize=11)

i120 = np.searchsorted(t_keep, 120.0)
for j, lab in [(0, 'R_int'), (j_r, 'r milieu'), (NR - 1, 'R_ext')]:
    ax[1, 1].plot(z * 1e3, TC[i120, :, j], label=f'2D, {lab}')
ax[1, 1].plot(z * 1e3, T_avg[i120], 'k--', label='2D, moyenne radiale')
if os.path.exists('brake_stack_data.npz'):
    i1 = np.searchsorted(d1['t'], 120.0)
    ax[1, 1].plot(d1['z'] * 1e3, d1['T'][i1] - 273.15, 'r:', lw=2, label='1D')
ax[1, 1].set(xlabel='z (mm)', ylabel='T (C)', title='Profils axiaux a t = 120 s')
ax[1, 1].legend(fontsize=11)

fig.tight_layout()

# ---------------------------------------------------------------- vues 3D
# (1) surface T(z, r) a t = 40 s ; (2) rendu de la pile en revolution : surface
# exterieure r = R_ext coloree par T, et coupe (r, z) a theta = 0 (demi-pile)
from matplotlib import cm
fig3 = plt.figure(figsize=(14, 6))
norm = plt.Normalize(TC[i40].min(), TC[i40].max())

ax3 = fig3.add_subplot(1, 2, 1, projection='3d')
ZZ, RR = np.meshgrid(z * 1e3, r * 1e3, indexing='ij')
ax3.plot_surface(ZZ, RR, TC[i40], cmap='inferno', norm=norm, rstride=3, cstride=1,
                 linewidth=0, antialiased=True)
ax3.set(xlabel='z (mm)', ylabel='r (mm)', zlabel='T (C)', title='T(z, r) a t = 40 s')
ax3.view_init(elev=28, azim=-55)

ax4 = fig3.add_subplot(1, 2, 2, projection='3d')
th = np.linspace(0, np.pi, 90)                       # demi-revolution : on voit la coupe
# surface exterieure (r = R_ext) sur la demi-revolution
ZE, TH = np.meshgrid(z, th, indexing='ij')
X = R_EXT * np.cos(TH); Y = R_EXT * np.sin(TH)
C = np.repeat(TC[i40, :, -1][:, None], len(th), axis=1)
ax4.plot_surface(ZE * 1e3, X * 1e3, Y * 1e3, facecolors=cm.inferno(norm(C)),
                 rstride=4, cstride=4, linewidth=0, shade=False)
# faces de coupe a theta = 0 et theta = pi (plans y = 0) montrant T(z, r)
for sgn in (+1, -1):
    ZC, RC = np.meshgrid(z, sgn * r, indexing='ij')
    ax4.plot_surface(ZC * 1e3, RC * 1e3, np.zeros_like(ZC), facecolors=cm.inferno(norm(TC[i40])),
                     rstride=4, cstride=1, linewidth=0, shade=False)
# faces d'extremite (z = 0 et z = L), demi-couronnes
RE, TE = np.meshgrid(r, th, indexing='ij')
for zi, Tend in ((0.0, TC[i40, 0]), (L, TC[i40, -1])):
    Cend = np.repeat(Tend[:, None], len(th), axis=1)
    ax4.plot_surface(np.full_like(RE, zi * 1e3), RE * np.cos(TE) * 1e3, RE * np.sin(TE) * 1e3,
                     facecolors=cm.inferno(norm(Cend)), rstride=2, cstride=4, linewidth=0, shade=False)
ax4.set(xlabel='z (mm)', ylabel='x (mm)', zlabel='y (mm)', title='Pile de freins coupee, t = 40 s')
ax4.set_box_aspect((L, 2 * R_EXT, R_EXT))
ax4.view_init(elev=22, azim=-60)
m = cm.ScalarMappable(norm=norm, cmap='inferno'); m.set_array([])
fig3.colorbar(m, ax=ax4, shrink=0.6, label='T (C)')
fig3.tight_layout()
plt.show()
