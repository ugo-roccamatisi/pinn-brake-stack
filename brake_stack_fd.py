"""
Solveur de reference 1D pour une pile de freins carbone (5 stators + 4 rotors).

Conduction axiale a travers les 9 disques (meme materiau, barreau continu de
longueur 9 * e_disc), sources de chaleur aux 8 interfaces de friction pendant
le freinage, convection aux deux extremites et puits lateral volumique.

    rho cp dT/dt = d/dz( k dT/dz ) + q(z,t) - s_lat (T - T_amb)
    -k dT/dz |_{z=0} =  h_end (T - T_amb)      (idem en z = L)

Schema : volumes finis, Crank-Nicolson, proprietes k(T) et cp(T) evaluees
au pas precedent (semi-implicite). Exporte T(z,t) complet + un capteur
"BTMS" bruite dans brake_stack_data.npz pour entrainer un PINN.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

# Tailles de police dimensionnees pour l'insertion dans le rapport : LaTeX
# ramene la figure a ~6.1 pouces de large, donc la police effective vaut
# fontsize x 6.1 / figsize_largeur. Objectif ~8.5 pt une fois imprime.
plt.rcParams.update({'font.size': 15, 'axes.titlesize': 15, 'axes.labelsize': 15,
                     'xtick.labelsize': 13, 'ytick.labelsize': 13,
                     'legend.fontsize': 10, 'lines.linewidth': 1.8})

# ---------------------------------------------------------------- geometrie
N_DISC = 9
E_DISC = 0.025                      # m, epaisseur d'un disque
L = N_DISC * E_DISC                 # longueur totale de la pile
R_EXT, R_INT = 0.20, 0.12           # m, couronne
A_SEC = np.pi * (R_EXT**2 - R_INT**2)      # section de conduction
PERIM = 2 * np.pi * (R_EXT + R_INT)        # perimetre lateral (ext + int)

# ---------------------------------------------------------------- materiau
RHO = 1750.0                        # kg/m3, carbone-carbone
K0, ALPHA_K = 15.0, 0.0             # k = K0 (1 + ALPHA_K (T - T_REF))
CP0, ALPHA_CP = 1000.0, 0.0         # cp = CP0 (1 + ALPHA_CP (T - T_REF))
T_REF = 293.15                      # K
# Pour un cas non lineaire realiste : ALPHA_K = -4e-4, ALPHA_CP = +8e-4

def k_of_T(T):
    return K0 * (1.0 + ALPHA_K * (T - T_REF))

def cp_of_T(T):
    return CP0 * (1.0 + ALPHA_CP * (T - T_REF))

# ---------------------------------------------------------------- echanges
T_AMB = 293.15                      # K
H_END = 20.0                        # W/m2K, extremites (piston / tube de couple)
H_LAT = 15.0                        # W/m2K, surface laterale
S_LAT = H_LAT * PERIM / A_SEC       # W/m3K, puits volumique equivalent
T_INIT = 293.15                     # K, temperature initiale uniforme

# ---------------------------------------------------------------- freinage
E_BRAKE = 12.0e6                    # J par frein (atterrissage normal)
T_BRAKE = 40.0                      # s, duree du roulement freine
N_INTERFACE = N_DISC - 1            # 8 interfaces de friction
SIGMA_Q = 2.0e-3                    # m, lissage gaussien de la source d'interface

def power(t):
    """Puissance de freinage (W) : deceleration constante -> profil triangulaire."""
    return np.where((t >= 0) & (t < T_BRAKE),
                    2 * E_BRAKE / T_BRAKE * (1 - t / T_BRAKE), 0.0)

# ---------------------------------------------------------------- maillage
NZ = 900
dz = L / NZ
z = (np.arange(NZ) + 0.5) * dz     # centres de cellules
z_int = np.arange(1, N_DISC) * E_DISC

# profil spatial normalise de la source (integrale = 1 sur chaque interface)
shape = np.zeros(NZ)
for zi in z_int:
    g = np.exp(-0.5 * ((z - zi) / SIGMA_Q)**2)
    shape += g / (g.sum() * dz)
shape /= N_INTERFACE                # somme des 8 interfaces -> integrale = 1

def q_vol(t):
    """Source volumique W/m3 : P(t) reparti sur les interfaces, / section."""
    return power(t) * shape / A_SEC

# ---------------------------------------------------------------- temps
T_END = 7200.0                      # s, 2 h de refroidissement
t = np.concatenate([np.arange(0.0, 60.0, 0.1),
                    np.arange(60.0, T_END + 1e-9, 2.0)])
NT = len(t)

# ---------------------------------------------------------------- assemblage
ones = np.ones(NZ)
b_const = S_LAT * T_AMB * ones
b_const[0] += H_END * T_AMB / dz
b_const[-1] += H_END * T_AMB / dz

def assemble(T):
    """Operateur lineaire A tel que rho cp dT/dt = A T + b_const + q."""
    kf = 0.5 * (k_of_T(T[1:]) + k_of_T(T[:-1])) / dz**2
    main = np.zeros(NZ)
    main[:-1] -= kf
    main[1:] -= kf
    main -= S_LAT
    main[0] -= H_END / dz
    main[-1] -= H_END / dz
    return sp.diags([kf, main, kf], [-1, 0, 1], format='csc')

# ---------------------------------------------------------------- integration
T = np.full(NZ, T_INIT)
T_hist = np.empty((NT, NZ), dtype=np.float32)
T_hist[0] = T
q_prev = q_vol(t[0])

for n in range(1, NT):
    dt = t[n] - t[n - 1]
    A = assemble(T)
    M = sp.diags(RHO * cp_of_T(T))
    q_new = q_vol(t[n])
    lhs = (M - 0.5 * dt * A).tocsc()
    rhs = (M + 0.5 * dt * A) @ T + dt * (b_const + 0.5 * (q_prev + q_new))
    T = spla.spsolve(lhs, rhs)
    T_hist[n] = T
    q_prev = q_new
    if n % 500 == 0:
        print(f"t = {t[n]:7.1f} s   T_max = {T.max() - 273.15:6.1f} C")

# ---------------------------------------------------------------- bilan
m_stack = RHO * A_SEC * L
dT_adiab = E_BRAKE / (m_stack * CP0)
print(f"\nMasse de la pile      : {m_stack:.1f} kg")
print(f"Elevation adiabatique : {dT_adiab:.0f} K  (E / m cp)")
print(f"Pic de temperature    : {T_hist.max() - 273.15:.0f} C")
print(f"T moyenne a t = 2 h   : {T_hist[-1].mean() - 273.15:.0f} C")

# ---------------------------------------------------------------- capteur
Z_SENSOR = 4.5 * E_DISC             # milieu du disque central
NOISE_C = 3.0                       # ecart-type du bruit, K
DT_SENSOR = 30.0                    # s, cadence type enregistreur de vol
rng = np.random.default_rng(0)
i_z = np.argmin(np.abs(z - Z_SENSOR))
t_sens = np.arange(0.0, T_END + 1e-9, DT_SENSOR)
i_t = np.searchsorted(t, t_sens)
T_sens = T_hist[i_t, i_z] + rng.normal(0, NOISE_C, len(t_sens))

np.savez('brake_stack_data.npz',
         z=z, t=t, T=T_hist, P=power(t),
         t_sens=t_sens, T_sens=T_sens, z_sensor=z[i_z],
         L=L, e_disc=E_DISC, rho=RHO, k0=K0, cp0=CP0,
         alpha_k=ALPHA_K, alpha_cp=ALPHA_CP, t_ref=T_REF,
         h_end=H_END, s_lat=S_LAT, t_amb=T_AMB, t_init=T_INIT,
         e_brake=E_BRAKE, t_brake=T_BRAKE, sigma_q=SIGMA_Q,
         a_sec=A_SEC, z_int=z_int)
print("\nDonnees ecrites dans brake_stack_data.npz")

# ---------------------------------------------------------------- figures
TC = T_hist - 273.15
fig, ax = plt.subplots(2, 2, figsize=(12, 8))

ax[0, 0].plot(t[t <= 60], power(t[t <= 60]) / 1e3)
ax[0, 0].set(xlabel='t (s)', ylabel='P (kW)', title='Puissance de freinage')

for tt in [10, 40, 60, 120, 300, 900, 3600, 7200]:
    i = np.searchsorted(t, tt)
    ax[0, 1].plot(z * 1e3, TC[i], label=f't = {tt:.0f} s')
for zi in z_int:
    ax[0, 1].axvline(zi * 1e3, color='k', lw=0.4, alpha=0.4)
ax[0, 1].set(xlabel='z (mm)', ylabel='T (C)', title='Profils axiaux')
ax[0, 1].legend(fontsize=10, ncol=2, loc='lower center', framealpha=0.85)

ax[1, 0].plot(t / 60, TC[:, i_z], label='modele au capteur')
ax[1, 0].plot(t_sens / 60, T_sens - 273.15, '.', ms=4, label='capteur bruite (30 s)')
ax[1, 0].plot(t / 60, TC.max(axis=1), '--', label='T max (interface)')
ax[1, 0].set(xlabel='t (min)', ylabel='T (C)', title='Historique')
ax[1, 0].legend()

pc = ax[1, 1].pcolormesh(t / 60, z * 1e3, TC.T, shading='auto', cmap='inferno')
# xlim explicite : depuis matplotlib 3.10, l'auto-limite d'un pcolormesh en
# echelle log dont la premiere arete vaut 0 se cale sur la decade du max
ax[1, 1].set(xlabel='t (min)', ylabel='z (mm)', title='T(z, t)', xscale='log',
             xlim=(t[1] / 60, t[-1] / 60))
fig.colorbar(pc, ax=ax[1, 1], label='T (C)')

fig.tight_layout()
plt.show()
