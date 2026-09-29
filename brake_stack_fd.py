"""
1D reference solver for a carbon brake stack (5 stators + 4 rotors).

Axial conduction through the 9 discs (same material, continuous bar of
length 9 * e_disc), heat sources at the 8 friction interfaces during
braking, convection at both ends and a volumetric lateral sink.

    rho cp dT/dt = d/dz( k dT/dz ) + q(z,t) - s_lat (T - T_amb)
    -k dT/dz |_{z=0} =  h_end (T - T_amb)      (same at z = L)

Scheme: finite volumes, Crank-Nicolson, properties k(T) and cp(T) evaluated
at the previous step (semi-implicit). Exports the full T(z,t) + a noisy
"BTMS" sensor to brake_stack_data.npz to train a PINN.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

# Font sizes chosen for the report: LaTeX scales the figure to ~6.1 inches
# wide, so the effective font size is
# fontsize x 6.1 / figsize_width. Target ~8.5 pt once printed.
plt.rcParams.update({'font.size': 15, 'axes.titlesize': 15, 'axes.labelsize': 15,
                     'xtick.labelsize': 13, 'ytick.labelsize': 13,
                     'legend.fontsize': 10, 'lines.linewidth': 1.8})

# ---------------------------------------------------------------- geometry
N_DISC = 9
E_DISC = 0.025                      # m, disc thickness
L = N_DISC * E_DISC                 # total stack length
R_EXT, R_INT = 0.20, 0.12           # m, annulus
A_SEC = np.pi * (R_EXT**2 - R_INT**2)      # conduction cross-section
PERIM = 2 * np.pi * (R_EXT + R_INT)        # lateral perimeter (outer + inner)

# ---------------------------------------------------------------- material
RHO = 1750.0                        # kg/m3, carbon-carbon
K0, ALPHA_K = 15.0, 0.0             # k = K0 (1 + ALPHA_K (T - T_REF))
CP0, ALPHA_CP = 1000.0, 0.0         # cp = CP0 (1 + ALPHA_CP (T - T_REF))
T_REF = 293.15                      # K
# For a realistic nonlinear case: ALPHA_K = -4e-4, ALPHA_CP = +8e-4

def k_of_T(T):
    return K0 * (1.0 + ALPHA_K * (T - T_REF))

def cp_of_T(T):
    return CP0 * (1.0 + ALPHA_CP * (T - T_REF))

# ---------------------------------------------------------------- heat exchange
T_AMB = 293.15                      # K
H_END = 20.0                        # W/m2K, ends (piston / torque tube)
H_LAT = 15.0                        # W/m2K, lateral surface
S_LAT = H_LAT * PERIM / A_SEC       # W/m3K, equivalent volumetric sink
T_INIT = 293.15                     # K, uniform initial temperature

# ---------------------------------------------------------------- braking
E_BRAKE = 12.0e6                    # J per brake (normal landing)
T_BRAKE = 40.0                      # s, braked roll duration
N_INTERFACE = N_DISC - 1            # 8 friction interfaces
SIGMA_Q = 2.0e-3                    # m, Gaussian smoothing of the interface source

def power(t):
    """Braking power (W): constant deceleration -> triangular profile."""
    return np.where((t >= 0) & (t < T_BRAKE),
                    2 * E_BRAKE / T_BRAKE * (1 - t / T_BRAKE), 0.0)

# ---------------------------------------------------------------- mesh
NZ = 900
dz = L / NZ
z = (np.arange(NZ) + 0.5) * dz     # cell centers
z_int = np.arange(1, N_DISC) * E_DISC

# normalized spatial profile of the source (integral = 1 over each interface)
shape = np.zeros(NZ)
for zi in z_int:
    g = np.exp(-0.5 * ((z - zi) / SIGMA_Q)**2)
    shape += g / (g.sum() * dz)
shape /= N_INTERFACE                # sum over the 8 interfaces -> integral = 1

def q_vol(t):
    """Volumetric source W/m3: P(t) spread over the interfaces, / cross-section."""
    return power(t) * shape / A_SEC

# ---------------------------------------------------------------- time
T_END = 7200.0                      # s, 2 h of cooling
t = np.concatenate([np.arange(0.0, 60.0, 0.1),
                    np.arange(60.0, T_END + 1e-9, 2.0)])
NT = len(t)

# ---------------------------------------------------------------- assembly
ones = np.ones(NZ)
b_const = S_LAT * T_AMB * ones
b_const[0] += H_END * T_AMB / dz
b_const[-1] += H_END * T_AMB / dz

def assemble(T):
    """Linear operator A such that rho cp dT/dt = A T + b_const + q."""
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

# ---------------------------------------------------------------- energy balance
m_stack = RHO * A_SEC * L
dT_adiab = E_BRAKE / (m_stack * CP0)
print(f"\nStack mass           : {m_stack:.1f} kg")
print(f"Adiabatic temperature rise: {dT_adiab:.0f} K  (E / m cp)")
print(f"Peak temperature     : {T_hist.max() - 273.15:.0f} C")
print(f"Mean T at t = 2 h    : {T_hist[-1].mean() - 273.15:.0f} C")

# ---------------------------------------------------------------- sensor
Z_SENSOR = 4.5 * E_DISC             # middle of the central disc
NOISE_C = 3.0                       # noise standard deviation, K
DT_SENSOR = 30.0                    # s, typical flight-data recorder rate
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
print("\nData written to brake_stack_data.npz")

# ---------------------------------------------------------------- figures
TC = T_hist - 273.15
fig, ax = plt.subplots(2, 2, figsize=(12, 8))

ax[0, 0].plot(t[t <= 60], power(t[t <= 60]) / 1e3)
ax[0, 0].set(xlabel='t (s)', ylabel='P (kW)', title='Braking power')

for tt in [10, 40, 60, 120, 300, 900, 3600, 7200]:
    i = np.searchsorted(t, tt)
    ax[0, 1].plot(z * 1e3, TC[i], label=f't = {tt:.0f} s')
for zi in z_int:
    ax[0, 1].axvline(zi * 1e3, color='k', lw=0.4, alpha=0.4)
ax[0, 1].set(xlabel='z (mm)', ylabel='T (C)', title='Axial profiles')
ax[0, 1].legend(fontsize=10, ncol=2, loc='lower center', framealpha=0.85)

ax[1, 0].plot(t / 60, TC[:, i_z], label='model at the sensor')
ax[1, 0].plot(t_sens / 60, T_sens - 273.15, '.', ms=4, label='noisy sensor (30 s)')
ax[1, 0].plot(t / 60, TC.max(axis=1), '--', label='T max (interface)')
ax[1, 0].set(xlabel='t (min)', ylabel='T (C)', title='Time history')
ax[1, 0].legend()

pc = ax[1, 1].pcolormesh(t / 60, z * 1e3, TC.T, shading='auto', cmap='inferno')
# explicit xlim: since matplotlib 3.10, the auto-limit of a log-scale pcolormesh
# whose first edge is 0 snaps to the decade of the max
ax[1, 1].set(xlabel='t (min)', ylabel='z (mm)', title='T(z, t)', xscale='log',
             xlim=(t[1] / 60, t[-1] / 60))
fig.colorbar(pc, ax=ax[1, 1], label='T (C)')

fig.tight_layout()
plt.show()
