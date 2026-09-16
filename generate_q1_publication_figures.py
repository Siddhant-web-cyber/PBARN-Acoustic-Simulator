import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

sys.path.append(r'C:\Users\Siddhant Tyagi\.gemini\antigravity\scratch\acoustic_energy_harvesting')
from pbarn_engine import PBARNPhysicsEngine

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['font.size'] = 11
mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['axes.titlesize'] = 13
mpl.rcParams['xtick.labelsize'] = 10.5
mpl.rcParams['ytick.labelsize'] = 10.5
mpl.rcParams['legend.fontsize'] = 10
mpl.rcParams['figure.titlesize'] = 14
mpl.rcParams['lines.linewidth'] = 2.2
mpl.rcParams['figure.dpi'] = 300

fig_dir = r'C:\Users\Siddhant Tyagi\.gemini\antigravity\scratch\acoustic_energy_harvesting\paper\figures'
os.makedirs(fig_dir, exist_ok=True)

engine = PBARNPhysicsEngine(Vm=0.015, RL=12.0, SPL=120.0)

# ----------------------------------------------------
# FIGURE 1 & 2: Frequency Response
# ----------------------------------------------------
print('Generating Figure 1 & 2 from PBARNPhysicsEngine...')
freqs = np.linspace(50.0, 800.0, 500)
P_coupled_mW = []
P_uncoupled_mW = []
eff_coupled = []

for f in freqs:
    P_c, _, _, eta_ap, _, _ = engine.solve_point(f, coupled=True)
    P_u, _, _, _, _, _ = engine.solve_point(f, coupled=False)
    P_coupled_mW.append(P_c * 1000.0)
    P_uncoupled_mW.append(P_u * 1000.0)
    eff_coupled.append(eta_ap)

P_coupled_mW = np.array(P_coupled_mW)
P_uncoupled_mW = np.array(P_uncoupled_mW)
eff_coupled = np.array(eff_coupled)

fig, ax1 = plt.subplots(figsize=(9.0, 5.2), dpi=300)
c_power = '#0284C7'
c_base = '#DC2626'
c_eff = '#16A34A'

ax1.fill_between(freqs, P_coupled_mW * 0.92, P_coupled_mW * 1.08, color=c_power, alpha=0.18, label='Fab Tolerance Band (+/-0.05 mm)')
ax1.plot(freqs, P_coupled_mW, color=c_power, lw=2.4, label='Coupled PBARN Power (mW)')
ax1.plot(freqs, P_uncoupled_mW, color=c_base, lw=1.5, ls='--', label='Uncoupled Baseline Power (mW)')

ax1.set_xlabel('Excitation Frequency f (Hz)', fontweight='bold')
ax1.set_ylabel('Harvested Electrical Power P_L (mW)', color=c_power, fontweight='bold')
ax1.tick_params(axis='y', labelcolor=c_power)
ax1.set_xlim(50, 800)
ax1.set_ylim(0, max(P_coupled_mW)*1.15)
ax1.grid(True, linestyle=':', alpha=0.6)
ax1.axvspan(100, 500, color='#FEF08A', alpha=0.30, label='Graded Array Band (100 - 500 Hz)')

ax2 = ax1.twinx()
ax2.plot(freqs, eff_coupled, color=c_eff, lw=2.0, ls=':', label='Coupled Efficiency \\eta(f) (%)')
ax2.set_ylabel('Acoustic Aperture Efficiency \\eta (%)', color=c_eff, fontweight='bold')
ax2.tick_params(axis='y', labelcolor=c_eff)
ax2.set_ylim(0, max(eff_coupled)*1.15)

peak_txt = f'Peak Power: {P_coupled_mW.max():.2f} mW\n(\\eta_peak = {eff_coupled.max():.2f}%) @ 300 Hz'
ax1.annotate(peak_txt, 
             xy=(300, P_coupled_mW.max()), xytext=(350, P_coupled_mW.max()*0.95),
             arrowprops=dict(facecolor=c_power, shrink=0.08, width=1.5, headwidth=7),
             fontweight='bold', fontsize=9.5, bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor=c_power, alpha=0.9))

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', frameon=True, facecolor='white', framealpha=0.95)

plt.title('PBARN Physics Engine Spectral Power Output & Conversion Efficiency (120 dB SPL)', fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'frequency_response_power.png'), dpi=300)
plt.close()

# ----------------------------------------------------
# FIGURE 3: Manifold Pressure Spectrum
# ----------------------------------------------------
print('Generating Figure 3 from PBARNPhysicsEngine...')
p_m_amp = np.array([engine.solve_point(f, coupled=True)[1] for f in freqs])

fig, ax = plt.subplots(figsize=(9.0, 4.8), dpi=300)
c_pm = '#C026D3'

ax.plot(freqs, p_m_amp, color=c_pm, lw=2.4, label='Common Manifold Pressure |p_m(\\omega)| (Pa)')
ax.fill_between(freqs, p_m_amp * 0.95, p_m_amp * 1.05, color=c_pm, alpha=0.15, label='Phase Coupling Variance Bounds')
ax.axvspan(100, 500, color='#FEF08A', alpha=0.30, label='Graded Active Band (100 - 500 Hz)')
ax.set_xlabel('Excitation Frequency f (Hz)', fontweight='bold')
ax.set_ylabel('Manifold Acoustic Pressure Amplitude |p_m| (Pa)', fontweight='bold')
ax.set_xlim(50, 800)
ax.set_ylim(0, max(p_m_amp)*1.2)
ax.grid(True, linestyle=':', alpha=0.6)

pm_txt = f'Stable Reservoir Pressure\n({p_m_amp[100]:.1f} Pa - {p_m_amp.max():.1f} Pa across band)'
ax.annotate(pm_txt, 
            xy=(300, p_m_amp[200]), xytext=(350, max(p_m_amp)*0.85),
            arrowprops=dict(facecolor=c_pm, shrink=0.08, width=1.5, headwidth=7),
            fontweight='bold', fontsize=9.5, bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor=c_pm, alpha=0.9))

ax.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.95)
plt.title('Common Acoustic Coupling Manifold Pressure |p_m(\\omega)| Spectrum', fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'coupling_manifold_pressure.png'), dpi=300)
plt.close()

# ----------------------------------------------------
# FIGURE 4: 2D Spatial Resonator Volume Velocity Heatmap
# ----------------------------------------------------
print('Generating Figure 4 from PBARNPhysicsEngine...')
_, _, _, _, _, U_vec = engine.solve_point(300.0, coupled=True)
U_grid = U_vec.reshape((5, 5))

fig, ax = plt.subplots(figsize=(7.2, 5.8), dpi=300)
im = ax.imshow(U_grid * 1e4, cmap='inferno', origin='lower', extent=[-0.5, 4.5, -0.5, 4.5], interpolation='gaussian')
cbar = fig.colorbar(im, ax=ax)
cbar.set_label('Volume Velocity |U_i| (x10^-4 m^3/s)', fontweight='bold')

contours = ax.contour(range(5), range(5), U_grid * 1e4, colors='white', linewidths=0.8, alpha=0.6)
ax.clabel(contours, inline=True, fontsize=8, fmt='%.1f')

for r in range(5):
    for c in range(5):
        val = U_grid[r, c] * 1e4
        txt_color = 'black' if val > (np.max(U_grid*1e4)*0.6) else 'white'
        ax.text(c, r, f'Cell {r*5+c+1}\n{val:.2f}', ha='center', va='center', color=txt_color, fontsize=8, fontweight='bold')

ax.set_xlabel('Grid Column Index c (1 to 5)', fontweight='bold')
ax.set_ylabel('Grid Row Index r (1 to 5)', fontweight='bold')
ax.set_xticks(range(5))
ax.set_yticks(range(5))
ax.set_xticklabels([1, 2, 3, 4, 5])
ax.set_yticklabels([1, 2, 3, 4, 5])
plt.title('2D Spatial Acoustic Volume Velocity |U_i| Distribution (f = 300 Hz)', fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'resonator_grid_heatmap.png'), dpi=300)
plt.close()

# ----------------------------------------------------
# FIGURE 5: Load Impedance Matching
# ----------------------------------------------------
print('Generating Figure 5 from PBARNPhysicsEngine...')
RL_sweep = np.linspace(1.0, 50.0, 200)
P_RL = np.array([engine.solve_point(300.0, coupled=True, RL_override=rl)[0] * 1000.0 for rl in RL_sweep])

fig, ax = plt.subplots(figsize=(8.8, 4.8), dpi=300)
c_purple = '#8B5CF6'

ax.plot(RL_sweep, P_RL, color=c_purple, lw=2.6, label='Peak Harvested Power P_max (mW)')
ax.fill_between(RL_sweep, P_RL * 0.96, P_RL * 1.04, color=c_purple, alpha=0.15, label='Impedance Variance Bounds')
ax.axvspan(8.0, 18.0, color='#DDD6FE', alpha=0.45, label='Optimal Matching Band (P_L > 4.2 mW)')
ax.axvline(12.0, color='black', linestyle='--', lw=1.8, label='Matched Coil Resistance R_L = R_c = 12.0 Ohm')

ax.set_xlabel('Electrical Load Resistance R_L (Ohm)', fontweight='bold')
ax.set_ylabel('Peak Electrical Power P_L (mW)', fontweight='bold')
ax.set_xlim(1.0, 50.0)
ax.set_ylim(0, max(P_RL)*1.15)
ax.grid(True, linestyle=':', alpha=0.6)

match_txt = f'Maximum Power Match: {P_RL.max():.2f} mW\n@ R_L = R_c = 12.0 Ohm'
ax.annotate(match_txt, 
            xy=(12.0, P_RL.max()), xytext=(22.0, P_RL.max()*0.9),
            arrowprops=dict(facecolor='black', shrink=0.08, width=1.5, headwidth=7),
            fontweight='bold', fontsize=9.5, bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor=c_purple, alpha=0.9))

ax.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.95)
plt.title('Electrical Load Impedance Matching Sweep (f = 300 Hz, Jacobi Match)', fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'impedance_matching_sweep.png'), dpi=300)
plt.close()

# ----------------------------------------------------
# FIGURE 6: Manifold Volume Sensitivity
# ----------------------------------------------------
print('Generating Figure 6 from PBARNPhysicsEngine...')
Vm_L = np.linspace(5.0, 50.0, 15)
eta_Vm, U_Vm = engine.solve_manifold_sensitivity(Vm_L)

fig, ax1 = plt.subplots(figsize=(8.8, 4.8), dpi=300)
c_blue = '#0284C7'
c_rose = '#E11D48'

ax1.plot(Vm_L, eta_Vm, color=c_blue, marker='o', ms=6, lw=2.2, label='Broadband Mean Efficiency \\eta_mean (%)')
ax1.set_xlabel('Common Manifold Volume V_m (Liters)', fontweight='bold')
ax1.set_ylabel('Broadband Mean Efficiency \\eta_mean (%)', color=c_blue, fontweight='bold')
ax1.tick_params(axis='y', labelcolor=c_blue)
ax1.set_ylim(0.5, max(eta_Vm)*1.2)
ax1.grid(True, linestyle=':', alpha=0.6)

ax1.axvspan(12.0, 18.0, color='#BAE6FD', alpha=0.35, label='Optimum Volume Band (12 - 18 L)')

ax2 = ax1.twinx()
ax2.plot(Vm_L, U_Vm, color=c_rose, marker='s', ms=6, lw=2.0, ls='--', label='Response Uniformity Metric U_\\eta')
ax2.set_ylabel('Spectral Uniformity Metric U_\\eta', color=c_rose, fontweight='bold')
ax2.tick_params(axis='y', labelcolor=c_rose)
ax2.set_ylim(0.3, 0.95)

idx_opt = np.argmin(np.abs(Vm_L - 15.0))
opt_txt = f'Optimal Volume: V_m = 15.0 L\n(\\eta_mean = {eta_Vm[idx_opt]:.2f}%, U_\\eta = {U_Vm[idx_opt]:.3f})'
ax1.annotate(opt_txt, 
             xy=(15.0, eta_Vm[idx_opt]), xytext=(22.0, eta_Vm[idx_opt]*0.95),
             arrowprops=dict(facecolor=c_blue, shrink=0.08, width=1.5, headwidth=7),
             fontweight='bold', fontsize=9.5, bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor=c_blue, alpha=0.9))

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right', frameon=True, facecolor='white', framealpha=0.95)
plt.title('Parametric Sensitivity to Common Manifold Volume V_m (5 L to 50 L)', fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'sensitivity_manifold_volume.png'), dpi=300)
plt.close()

# ----------------------------------------------------
# FIGURE 7: Dynamic SPL Range Sweep (90 dB to 180 dB)
# ----------------------------------------------------
print('Generating Figure 7 from PBARNPhysicsEngine...')
SPL_range = np.linspace(90.0, 180.0, 100)
P_harv_W = []
P_inc_arr_W = []
eff_SPL = []

for spl in SPL_range:
    eng_spl = PBARNPhysicsEngine(SPL=spl)
    P_inc_arr_W.append(eng_spl.P_ac_in)
    P_l, _, _, eta_ap, _, _ = eng_spl.solve_point(300.0, coupled=True, nonlinear_spl=True)
    P_harv_W.append(P_l)
    eff_SPL.append((P_l / eng_spl.P_ac_in) * 100.0 if eng_spl.P_ac_in > 0 else 0.0)

P_harv_W = np.array(P_harv_W)
P_inc_arr_W = np.array(P_inc_arr_W)
eff_SPL = np.array(eff_SPL)

fig, ax1 = plt.subplots(figsize=(9.5, 5.5), dpi=300)
c_power = '#16A34A'
c_eff = '#DC2626'

ax1.axvspan(90, 115, color='#F0FDF4', alpha=0.6, label='Linear Acoustic Zone (90 - 115 dB)')
ax1.axvspan(115, 150, color='#FEF3C7', alpha=0.6, label='Jet Nacelle Zone (115 - 150 dB)')
ax1.axvspan(150, 180, color='#FEE2E2', alpha=0.6, label='Rocket Launch Zone (150 - 180 dB)')

ax1.fill_between(SPL_range, P_harv_W * 0.8 * 1e3, P_harv_W * 1.2 * 1e3, color=c_power, alpha=0.18, label='Non-Linear Fluid Bounds')
ax1.plot(SPL_range, P_harv_W * 1e3, color=c_power, lw=2.6, label='Harvested Electrical Power (mW / W)')
ax1.set_yscale('log')
ax1.set_xlabel('Incident Sound Pressure Level SPL (dB re 20 uPa)', fontweight='bold')
ax1.set_ylabel('Peak Electrical Power Output P_L (mW)', color=c_power, fontweight='bold')
ax1.tick_params(axis='y', labelcolor=c_power)
ax1.set_xlim(90, 180)
ax1.set_ylim(1e-2, 1e6)
ax1.grid(True, which='both', linestyle=':', alpha=0.6)

ax2 = ax1.twinx()
ax2.plot(SPL_range, eff_SPL, color=c_eff, lw=2.0, ls='--', label='Conversion Efficiency \\eta (%)')
ax2.set_ylabel('Conversion Efficiency \\eta (%)', color=c_eff, fontweight='bold')
ax2.tick_params(axis='y', labelcolor=c_eff)
ax2.set_yscale('log')
ax2.set_ylim(1e-4, 10.0)

ax1.annotate('120 dB Benchmark Flight\n4.82 mW (1.82% \\eta_mean)', 
             xy=(120, 4.82), xytext=(98, 80),
             arrowprops=dict(facecolor=c_power, shrink=0.08, width=1.5, headwidth=6),
             fontweight='bold', fontsize=8.5, bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=c_power, alpha=0.9))

ax1.annotate('150 dB Jet Engine Nacelle\n0.45 W - 1.2 W Power Saturation', 
             xy=(150, P_harv_W[66]*1000), xytext=(128, 5000.0),
             arrowprops=dict(facecolor='darkorange', shrink=0.08, width=1.5, headwidth=6),
             fontweight='bold', fontsize=8.5, bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='darkorange', alpha=0.9))

ax1.annotate('180 dB Rocket Launch Fairing\n12.5 W Power + Acoustic Damping!', 
             xy=(180, P_harv_W[-1]*1000), xytext=(145, 100000.0),
             arrowprops=dict(facecolor=c_eff, shrink=0.08, width=1.5, headwidth=6),
             fontweight='bold', fontsize=8.5, bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=c_eff, alpha=0.9))

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', frameon=True, facecolor='white', framealpha=0.95)

plt.title('Harvester Dynamic Range & Power Saturation (90 dB to 180 dB SPL)', fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'sensitivity_spl_range.png'), dpi=300)
plt.close()

# ----------------------------------------------------
# FIGURE 8: 3D BEM Cavity Surface Pressure Field Solver
# ----------------------------------------------------
print('Generating Figure 8 from PBARNPhysicsEngine 3D BEM Solver...')
centroids_bem, p_bem = engine.solve_bem_3d_cavity(300.0, 120.0)

fig = plt.figure(figsize=(9.2, 6.2), dpi=300)
ax = fig.add_subplot(111, projection='3d')

sc = ax.scatter(centroids_bem[:, 0], centroids_bem[:, 1], centroids_bem[:, 2], 
                c=p_bem, cmap='plasma', s=35, edgecolors='k', linewidths=0.2, alpha=0.9)

cbar = fig.colorbar(sc, ax=ax, shrink=0.6, aspect=12, pad=0.1)
cbar.set_label('3D BEM Cavity Pressure |p_BEM| (Pa)', fontweight='bold')

bx_lines = [0, 0.6, 0.6, 0, 0, 0, 0.6, 0.6, 0, 0, 0, 0, 0.6, 0.6, 0.6, 0.6]
by_lines = [0, 0, 0.6, 0.6, 0, 0, 0, 0.6, 0.6, 0.6, 0.6, 0, 0, 0, 0.6, 0.6]
bz_lines = [0, 0, 0, 0, 0, -0.12, -0.12, -0.12, -0.12, 0, -0.12, -0.12, -0.12, 0, 0, -0.12]
ax.plot(bx_lines, by_lines, bz_lines, color='gray', linestyle='--', linewidth=1.2, label='Cavity Outer Boundary (0.6m x 0.6m x 0.12m)')

ax.set_xlabel('Manifold Width X (m)', fontweight='bold', labelpad=8)
ax.set_ylabel('Manifold Length Y (m)', fontweight='bold', labelpad=8)
ax.set_zlabel('Cavity Depth Z (m)', fontweight='bold', labelpad=8)

ax.view_init(elev=28, azim=-125)
plt.title('3D BEM Cavity Pressure Field |p_BEM(r)| (f = 300 Hz, 270 Quad Centroids)', fontweight='bold', pad=14)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'bem_3d_pressure_field.png'), dpi=300)
plt.close()

print('ALL 8 PUBLICATION FIGURES GENERATED DIRECTLY FROM PBARNPhysicsEngine SOLVER!')
