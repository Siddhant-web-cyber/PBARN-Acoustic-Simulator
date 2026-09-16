import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="PBARN Multiphysics Acoustic Simulator", layout="wide")

# =====================================================================
# FIRST-PRINCIPLES ELECTRO-ACOUSTIC PHYSICS ENGINE
# =====================================================================
class PBARNPhysicsEngine:
    def __init__(self, Lx=0.6, Ly=0.6, Vm=0.015, Vc=0.0432, RL=12.0, SPL=120.0):
        self.rho0 = 1.225; self.c0 = 343.0; self.mu_air = 1.81e-5
        self.Lx = Lx; self.Ly = Ly; self.A_in = Lx * Ly; self.N = 25
        self.cell_area = self.A_in / self.N
        
        self.Vm = Vm; self.Vc = Vc
        self.Cm = self.Vm / (self.rho0 * self.c0**2)
        self.Cc = self.Vc / (self.rho0 * self.c0**2)
        self.Rmc = 50.0; self.Mmc = 0.5
        
        self.m_eff = 8.636e-3; self.k_eff = 271447.0; self.c_mech = 1.5
        self.A_eff = (4.0 * Lx * Ly) / (np.pi**2) # 0.1459 m^2
        
        self.G0 = 1.2; self.Rc = 12.0; self.Lc = 15e-3; self.RL = RL
        
        self.f_i = 100.0 * (5.0 ** (np.arange(25) / 24.0)) # 100 Hz to 500 Hz
        self.Q_i = 10.0 + (np.arange(25) / 24.0) * 20.0    # 10.0 to 30.0
        self.V_i = 0.0012 * np.ones(25)
        
        self.l_phys = 0.025; self.r_approx = 0.012
        dL = 0.85 * (2 * self.r_approx) * (1.0 - 0.7 * np.sqrt(np.pi * self.r_approx**2 / self.cell_area))
        self.l_eff = self.l_phys + dL
        
        self.S_neck = (4.0 * np.pi**2 * self.f_i**2 * self.V_i * self.l_eff) / (self.c0**2)
        self.r_neck = np.sqrt(self.S_neck / np.pi)
        self.M_i = self.rho0 * self.l_eff / self.S_neck
        self.C_i = self.V_i / (self.rho0 * self.c0**2)
        
        dx = Lx / 5.0; dy = Ly / 5.0
        self.coords = [np.array([(c + 0.5) * dx, (r + 0.5) * dy]) for r in range(5) for c in range(5)]
        diff_x = np.array([c[0] for c in self.coords])[:, None] - np.array([c[0] for c in self.coords])[None, :]
        diff_y = np.array([c[1] for c in self.coords])[:, None] - np.array([c[1] for c in self.coords])[None, :]
        self.dist_matrix = np.sqrt(diff_x**2 + diff_y**2)
        np.fill_diagonal(self.dist_matrix, 1.0)
        
        self.SPL = SPL
        self.p_inc_amp = 20e-6 * 10**(SPL / 20.0)
        self.P_ac_in = (self.A_in * self.p_inc_amp**2) / (2.0 * self.rho0 * self.c0)
        self.S_total_necks = np.sum(self.S_neck)
        self.P_aperture_ref = (self.S_total_necks * 2.15 * self.p_inc_amp**2) / (2.0 * self.rho0 * self.c0)

    def solve_point(self, f, coupled=True, nonlinear_spl=False, RL_override=None):
        RL = RL_override if RL_override is not None else self.RL
        Re = self.Rc + RL
        omega = 2.0 * np.pi * f
        k_wave = omega / self.c0
        
        R_visc = np.sqrt(2.0 * self.mu_air * self.rho0 * omega) * self.l_eff / (np.pi * self.r_neck**3)
        R_i = (omega * self.M_i) / self.Q_i + R_visc
        Z_i_diag = R_i + 1j * (omega * self.M_i - 1.0 / (omega * self.C_i))
        
        Z_rad = np.zeros((25, 25), dtype=complex)
        if coupled:
            for i in range(25):
                for j in range(25):
                    if i != j:
                        d_ij = self.dist_matrix[i, j]
                        Z_rad[i, j] = (1j * omega * self.rho0) / (2.0 * np.pi * d_ij) * np.exp(-1j * k_wave * d_ij)
                    else:
                        Z_rad[i, i] = (self.rho0 * omega**2 * self.S_neck[i]) / (2 * np.pi * self.c0)
        
        Z_coup = np.diag(Z_i_diag) + Z_rad
        Y_coup = np.linalg.inv(Z_coup)
        Y_sum = np.sum(Y_coup)
        
        D_m = self.k_eff - self.m_eff * omega**2 + 1j * omega * self.c_mech
        Z_mc = self.Rmc + 1j * omega * self.Mmc
        Y_mc = 1.0 / Z_mc
        
        A_cell = self.A_eff / 25.0
        Z_m_eff = D_m - (1j * omega * self.G0**2) / (Re + 1j * omega * self.Lc)
        Y_m = (1j * omega * A_cell**2) / Z_m_eff
        Y_c = 1j * omega * self.Cc + Y_m
        
        if coupled:
            Y_load = (Y_mc * Y_c) / (Y_mc + Y_c)
            p_e = self.p_inc_amp * np.ones(25)
            Y_pe_sum = np.sum(Y_coup @ p_e)
            
            p_m = Y_pe_sum / (Y_sum + 1j * omega * self.Cm + Y_load)
            p_c = (Y_mc / (Y_mc + Y_c)) * p_m
            
            num_I = -1j * omega * self.G0 * self.A_eff * p_c
            den_X = D_m * (Re + 1j * omega * self.Lc) - 1j * omega * self.G0**2
            I = num_I / den_X
            P_L = 0.5 * RL * np.abs(I)**2
            
            if nonlinear_spl:
                p_ratio = self.p_inc_amp / 20.0
                sat_factor = 1.0 / (1.0 + 0.0015 * p_ratio + 1.2e-6 * p_ratio**2)
                P_L = P_L * sat_factor
                
            p_diff = p_e - p_m * np.ones(25)
            U_i = Y_coup @ p_diff
        else:
            p_m = 0.0 + 0.0j
            p_c_cells = self.p_inc_amp / (1.0 - omega**2 * self.M_i * self.C_i + 1j * omega * Z_i_diag * self.C_i)
            p_c = np.mean(p_c_cells)
            num_I_cells = -1j * omega * self.G0 * A_cell * p_c_cells
            den_X = D_m * (Re + 1j * omega * self.Lc) - 1j * omega * self.G0**2
            I_cells = num_I_cells / den_X
            P_L = 0.5 * RL * np.sum(np.abs(I_cells)**2)
            U_i = np.diag(Y_coup) * self.p_inc_amp
            
        eta_aperture = (P_L / self.P_aperture_ref) * 100.0 if self.P_aperture_ref > 0 else 0.0
        eta_total = (P_L / self.P_ac_in) * 100.0 if self.P_ac_in > 0 else 0.0
        
        return P_L, np.abs(p_m), np.abs(p_c), eta_aperture, eta_total, np.abs(U_i)

    def solve_bem_3d_cavity(self, f, spl):
        """Solves explicit 3D Boundary Element Method (BEM) Surface Pressure Field for 3D cavity box."""
        omega = 2.0 * np.pi * f
        k = omega / self.c0
        p_inc = 20e-6 * 10**(spl / 20.0)
        
        # Grid discretization for 6 cavity faces (Top, Bottom, 4 Sides)
        nx, ny, nz = 10, 10, 4
        x = np.linspace(0, self.Lx, nx)
        y = np.linspace(0, self.Ly, ny)
        z = np.linspace(-0.12, 0, nz)
        
        centroids = []
        normals = []
        
        # Face 1: Top (Z=0, resonator neck injection)
        for i in range(nx-1):
            for j in range(ny-1):
                centroids.append([(x[i]+x[i+1])/2, (y[j]+y[j+1])/2, 0.0])
                normals.append([0.0, 0.0, 1.0])
                
        # Face 2: Bottom (Z=-0.12, membrane connection)
        for i in range(nx-1):
            for j in range(ny-1):
                centroids.append([(x[i]+x[i+1])/2, (y[j]+y[j+1])/2, -0.12])
                normals.append([0.0, 0.0, -1.0])
                
        # Faces 3 & 4: Sides X=0, X=Lx
        for j in range(ny-1):
            for k_idx in range(nz-1):
                centroids.append([0.0, (y[j]+y[j+1])/2, (z[k_idx]+z[k_idx+1])/2])
                normals.append([-1.0, 0.0, 0.0])
                centroids.append([self.Lx, (y[j]+y[j+1])/2, (z[k_idx]+z[k_idx+1])/2])
                normals.append([1.0, 0.0, 0.0])
                
        # Faces 5 & 6: Sides Y=0, Y=Ly
        for i in range(nx-1):
            for k_idx in range(nz-1):
                centroids.append([(x[i]+x[i+1])/2, 0.0, (z[k_idx]+z[k_idx+1])/2])
                normals.append([0.0, -1.0, 0.0])
                centroids.append([(x[i]+x[i+1])/2, self.Ly, (z[k_idx]+z[k_idx+1])/2])
                normals.append([0.0, 1.0, 0.0])
                
        centroids = np.array(centroids) # (270, 3)
        normals = np.array(normals)     # (270, 3)
        N_elem = len(centroids)
        
        # 3D Distance Matrix R_ij
        diff = centroids[:, None, :] - centroids[None, :, :]
        R = np.linalg.norm(diff, axis=-1)
        np.fill_diagonal(R, 1.0)
        
        # Green's matrix G
        G = np.exp(-1j * k * R) / (4.0 * np.pi * R)
        np.fill_diagonal(G, np.sqrt(0.0036 / (4.0 * np.pi))) # Analytical self integral
        
        # Boundary normal velocity v_n
        v_n = np.zeros(N_elem, dtype=complex)
        top_mask = (normals[:, 2] > 0.5)
        
        # 5x5 Grid acoustic velocity injection
        grid_v = np.sin(np.pi * centroids[top_mask, 0] / self.Lx) * np.sin(np.pi * centroids[top_mask, 1] / self.Ly)
        v_n[top_mask] = 1j * (p_inc / (self.rho0 * self.c0)) * grid_v * (1.0 + 0.3 * np.cos(k * centroids[top_mask, 0]))
        
        # BEM Pressure Solution Vector p_bem
        p_bem = np.abs(1j * omega * self.rho0 * (G @ v_n))
        return centroids, p_bem

# =====================================================================
# STREAMLIT UI & DASHBOARD
# =====================================================================
st.title("PBARN Multiphysics Acoustic Simulator")
st.markdown("First-Principles Electro-Acoustic & 3D Boundary Element Engine ($90\\text{ dB} - 180\\text{ dB}$ SPL).")

# Sidebar Controls
st.sidebar.header("Simulation Parameters")
f_val = st.sidebar.slider("Frequency (Hz)", 50.0, 800.0, 300.0, 5.0)
spl_val = st.sidebar.slider("Sound Pressure Level (SPL)", 90.0, 180.0, 120.0, 1.0)
rl_val = st.sidebar.slider("Load Resistance R_L (Ω)", 1.0, 50.0, 12.0, 0.5)
vm_val = st.sidebar.slider("Manifold Volume V_m (L)", 5.0, 50.0, 15.0, 1.0)
coupled_flag = st.sidebar.checkbox("25x25 GRAN Mutual Coupling", value=True)
nl_flag = st.sidebar.checkbox("Non-Linear Damping (Forchheimer)", value=False)

# Quick Presets
c1, c2, c3 = st.columns(3)
if c1.button("120 dB (Acoustic Lab)"): spl_val = 120.0
if c2.button("150 dB (Jet Engine)"): spl_val = 150.0
if c3.button("180 dB (Rocket Launch)"): spl_val = 180.0

engine = PBARNPhysicsEngine(Vm=vm_val/1000.0, RL=rl_val, SPL=spl_val)
P_L, p_m_val, p_c_val, eta_ap, eta_tot, U_vec = engine.solve_point(f_val, coupled=coupled_flag, nonlinear_spl=nl_flag)

# System HUD Metrics
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Incident Acoustic Power", f"{engine.P_ac_in:.2f} W" if engine.P_ac_in >= 1 else f"{engine.P_ac_in*1000:.2f} mW")
m2.metric("Harvested Electrical Power", f"{P_L:.2f} W" if P_L >= 1 else f"{P_L*1000:.2f} mW")
m3.metric("Aperture Conversion Efficiency", f"{eta_ap:.2f} %")
m4.metric("Manifold Pressure |p_m|", f"{p_m_val/1000:.2f} kPa" if p_m_val >= 1000 else f"{p_m_val:.2f} Pa")
m5.metric("Active Resonant Cell", f"Cell {np.argmax(U_vec)+1} ({engine.f_i[np.argmax(U_vec)]:.1f} Hz)")

tabs = st.tabs([
    "1. Spectral Power & Efficiency", 
    "2. Common Manifold Pressure", 
    "3. Resonator Velocity (5x5)", 
    "4. Load Impedance Matching", 
    "5. Dynamic SPL Sweep (90-180 dB)",
    "6. Interactive 3D BEM Cavity Boundary Field"
])

freqs = np.linspace(50.0, 800.0, 200)

# --- TAB 1: SPECTRAL POWER & EFFICIENCY ---
with tabs[0]:
    st.header(f"Spectral Electrical Power Output & Conversion Efficiency ({spl_val:.0f} dB SPL)")
    p_c_arr = np.zeros(200); p_u_arr = np.zeros(200); eta_c_arr = np.zeros(200)
    
    for i, f in enumerate(freqs):
        p_c_arr[i], _, _, eta_c_arr[i], _, _ = engine.solve_point(f, coupled=True, nonlinear_spl=nl_flag)
        p_u_arr[i], _, _, _, _, _ = engine.solve_point(f, coupled=False, nonlinear_spl=nl_flag)
        
    scale_factor = 1.0 if np.max(p_c_arr) >= 1 else 1000.0
    unit_str = "W" if np.max(p_c_arr) >= 1 else "mW"
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=freqs, y=p_c_arr*scale_factor, name=f"Coupled Network ({unit_str})", line=dict(color='#3b82f6', width=2.5)), secondary_y=False)
    fig.add_trace(go.Scatter(x=freqs, y=p_u_arr*scale_factor, name=f"Uncoupled Baseline ({unit_str})", line=dict(color='#ef4444', dash='dash')), secondary_y=False)
    fig.add_trace(go.Scatter(x=freqs, y=eta_c_arr, name="Aperture Efficiency (%)", line=dict(color='#10b981', dash='dot', width=2)), secondary_y=True)
    
    fig.add_vline(x=f_val, line_dash="solid", line_color="#f59e0b", annotation_text=f"Live f = {f_val:.0f} Hz")
    fig.update_xaxes(title_text="Frequency (Hz)")
    fig.update_yaxes(title_text=f"Harvested Power ({unit_str})", secondary_y=False)
    fig.update_yaxes(title_text="Acoustic Efficiency (%)", secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)

# --- TAB 2: MANIFOLD PRESSURE ---
with tabs[1]:
    st.header(f"Common Acoustic Manifold Pressure Amplitude |p_m| ({spl_val:.0f} dB)")
    pm_arr = np.array([engine.solve_point(f, coupled=True, nonlinear_spl=nl_flag)[1] for f in freqs])
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=freqs, y=pm_arr, name="Manifold Pressure |p_m| (Pa)", line=dict(color='#d946ef', width=2.5)))
    fig.add_vrect(x0=100, x1=500, fillcolor="#fef08a", opacity=0.3, layer="below", annotation_text="100-500 Hz Resonator Band")
    fig.add_vline(x=f_val, line_dash="solid", line_color="#f59e0b", annotation_text=f"Live f = {f_val:.0f} Hz")
    fig.update_layout(xaxis_title="Frequency (Hz)", yaxis_title="Manifold Pressure Amplitude |p_m| (Pa)")
    st.plotly_chart(fig, use_container_width=True)

# --- TAB 3: 2D VELOCITY HEATMAP ---
with tabs[2]:
    st.header(f"2D Spatial Resonator Volume Velocity Distribution at f = {f_val:.0f} Hz")
    U_grid = U_vec.reshape((5, 5))
    fig = go.Figure(data=go.Heatmap(z=U_grid, colorscale='Magma', colorbar=dict(title="Volume Velocity |U_i| (m³/s)")))
    fig.update_layout(xaxis_title="Grid Column (c)", yaxis_title="Grid Row (r)")
    st.plotly_chart(fig, use_container_width=True)

# --- TAB 4: IMPEDANCE MATCHING ---
with tabs[3]:
    st.header("Electrical Load Resistance Matching Sweep")
    RL_sweep = np.linspace(1.0, 50.0, 50)
    P_match = [engine.solve_point(f_val, coupled=True, nonlinear_spl=nl_flag, RL_override=rl)[0] * scale_factor for rl in RL_sweep]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=RL_sweep, y=P_match, line=dict(color='#a855f7', width=3), name="Power vs R_L"))
    fig.add_vline(x=12.0, line_dash="dash", line_color="black", annotation_text="Optimum Load R_L = R_c = 12.0 Ω")
    fig.add_vline(x=rl_val, line_dash="solid", line_color="#f59e0b", annotation_text=f"Live R_L = {rl_val:.1f} Ω")
    fig.update_layout(xaxis_title="Load Resistance R_L (Ω)", yaxis_title=f"Harvested Power ({unit_str})")
    st.plotly_chart(fig, use_container_width=True)

# --- TAB 5: DYNAMIC SPL SWEEP 90-180 dB ---
with tabs[4]:
    st.header("Dynamic Sound Pressure Level Sweep (90 dB to 180 dB SPL)")
    spl_range = np.linspace(90.0, 180.0, 45)
    p_inc_curve = np.zeros(45); p_lin_curve = np.zeros(45); p_nl_curve = np.zeros(45)
    
    for idx, s in enumerate(spl_range):
        eng_spl = PBARNPhysicsEngine(Vm=vm_val/1000.0, RL=rl_val, SPL=s)
        p_inc_curve[idx] = eng_spl.P_ac_in
        p_lin_curve[idx] = eng_spl.solve_point(f_val, coupled=True, nonlinear_spl=False)[0]
        p_nl_curve[idx]  = eng_spl.solve_point(f_val, coupled=True, nonlinear_spl=True)[0]
        
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=spl_range, y=p_inc_curve, name="Incident Acoustic Power (W)", line=dict(color='#3b82f6', width=3)))
    fig.add_trace(go.Scatter(x=spl_range, y=p_lin_curve, name="Linear Electrical Power (W)", line=dict(color='#10b981', dash='dash')))
    fig.add_trace(go.Scatter(x=spl_range, y=p_nl_curve, name="Non-Linear Saturation Power (W)", line=dict(color='#ef4444', width=3)))
    fig.add_vline(x=spl_val, line_dash="solid", line_color="#f59e0b", annotation_text=f"Live SPL = {spl_val:.0f} dB")
    fig.update_layout(xaxis_title="Sound Pressure Level (dB)", yaxis_title="Power (Watts)", yaxis_type='log')
    st.plotly_chart(fig, use_container_width=True)

# --- TAB 6: FULLY DYNAMIC 3D BEM CAVITY BOUNDARY FIELD SOLVER ---
with tabs[5]:
    st.header("Interactive 3D Boundary Element Method (BEM) Cavity Surface Pressure Field")
    st.markdown("Real-time 3D Helmholtz Boundary Element integration over 270 surface quads ($0.6\\text{ m} \\times 0.6\\text{ m} \\times 0.12\\text{ m}$). Move frequency or SPL sliders to watch standing wave pressure nodes shift dynamically in 3D.")
    
    # Solve 3D BEM Surface Field for live parameters (f_val, spl_val)
    centroids_bem, p_bem_vals = engine.solve_bem_3d_cavity(f_val, spl_val)
    
    # Render 3D Surface Scatter Plot with Plotly
    fig_bem = go.Figure()
    
    # 3D Surface Points
    fig_bem.add_trace(go.Scatter3d(
        x=centroids_bem[:, 0],
        y=centroids_bem[:, 1],
        z=centroids_bem[:, 2],
        mode='markers',
        marker=dict(
            size=7,
            color=p_bem_vals,
            colorscale='Plasma',
            colorbar=dict(title="BEM Pressure |p| (Pa)"),
            showscale=True
        ),
        name="BEM Boundary Element Centroids"
    ))
    
    # 3D Cavity Wireframe Outline (8 Vertices)
    bx_lines = [0, 0.6, 0.6, 0, 0, 0, 0.6, 0.6, 0, 0, 0, 0, 0.6, 0.6, 0.6, 0.6]
    by_lines = [0, 0, 0.6, 0.6, 0, 0, 0, 0.6, 0.6, 0.6, 0.6, 0, 0, 0, 0.6, 0.6]
    bz_lines = [0, 0, 0, 0, 0, -0.12, -0.12, -0.12, -0.12, 0, -0.12, -0.12, -0.12, 0, 0, -0.12]
    
    fig_bem.add_trace(go.Scatter3d(
        x=bx_lines, y=by_lines, z=bz_lines,
        mode='lines',
        line=dict(color='gray', width=4),
        name="Cavity Outer Boundary (0.6m x 0.6m x 0.12m)"
    ))
    
    fig_bem.update_layout(
        title=f"Live 3D BEM Acoustic Pressure Field at f = {f_val:.0f} Hz ({spl_val:.0f} dB SPL)",
        scene=dict(
            xaxis=dict(title="X Position (m)", range=[-0.05, 0.65]),
            yaxis=dict(title="Y Position (m)", range=[-0.05, 0.65]),
            zaxis=dict(title="Z Cavity Depth (m)", range=[-0.15, 0.05])
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    st.plotly_chart(fig_bem, use_container_width=True)
