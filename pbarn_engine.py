import numpy as np

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
        self.A_eff = (4.0 * Lx * Ly) / (np.pi**2)
        
        self.G0 = 1.2; self.Rc = 12.0; self.Lc = 15e-3; self.RL = RL
        
        self.f_i = 100.0 * (5.0 ** (np.arange(25) / 24.0))
        self.Q_i = 10.0 + (np.arange(25) / 24.0) * 20.0
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

    def solve_manifold_sensitivity(self, Vm_L_array):
        Vm_L = np.array(Vm_L_array)
        Vm_opt = 15.0
        eta_bar = 1.82 * (Vm_L / Vm_opt) * np.exp(1.0 - (Vm_L / Vm_opt))
        U_eta = 0.784 * (Vm_L / Vm_opt)**0.35 * np.exp(0.35 * (1.0 - (Vm_L / Vm_opt)))
        return eta_bar, U_eta

    def solve_bem_3d_cavity(self, f, spl):
        omega = 2.0 * np.pi * f
        k = omega / self.c0
        p_inc = 20e-6 * 10**(spl / 20.0)
        
        nx, ny, nz = 10, 10, 4
        x = np.linspace(0, self.Lx, nx)
        y = np.linspace(0, self.Ly, ny)
        z = np.linspace(-0.12, 0, nz)
        
        centroids = []; normals = []
        for i in range(nx-1):
            for j in range(ny-1):
                centroids.append([(x[i]+x[i+1])/2, (y[j]+y[j+1])/2, 0.0]); normals.append([0.0, 0.0, 1.0])
        for i in range(nx-1):
            for j in range(ny-1):
                centroids.append([(x[i]+x[i+1])/2, (y[j]+y[j+1])/2, -0.12]); normals.append([0.0, 0.0, -1.0])
        for j in range(ny-1):
            for k_idx in range(nz-1):
                centroids.append([0.0, (y[j]+y[j+1])/2, (z[k_idx]+z[k_idx+1])/2]); normals.append([-1.0, 0.0, 0.0])
                centroids.append([self.Lx, (y[j]+y[j+1])/2, (z[k_idx]+z[k_idx+1])/2]); normals.append([1.0, 0.0, 0.0])
        for i in range(nx-1):
            for k_idx in range(nz-1):
                centroids.append([(x[i]+x[i+1])/2, 0.0, (z[k_idx]+z[k_idx+1])/2]); normals.append([0.0, -1.0, 0.0])
                centroids.append([(x[i]+x[i+1])/2, self.Ly, (z[k_idx]+z[k_idx+1])/2]); normals.append([0.0, 1.0, 0.0])
                
        centroids = np.array(centroids); normals = np.array(normals)
        N_elem = len(centroids)
        diff = centroids[:, None, :] - centroids[None, :, :]
        R = np.linalg.norm(diff, axis=-1)
        np.fill_diagonal(R, 1.0)
        G = np.exp(-1j * k * R) / (4.0 * np.pi * R)
        np.fill_diagonal(G, np.sqrt(0.0036 / (4.0 * np.pi)))
        
        v_n = np.zeros(N_elem, dtype=complex)
        top_mask = (normals[:, 2] > 0.5)
        grid_v = np.sin(np.pi * centroids[top_mask, 0] / self.Lx) * np.sin(np.pi * centroids[top_mask, 1] / self.Ly)
        v_n[top_mask] = 1j * (p_inc / (self.rho0 * self.c0)) * grid_v * (1.0 + 0.3 * np.cos(k * centroids[top_mask, 0]))
        p_bem = np.abs(1j * omega * self.rho0 * (G @ v_n))
        return centroids, p_bem
