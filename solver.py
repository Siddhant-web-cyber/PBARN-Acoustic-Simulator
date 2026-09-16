"""
PBARN Mathematical Matrix Solver Engine
Implements the exact 25-page handwritten derivation with Electromagnetic Transducer & GRAN matrix.
"""

import numpy as np
from typing import Dict, Tuple
from parameters import PBARNConfig

class PBARNMathSolver:
    def __init__(self, config: PBARNConfig = None):
        self.cfg = config if config is not None else PBARNConfig()
        self.dist = self.cfg.resonator.generate_distribution(self.cfg.geom, self.cfg.air)
        self.Cm = self.cfg.mc.Vm / (self.cfg.air.rho0 * self.cfg.air.c0**2)
        self.Cc = self.cfg.mc.Vc / (self.cfg.air.rho0 * self.cfg.air.c0**2)
        self.m_eff, self.k_eff, self.A_eff = self.cfg.membrane.get_modal_properties(self.cfg.geom)
        
    def solve_frequency_response(
        self, 
        freqs: np.ndarray, 
        p_inc_amp: float = 20.0, # Incident pressure amplitude [Pa] (~120 dB)
        coupled: bool = True,
        RL_override: float = None
    ) -> Dict[str, np.ndarray]:
        
        N = self.cfg.geom.N
        num_f = len(freqs)
        
        RL = RL_override if RL_override is not None else self.cfg.transducer.RL
        Rc = self.cfg.transducer.Rc
        Lc = self.cfg.transducer.Lc
        G0 = self.cfg.transducer.G0
        Re = Rc + RL
        
        M_arr = self.dist['M']
        C_arr = self.dist['C']
        R_arr = self.dist['R']
        
        c_mech = self.cfg.membrane.c_mech
        
        p_m_all = np.zeros(num_f, dtype=complex)
        p_c_all = np.zeros(num_f, dtype=complex)
        X_all = np.zeros(num_f, dtype=complex)
        I_all = np.zeros(num_f, dtype=complex)
        P_harv = np.zeros(num_f)
        P_inc = np.zeros(num_f)
        eta_eff = np.zeros(num_f)
        U_hat_all = np.zeros((N, num_f), dtype=complex)
        
        # Incident acoustic power P_ac_in = (A_in * |p_inc|^2) / (2 * rho0 * c0)
        A_in = self.cfg.geom.membrane_area
        P_ac_in_val = (A_in * p_inc_amp**2) / (2.0 * self.cfg.air.rho0 * self.cfg.air.c0)
        
        S_neck = self.dist['S_neck']
        r_neck = self.dist['r_neck']
        dist_matrix = self.dist['dist_matrix']
        rho0 = self.cfg.air.rho0
        c0 = self.cfg.air.c0
        
        P_visc_all = np.zeros(num_f)
        P_mech_all = np.zeros(num_f)
        P_balance_err = np.zeros(num_f)

        for k, f in enumerate(freqs):
            omega = 2.0 * np.pi * f
            k_wave = omega / c0
            
            # Intrinsic Resonator Impedance Z_i
            Z_i_diag = R_arr + 1j * (omega * M_arr - 1.0 / (omega * C_arr))
            
            # Mutual Radiation Impedance Matrix Z_rad (25x25)
            Z_rad = np.zeros((N, N), dtype=complex)
            for i_idx in range(N):
                for j_idx in range(N):
                    if i_idx != j_idx:
                        # Inter-orifice mutual radiation impedance Z_ij = j * omega * rho0 / (2 * pi * d_ij) * exp(-j * k * d_ij)
                        d_ij = dist_matrix[i_idx, j_idx]
                        Z_rad[i_idx, j_idx] = (1j * omega * rho0) / (2.0 * np.pi * d_ij) * np.exp(-1j * k_wave * d_ij)
            
            # Coupled Resonator Impedance Matrix Z_coup = diag(Z_i) + Z_rad
            Z_coup = np.diag(Z_i_diag) + Z_rad
            Y_coup = np.linalg.inv(Z_coup)
            Y_i = np.diag(Y_coup) # Effective admittance vector
            Y_sum = np.sum(Y_coup)
            
            # Membrane Mechanical Impedance D_m = k_eff - m_eff * omega^2 + j * omega * c_mech
            D_m = self.k_eff - self.m_eff * omega**2 + 1j * omega * c_mech
            
            # Effective Mechanical Impedance Z_m_eff = D_m - (j * omega * G0^2) / (Re + j * omega * Lc)
            Z_m_eff = D_m - (1j * omega * G0**2) / (Re + 1j * omega * Lc)
            
            # Membrane Acoustic Admittance Y_m = (j * omega * A_cell^2) / Z_m_eff
            A_cell = self.A_eff / self.cfg.geom.N
            Y_m = (1j * omega * A_cell**2) / Z_m_eff
            
            # Chamber Admittance Y_c = j * omega * Cc + Y_m
            Y_c = 1j * omega * self.Cc + Y_m
            
            # Manifold to Chamber Connection Admittance Y_mc = 1 / (Rmc + j * omega * Mmc)
            Z_mc = self.cfg.mc.Rmc + 1j * omega * self.cfg.mc.Mmc
            Y_mc = 1.0 / Z_mc
            
            # Manifold Load Admittance Y_load = (Y_mc * Y_c) / (Y_mc + Y_c)
            if coupled:
                Y_load = (Y_mc * Y_c) / (Y_mc + Y_c)
            else:
                Y_load = 0.0 + 0.0j
                
            # Driving pressure vector p_e (assumed uniform acoustic illumination p_e_i = p_inc_amp)
            p_e = p_inc_amp * np.ones(N)
            Y_pe_sum = np.sum(np.dot(Y_coup, p_e))
            
            # Common Manifold Pressure p_m = (1^T Y_coup p_e) / (Y_sum + j * omega * Cm + Y_load)
            if coupled:
                p_m = Y_pe_sum / (Y_sum + 1j * omega * self.Cm + Y_load)
                p_c = (Y_mc / (Y_mc + Y_c)) * p_m
                num_X = self.A_eff * p_c * (Re + 1j * omega * Lc)
                den_X = D_m * (Re + 1j * omega * Lc) - 1j * omega * G0**2
                X = num_X / den_X
                num_I = -1j * omega * G0 * self.A_eff * p_c
                I = num_I / den_X
                P_L = 0.5 * RL * np.abs(I)**2
                p_diff = p_e - p_m * np.ones(N)
                U_i = np.dot(Y_coup, p_diff)
            else:
                p_m = 0.0 + 0.0j
                # Individual cell chamber acoustic pressure for uncoupled baseline array
                p_c_cells = p_e / (1.0 - omega**2 * M_arr * C_arr + 1j * omega * R_arr * C_arr)
                p_c = np.mean(p_c_cells)
                A_cell = self.A_eff / N
                num_I_cells = -1j * omega * G0 * A_cell * p_c_cells
                den_X = D_m * (Re + 1j * omega * Lc) - 1j * omega * G0**2
                I_cells = num_I_cells / den_X
                I = np.sum(I_cells)
                X = np.mean((A_cell * p_c_cells * (Re + 1j * omega * Lc)) / den_X)
                P_L = 0.5 * RL * np.sum(np.abs(I_cells)**2)
                U_i = Y_i * p_e

            # Power Flow & Strict Energy Balance Verification
            P_array_in = 0.5 * np.real(np.sum(p_e * np.conj(U_i)))
            P_visc = 0.5 * np.sum(R_arr * np.abs(U_i)**2)
            P_mech = 0.5 * c_mech * (omega * np.abs(X))**2
            P_elec_coil = 0.5 * Rc * np.abs(I)**2
            P_mc_loss = 0.5 * self.cfg.mc.Rmc * np.abs((Y_mc / (Y_mc + Y_c)) * p_m * Y_c)**2
            
            P_dissipated = P_visc + P_mech + P_elec_coil + P_L + P_mc_loss
            P_err_pct = np.abs(P_array_in - P_dissipated) / P_array_in * 100.0 if P_array_in > 0 else 0.0
            
            p_m_all[k] = p_m
            p_c_all[k] = p_c
            X_all[k] = X
            I_all[k] = I
            P_harv[k] = P_L
            P_inc[k] = P_ac_in_val
            eta_eff[k] = (P_L / P_ac_in_val) * 100.0 if P_ac_in_val > 0 else 0.0
            U_hat_all[:, k] = U_i
            P_visc_all[k] = P_visc
            P_mech_all[k] = P_mech
            P_balance_err[k] = P_err_pct
            
        return {
            'freqs': freqs,
            'p_m': p_m_all,
            'p_c': p_c_all,
            'X': X_all,
            'I': I_all,
            'P_harv': P_harv,
            'P_inc': P_inc,
            'eta_eff': eta_eff,
            'U_hat': U_hat_all,
            'P_visc': P_visc_all,
            'P_mech': P_mech_all,
            'P_balance_err': P_balance_err
        }

    def compute_broadband_metrics(self, freqs: np.ndarray, eta_eff: np.ndarray) -> Tuple[float, float]:
        band_mask = (freqs >= 100.0) & (freqs <= 500.0)
        f_band = freqs[band_mask]
        eta_band = eta_eff[band_mask]
        bw = 400.0
        bar_eta = np.trapz(eta_band, f_band) / bw
        sigma_eta = np.sqrt(np.trapz((eta_band - bar_eta)**2, f_band) / bw)
        U_eta = 1.0 - (sigma_eta / (bar_eta + 1e-6)) if bar_eta > 0 else 0.0
        return float(bar_eta), float(U_eta)
