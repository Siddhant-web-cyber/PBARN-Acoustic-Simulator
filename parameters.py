"""
PBARN Parameter Definitions based on Handwritten Electromagnetic Derivation
"""

import numpy as np
from dataclasses import dataclass, field

@dataclass
class AirProperties:
    rho0: float = 1.225        # Equilibrium air density [kg/m^3]
    c0: float = 343.0          # Speed of sound [m/s]
    mu: float = 1.81e-5        # Dynamic viscosity [Pa s]
    kp: float = 0.0257         # Thermal conductivity [W/(m K)]
    cp: float = 1005.0         # Specific heat [J/(kg K)]

@dataclass
class NetworkGeometry:
    Nx: int = 5
    Ny: int = 5
    Lx: float = 0.60           # Membrane length [m]
    Ly: float = 0.60           # Membrane width [m]
    Lz: float = 0.12           # Chamber depth [m]
    
    @property
    def N(self) -> int:
        return self.Nx * self.Ny
    
    @property
    def cell_area(self) -> float:
        return (self.Lx / self.Nx) * (self.Ly / self.Ny)
        
    @property
    def membrane_area(self) -> float:
        return self.Lx * self.Ly

@dataclass
class ResonatorParameters:
    f_min: float = 100.0       # Minimum frequency [Hz]
    f_max: float = 500.0       # Maximum frequency [Hz]
    Q_min: float = 10.0        # Minimum Q factor
    Q_max: float = 30.0        # Maximum Q factor
    
    V_cavity_base: float = 0.0012 # Cavity volume base [m^3]
    l_physical: float = 0.025   # Neck physical length [m]
    
    def generate_distribution(self, geom: NetworkGeometry, air: AirProperties):
        N = geom.N
        # Logarithmic frequency spacing: f_i = f_min * (f_max / f_min)^((i-1)/(N-1))
        f_dist = self.f_min * (self.f_max / self.f_min) ** (np.arange(N) / (N - 1))
        omega_dist = 2 * np.pi * f_dist
        
        # Linear Q-factor distribution: Q_i = Q_min + (i-1)/(N-1) * (Q_max - Q_min)
        Q_dist = self.Q_min + (np.arange(N) / (N - 1)) * (self.Q_max - self.Q_min)
        
        # Fixed cavity volume V_i
        V_cavity = self.V_cavity_base * np.ones(N)
        
        # End correction delta_l
        r_approx = 0.012
        dL = 0.85 * (2 * r_approx) * (1.0 - 0.7 * np.sqrt(np.pi * r_approx**2 / geom.cell_area))
        l_eff = self.l_physical + dL
        
        # Calculate neck cross-sectional area S_i = (4 pi^2 f_i^2 V_i l_eff) / c0^2
        S_neck = (4.0 * np.pi**2 * f_dist**2 * V_cavity * l_eff) / (air.c0**2)
        r_neck = np.sqrt(S_neck / np.pi)
        
        # Acoustic Inertance M_i = rho0 * l_eff / S_i
        M = air.rho0 * l_eff / S_neck
        
        # Acoustic Compliance C_i = V_i / (rho0 * c0^2)
        C = V_cavity / (air.rho0 * air.c0**2)
        
        # Acoustic Resistance R_i = omega_i * M_i / Q_i (incorporating Crandall thermo-viscous boundary layer loss)
        R_visc_base = np.sqrt(2.0 * air.mu * air.rho0 * omega_dist) * l_eff / (np.pi * r_neck**3)
        R = (omega_dist * M) / Q_dist + R_visc_base
        
        # 5x5 Spatial Grid Coordinates for aperture locations (x_i, y_i)
        dx = geom.Lx / geom.Nx
        dy = geom.Ly / geom.Ny
        x_coords = np.zeros(N)
        y_coords = np.zeros(N)
        for idx in range(N):
            row = idx // geom.Nx
            col = idx % geom.Nx
            x_coords[idx] = (col + 0.5) * dx
            y_coords[idx] = (row + 0.5) * dy

        # Inter-orifice mutual distance matrix D_ij
        diff_x = x_coords[:, np.newaxis] - x_coords[np.newaxis, :]
        diff_y = y_coords[:, np.newaxis] - y_coords[np.newaxis, :]
        dist_matrix = np.sqrt(diff_x**2 + diff_y**2)
        np.fill_diagonal(dist_matrix, 1.0) # Avoid zero division on diagonal
        
        return {
            'f_i': f_dist,
            'omega_i': omega_dist,
            'Q_i': Q_dist,
            'V_cavity': V_cavity,
            'S_neck': S_neck,
            'r_neck': r_neck,
            'l_eff': l_eff,
            'M': M,
            'C': C,
            'R': R,
            'R_visc_base': R_visc_base,
            'x_coords': x_coords,
            'y_coords': y_coords,
            'dist_matrix': dist_matrix
        }

@dataclass
class ManifoldChamberParameters:
    Vm: float = 0.015          # Common manifold volume [m^3]
    Vc: float = 0.0432         # Rear chamber volume [m^3] (0.6 x 0.6 x 0.12)
    Rmc: float = 50.0          # Manifold-to-chamber acoustic resistance [Pa s / m^3]
    Mmc: float = 0.5           # Manifold-to-chamber acoustic inertance [kg / m^4]

@dataclass
class GrapheneKaptonMembrane:
    # Kapton properties
    rho_k: float = 1420.0      # Density [kg/m^3]
    E_k: float = 2.5e9         # Young's modulus [Pa]
    h_k: float = 50e-6         # Thickness [m] (50 um)
    
    # Graphene properties
    rho_g: float = 2200.0      # Density [kg/m^3]
    E_g: float = 1.0e12        # Young's modulus [Pa] (1 TPa)
    h_g: float = 2e-6          # Thickness [m] (2 um)
    
    tension: float = 6218.0    # Membrane tension T [N/m] tuned to 300 Hz center of harvesting band
    nu: float = 0.34           # Poisson's ratio
    c_mech: float = 1.5        # Mechanical damping [N s / m]

    @property
    def hm(self) -> float:
        return self.h_k + self.h_g

    @property
    def rho_m(self) -> float:
        return (self.rho_k * self.h_k + self.rho_g * self.h_g) / self.hm

    @property
    def E_eff(self) -> float:
        return (self.E_k * self.h_k + self.E_g * self.h_g) / self.hm

    def get_modal_properties(self, geom: NetworkGeometry):
        Lx, Ly = geom.Lx, geom.Ly
        # Modal mass m_eff = (rho_m * h_m * Lx * Ly) / 4 + m_coil (voice coil mass loading 1.85 g)
        m_coil = 1.85e-3
        m_eff = (self.rho_m * self.hm * Lx * Ly) / 4.0 + m_coil
        
        # Modal tension stiffness k_tension = (T * pi^2 * Lx * Ly / 4) * (1/Lx^2 + 1/Ly^2) = T * pi^2 / 2 for square
        k_tension = (self.tension * np.pi**2) / 2.0
        
        # Bending rigidity D = E_eff * hm^3 / (12 * (1 - nu^2))
        D = (self.E_eff * self.hm**3) / (12.0 * (1.0 - self.nu**2))
        k_bend = D * np.pi**4 * (1.0/Lx**2 + 1.0/Ly**2)**2 * (Lx * Ly / 4.0)
        
        k_eff = k_tension + k_bend
        
        # Modal acoustic area A_eff = 4 * Lx * Ly / pi^2
        A_eff = 4.0 * Lx * Ly / np.pi**2
        
        return m_eff, k_eff, A_eff

@dataclass
class ElectromagneticTransducer:
    G0: float = 1.2            # Electromagnetic coupling coefficient [Wb/m or N/A]
    Rc: float = 12.0           # Coil resistance [Ohm]
    Lc: float = 15e-3          # Coil inductance [H] (15 mH)
    RL: float = 12.0           # External electrical load resistance [Ohm]

@dataclass
class PBARNConfig:
    air: AirProperties = field(default_factory=AirProperties)
    geom: NetworkGeometry = field(default_factory=NetworkGeometry)
    resonator: ResonatorParameters = field(default_factory=ResonatorParameters)
    mc: ManifoldChamberParameters = field(default_factory=ManifoldChamberParameters)
    membrane: GrapheneKaptonMembrane = field(default_factory=GrapheneKaptonMembrane)
    transducer: ElectromagneticTransducer = field(default_factory=ElectromagneticTransducer)
