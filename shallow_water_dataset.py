"""
Shallow Water Dataset - Generate perturbation samples and vorticity evolution
Uses PyTorch IterableDataset for on-demand sample generation.

This dataset simulates the viscous shallow water equations on a sphere using Dedalus.
It generates random perturbations to a balanced jet and evolves the system to produce
vorticity fields over time.

Equations solved:
- Momentum: ∂u/∂t + ν∇⁴u + g∇h + 2Ω × u = -u·∇u
- Continuity: ∂h/∂t + ν∇⁴h + H∇·u = -∇·(hu)

where u is velocity, h is height perturbation, ν is viscosity, g is gravity,
Ω is rotation vector, and H is mean height.
"""

import numpy as np
from torch.utils.data import IterableDataset
import dedalus.public as d3
import logging

logger = logging.getLogger(__name__)


def build_s2_coord_vertices(phi, theta):
    """Build vertices for spherical coordinate plotting."""
    phi = phi.ravel()
    phi_vert = np.concatenate([phi, [2 * np.pi]])
    phi_vert -= phi_vert[1] / 2
    theta = theta.ravel()
    theta_mid = (theta[:-1] + theta[1:]) / 2
    theta_vert = np.concatenate([[np.pi], theta_mid, [0]])
    return np.meshgrid(phi_vert, theta_vert, indexing="ij")


class ShallowWaterDataset(IterableDataset):
    def __init__(
        self,
        Nphi=256,
        Ntheta=128,
        dealias=3 / 2,
        stop_sim_time=360,  # hours
        timestep=600,  # seconds
        timestepper=d3.RK222,
        dtype=np.float64,
        alpha_range=(0.05, 0.6),
        beta_range=(0.02, 0.3),
        save_interval=5,  # Save every 5 hours instead of every hour
    ):
        """
        Dataset for shallow water equation simulations on a sphere.

        Args:
            Nphi: Number of longitudinal grid points
            Ntheta: Number of latitudinal grid points
            dealias: Dealiasing factor
            stop_sim_time: Simulation time in hours
            timestep: Time step size in seconds
            timestepper: Dedalus timestepper
            dtype: Data type
            alpha_range: Range for alpha perturbation parameter
            beta_range: Range for beta perturbation parameter
            save_interval: Save output every N hours
        """
        super().__init__()
        self.Nphi = Nphi
        self.Ntheta = Ntheta
        self.dealias = dealias
        self.stop_sim_time = stop_sim_time
        self.timestep = timestep
        self.timestepper = timestepper
        self.dtype = dtype
        self.alpha_range = alpha_range
        self.beta_range = beta_range
        self.save_interval = save_interval

        # Simulation units
        self.meter = 1 / 6.37122e6
        self.hour = 1
        self.second = self.hour / 3600

        # Physical parameters
        self.R = 6.37122e6 * self.meter
        self.Omega = 7.292e-5 / self.second
        self.nu = 1e5 * self.meter**2 / self.second / 32**2
        self.g = 9.80616 * self.meter / self.second**2
        self.H = 1e4 * self.meter
        self.timestep_scaled = timestep * self.second
        self.stop_sim_time_scaled = stop_sim_time * self.hour

        # Setup Dedalus components
        self.coords = d3.S2Coordinates("phi", "theta")
        self.dist = d3.Distributor(self.coords, dtype=dtype)
        self.basis = d3.SphereBasis(
            self.coords, (Nphi, Ntheta), radius=self.R, dealias=dealias, dtype=dtype
        )

        # Get grids for initial conditions
        self.phi, self.theta = self.dist.local_grids(self.basis)
        self.lat = np.pi / 2 - self.theta + 0 * self.phi

        # Setup jet parameters
        self.umax = 80 * self.meter / self.second
        self.lat0 = np.pi / 7
        self.lat1 = np.pi / 2 - self.lat0
        self.en = np.exp(-4 / (self.lat1 - self.lat0) ** 2)
        self.jet = (self.lat0 <= self.lat) * (self.lat <= self.lat1)
        self.u_jet = (
            self.umax
            / self.en
            * np.exp(
                1 / (self.lat[self.jet] - self.lat0) / (self.lat[self.jet] - self.lat1)
            )
        )

        # Perturbation parameters
        self.lat2 = np.pi / 4
        self.hpert = 120 * self.meter

        # Substitutions for equations
        self.zcross = lambda A: d3.MulCosine(d3.skew(A))

    def __iter__(self):
        """Generate infinite samples from the dataset."""
        while True:
            # Sample-varying parameters
            alpha = np.random.uniform(*self.alpha_range)
            beta = np.random.uniform(*self.beta_range)

            # Create fields
            u = self.dist.VectorField(self.coords, name="u", bases=self.basis)
            h = self.dist.Field(name="h", bases=self.basis)

            # Set up zonal jet
            u["g"][0][self.jet] = self.u_jet

            # Solve for balanced height field
            c = self.dist.Field(name="c")
            g = self.g
            Omega = self.Omega
            zcross = self.zcross
            problem = d3.LBVP([h, c], namespace=locals())
            problem.add_equation("g*lap(h) + c = - div(u@grad(u) + 2*Omega*zcross(u))")
            problem.add_equation("ave(h) = 0")
            solver = problem.build_solver()
            solver.solve()

            # Add perturbation
            h["g"] += (
                self.hpert
                * np.cos(self.lat)
                * np.exp(-((self.phi / alpha) ** 2))
                * np.exp(-(((self.lat2 - self.lat) / beta) ** 2))
            )

            # Store initial perturbation for output
            h_initial = np.copy(h["g"])

            # Solve the shallow water equations
            nu = self.nu
            g = self.g
            Omega = self.Omega
            H = self.H
            zcross = self.zcross
            problem = d3.IVP([u, h], namespace=locals())
            problem.add_equation(
                "dt(u) + nu*lap(lap(u)) + g*grad(h) + 2*Omega*zcross(u) = - u@grad(u)"
            )
            problem.add_equation("dt(h) + nu*lap(lap(h)) + H*div(u) = - div(h*u)")

            # Build solver
            solver = problem.build_solver(self.timestepper)
            solver.stop_sim_time = self.stop_sim_time_scaled

            # Storage for solution
            vorticity_list = []
            u_list = []
            h_list = []
            time_list = []

            # Initial state - compute vorticity and store fields
            vorticity_data = -d3.div(d3.skew(u)).evaluate()["g"]
            u.change_scales(1)
            h.change_scales(1)

            vorticity_list.append(np.copy(vorticity_data))
            u_list.append(np.copy(u["g"]))
            h_list.append(np.copy(h["g"]))
            time_list.append(solver.sim_time / self.hour)  # Convert to hours

            # Main loop
            save_counter = 0
            while solver.proceed:
                solver.step(self.timestep_scaled)
                save_counter += 1

                # Save every save_interval hours (convert timestep from seconds to hours)
                if (
                    save_counter
                    % (self.save_interval * self.hour / self.timestep_scaled)
                    == 0
                ):
                    vorticity_data = -d3.div(d3.skew(u)).evaluate()["g"]
                    u.change_scales(1)
                    h.change_scales(1)

                    vorticity_list.append(np.copy(vorticity_data))
                    u_list.append(np.copy(u["g"]))
                    h_list.append(np.copy(h["g"]))
                    time_list.append(solver.sim_time / self.hour)

            # Convert lists to arrays
            u_trajectory = np.array(u_list)  # Shape: (time, 2, Nphi, Ntheta)
            h_trajectory = np.array(h_list)  # Shape: (time, Nphi, Ntheta)
            vorticity_trajectory = np.array(
                vorticity_list
            )  # Shape: (time, Nphi, Ntheta)
            time_coords = np.array(time_list)

            # Create spatial coordinates
            Nphi, Ntheta = self.Nphi, self.Ntheta
            phi_1d = np.linspace(0, 2 * np.pi, Nphi, endpoint=False)
            theta_1d = np.linspace(0, np.pi, Ntheta)

            phi_vert, theta_vert = build_s2_coord_vertices(phi_1d, theta_1d)
            spatial_coords = np.column_stack([phi_vert.ravel(), theta_vert.ravel()])

            # Format and yield comprehensive output
            yield {
                # Spatial coordinates
                "spatial_coordinates": spatial_coords,  # (Nphi*Ntheta, 2) - phi, theta pairs
                "phi_coords": self.phi,  # Raw phi grid
                "theta_coords": self.theta,  # Raw theta grid
                "lat_coords": self.lat,  # Latitude grid
                # Time coordinates
                "time_coordinates": time_coords,  # Time points where solution was saved
                # Initial conditions
                "u_initial": u_list[0],  # Initial velocity field (2, Nphi, Ntheta)
                "h_initial": h_initial,  # Initial height perturbation (Nphi, Ntheta)
                "vorticity_initial": vorticity_list[0],  # Initial vorticity (Nphi, Ntheta)
                # Solution trajectories
                "u_trajectory": u_trajectory,  # Velocity evolution (time, 2, Nphi, Ntheta)
                "h_trajectory": h_trajectory,  # Height evolution (time, Nphi, Ntheta)
                "vorticity_trajectory": vorticity_trajectory,  # Vorticity evolution (time, Nphi, Ntheta)
                # Physical parameters
                "alpha": alpha,  # Perturbation width parameter
                "beta": beta,  # Perturbation latitude parameter
                "R": self.R,  # Earth radius
                "Omega": self.Omega,  # Earth rotation rate
                "nu": self.nu,  # Viscosity
                "g": self.g,  # Gravity
                "H": self.H,  # Mean height
                # Grid parameters
                "Nphi": Nphi,
                "Ntheta": Ntheta,
            }
