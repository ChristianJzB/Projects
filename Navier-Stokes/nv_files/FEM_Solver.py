import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from ufl import (TestFunction, TrialFunction, grad, inner, dx, as_vector, dot,lhs, rhs)
from nv_files.Field_Generator import compute_seq_pairs,generate_omega0
from dolfinx.fem.petsc import LinearProblem


class VorticitySolver:
    def __init__(self, nx=32, ny=32, N_KL=10000, dt=0.01, T=1.0, nu=1e-3, force_func=None, d= 5, tau =7, seed = 108):
        self.nx = nx
        self.ny = ny
        self.N_KL = N_KL
        self.dt = dt
        self.T = T
        self.nu = nu
        self.t = 0.0
        self.seed = seed
        self.d = d
        self.tau = tau

        # Initialize the mesh with periodic boundary conditions
        self.mesh = self.create_periodic_mesh()
        
        # Define the function space
        self.V = fem.functionspace(self.mesh, ("Lagrange", 1))

        # Initialize functions
        self.w0 = fem.Function(self.V)
        self.f = fem.Function(self.V)
        self.w = fem.Function(self.V)
        self.psi = fem.Function(self.V)

        # Define the initial vorticity
        self.initialize_vorticity()

        # Define the forcing term
        if force_func is None:
            self.f.interpolate(lambda x: 0 * x[0])  # Default to zero forcing
        else:
            self.define_forcing_term(force_func)

        # Solver setup
        self.psi_solver = None
        self.setup_stream_function_solver()

    def create_periodic_mesh(self):
        # Create a unit square mesh and set periodicity
        mesh_ = mesh.create_unit_square(MPI.COMM_WORLD, self.nx, self.ny, cell_type=mesh.CellType.triangle)

        #pbc_mesh = mesh.create_periodic_mesh(mesh_, [0, 1])  # 0 maps to 1 and vice versa (periodicity along both axes)
        return mesh_

    def initialize_vorticity(self):
        # Compute the sequence pairs for w0
        seq_pairs = compute_seq_pairs(self.N_KL)

        # Define w0 using the generate_omega0 function
        self.w0.interpolate(lambda x: generate_omega0(x[0], x[1], seq_pairs,d = self.d, tau = self.tau, seed=self.seed))

        # Set initial condition for w
        self.w.x.array[:] = self.w0.x.array[:]

    def define_forcing_term(self, force_func):
        # Apply the user-defined forcing function
        self.f.interpolate(force_func)

    def setup_stream_function_solver(self):
        # Define variational forms for the Poisson equation
        psi_trial = TrialFunction(self.V)
        psi_test = TestFunction(self.V)
        stream_lhs = inner(grad(psi_trial), grad(psi_test)) * dx
        stream_rhs = -self.w * psi_test * dx
        self.psi_solver = fem.petsc.LinearProblem(stream_lhs, stream_rhs, u=self.psi)

    def time_step(self):
        # Solve for the stream function psi
        self.psi_solver.solve()

        # Define the velocity field
        u = as_vector((-self.psi.dx(1), self.psi.dx(0)))

        # Define the variational problem for w
        w_new = TrialFunction(self.V)
        phi = TestFunction(self.V)

        F = (w_new - self.w) / self.dt * phi * dx \
            + self.nu * dot(grad((self.w + w_new) / 2), grad(phi)) * dx \
            + dot(u, grad((self.w + w_new) / 2)) * phi * dx \
            - self.f * phi * dx

        a = fem.form(lhs(F))
        L = fem.form(rhs(F))
        
        # Solve for the new w
        w_new_func = fem.Function(self.V)
        w_solver = fem.petsc.LinearProblem(a, L, u=w_new_func)
        w_solver.solve()

        # Update w for the next time step
        self.w.x.array[:] = w_new_func.x.array[:]
        self.t += self.dt

    def run(self):
        while self.t <= self.T:
            self.time_step()
    

    def evaluate_at_points(self, points):
        """
        Evaluate the current and initial vorticity fields at specified points.

        Parameters:
        points (np.ndarray): Array of shape (N, 3) containing the points where values are evaluated.

        Returns:
        tuple: A tuple containing:
            - u_values: The evaluated values of `w` (current vorticity) at the points.
            - u0_values: The evaluated values of `w0` (initial vorticity) at the points.

        Raises:
        ValueError: If the `points` array is not of shape `(N, 3)` or if no points are found on this process.
        """
        # Check if points array has correct shape
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"Points array must have shape (N, 3). Provided shape: {points.shape}")

        # Build the bounding box tree for the mesh
        bb_tree = geometry.bb_tree(self.mesh, self.mesh.topology.dim)

        # Find cells whose bounding boxes collide with the points
        cell_candidates = geometry.compute_collisions_points(bb_tree, points)

        # Compute colliding cells
        colliding_cells = geometry.compute_colliding_cells(self.mesh, cell_candidates, points)

        # Store points and their corresponding cells
        cells = []
        points_on_proc = []

        for i, point in enumerate(points):
            if len(colliding_cells.links(i)) > 0:
                points_on_proc.append(point)
                cells.append(colliding_cells.links(i)[0])  # Select the first colliding cell

        points_on_proc = np.array(points_on_proc, dtype=np.float64)

        if len(points_on_proc) > 0:
            # Evaluate `w` (current vorticity) and `w0` (initial vorticity) at the points
            u_values = self.w.eval(points_on_proc, cells)
            u_0 = self.w0.eval(points_on_proc, cells)
            return u_0,u_values
        else:
            raise ValueError("No points found on this process for evaluation.")