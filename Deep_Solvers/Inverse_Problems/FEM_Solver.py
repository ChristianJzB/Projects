import ufl
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, plot
from dolfinx.fem.petsc import LinearProblem
from dolfinx import geometry

import numpy as np
import scipy.optimize as opt
from scipy.optimize import fsolve, brentq
import matplotlib.pyplot as plt


class RootFinder2:
    def __init__(self, n, equation, intervals):
        """
        n: number of roots to find
        equation: the transcendental equation function
        intervals: list of tuples defining initial search intervals
        """
        self.n = n
        self.equation = equation
        self.intervals = intervals  # List of intervals where we expect roots
    
    def find_roots(self):
        """Find the first n roots using bisection method."""
        roots = []
        for interval in self.intervals:
            try:
                root = opt.bisect(self.equation, interval[0], interval[1])
                roots.append(root)
                if len(roots) == self.n:
                    break
            except ValueError as e:
                print(f"No root found in interval {interval}: {e}")
        return roots
    
    def plot_equation(self, points = 100):
        x = np.linspace(self.intervals[0][0]-1,self.intervals[-1][-1],points)
        plt.plot(x, self.equation(x))
        plt.axhline(y = 0, color = 'r', linestyle = '-') 
        plt.ylim(-10,10)

class RootFinder:
    def __init__(self, lam , M, equation=None):
        """
        lam: parameter lambda (for your equation)
        M: number of intervals to search for roots
        equation: optional, if you want to provide a custom equation
        """
        self.lam = lam
        self.M = M
        self.c = 1 / lam
        self.equation = equation if equation else self.default_equation

    def default_equation(self, x):
        """Default transcendental equation: tan(x) = (2c*x)/(x^2 - c^2)"""
        c = self.c
        return np.tan(x) - (2 * c * x) / (x**2 - c**2)

    def find_roots(self):
        """Find the roots using the Brent or fsolve method depending on the case."""
        roots = []
        for i in range(self.M):
            wmin = (i - 0.499) * np.pi
            wmax = (i + 0.499) * np.pi

            # Handle the singularity around c
            if wmin <= self.c <= wmax:  
                if wmin > 0:
                    root = fsolve(self.equation, (self.c + wmin) / 2)[0]
                    roots.append(root)
                root = fsolve(self.equation, (self.c + wmax) / 2)[0]
                roots.append(root)
            elif wmin > 0:  
                root = brentq(self.equation, wmin, wmax)
                roots.append(root)
        
        return np.array(roots)


class Parametric_K:
    def __init__(self, theta, lam , M):
        self.theta = np.array(theta)
        self.finder = RootFinder(lam, M)
        self.roots = np.array(self.finder.find_roots())

    @property
    def A(self):
        """Compute the A coefficients."""
        return np.sqrt(1 / ((1/8)*(5 + (self.roots / 2)**2) + 
                            (np.sin(2*self.roots) / (4*self.roots)) * ((self.roots / 4)**2 - 1) - (np.cos(2*self.roots)/8)))
    @property
    def an(self):
        """Compute the an values."""
        return np.sqrt(8 / (self.roots**2 + 16))
    
    def eval(self, x):
        """
        Evaluate the sum for a given x, summing over all terms defined by theta and roots.
        x: 2D array of points (n, dim), where n is the number of points and dim is the dimension.
        Returns an array of evaluations for each point.
        """
        # Initialize the result array (same size as the number of points)
        result = np.zeros(x.shape[1])
        
        # Compute the sum over all terms
        for i in range(len(self.theta)):
            result += self.theta[i] * self.an[i] * self.A[i] * (
                np.sin(self.roots[i] * x[0]) + (self.roots[i] / 4) * np.cos(self.roots[i] * x[0])
            )
        return np.exp(result)
     

class FEMSolver:
    def __init__(self, theta, lam = 1 /4, M = 2, vert=30, l_bc = 0, r_bc = 2 ):
        self.l_bc, self.r_bc = l_bc, r_bc
        self.theta = theta
        self.lam = lam
        self.M = M

        # Enable GPU-aware PETSc options
        # PETSc.Options().setValue('mat_type', 'aijcusparse')  # Use CUDA sparse matrix type
        # PETSc.Options().setValue('vec_type', 'cuda')  # Use CUDA vector type

        self.domain = mesh.create_unit_interval(MPI.COMM_WORLD, vert)
        self.V = fem.functionspace(self.domain, ("Lagrange", 1))
        self.bc_l, self.bc_r = self.set_boundary_conditions()
        self.k = self.interpolate_k()
        self.f = self.interpolate_f()
        self.uh = None
    
    
    def interpolate_k(self):
        """Interpolate the k function based on the provided equation."""
        k = fem.Function(self.V)
        k_an = Parametric_K(self.theta,self.lam,self.M)
        k.interpolate(k_an.eval)
        return k


    def set_boundary_conditions(self):
        """Set the Dirichlet boundary conditions for the problem."""
        dofl = fem.locate_dofs_geometrical(self.V, lambda x: np.isclose(x[0], 0))
        bc_l = fem.dirichletbc(PETSc.ScalarType(self.l_bc), dofl, self.V)

        dofr = fem.locate_dofs_geometrical(self.V, lambda x: np.isclose(x[0], 1))
        bc_r = fem.dirichletbc(PETSc.ScalarType(self.r_bc), dofr, self.V)
        
        return bc_l, bc_r

    def interpolate_f(self):
        """Interpolate the f function."""
        f = fem.Function(self.V)
        f.interpolate(lambda x: 4 * x[0])
        return f

    def solve(self):
        """Define and solve the linear variational problem."""
        # Interpolate k and f functions every time we solve
        self.k = self.interpolate_k()  # Ensure k is updated based on current theta
        self.f = self.interpolate_f()  # f can remain static, but can be updated if needed

        u = ufl.TrialFunction(self.V)
        v = ufl.TestFunction(self.V)

        # Define the bilinear form (a) and the linear form (L)
        a = self.k * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
        L = self.f * v * ufl.dx

        # Solve the linear problem
        problem = LinearProblem(a, L, bcs=[self.bc_l, self.bc_r])
        self.uh = problem.solve()
        return self.uh
    
    
    def solution_array(self):
        cells, types, x = plot.vtk_mesh(self.V)
        return (x, self.uh.x.array)
    
    def eval_at_points(self, points):
        """Evaluate the solution at arbitrary points using interpolation."""
        
        if self.uh is None:
            raise ValueError("Solve the problem first by calling solve().")
        
        # Ensure points are in the shape (N, 3) for DOLFINx
        if points.shape[1] == 1:
            # If points are 1D, pad with zeros for 2nd and 3rd dimensions
            points = np.hstack((points, np.zeros((points.shape[0], 2))))  # shape (N, 3)

        bb_tree = geometry.bb_tree(self.domain, self.domain.topology.dim)

        cells,points_on_proc = [],[]

        # Find cells whose bounding-box collide with the the points
        cell_candidates = geometry.compute_collisions_points(bb_tree, points)
        # Choose one of the cells that contains the point
        colliding_cells = geometry.compute_colliding_cells(self.domain, cell_candidates, points)

        for i, point in enumerate(points):
            if len(colliding_cells.links(i)) > 0:
                points_on_proc.append(point)
                cells.append(colliding_cells.links(i)[0])

        points_on_proc = np.array(points_on_proc, dtype=np.float64)
        u_values = self.uh.eval(points_on_proc, cells)
        
        return u_values