from fenics import *
import numpy as np

def setup_PDE(m, pol_degree, a):
    '''
    Set up the mesh, function space and boundary conditions for FEM solution of 
    the PDE $- \nabla \cdot (k \nabla p) = a$, where a is a real constant.

    :param m: the mesh size to use in x and y direction to create finite 
    element grid.
    :param pol_degree: the degree of the polynomial to be used in FEM approximation.
    :param a: the RHS constant of the PDE.

    :return: 
        V - the finite element function space.
        f - the RHS.
        bc - the Dirichlet boundary conditions. 
    '''

    # Create mesh and define function space
    mesh = UnitSquareMesh(m, m)
    V = FunctionSpace(mesh, 'P', pol_degree)

    # Define boundary condition (only Dirichlet as Neumann are defined as part 
    # of the variational form)
    high_pressure = 'near(x[0], 0)'
    low_pressure = 'near(x[0], 1)'
    bc_hp = DirichletBC(V, Constant(1), high_pressure)
    bc_lp = DirichletBC(V, Constant(0), low_pressure)
    bc = [bc_hp, bc_lp]
    
    # Define RHS
    f = Constant(a)
    
    return V, f, bc

class k_RF(UserExpression):
    '''
    New type of User Expression for selecting the value of the random field at a specific grid point (for building discretisation system).

    :param Z: array containing the values of the Gaussian random field at each discretisation point.
    :param m: mesh size used in discretisation.

    :return:
        values[0] - the value of the random field at specific point x[0],x[1]
    '''
    def __init__(self, Z, m, **kwargs):
        super().__init__(**kwargs)
        self.Z = Z
        self.m = m
    def eval(self,values,x):
        # select the correct element in the vector
        z = self.Z[int((self.m+1)*x[0]*self.m + x[1]*self.m)]
        # k = e^z
        values[0] = np.exp(z.item())
    def value_shape(self):
        return()