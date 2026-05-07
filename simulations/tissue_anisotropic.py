"""
Simulation: Anisotropic muscle tissue under uniaxial compression
Material:   Holzapfel-Ogden (Neo-Hookean + fiber reinforcement)
Fiber dir:  along x-axis [1, 0, 0]
"""
import sys
sys.path.insert(0, "/workspace")

from dolfinx import mesh, fem, io
from dolfinx.fem.petsc import NonlinearProblem
import ufl
from ufl import derivative, dx, grad, Identity, det, tr, inner
from mpi4py import MPI
import numpy as np

from src.models.neo_hookean    import lame_parameters
from src.models.holzapfel_ogden import strain_energy

# Mesh
domain = mesh.create_unit_cube(MPI.COMM_WORLD, 10, 10, 10, mesh.CellType.tetrahedron)
gdim   = domain.geometry.dim

# Function spaces
V = fem.functionspace(domain, ("Lagrange", 1, (gdim,)))
W = fem.functionspace(domain, ("DG", 0))

# Material parameters
mu, lmbda = lame_parameters(domain, E=100e3, nu=0.49)
k1 = fem.Constant(domain, 50.0e3)    # fiber stiffness   [Pa]
k2 = fem.Constant(domain, 10.0)     # fiber nonlinearity [-]
fiber_dir = [1.0, 0.0, 0.0]         # fibers along x-axis

# Fields
u = fem.Function(V, name="Displacement")
v = ufl.TestFunction(V)

# Variational problem
psi = strain_energy(u, mu, lmbda, k1, k2, fiber_dir)
Pi  = psi * dx
R   = derivative(Pi, u, v)
K   = derivative(R, u, ufl.TrialFunction(V))

# Boundary conditions
def bottom(x): return np.isclose(x[2], 0.0)
def top(x):    return np.isclose(x[2], 1.0)

bc_bot = fem.dirichletbc(np.zeros(gdim),           fem.locate_dofs_geometrical(V, bottom), V)
bc_top = fem.dirichletbc(np.array([0.1, 0., 0.]), fem.locate_dofs_geometrical(V, top),    V)

# Solve
problem = NonlinearProblem(R, u, petsc_options_prefix="nls_",
                           bcs=[bc_bot, bc_top], J=K,
                           petsc_options={
                               "ksp_type": "preonly",
                               "pc_type": "lu",
                               "pc_factor_mat_solver_type": "mumps",
                               "snes_linesearch_type": "bt",
                           })
problem.solver.solve(None, u.x.petsc_vec)
u.x.scatter_forward()

# Von Mises stress (full: isotropic + fiber contribution)
from ufl import outer, inv, exp as uexp
I     = Identity(gdim)
F     = I + grad(u)
C     = F.T * F
J     = det(F)
a     = ufl.as_vector(fiber_dir)
I4    = inner(a, C * a)

# Second Piola-Kirchhoff (full)
S = (mu * I
     - mu * inv(C)
     + lmbda * ufl.ln(J) * inv(C)
     + 2 * k1 * (I4 - 1) * uexp(k2 * (I4 - 1)**2) * outer(a, a))

# Cauchy stress
sigma = (1/J) * F * S * F.T
s     = sigma - (1/3) * tr(sigma) * I
vm    = ufl.sqrt((3/2) * inner(s, s))

vm_expr = fem.Expression(vm, W.element.interpolation_points)
vm_h    = fem.Function(W, name="VonMises")
vm_h.interpolate(vm_expr)

# Output
with io.XDMFFile(domain.comm, "results/tissue_anisotropic.xdmf", "w") as f:
    f.write_mesh(domain)
    f.write_function(u)
    f.write_function(vm_h)

print(f"Max displacement z:   {u.x.array.min():.4f} m")
print(f"Max von Mises stress: {vm_h.x.array.max():.1f} Pa")
print("Results written to results/tissue_anisotropic.xdmf")
