"""
Simulation: Muscle tissue under uniaxial compression
Material:   Neo-Hookean (hyperelastic)
Output:     Displacement field + von Mises stress
"""
import sys
sys.path.insert(0, "/workspace")

from dolfinx import mesh, fem, io
from dolfinx.fem.petsc import NonlinearProblem
import ufl
from ufl import derivative, dx, grad, Identity, det, tr, inner, inv
from mpi4py import MPI
import numpy as np

from src.models.neo_hookean import strain_energy, lame_parameters

# Mesh
domain = mesh.create_unit_cube(MPI.COMM_WORLD, 10, 10, 10, mesh.CellType.tetrahedron)
gdim   = domain.geometry.dim

# Function spaces
V = fem.functionspace(domain, ("Lagrange", 1, (gdim,)))
W = fem.functionspace(domain, ("DG", 0))          # scalar, piecewise constant

# Material: muscle tissue
mu, lmbda = lame_parameters(domain, E=100e3, nu=0.49)

# Fields
u = fem.Function(V, name="Displacement")
v = ufl.TestFunction(V)

# Variational problem
psi = strain_energy(u, mu, lmbda)
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

# Cauchy stress + von Mises
I  = Identity(gdim)
F  = I + grad(u)
J  = det(F)

# Cauchy stress tensor
sigma = (1/J) * (mu * (F * F.T - I) + lmbda * ufl.ln(J) * I)

# Deviatoric part
s        = sigma - (1/3) * tr(sigma) * I
von_mises = ufl.sqrt((3/2) * inner(s, s))

# Project onto DG0 space
vm_expr = fem.Expression(von_mises, W.element.interpolation_points)
vm_h    = fem.Function(W, name="VonMises")
vm_h.interpolate(vm_expr)

# Output
with io.XDMFFile(domain.comm, "results/tissue_compression.xdmf", "w") as f:
    f.write_mesh(domain)
    f.write_function(u)
    f.write_function(vm_h)

print(f"Max displacement z:      {u.x.array.min():.4f} m")
print(f"Max von Mises stress:    {vm_h.x.array.max():.1f} Pa")
print("Results written to results/tissue_compression.xdmf")
