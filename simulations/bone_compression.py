"""
Simulation: Cortical bone under uniaxial compression
Material:   Linear elastic, orthotropic
Bone axis:  z-direction (stiffest)
"""
import sys
sys.path.insert(0, "/workspace")

from dolfinx import mesh, fem, io
from dolfinx.fem.petsc import LinearProblem
from dolfinx.fem import form
import ufl
from ufl import inner, dx, TestFunction, TrialFunction
from mpi4py import MPI
import numpy as np

from src.models.bone_orthotropic import orthotropic_elasticity_tensor, epsilon, sigma_orthotropic

# Mesh
domain = mesh.create_unit_cube(MPI.COMM_WORLD, 10, 10, 10, mesh.CellType.tetrahedron)
gdim   = domain.geometry.dim

V = fem.functionspace(domain, ("Lagrange", 1, (gdim,)))

# Cortical bone parameters [Pa]
C = orthotropic_elasticity_tensor(
    E1=20e9, E2=12e9, E3=12e9,
    nu12=0.25, nu23=0.25, nu13=0.25,
    G12=4e9,  G23=4e9,  G13=4e9
)

# Variational problem (linear)
u = TrialFunction(V)
v = TestFunction(V)

sigma = lambda u: sigma_orthotropic(u, C)
a     = inner(sigma(u), epsilon(v)) * dx
L     = inner(fem.Constant(domain, np.zeros(gdim)), v) * dx

# Boundary conditions
def bottom(x): return np.isclose(x[2], 0.0)
def top(x):    return np.isclose(x[2], 1.0)

bc_bot = fem.dirichletbc(np.zeros(gdim), fem.locate_dofs_geometrical(V, bottom), V)
bc_top = fem.dirichletbc(np.array([0.0, 0.0, -0.001]), fem.locate_dofs_geometrical(V, top), V)

# Solve
problem = LinearProblem(a, L, bcs=[bc_bot, bc_top], petsc_options_prefix="lin_",
                        petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
u_h = problem.solve()

# Von Mises
sigma_h = sigma_orthotropic(u_h, C)
s       = sigma_h - (1/3) * ufl.tr(sigma_h) * ufl.Identity(gdim)
vm      = ufl.sqrt((3/2) * inner(s, s))

W    = fem.functionspace(domain, ("DG", 0))
vm_h = fem.Function(W, name="VonMises")
vm_h.interpolate(fem.Expression(vm, W.element.interpolation_points))

# Output
with io.XDMFFile(domain.comm, "results/bone_compression.xdmf", "w") as f:
    f.write_mesh(domain)
    u_h.name = "Displacement"
    f.write_function(u_h)
    f.write_function(vm_h)

print(f"Max displacement z:   {u_h.x.array.min():.6f} m")
print(f"Max von Mises stress: {vm_h.x.array.max()/1e6:.1f} MPa")
print("Results written to results/bone_compression.xdmf")
