"""
Force-displacement comparison: Neo-Hookean vs Holzapfel-Ogden
Load stepping: 15 increments, 0 to 30% compression.
"""
import sys
sys.path.insert(0, "/workspace")

from dolfinx import mesh, fem
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.fem import form, assemble_scalar
from dolfinx.mesh import locate_entities_boundary, meshtags
import ufl
from ufl import derivative, dx, ds, grad, Identity, det, ln, inv, inner, as_vector, FacetNormal, tr, exp
from mpi4py import MPI
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.models.neo_hookean import lame_parameters

def run_simulation(domain, V, mu, lmbda, fiber=False, k1=None, k2=None, fiber_dir=None):
    gdim = domain.geometry.dim
    domain.topology.create_connectivity(gdim - 1, gdim)

    def top(x):    return np.isclose(x[2], 1.0)
    def bottom(x): return np.isclose(x[2], 0.0)

    top_facets = locate_entities_boundary(domain, gdim - 1, top)
    facet_tags = meshtags(domain, gdim - 1, top_facets,
                          np.ones(len(top_facets), dtype=np.int32))
    ds_top = ds(domain=domain, subdomain_data=facet_tags, subdomain_id=1)

    u = fem.Function(fem.functionspace(domain, ("Lagrange", 1, (gdim,))), name="Displacement")
    v = ufl.TestFunction(V)

    I_  = Identity(gdim)
    F_  = I_ + grad(u)
    C_  = F_.T * F_
    J_  = det(F_)

    psi = (mu / 2) * (tr(C_) - 3) - mu * ln(J_) + (lmbda / 2) * (ln(J_))**2

    if fiber:
        a  = ufl.as_vector(fiber_dir)
        I4 = inner(a, C_ * a)
        psi = psi + k1 / (2 * k2) * (exp(k2 * (I4 - 1)**2) - 1)

    Pi = psi * dx
    R  = derivative(Pi, u, v)
    K  = derivative(R, u, ufl.TrialFunction(V))

    u_top_val = fem.Constant(domain, np.array([0.0, 0.0, 0.0]))
    bc_bot    = fem.dirichletbc(np.zeros(gdim), fem.locate_dofs_geometrical(V, bottom), V)
    bc_top    = fem.dirichletbc(u_top_val, fem.locate_dofs_geometrical(V, top), V)

    problem = NonlinearProblem(R, u, petsc_options_prefix="nls_",
                               bcs=[bc_bot, bc_top], J=K,
                               petsc_options={
                                   "ksp_type": "preonly",
                                   "pc_type": "lu",
                                   "pc_factor_mat_solver_type": "mumps",
                                   "snes_linesearch_type": "bt",
                               })

    N        = FacetNormal(domain)
    P_stress = mu * F_ - mu * inv(F_.T) + lmbda * ln(J_) * inv(F_.T)
    f_react  = form(-inner(P_stress * N, as_vector([0.0, 0.0, 1.0])) * ds_top)

    n_steps       = 15
    displacements = np.linspace(0, 0.30, n_steps + 1)[1:]
    forces        = []

    for disp in displacements:
        u_top_val.value[2] = -disp
        u.x.array[:] = 0.0
        problem.solver.solve(None, u.x.petsc_vec)
        u.x.scatter_forward()
        F_react = domain.comm.allreduce(assemble_scalar(f_react), op=MPI.SUM)
        forces.append(F_react)

    return displacements, np.array(forces)

# Setup
domain = mesh.create_unit_cube(MPI.COMM_WORLD, 10, 10, 10, mesh.CellType.tetrahedron)
gdim   = domain.geometry.dim
V      = fem.functionspace(domain, ("Lagrange", 1, (gdim,)))
mu, lmbda = lame_parameters(domain, E=100e3, nu=0.49)
k1 = fem.Constant(domain, 50e3)
k2 = fem.Constant(domain, 10.0)

print("Running Neo-Hookean...")
disp_nh, force_nh = run_simulation(domain, V, mu, lmbda)

print("Running Holzapfel-Ogden...")
disp_ho, force_ho = run_simulation(domain, V, mu, lmbda,
                                   fiber=True, k1=k1, k2=k2,
                                   fiber_dir=[0.0, 0.0, 1.0])

# Plot
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(disp_nh * 100, force_nh, "o-", color="#2563eb", linewidth=2, markersize=5, label="Neo-Hookean (isotropic)")
ax.plot(disp_ho * 100, force_ho, "s-", color="#dc2626", linewidth=2, markersize=5, label="Holzapfel-Ogden (fiber along z, k₁=50 kPa)")
ax.set_xlabel("Compression [%]")
ax.set_ylabel("Reaction force [N]")
ax.set_title("Force-displacement — muscle tissue models")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig("results/force_displacement_comparison.png", dpi=150)
print("Plot saved to results/force_displacement_comparison.png")
