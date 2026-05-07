import ufl
from ufl import grad, Identity, det, tr, ln, inner, exp, dx, as_vector
from dolfinx import fem


def strain_energy(u, mu, lmbda, k1, k2, fiber_direction):
    """
    Holzapfel-Ogden strain energy for fibered soft tissue.
    Isotropic Neo-Hookean base + anisotropic fiber term.

    Parameters
    ----------
    u               : displacement field
    mu, lmbda       : isotropic Lame parameters
    k1              : fiber stiffness [Pa]
    k2              : fiber nonlinearity [-] (dimensionless)
    fiber_direction : unit vector along fiber axis (list of 3 floats)
    """
    gdim = len(u)
    I    = Identity(gdim)
    F    = I + grad(u)
    C    = F.T * F
    J    = det(F)

    # Isotropic base (Neo-Hookean)
    psi_iso = (mu / 2) * (tr(C) - 3) - mu * ln(J) + (lmbda / 2) * (ln(J))**2

    # Anisotropic fiber term
    a  = as_vector(fiber_direction)
    I4 = inner(a, C * a)                              # stretch along fiber
    psi_fiber = k1 / (2 * k2) * (exp(k2 * (I4 - 1)**2) - 1)

    return psi_iso + psi_fiber
