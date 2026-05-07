import ufl
from ufl import grad, Identity, tr, det, ln, dx
from dolfinx import fem


def strain_energy(u, mu, lmbda):
    """Neo-Hookean Verzerrungsenergie fuer weiches Gewebe."""
    gdim = len(u)
    I = Identity(gdim)
    F = I + grad(u)
    C = F.T * F
    J = det(F)
    return (mu / 2) * (tr(C) - 3) - mu * ln(J) + (lmbda / 2) * (ln(J))**2


def lame_parameters(domain, E, nu):
    """Lame-Parameter aus E-Modul und Querkontraktionszahl."""
    mu    = fem.Constant(domain, E / (2 * (1 + nu)))
    lmbda = fem.Constant(domain, E * nu / ((1 + nu) * (1 - 2 * nu)))
    return mu, lmbda
