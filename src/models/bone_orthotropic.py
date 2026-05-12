import ufl
from ufl import grad, sym, tr, inner, as_tensor, dx
from dolfinx import fem
import numpy as np


def orthotropic_elasticity_tensor(E1, E2, E3, nu12, nu23, nu13, G12, G23, G13):
    """
    Orthotropic stiffness tensor C in Voigt notation.
    Axis 1 = bone axis (stiffest), 2 and 3 = transverse.
    """
    # Compliance matrix S
    S = np.zeros((6, 6))
    S[0, 0] =  1.0 / E1
    S[1, 1] =  1.0 / E2
    S[2, 2] =  1.0 / E3
    S[0, 1] = S[1, 0] = -nu12 / E1
    S[1, 2] = S[2, 1] = -nu23 / E2
    S[0, 2] = S[2, 0] = -nu13 / E1
    S[3, 3] =  1.0 / G23
    S[4, 4] =  1.0 / G13
    S[5, 5] =  1.0 / G12
    return np.linalg.inv(S)


def epsilon(u):
    """Linearized strain tensor."""
    return sym(grad(u))


def sigma_orthotropic(u, C):
    """
    Cauchy stress from orthotropic stiffness tensor C (6x6 numpy array).
    Maps strain vector (Voigt) to stress vector, returns as UFL tensor.
    """
    eps = epsilon(u)

    # Voigt strain vector [e11, e22, e33, 2e23, 2e13, 2e12]
    e = [eps[0,0], eps[1,1], eps[2,2],
         2*eps[1,2], 2*eps[0,2], 2*eps[0,1]]

    # Stress vector via C * e
    s = [sum(float(C[i, j]) * e[j] for j in range(6)) for i in range(6)]

    # Back to tensor
    return as_tensor([[s[0], s[5], s[4]],
                      [s[5], s[1], s[3]],
                      [s[4], s[3], s[2]]])
