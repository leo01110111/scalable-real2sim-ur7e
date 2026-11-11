"""Rigid-body payload regressor for a wrist force/torque sensor.

Parameter vector (about the flange origin, expressed in the flange frame F):
    phi = [m, hx, hy, hz, Ixx, Ixy, Ixz, Iyy, Iyz, Izz],  h = m * c

Newton-Euler for the body attached after the sensor (all quantities in F):
    f = m (a - g) + (alpha x h) + omega x (omega x h)
    n = I alpha + omega x (I omega) + h x (a - g)
which is linear in phi: [f; n] = Y(g, a, omega, alpha) @ phi.
"""

import numpy as np

PARAM_NAMES = ["m", "hx", "hy", "hz", "Ixx", "Ixy", "Ixz", "Iyy", "Iyz", "Izz"]


def skew(v: np.ndarray) -> np.ndarray:
    """(T,3) -> (T,3,3) cross-product matrices."""
    v = np.atleast_2d(v)
    S = np.zeros((v.shape[0], 3, 3))
    S[:, 0, 1], S[:, 0, 2] = -v[:, 2], v[:, 1]
    S[:, 1, 0], S[:, 1, 2] = v[:, 2], -v[:, 0]
    S[:, 2, 0], S[:, 2, 1] = -v[:, 1], v[:, 0]
    return S


def inertia_dot(v: np.ndarray) -> np.ndarray:
    """(T,3) -> (T,3,6) matrix L(v) with I @ v = L(v) @ [Ixx,Ixy,Ixz,Iyy,Iyz,Izz]."""
    v = np.atleast_2d(v)
    L = np.zeros((v.shape[0], 3, 6))
    x, y, z = v[:, 0], v[:, 1], v[:, 2]
    L[:, 0, 0], L[:, 0, 1], L[:, 0, 2] = x, y, z
    L[:, 1, 1], L[:, 1, 3], L[:, 1, 4] = x, y, z
    L[:, 2, 2], L[:, 2, 4], L[:, 2, 5] = x, y, z
    return L


def payload_regressor(
    g: np.ndarray, a: np.ndarray, omega: np.ndarray, alpha: np.ndarray
) -> np.ndarray:
    """Returns Y with shape (T, 6, 10). Inputs are (T,3), expressed in F."""
    T = np.atleast_2d(a).shape[0]
    a_g = np.atleast_2d(a) - np.atleast_2d(g)
    W, Al = skew(omega), skew(alpha)
    Y = np.zeros((T, 6, 10))
    Y[:, :3, 0] = a_g
    Y[:, :3, 1:4] = Al + W @ W
    Y[:, 3:, 1:4] = -skew(a_g)
    Y[:, 3:, 4:10] = inertia_dot(alpha) + W @ inertia_dot(omega)
    return Y


def inertia_matrix(phi: np.ndarray) -> np.ndarray:
    Ixx, Ixy, Ixz, Iyy, Iyz, Izz = phi[4:10]
    return np.array([[Ixx, Ixy, Ixz], [Ixy, Iyy, Iyz], [Ixz, Iyz, Izz]])


def pseudo_inertia(phi: np.ndarray) -> np.ndarray:
    """4x4 pseudo-inertia matrix; PSD <=> physically consistent (Wensing 2017)."""
    I = inertia_matrix(phi)
    h = phi[1:4]
    J = np.zeros((4, 4))
    J[:3, :3] = 0.5 * np.trace(I) * np.eye(3) - I
    J[:3, 3] = h
    J[3, :3] = h
    J[3, 3] = phi[0]
    return J


def phi_from_body(mass: float, com: np.ndarray, I_com: np.ndarray) -> np.ndarray:
    """Builds phi (about the frame origin) from mass, CoM and inertia about the CoM."""
    com = np.asarray(com, dtype=float)
    I_o = I_com + mass * (np.dot(com, com) * np.eye(3) - np.outer(com, com))
    h = mass * com
    return np.array(
        [mass, *h, I_o[0, 0], I_o[0, 1], I_o[0, 2], I_o[1, 1], I_o[1, 2], I_o[2, 2]]
    )


def body_from_phi(phi: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Inverse of `phi_from_body`: returns (mass, com, inertia about com)."""
    m = float(phi[0])
    com = phi[1:4] / m
    I_o = inertia_matrix(phi)
    I_com = I_o - m * (np.dot(com, com) * np.eye(3) - np.outer(com, com))
    return m, com, I_com
