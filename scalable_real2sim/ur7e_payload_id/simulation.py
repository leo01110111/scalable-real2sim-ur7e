"""Synthetic F/T data for testing the pipeline and predicting accuracy."""

from typing import Optional

import numpy as np

from .regressor import payload_regressor, phi_from_body
from .signals import GRAVITY_BASE, RunData, flange_kinematics


def box_inertia(mass: float, size) -> np.ndarray:
    x, y, z = size
    return mass / 12.0 * np.diag([y * y + z * z, x * x + z * z, x * x + y * y])


# Robotiq 2F-85 with coupling: ~0.925 kg, CoM ~6 cm along flange z.
GRIPPER_PHI = phi_from_body(0.925, [0.0, 0.0, 0.06], box_inertia(0.925, [0.08, 0.15, 0.12]))


def box_payload_phi(mass: float, size, center_F) -> np.ndarray:
    return phi_from_body(mass, center_F, box_inertia(mass, size))


def simulate_run(
    t: np.ndarray,
    q: np.ndarray,
    phi: np.ndarray,
    force_noise: float = 3.5,
    torque_noise: float = 0.2,
    bias: Optional[np.ndarray] = None,
    frame: str = "flange",
    sign: float = 1.0,
    q_noise: float = 1e-5,
    tcp_offset: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
) -> RunData:
    """Simulates ft_raw_wrench for a body with params `phi` (about flange origin)."""
    rng = rng or np.random.default_rng()
    kin = flange_kinematics(t, q, cutoff_hz=None, gravity_base=GRAVITY_BASE)
    Y = payload_regressor(kin.g_F, kin.a_F, kin.omega_F, kin.alpha_F)
    w = Y @ phi  # (T,6) in F, torque about flange origin
    if bias is not None:
        w = w + bias  # sensor bias is fixed in the sensor (flange) frame
    if frame == "base":
        w = np.concatenate(
            [
                np.einsum("tij,tj->ti", kin.R_BF, w[:, :3]),
                np.einsum("tij,tj->ti", kin.R_BF, w[:, 3:]),
            ],
            axis=1,
        )
    w = sign * w
    noise = rng.standard_normal(w.shape) * np.r_[[force_noise] * 3, [torque_noise] * 3]
    return RunData(
        t=t.copy(),
        q=q + q_noise * rng.standard_normal(q.shape),
        ft_raw=w + noise,
        tcp_offset=np.zeros(6) if tcp_offset is None else tcp_offset,
    )
