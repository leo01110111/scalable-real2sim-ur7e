"""UR7e forward kinematics and flange Jacobian (standard DH, vectorized over time).

The UR7e uses the UR5e kinematic chain (850 mm reach). The nominal DH values below
are UR's published UR5e values. Each arm has small factory calibration offsets
(see urcontrol.conf / the calibration file on the controller), so the collector
also records the controller's own TCP pose. `check_fk_against_controller` uses it
to confirm that this model is good enough.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class DHParams:
    a: np.ndarray = field(
        default_factory=lambda: np.array([0.0, -0.425, -0.3922, 0.0, 0.0, 0.0])
    )
    d: np.ndarray = field(
        default_factory=lambda: np.array([0.1625, 0.0, 0.0, 0.1333, 0.0997, 0.0996])
    )
    alpha: np.ndarray = field(
        default_factory=lambda: np.array(
            [np.pi / 2, 0.0, 0.0, np.pi / 2, -np.pi / 2, 0.0]
        )
    )


UR7E_DH = DHParams()


def _dh_transforms(q: np.ndarray, dh: DHParams) -> np.ndarray:
    """Returns the per-joint transforms T_{i-1,i}, shape (T, 6, 4, 4)."""
    q = np.atleast_2d(q)
    T = q.shape[0]
    ct, st = np.cos(q), np.sin(q)
    ca, sa = np.cos(dh.alpha), np.sin(dh.alpha)
    A = np.zeros((T, 6, 4, 4))
    A[:, :, 0, 0] = ct
    A[:, :, 0, 1] = -st * ca
    A[:, :, 0, 2] = st * sa
    A[:, :, 0, 3] = dh.a * ct
    A[:, :, 1, 0] = st
    A[:, :, 1, 1] = ct * ca
    A[:, :, 1, 2] = -ct * sa
    A[:, :, 1, 3] = dh.a * st
    A[:, :, 2, 1] = sa
    A[:, :, 2, 2] = ca
    A[:, :, 2, 3] = dh.d
    A[:, :, 3, 3] = 1.0
    return A


def fk_frames(q: np.ndarray, dh: DHParams = UR7E_DH) -> np.ndarray:
    """Base-frame poses of frames 0..6 (frame 6 = tool flange), shape (T, 7, 4, 4)."""
    A = _dh_transforms(q, dh)
    T = A.shape[0]
    frames = np.zeros((T, 7, 4, 4))
    frames[:, 0] = np.eye(4)
    for i in range(6):
        frames[:, i + 1] = frames[:, i] @ A[:, i]
    return frames


def flange_pose(q: np.ndarray, dh: DHParams = UR7E_DH) -> np.ndarray:
    """Base-frame pose of the tool flange, shape (T, 4, 4)."""
    return fk_frames(q, dh)[:, 6]


def flange_jacobian(q: np.ndarray, dh: DHParams = UR7E_DH) -> np.ndarray:
    """Geometric Jacobian of the flange origin in the base frame, shape (T, 6, 6).

    Rows 0..2: linear velocity of the flange origin. Rows 3..5: angular velocity.
    """
    frames = fk_frames(q, dh)
    p_e = frames[:, 6, :3, 3]
    J = np.zeros((frames.shape[0], 6, 6))
    for i in range(6):
        z = frames[:, i, :3, 2]
        o = frames[:, i, :3, 3]
        J[:, :3, i] = np.cross(z, p_e - o)
        J[:, 3:, i] = z
    return J


def rotvec_to_matrix(rv: np.ndarray) -> np.ndarray:
    """UR axis-angle (rotation vector) to rotation matrices, (T,3) -> (T,3,3)."""
    rv = np.atleast_2d(rv)
    theta = np.linalg.norm(rv, axis=1, keepdims=True)
    k = np.divide(rv, theta, out=np.zeros_like(rv), where=theta > 1e-12)
    K = np.zeros((rv.shape[0], 3, 3))
    K[:, 0, 1], K[:, 0, 2] = -k[:, 2], k[:, 1]
    K[:, 1, 0], K[:, 1, 2] = k[:, 2], -k[:, 0]
    K[:, 2, 0], K[:, 2, 1] = -k[:, 1], k[:, 0]
    th = theta[:, :, None]
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def pose6_to_matrix(pose: np.ndarray) -> np.ndarray:
    """UR pose [x, y, z, rx, ry, rz] to homogeneous transforms, (T,6) -> (T,4,4)."""
    pose = np.atleast_2d(pose)
    X = np.tile(np.eye(4), (pose.shape[0], 1, 1))
    X[:, :3, :3] = rotvec_to_matrix(pose[:, 3:])
    X[:, :3, 3] = pose[:, :3]
    return X


def check_fk_against_controller(
    q: np.ndarray, tcp_pose: np.ndarray, tcp_offset: np.ndarray, dh: DHParams = UR7E_DH
) -> tuple[float, float]:
    """Compares this FK with the controller's actual_TCP_pose.

    Returns (max position error [m], max rotation error [rad]).
    """
    X_BF = flange_pose(q, dh)
    X_FT = pose6_to_matrix(np.asarray(tcp_offset))[0]
    X_BT = X_BF @ X_FT
    X_ctrl = pose6_to_matrix(tcp_pose)
    pos_err = np.linalg.norm(X_BT[:, :3, 3] - X_ctrl[:, :3, 3], axis=1)
    R_err = np.einsum("tji,tjk->tik", X_BT[:, :3, :3], X_ctrl[:, :3, :3])
    cos = np.clip((np.trace(R_err, axis1=1, axis2=2) - 1) / 2, -1.0, 1.0)
    return float(pos_err.max()), float(np.arccos(cos).max())
