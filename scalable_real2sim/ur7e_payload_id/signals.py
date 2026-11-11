"""Run data I/O, filtering, and flange kinematics from recorded joint positions."""

import json

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from scipy.signal import butter, filtfilt

from .kinematics import UR7E_DH, DHParams, fk_frames, flange_jacobian

GRAVITY_BASE = np.array([0.0, 0.0, -9.81])


@dataclass
class RunData:
    """One trajectory execution, as written by `collect_payload_data.py`."""

    t: np.ndarray  # (T,) controller timestamps [s]
    q: np.ndarray  # (T,6) actual joint positions [rad]
    ft_raw: np.ndarray  # (T,6) ft_raw_wrench [N, Nm]
    tcp_force: Optional[np.ndarray] = None  # (T,6) actual_TCP_force
    tcp_pose: Optional[np.ndarray] = None  # (T,6) actual_TCP_pose
    tcp_offset: Optional[np.ndarray] = None  # (6,) TCP offset set on controller
    gripper_position: Optional[int] = None  # Robotiq 0..255

    @staticmethod
    def load(run_dir: Path) -> "RunData":
        run_dir = Path(run_dir)

        def opt(name):
            p = run_dir / name
            return np.load(p) if p.exists() else None

        meta_path = run_dir / "meta.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        return RunData(
            t=np.load(run_dir / "sample_times_s.npy"),
            q=np.load(run_dir / "joint_positions.npy"),
            ft_raw=np.load(run_dir / "ft_raw_wrench.npy"),
            tcp_force=opt("tcp_force.npy"),
            tcp_pose=opt("tcp_pose.npy"),
            tcp_offset=opt("tcp_offset.npy"),
            gripper_position=meta.get("gripper_position"),
        )

    def save(self, run_dir: Path, meta: Optional[dict] = None) -> None:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        np.save(run_dir / "sample_times_s.npy", self.t)
        np.save(run_dir / "joint_positions.npy", self.q)
        np.save(run_dir / "ft_raw_wrench.npy", self.ft_raw)
        for name, arr in [
            ("tcp_force.npy", self.tcp_force),
            ("tcp_pose.npy", self.tcp_pose),
            ("tcp_offset.npy", self.tcp_offset),
        ]:
            if arr is not None:
                np.save(run_dir / name, arr)
        meta = dict(meta or {})
        meta["gripper_position"] = self.gripper_position
        (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))


def to_uniform(run: RunData, rate_hz: float) -> RunData:
    """Drops duplicate RTDE packets and resamples all signals on a uniform grid."""
    _, idx = np.unique(run.t, return_index=True)
    t = run.t[idx]
    grid = np.arange(t[0], t[-1], 1.0 / rate_hz)

    def interp(x):
        if x is None:
            return None
        x = x[idx]
        return np.stack([np.interp(grid, t, x[:, j]) for j in range(x.shape[1])], 1)

    return RunData(
        t=grid,
        q=interp(run.q),
        ft_raw=interp(run.ft_raw),
        tcp_force=interp(run.tcp_force),
        tcp_pose=None,  # rotation vectors do not interpolate safely; unused here
        tcp_offset=run.tcp_offset,
        gripper_position=run.gripper_position,
    )


def lowpass(x: np.ndarray, fs: float, cutoff_hz: Optional[float], order: int = 4):
    """Zero-phase Butterworth low-pass along axis 0 (identity if cutoff is None)."""
    if cutoff_hz is None:
        return x
    b, a = butter(order, cutoff_hz / (0.5 * fs))
    return filtfilt(b, a, x, axis=0)


@dataclass
class FlangeKinematics:
    R_BF: np.ndarray  # (T,3,3)
    p_BF: np.ndarray  # (T,3)
    g_F: np.ndarray  # (T,3) gravity in F
    a_F: np.ndarray  # (T,3) linear acceleration of the flange origin, in F
    omega_F: np.ndarray  # (T,3)
    alpha_F: np.ndarray  # (T,3)


def flange_kinematics(
    t: np.ndarray,
    q: np.ndarray,
    cutoff_hz: Optional[float] = 6.0,
    gravity_base: np.ndarray = GRAVITY_BASE,
    dh: DHParams = UR7E_DH,
) -> FlangeKinematics:
    """Flange velocities/accelerations from joint positions on a uniform grid."""
    dt = float(np.mean(np.diff(t)))
    fs = 1.0 / dt
    qf = lowpass(q, fs, cutoff_hz)
    qd = np.gradient(qf, dt, axis=0)
    J = flange_jacobian(qf, dh)
    V = np.einsum("tij,tj->ti", J, qd)  # [v; omega] in B
    Vd = np.gradient(V, dt, axis=0)  # [a; alpha] in B (time derivative in B)
    X = fk_frames(qf, dh)[:, 6]
    R = X[:, :3, :3]
    Rt = np.transpose(R, (0, 2, 1))
    rot = lambda v: np.einsum("tij,tj->ti", Rt, v)
    return FlangeKinematics(
        R_BF=R,
        p_BF=X[:, :3, 3],
        g_F=rot(np.broadcast_to(gravity_base, (len(t), 3))),
        a_F=rot(Vd[:, :3]),
        omega_F=rot(V[:, 3:]),
        alpha_F=rot(Vd[:, 3:]),
    )


def wrench_in_flange(
    w: np.ndarray, R_BF: np.ndarray, frame: str, p_F_ref: Optional[np.ndarray] = None
) -> np.ndarray:
    """Expresses a measured wrench in F, with torque about the flange origin.

    frame: "flange" if w is already expressed in F, "base" if expressed in B.
    p_F_ref: point (in F) about which the measured torque is taken; None = origin.
    """
    f, n = w[:, :3], w[:, 3:]
    if frame == "base":
        Rt = np.transpose(R_BF, (0, 2, 1))
        f = np.einsum("tij,tj->ti", Rt, f)
        n = np.einsum("tij,tj->ti", Rt, n)
    elif frame != "flange":
        raise ValueError(f"Unknown wrench frame '{frame}'.")
    if p_F_ref is not None and np.linalg.norm(p_F_ref) > 0:
        n = n + np.cross(np.broadcast_to(p_F_ref, f.shape), f)
    return np.concatenate([f, n], axis=1)
