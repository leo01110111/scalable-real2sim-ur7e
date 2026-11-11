"""Excitation trajectory design for wrist F/T payload identification.

Each joint follows q0 + w(t) * sum_k [a_k sin(k w0 t) + b_k cos(k w0 t)], where
w(t) is a quintic on/off window, so the motion starts and ends at q0 with zero
velocity and acceleration. A random search keeps the candidate with the best
(lowest) condition number of the column-normalized payload regressor that meets
joint, velocity, acceleration and simple workspace limits.

The iiwa pipeline designs trajectories with an augmented-Lagrangian black-box
optimizer and full collision checks in Drake. This module only checks a table
plane and a keep-out cylinder around the base. Check every new trajectory at low
speed (--speed-scale) before you run it at full speed.
"""

from dataclasses import asdict, dataclass, field

import numpy as np

from .kinematics import fk_frames
from .regressor import payload_regressor
from .signals import GRAVITY_BASE, flange_kinematics


@dataclass
class ExcitationConfig:
    q0: list = field(
        default_factory=lambda: [-1.57, -1.57, -1.57, -1.57, 1.57, 1.57]
    )
    amp_max: list = field(default_factory=lambda: [0.3, 0.25, 0.3, 1.0, 1.0, 1.5])
    q_min: list = field(default_factory=lambda: [-6.2] * 6)
    q_max: list = field(default_factory=lambda: [6.2] * 6)
    v_max: list = field(default_factory=lambda: [0.8, 0.8, 0.8, 1.5, 1.5, 1.5])
    a_max: list = field(default_factory=lambda: [2.0, 2.0, 2.0, 4.0, 4.0, 4.0])
    duration_s: float = 20.0
    base_freq_hz: float = 0.1
    num_harmonics: int = 5
    ramp_s: float = 2.0
    dt: float = 0.002
    tool_length_m: float = 0.20  # flange to fingertip, along flange z
    z_min_m: float = 0.05  # lowest allowed height of flange/fingertip in base frame
    r_min_m: float = 0.20  # keep-out radius around the base z axis
    num_candidates: int = 300
    seed: int = 0


def _window(t: np.ndarray, T: float, ramp: float) -> np.ndarray:
    def smooth(s):
        s = np.clip(s, 0.0, 1.0)
        return s**3 * (10 - 15 * s + 6 * s**2)

    return smooth(t / ramp) * smooth((T - t) / ramp)


def trajectory_from_coeffs(coeffs: np.ndarray, cfg: ExcitationConfig):
    """coeffs: (6, K, 2). Returns t (T,), q (T,6)."""
    t = np.arange(0.0, cfg.duration_s + 1e-9, cfg.dt)
    k = np.arange(1, coeffs.shape[1] + 1)
    ph = 2 * np.pi * cfg.base_freq_hz * np.outer(t, k)  # (T,K)
    dev = np.sin(ph) @ coeffs[:, :, 0].T + np.cos(ph) @ coeffs[:, :, 1].T
    dev *= _window(t, cfg.duration_s, cfg.ramp_s)[:, None]
    return t, np.asarray(cfg.q0)[None, :] + dev


def check_limits(t: np.ndarray, q: np.ndarray, cfg: ExcitationConfig) -> dict:
    dt = t[1] - t[0]
    qd = np.gradient(q, dt, axis=0)
    qdd = np.gradient(qd, dt, axis=0)
    X = fk_frames(q)[:, 6]
    p_flange = X[:, :3, 3]
    p_tip = p_flange + cfg.tool_length_m * X[:, :3, 2]
    pts = np.concatenate([p_flange, p_tip])
    return {
        "q_ok": bool(np.all(q >= cfg.q_min) and np.all(q <= cfg.q_max)),
        "v_ratio": float(np.max(np.abs(qd) / cfg.v_max)),
        "a_ratio": float(np.max(np.abs(qdd) / cfg.a_max)),
        "z_min": float(pts[:, 2].min()),
        "r_min": float(np.linalg.norm(pts[:, :2], axis=1).min()),
    }


def limits_ok(lim: dict, cfg: ExcitationConfig) -> bool:
    return (
        lim["q_ok"]
        and lim["v_ratio"] <= 1.0
        and lim["a_ratio"] <= 1.0
        and lim["z_min"] >= cfg.z_min_m
        and lim["r_min"] >= cfg.r_min_m
    )


def excitation_cost(t: np.ndarray, q: np.ndarray) -> float:
    """Condition number of the column-normalized regressor incl. bias columns."""
    sub = slice(None, None, 10)  # 50 Hz is plenty for the metric
    kin = flange_kinematics(t[sub], q[sub], cutoff_hz=None, gravity_base=GRAVITY_BASE)
    Y = payload_regressor(kin.g_F, kin.a_F, kin.omega_F, kin.alpha_F)
    T = Y.shape[0]
    A = np.concatenate([Y, np.broadcast_to(np.eye(6), (T, 6, 6))], axis=2)
    A = A.reshape(6 * T, 16)
    A = A / (np.linalg.norm(A, axis=0) + 1e-12)
    return float(np.linalg.cond(A))


def design_excitation(cfg: ExcitationConfig, verbose: bool = True):
    """Random search. Returns dict with t, q, coeffs, cost, limits, config."""
    rng = np.random.default_rng(cfg.seed)
    K = cfg.num_harmonics
    best = None
    for i in range(cfg.num_candidates):
        coeffs = rng.uniform(-1, 1, size=(6, K, 2)) / np.arange(1, K + 1)[None, :, None]
        _, q_unit = trajectory_from_coeffs(coeffs, cfg)
        dev = np.max(np.abs(q_unit - np.asarray(cfg.q0)), axis=0) + 1e-12
        scale = np.asarray(cfg.amp_max) * rng.uniform(0.6, 1.0, 6) / dev
        coeffs = coeffs * scale[:, None, None]
        t, q = trajectory_from_coeffs(coeffs, cfg)
        lim = check_limits(t, q, cfg)
        # Shrink until velocity/acceleration limits hold.
        shrink = 1.0 / max(1.0, lim["v_ratio"], np.sqrt(lim["a_ratio"]))
        if shrink < 1.0:
            coeffs = coeffs * shrink * 0.98
            t, q = trajectory_from_coeffs(coeffs, cfg)
            lim = check_limits(t, q, cfg)
        if not limits_ok(lim, cfg):
            continue
        cost = excitation_cost(t, q)
        if best is None or cost < best["cost"]:
            best = {"t": t, "q": q, "coeffs": coeffs, "cost": cost, "limits": lim}
            if verbose:
                print(f"[{i}/{cfg.num_candidates}] cond={cost:.1f} limits={lim}")
    if best is None:
        raise RuntimeError(
            "No candidate met the limits. Relax amp_max/z_min_m/r_min_m or move q0."
        )
    best["config"] = asdict(cfg)
    return best


def time_scale(t: np.ndarray, q: np.ndarray, speed_scale: float, dt: float):
    """Replays the same path slower (speed_scale < 1) on a grid with step dt."""
    t_new = np.arange(0.0, t[-1] / speed_scale + 1e-9, dt)
    q_new = np.stack(
        [np.interp(t_new * speed_scale, t, q[:, j]) for j in range(q.shape[1])], 1
    )
    return t_new, q_new
