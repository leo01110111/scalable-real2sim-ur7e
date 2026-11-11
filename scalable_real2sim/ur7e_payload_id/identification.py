"""Payload identification from the UR7e wrist F/T sensor.

Method
------
Two runs of the same excitation trajectory are used:
  * baseline run: gripper at the grasp opening, no object  -> phi_g (gripper)
  * object run:   gripper holding the object                -> phi_g + phi_p
Each run also gets its own constant 6D sensor bias, so F/T drift between runs
cancels. Both runs are solved jointly as one least-squares problem with
physical-consistency constraints (pseudo-inertia matrices PSD, Wensing et al. 2017)
and an optional bounding ellipsoid for the object (from its point cloud), the same
constraints the iiwa pipeline uses. The SDP is solved with cvxpy (Clarabel by
default, MOSEK if installed); no MOSEK license is needed.

Unlike the iiwa pipeline, no robot identification is needed: the F/T sensor sits
after the arm, so arm dynamics and joint friction do not enter the model.
"""

import json
import logging

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import cvxpy as cp
import numpy as np

from .regressor import PARAM_NAMES, body_from_phi, payload_regressor
from .signals import (
    GRAVITY_BASE,
    RunData,
    flange_kinematics,
    lowpass,
    to_uniform,
    wrench_in_flange,
)

# Hypotheses for how ft_raw_wrench is expressed: (frame, torque reference point).
HYPOTHESES = [("flange", "origin"), ("base", "origin"), ("flange", "tcp"), ("base", "tcp")]


@dataclass
class IdentConfig:
    rate_hz: float = 500.0
    cutoff_hz: float = 6.0  # low-pass for joint positions and wrenches
    trim_s: float = 0.5  # drop filter edge effects at both ends
    gravity_base: list = field(default_factory=lambda: GRAVITY_BASE.tolist())
    wrench_frame: str = "auto"  # "auto" | "flange" | "base"
    wrench_point: str = "auto"  # "auto" | "origin" | "tcp"
    min_payload_mass: float = 1e-3
    ellipsoid_margin: float = 1.1  # scale on the point-cloud bounding ellipsoid
    solver: str = "auto"  # "auto" (MOSEK if present, else CLARABEL) or a cvxpy name


@dataclass
class PreparedRun:
    Y: np.ndarray  # (T,6,10) regressor
    w_raw: np.ndarray  # (T,6) filtered raw wrench
    R_BF: np.ndarray  # (T,3,3)
    tcp_offset: Optional[np.ndarray]
    fs: float


def prepare_run(run: RunData, cfg: IdentConfig) -> PreparedRun:
    u = to_uniform(run, cfg.rate_hz)
    kin = flange_kinematics(
        u.t, u.q, cfg.cutoff_hz, np.asarray(cfg.gravity_base, dtype=float)
    )
    w = lowpass(u.ft_raw, cfg.rate_hz, cfg.cutoff_hz)
    n_trim = int(cfg.trim_s * cfg.rate_hz)
    sl = slice(n_trim, len(u.t) - n_trim)
    Y = payload_regressor(kin.g_F[sl], kin.a_F[sl], kin.omega_F[sl], kin.alpha_F[sl])
    return PreparedRun(Y, w[sl], kin.R_BF[sl], u.tcp_offset, cfg.rate_hz)


def measured_wrench(p: PreparedRun, frame: str, point: str) -> np.ndarray:
    ref = None
    if point == "tcp":
        if p.tcp_offset is None:
            raise ValueError("TCP reference point requested but no tcp_offset saved.")
        ref = np.asarray(p.tcp_offset[:3], dtype=float)
    return wrench_in_flange(p.w_raw, p.R_BF, frame, ref)


def _fit_with_bias(Y: np.ndarray, w: np.ndarray):
    """Unconstrained LS of w = Y phi + b. Returns phi, b, residual (T,6)."""
    T = Y.shape[0]
    A = np.concatenate([Y, np.broadcast_to(np.eye(6), (T, 6, 6))], axis=2).reshape(
        6 * T, 16
    )
    x, *_ = np.linalg.lstsq(A, w.reshape(-1), rcond=None)
    res = (w.reshape(-1) - A @ x).reshape(T, 6)
    return x[:10], x[10:], res


def select_wrench_convention(baseline: PreparedRun, cfg: IdentConfig):
    """Picks frame/point/sign of ft_raw_wrench using the empty-gripper run.

    The gripper is a rigid body with positive mass, so the correct convention fits
    it far better and gives m > 0. Returns (frame, point, sign, report).
    """
    has_tcp = baseline.tcp_offset is not None and np.linalg.norm(
        baseline.tcp_offset[:3]
    ) > 1e-6
    report = {}
    for frame, point in HYPOTHESES:
        if cfg.wrench_frame != "auto" and frame != cfg.wrench_frame:
            continue
        if cfg.wrench_point != "auto" and point != cfg.wrench_point:
            continue
        if point == "tcp" and not has_tcp:
            continue
        w = measured_wrench(baseline, frame, point)
        phi, _, res = _fit_with_bias(baseline.Y, w)
        report[f"{frame}/{point}"] = {
            "rms_force": float(np.sqrt(np.mean(res[:, :3] ** 2))),
            "rms_torque": float(np.sqrt(np.mean(res[:, 3:] ** 2))),
            "mass": float(phi[0]),
        }
    if not report:
        raise ValueError("No wrench convention left to test; check the config.")
    # Force residual is the most informative score (torque scale is ~10x smaller).
    best = min(report, key=lambda k: report[k]["rms_force"] + 10 * report[k]["rms_torque"])
    frame, point = best.split("/")
    sign = 1.0 if report[best]["mass"] > 0 else -1.0
    return frame, point, sign, report


def min_volume_ellipsoid(points: np.ndarray, tol: float = 1e-4, max_iter: int = 2000):
    """Khachiyan's algorithm. Returns (center c, shape A) with (x-c)^T A (x-c) <= 1."""
    P = np.asarray(points, dtype=float)
    if len(P) > 4000:
        P = P[np.random.default_rng(0).choice(len(P), 4000, replace=False)]
    N, d = P.shape
    Q = np.vstack([P.T, np.ones(N)])
    u = np.full(N, 1.0 / N)
    for _ in range(max_iter):
        X = Q @ np.diag(u) @ Q.T
        M = np.einsum("ij,ji->i", Q.T @ np.linalg.inv(X), Q)
        j = int(np.argmax(M))
        step = (M[j] - d - 1.0) / ((d + 1.0) * (M[j] - 1.0))
        new_u = (1 - step) * u
        new_u[j] += step
        if np.linalg.norm(new_u - u) < tol:
            u = new_u
            break
        u = new_u
    c = P.T @ u
    A = np.linalg.inv(P.T @ np.diag(u) @ P - np.outer(c, c)) / d
    return c, A


def _phi_from_pseudo(J) -> list:
    """Linear map from a 4x4 pseudo-inertia (cvxpy expr) to phi entries."""
    S = J[:3, :3]
    return [
        J[3, 3],
        J[0, 3],
        J[1, 3],
        J[2, 3],
        S[1, 1] + S[2, 2],
        -S[0, 1],
        -S[0, 2],
        S[0, 0] + S[2, 2],
        -S[1, 2],
        S[0, 0] + S[1, 1],
    ]


def _pick_solver(name: str) -> str:
    if name != "auto":
        return name
    return "MOSEK" if "MOSEK" in cp.installed_solvers() else "CLARABEL"


@dataclass
class PayloadResult:
    phi_payload_F: np.ndarray
    phi_gripper_F: np.ndarray
    mass: float
    com_F: np.ndarray
    inertia_com_F: np.ndarray
    mass_std: float  # approximate 1-sigma from the unconstrained fit
    com_std_F: np.ndarray
    convention: dict
    diagnostics: dict

    def in_object_frame(self, X_FO: np.ndarray):
        """Mass, CoM and inertia about the CoM, expressed in object frame O."""
        R, p = X_FO[:3, :3], X_FO[:3, 3]
        com_O = R.T @ (self.com_F - p)
        I_O = R.T @ self.inertia_com_F @ R
        return self.mass, com_O, I_O


def identify_payload(
    baseline: RunData,
    obj: RunData,
    cfg: IdentConfig = IdentConfig(),
    ellipsoid_points_F: Optional[np.ndarray] = None,
) -> PayloadResult:
    pb, po = prepare_run(baseline, cfg), prepare_run(obj, cfg)
    frame, point, sign, conv_report = select_wrench_convention(pb, cfg)
    logging.info(f"ft_raw_wrench convention: frame={frame}, point={point}, sign={sign}")
    logging.info(f"Convention scores: {json.dumps(conv_report, indent=1)}")
    wb = sign * measured_wrench(pb, frame, point)
    wo = sign * measured_wrench(po, frame, point)

    # Per-axis noise level from unconstrained fits -> row weights.
    _, _, res_b = _fit_with_bias(pb.Y, wb)
    _, _, res_o = _fit_with_bias(po.Y, wo)
    sig_b = res_b.std(axis=0) + 1e-9
    sig_o = res_o.std(axis=0) + 1e-9

    # Stacked design: x = [phi_g(10), phi_p(10), b_b(6), b_o(6)].
    def block(Y, sig, payload, bias_slot):
        T = Y.shape[0]
        D = np.zeros((T, 6, 32))
        D[:, :, 0:10] = Y
        if payload:
            D[:, :, 10:20] = Y
        D[:, :, 20 + 6 * bias_slot : 26 + 6 * bias_slot] = np.eye(6)
        return (D / sig[None, :, None]).reshape(6 * T, 32)

    D = np.vstack([block(pb.Y, sig_b, False, 0), block(po.Y, sig_o, True, 1)])
    y = np.concatenate([(wb / sig_b).reshape(-1), (wo / sig_o).reshape(-1)])
    N = len(y)

    # Unconstrained solution and approximate covariance. Filtered samples are
    # correlated, so the effective sample count is ~ N * 2 * cutoff / fs.
    x_ls, *_ = np.linalg.lstsq(D, y, rcond=None)
    rss = float(np.sum((y - D @ x_ls) ** 2))
    corr = cfg.rate_hz / (2.0 * cfg.cutoff_hz)
    cov = np.linalg.pinv(D.T @ D) * rss / max(N - 32, 1) * corr
    std = np.sqrt(np.clip(np.diag(cov), 0, None))

    # Compress with QR, then solve the constrained problem.
    Qm, Rm = np.linalg.qr(D)
    z = Qm.T @ y
    Jg = cp.Variable((4, 4), PSD=True)
    Jp = cp.Variable((4, 4), PSD=True)
    b = cp.Variable(12)
    x = cp.hstack(_phi_from_pseudo(Jg) + _phi_from_pseudo(Jp) + [b])
    constraints = [Jp[3, 3] >= cfg.min_payload_mass]
    if ellipsoid_points_F is not None:
        c, A = min_volume_ellipsoid(ellipsoid_points_F)
        A = A / cfg.ellipsoid_margin**2
        Qe = np.zeros((4, 4))
        Qe[:3, :3] = -A
        Qe[:3, 3] = A @ c
        Qe[3, :3] = A @ c
        Qe[3, 3] = 1.0 - c @ A @ c
        constraints.append(cp.trace(Jp @ Qe) >= 0)
    prob = cp.Problem(cp.Minimize(cp.sum_squares(Rm @ x - z) / N), constraints)
    solver = _pick_solver(cfg.solver)
    prob.solve(solver=solver)
    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"Payload SDP failed: status={prob.status}")
    xv = np.asarray(x.value).ravel()
    phi_g, phi_p = xv[:10], xv[10:20]
    m, com, I_com = body_from_phi(phi_p)

    # Delta-method std of the CoM: c = h / m.
    J_c = np.zeros((3, 32))
    J_c[:, 10] = -x_ls[11:14] / x_ls[10] ** 2
    J_c[:, 11:14] = np.eye(3) / x_ls[10]
    com_std = np.sqrt(np.clip(np.diag(J_c @ cov @ J_c.T), 0, None))

    Yp = np.vstack([pb.Y.reshape(-1, 10), po.Y.reshape(-1, 10)])
    Yn = Yp / (np.linalg.norm(Yp, axis=0) + 1e-12)
    diagnostics = {
        "solver": solver,
        "status": prob.status,
        "num_samples": int(N // 6),
        "noise_std_baseline": sig_b.tolist(),
        "noise_std_object": sig_o.tolist(),
        "regressor_condition_number": float(np.linalg.cond(Yn)),
        "unconstrained_payload_phi": dict(zip(PARAM_NAMES, x_ls[10:20].tolist())),
        "unconstrained_payload_phi_std": dict(zip(PARAM_NAMES, std[10:20].tolist())),
        "used_bounding_ellipsoid": ellipsoid_points_F is not None,
    }
    return PayloadResult(
        phi_payload_F=phi_p,
        phi_gripper_F=phi_g,
        mass=m,
        com_F=com,
        inertia_com_F=I_com,
        mass_std=float(std[10]),
        com_std_F=com_std,
        convention={"frame": frame, "point": point, "sign": sign, "scores": conv_report},
        diagnostics=diagnostics,
    )


def closest_baseline(baseline_root: Path, gripper_position: int) -> Path:
    """Picks <root>/gripper_position_<P> closest to the object run's opening."""
    cands = [
        d
        for d in Path(baseline_root).iterdir()
        if d.is_dir() and d.name.startswith("gripper_position_")
    ]
    if not cands:
        raise FileNotFoundError(f"No gripper_position_* dirs in {baseline_root}")
    return min(cands, key=lambda d: abs(int(d.name.split("_")[-1]) - gripper_position))


def identify_grasped_object_payload_ur7e(
    object_data_path: str,
    json_output_path: str,
    object_mesh_path: Optional[str] = None,
    baseline_root: Optional[str] = None,
    cfg: IdentConfig = IdentConfig(),
) -> dict:
    """Drop-in replacement for the iiwa `identify_grasped_object_payload`.

    Expected layout of `object_data_path` (written by collect_payload_data.py):
        object_run/                    object in gripper
        baseline_run/                  same trajectory, empty gripper (optional if
                                       `baseline_root` is given)
        X_flange_object.npy            (optional) 4x4 pose of the object frame in F
        manipuland_cloud_flange_frame.npy  (optional) (N,3) object points in F

    The output JSON matches the iiwa pipeline: mass, center_of_mass, and
    inertia_matrix (about the CoM), in the object frame if the object pose is
    known (from X_flange_object.npy, or by registering the point cloud to the
    mesh), else in the flange frame.
    """
    root = Path(object_data_path)
    obj = RunData.load(root / "object_run")
    if (root / "baseline_run").exists():
        base_dir = root / "baseline_run"
    elif baseline_root is not None:
        base_dir = closest_baseline(Path(baseline_root), obj.gripper_position or 0)
    else:
        raise FileNotFoundError(f"No baseline_run in {root} and no baseline_root.")
    logging.info(f"Baseline run: {base_dir}")
    baseline = RunData.load(base_dir)

    cloud_path = root / "manipuland_cloud_flange_frame.npy"
    cloud = np.load(cloud_path) if cloud_path.exists() else None
    res = identify_payload(baseline, obj, cfg, ellipsoid_points_F=cloud)

    X_FO = None
    if (root / "X_flange_object.npy").exists():
        X_FO = np.load(root / "X_flange_object.npy")
    elif cloud is not None and object_mesh_path is not None:
        # Same RANSAC + ICP registration as the iiwa pipeline (needs open3d).
        from scalable_real2sim.robot_payload_id.scripts.identify_grasped_object_payload import (  # noqa: E501
            get_object_pose_in_link_frame,
        )

        X_FO = get_object_pose_in_link_frame(Path(object_mesh_path), cloud)

    if X_FO is not None:
        mass, com, I = res.in_object_frame(X_FO)
        frame = "object"
    else:
        logging.warning("Object pose unknown: writing the result in the flange frame.")
        mass, com, I = res.mass, res.com_F, res.inertia_com_F
        frame = "flange"

    out = {
        "mass": float(mass),
        "center_of_mass": np.asarray(com).tolist(),
        "inertia_matrix": np.asarray(I).tolist(),
        "frame": frame,
        "mass_std": res.mass_std,
        "com_std_flange": res.com_std_F.tolist(),
        "flange_frame": {
            "center_of_mass": res.com_F.tolist(),
            "inertia_matrix": res.inertia_com_F.tolist(),
        },
        "wrench_convention": res.convention,
        "diagnostics": res.diagnostics,
        "config": asdict(cfg),
        "baseline_run": str(base_dir),
    }
    Path(json_output_path).write_text(json.dumps(out, indent=2))
    logging.info(
        f"Payload: m={mass:.4f} kg (+/- {res.mass_std:.4f}), CoM_F={res.com_F}"
    )
    return out
