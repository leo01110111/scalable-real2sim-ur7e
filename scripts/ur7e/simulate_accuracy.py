"""Predicts identification accuracy with simulated UR7e F/T noise (no robot needed).

    python scripts/ur7e/simulate_accuracy.py --trajectory data/ur7e_right_excitation.npz

Noise is white Gaussian with the datasheet precision as 1 sigma (conservative for
white noise, optimistic if the real noise is low-frequency drift). Each trial also
gets a random constant bias per run.
"""

import argparse
import logging

import numpy as np

from _common import load_config

from scalable_real2sim.ur7e_payload_id.identification import IdentConfig, identify_payload
from scalable_real2sim.ur7e_payload_id.simulation import (
    GRIPPER_PHI,
    box_inertia,
    box_payload_phi,
    simulate_run,
)


def main():
    logging.basicConfig(level=logging.WARNING)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="ur7e_right")
    ap.add_argument("--trajectory", required=True)
    ap.add_argument("--masses", type=float, nargs="+", default=[0.1, 0.25, 0.5, 1.0, 2.0])
    ap.add_argument("--size", type=float, nargs=3, default=[0.07, 0.07, 0.15],
                    help="object box size [m]")
    ap.add_argument("--force-noise", type=float, default=3.5)
    ap.add_argument("--torque-noise", type=float, default=0.2)
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--speed-scale", type=float, default=1.0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    ident = IdentConfig(rate_hz=cfg["servo"]["frequency"], **cfg["identification"])
    tr = np.load(args.trajectory)
    t, q = tr["t"], tr["q"]
    if args.speed_scale != 1.0:
        from scalable_real2sim.ur7e_payload_id.trajectory import time_scale
        t, q = time_scale(t, q, args.speed_scale, t[1] - t[0])
    rng = np.random.default_rng(0)
    print(f"{'mass':>6} | {'|dm| mean':>9} {'rel':>6} | {'|dCoM| mm':>9} | "
          f"{'I rel err':>9} | {'pred std m':>10}")
    for m in args.masses:
        dm, dc, dI, sd = [], [], [], []
        for _ in range(args.trials):
            com = np.array([0.0, 0.0, 0.16]) + rng.normal(0, 0.01, 3)
            phi_p = box_payload_phi(m, args.size, com)
            kw = dict(force_noise=args.force_noise, torque_noise=args.torque_noise, rng=rng)
            base = simulate_run(t, q, GRIPPER_PHI, bias=rng.normal(0, 2, 6) * [1, 1, 1, .1, .1, .1], **kw)
            obj = simulate_run(t, q, GRIPPER_PHI + phi_p, bias=rng.normal(0, 2, 6) * [1, 1, 1, .1, .1, .1], **kw)
            res = identify_payload(base, obj, ident)
            I_true = box_inertia(m, args.size)
            dm.append(abs(res.mass - m))
            dc.append(np.linalg.norm(res.com_F - com) * 1e3)
            dI.append(np.linalg.norm(res.inertia_com_F - I_true) / np.linalg.norm(I_true))
            sd.append(res.mass_std)
        print(f"{m:6.2f} | {np.mean(dm):9.4f} {np.mean(dm)/m:6.1%} | {np.mean(dc):9.1f} | "
              f"{np.mean(dI):9.2f} | {np.mean(sd):10.4f}")


if __name__ == "__main__":
    main()
