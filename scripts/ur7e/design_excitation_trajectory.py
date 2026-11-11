"""Designs an excitation trajectory for one UR7e and saves it as .npz.

Run once per workcell (and again if q0 or the workspace changes):
    python scripts/ur7e/design_excitation_trajectory.py --config ur7e_right \
        --out data/ur7e_right_excitation.npz
"""

import argparse
import json

import numpy as np

from _common import load_config

from scalable_real2sim.ur7e_payload_id.trajectory import (
    ExcitationConfig,
    design_excitation,
)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="ur7e_right", help="config name or yaml path")
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-candidates", type=int, default=None)
    args = ap.parse_args()

    cfg_dict = load_config(args.config)
    ex = ExcitationConfig(**cfg_dict["excitation"])
    if args.num_candidates is not None:
        ex.num_candidates = args.num_candidates
    expected_dt = 1.0 / cfg_dict["servo"]["frequency"]
    if abs(ex.dt - expected_dt) > 1e-9:
        raise ValueError(f"excitation.dt must be {expected_dt} (1/servo.frequency).")
    best = design_excitation(ex)
    np.savez(
        args.out,
        t=best["t"],
        q=best["q"],
        coeffs=best["coeffs"],
        cost=best["cost"],
        config=json.dumps(best["config"]),
        limits=json.dumps(best["limits"]),
    )
    print(f"Saved {args.out}: cond={best['cost']:.1f}, limits={best['limits']}")


if __name__ == "__main__":
    main()
