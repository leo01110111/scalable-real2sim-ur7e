"""Identifies mass, CoM and inertia of a grasped object from UR7e F/T runs.

    python scripts/ur7e/identify_payload.py \
        --object-data-dir data/object_data/mug/system_id_data \
        [--object-mesh mesh.obj] [--baseline-root data/baselines] \
        --json-out mug_inertial_params.json
"""

import argparse
import logging

import numpy as np

from _common import load_config

from scalable_real2sim.ur7e_payload_id.identification import (
    IdentConfig,
    identify_grasped_object_payload_ur7e,
)
from scalable_real2sim.ur7e_payload_id.kinematics import check_fk_against_controller
from scalable_real2sim.ur7e_payload_id.signals import RunData


def main():
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="ur7e_right")
    ap.add_argument("--object-data-dir", required=True)
    ap.add_argument("--object-mesh", default=None)
    ap.add_argument("--baseline-root", default=None)
    ap.add_argument("--json-out", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    ident = IdentConfig(rate_hz=cfg["servo"]["frequency"], **cfg["identification"])

    run = RunData.load(f"{args.object_data_dir}/object_run")
    if run.tcp_pose is not None and run.tcp_offset is not None:
        pos_err, rot_err = check_fk_against_controller(run.q, run.tcp_pose, run.tcp_offset)
        logging.info(f"FK vs controller: pos {pos_err*1e3:.2f} mm, rot {np.degrees(rot_err):.3f} deg")
        if pos_err > 0.005 or rot_err > np.radians(0.5):
            logging.warning("Nominal DH differs from the controller; check kinematics.")

    identify_grasped_object_payload_ur7e(
        object_data_path=args.object_data_dir,
        json_output_path=args.json_out,
        object_mesh_path=args.object_mesh,
        baseline_root=args.baseline_root,
        cfg=ident,
    )


if __name__ == "__main__":
    main()
