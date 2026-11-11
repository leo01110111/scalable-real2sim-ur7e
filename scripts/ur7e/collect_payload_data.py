"""Collects F/T data for one grasped object on a UR7e (object run + empty baseline).

Order: start this script, then press Play on the ExternalControl program.

    python scripts/ur7e/collect_payload_data.py --config ur7e_right \
        --trajectory data/ur7e_right_excitation.npz \
        --out-dir data/object_data/mug/system_id_data

Writes <out-dir>/object_run and <out-dir>/baseline_run. The baseline run uses the
same trajectory with the gripper at the same opening and no object, right after
the object run, so slow F/T drift and finger position effects cancel.

Use --baseline-only --gripper-position P --out-dir <root>/gripper_position_P to
build a library of empty-gripper runs (pass it later as --baseline-root).

Safety: set a payload on the pendant that is close to gripper + object, and run a
new trajectory first with --speed-scale 0.25 while you hold the E-stop.
"""

import argparse
import json
import time

from pathlib import Path

import numpy as np

from _common import load_config

from scalable_real2sim.ur7e_payload_id.trajectory import time_scale
from scalable_real2sim.ur7e_payload_id.ur7e_io import ArmAddress, RobotiqGripper, UR7e


def run_once(robot: UR7e, q_traj, approach_speed, label):
    print(f"[{label}] Approaching start pose...")
    robot.servo_approach(q_traj[0], speed=approach_speed)
    print(f"[{label}] Running excitation trajectory ({len(q_traj) * robot.dt:.1f} s)...")
    run = robot.run_trajectory(q_traj)
    print(f"[{label}] Done: {len(run.t)} samples.")
    return run


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="ur7e_right")
    ap.add_argument("--trajectory", required=True, help=".npz from design script")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--speed-scale", type=float, default=1.0)
    ap.add_argument("--baseline-only", action="store_true")
    ap.add_argument("--gripper-position", type=int, default=None,
                    help="0..255; required with --baseline-only")
    ap.add_argument("--no-baseline", action="store_true",
                    help="object run only (use a baseline library at ID time)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    sv, gc = cfg["servo"], cfg["gripper"]
    traj = np.load(args.trajectory)
    t, q = traj["t"], traj["q"]
    if abs((t[1] - t[0]) - 1.0 / sv["frequency"]) > 1e-9:
        raise ValueError("Trajectory dt does not match servo.frequency.")
    if args.speed_scale != 1.0:
        t, q = time_scale(t, q, args.speed_scale, 1.0 / sv["frequency"])
    if args.baseline_only and args.gripper_position is None:
        ap.error("--baseline-only needs --gripper-position")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    addr = ArmAddress(**cfg["arm"])
    meta = {
        "config": cfg,
        "trajectory": str(args.trajectory),
        "speed_scale": args.speed_scale,
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    gripper = RobotiqGripper(addr.ip, addr.gripper_port)
    gripper.activate(speed=gc["speed"], force=gc["force"])
    robot = UR7e(
        addr,
        frequency=sv["frequency"],
        watchdog_min_frequency=sv["watchdog_min_frequency"],
        lookahead_time=sv["lookahead_time"],
        gain=sv["gain"],
    )
    try:
        if args.baseline_only:
            gripper.move_and_wait(args.gripper_position)
            run = run_once(robot, q, sv["approach_speed"], "baseline")
            run.gripper_position = gripper.get_position()
            run.save(out, meta)
            print(f"Saved {out}")
            return

        gripper.move_and_wait(0)
        input("Place the object between the fingers, then press ENTER to grasp... ")
        obj_status = gripper.move_and_wait(255)
        grip_pos = gripper.get_position()
        if obj_status != 2:
            print(f"WARNING: gripper OBJ={obj_status} (2 = object detected).")
        input(f"Grasped at position {grip_pos}. Clear the area, press ENTER to run... ")
        run = run_once(robot, q, sv["approach_speed"], "object")
        run.gripper_position = grip_pos
        run.save(out / "object_run", {**meta, "gripper_obj_status": obj_status})

        if not args.no_baseline:
            input("Hold the object and press ENTER to open the gripper... ")
            gripper.move_and_wait(0)
            input("Remove the object, then press ENTER (gripper goes to the grasp "
                  "opening and runs the baseline)... ")
            gripper.move_and_wait(grip_pos)
            run = run_once(robot, q, sv["approach_speed"], "baseline")
            run.gripper_position = gripper.get_position()
            run.save(out / "baseline_run", meta)
        (out / "collection.json").write_text(json.dumps(meta, indent=2, default=str))
        print(f"Saved runs to {out}")
    finally:
        robot.close()
        gripper.close()


if __name__ == "__main__":
    main()
