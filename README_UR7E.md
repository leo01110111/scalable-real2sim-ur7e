# UR7e payload identification

This fork adds inertial-parameter identification (mass, CoM, inertia) for a
**UR7e with a Robotiq 2F-85**, as a replacement for the iiwa joint-torque method in
`robot_payload_id`. Code: `scalable_real2sim/ur7e_payload_id/`, scripts:
`scripts/ur7e/`.

## Method

The iiwa method needs joint torque sensors. The UR7e has none (RTDE gives only
joint currents), but it has a 6-axis F/T sensor at the flange. So this version:

1. Moves the arm on an excitation trajectory (Fourier series, quintic on/off
   window) and records `ft_raw_wrench` + joint positions at 500 Hz over RTDE.
2. Runs the same trajectory twice: once with the object, once with the gripper
   empty at the same opening (`baseline_run`).
3. Fits the standard linear wrist-F/T model `w = Y(g, a, omega, alpha) phi + b`
   for both runs together: gripper params, payload params, and one sensor bias per
   run. Constraints: pseudo-inertia PSD (physical consistency) and, if a point
   cloud of the object is given, a bounding ellipsoid. These are the same
   constraints as the iiwa pipeline. Solver: cvxpy + Clarabel (MOSEK optional;
   no license needed).

No robot identification step is needed: arm dynamics and joint friction are
before the sensor. The frame/sign convention of `ft_raw_wrench` is detected
automatically from the empty-gripper run (it is reported in the output JSON).

## Expected accuracy (simulation)

`scripts/ur7e/simulate_accuracy.py`, datasheet noise (3.5 N / 0.2 Nm, 1 sigma,
white), 20 s trajectory, box 7x7x15 cm, 8 trials:

| mass [kg] | mass error | CoM error | inertia rel. error |
|---|---|---|---|
| 0.10 | 6.5 g (6.5 %) | 15 mm | 3.7 |
| 0.25 | 6.5 g (2.6 %) | 7 mm | 2.1 |
| 0.50 | 5.6 g (1.1 %) | 3 mm | 1.6 |
| 1.00 | 6.0 g (0.6 %) | 1.3 mm | 0.9 |
| 2.00 | 4.8 g (0.2 %) | 0.7 mm | 0.6 |

Mass and CoM: good. Rotational inertia: **not identifiable** for household
objects with this sensor. Use the mesh-based inertia (uniform density, scaled to
the identified mass) instead. Real sensor noise is not white (drift, temperature,
and a reported Tz coupling with wrist-3 motor current), so expect worse results on
hardware. Check with a kitchen scale.

## Usage

Install (in the poetry env, or standalone): `pip install -r requirements-ur7e.txt`.
Network settings come from FACTR_Teleop: right arm `192.168.2.2` (ExternalControl
port 50003), left arm `192.168.1.2` (port 50002), Robotiq socket 63352. Edit
`scalable_real2sim/ur7e_payload_id/configs/ur7e_{right,left}.yaml`.

```bash
# 1. Once per workcell: design the trajectory. Edit q0 and the workspace limits first.
python scripts/ur7e/design_excitation_trajectory.py --config ur7e_right \
    --out data/ur7e_right_excitation.npz

# 2. Optional: predict accuracy for your objects.
python scripts/ur7e/simulate_accuracy.py --trajectory data/ur7e_right_excitation.npz

# 3. Per object. Start the script, then press Play on ExternalControl.
#    Run a new trajectory first with --speed-scale 0.25 and a hand on the E-stop.
python scripts/ur7e/collect_payload_data.py --config ur7e_right \
    --trajectory data/ur7e_right_excitation.npz \
    --out-dir data/object_data/<object>/system_id_data

# 4. Identify.
python scripts/ur7e/identify_payload.py --config ur7e_right \
    --object-data-dir data/object_data/<object>/system_id_data --json-out out.json

# Or in the full asset pipeline:
python run_asset_generation.py --robot ur7e --data-dir data/object_data \
    --output-dir out
```

To express the result in the object (mesh) frame, put one of these in
`system_id_data/`: `X_flange_object.npy` (4x4 pose of the object in the flange
frame), or `manipuland_cloud_flange_frame.npy` (object points in the flange
frame; registered to the mesh with the original RANSAC+ICP code). Otherwise the
output is in the flange frame. The point cloud also enables the bounding-ellipsoid
constraint.

Baseline library (optional, saves one run per object):
`collect_payload_data.py --baseline-only --gripper-position P --out-dir
data/baselines/gripper_position_P`, then `--no-baseline` per object and
`--baseline-root data/baselines`. A fresh baseline right after each object run
cancels F/T drift better.

## Safety notes

- Set a pendant payload close to gripper + object. The controller uses it for
  collision detection.
- The trajectory check covers only joint/velocity/acceleration limits, a table
  plane (`z_min_m`), and a keep-out cylinder around the base (`r_min_m`). It does
  **not** check collisions with the workcell. Check it at low speed.
- Motion uses servoJ only, with an RTDE watchdog (as in FACTR_Teleop). A thread
  kicks the watchdog while the script waits for the operator.

## Not tested on hardware yet

Tested: unit tests (`tests/test_ur7e_payload_id.py`: Jacobian, regressor,
recovery, frame/sign detection) and simulated end-to-end runs. Not yet tested:
real RTDE streaming, the Robotiq flow, and the real `ft_raw_wrench` convention.
The UR7e kinematics use the nominal UR5e DH values; `identify_payload.py` compares
them with the controller's `actual_TCP_pose` and warns if they differ.
