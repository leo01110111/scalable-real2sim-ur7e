"""UR7e + Robotiq 2F-85 hardware I/O over ur_rtde.

Connection pattern follows FACTR_Teleop (UR7e fork):
  * RTDEReceiveInterface for state, RTDEControlInterface with the ExternalControl
    URCap (FLAG_USE_EXT_UR_CAP + per-arm custom port). Start this script first,
    then press Play on the ExternalControl program on the pendant.
  * servoJ only (no moveJ): mixing moveJ and servoJ makes the controller reject
    the following servoJ.
  * A watchdog stops the arm if commands stop. Between runs (operator prompts) a
    background thread kicks it.
  * Robotiq 2F-85 through the Robotiq URCap ASCII socket (port 63352).
"""

import socket
import threading
import time

from dataclasses import dataclass

import numpy as np

from .signals import RunData


@dataclass
class ArmAddress:
    ip: str
    ur_cap_port: int
    gripper_port: int = 63352


class RobotiqGripper:
    """Minimal client for the Robotiq URCap socket. Positions 0 (open)..255 (closed)."""

    def __init__(self, ip: str, port: int = 63352, timeout: float = 2.0):
        self._sock = socket.create_connection((ip, port), timeout=timeout)
        self._lock = threading.Lock()

    def _cmd(self, cmd: str) -> str:
        with self._lock:
            self._sock.sendall((cmd + "\n").encode())
            return self._sock.recv(1024).decode().strip()

    def _get_var(self, name: str) -> int:
        return int(self._cmd(f"GET {name}").split()[1])

    def activate(self, speed: int = 255, force: int = 128, wait_timeout: float = 5.0):
        self._cmd("SET ACT 1")
        self._cmd(f"SET SPE {int(np.clip(speed, 0, 255))}")
        self._cmd(f"SET FOR {int(np.clip(force, 0, 255))}")
        self._cmd("SET GTO 1")
        t0 = time.time()
        while self._get_var("STA") != 3 and time.time() - t0 < wait_timeout:
            time.sleep(0.1)

    def set_position(self, pos: int):
        self._cmd(f"SET POS {int(np.clip(pos, 0, 255))}")

    def get_position(self) -> int:
        return self._get_var("POS")

    def get_object(self) -> int:
        """0 moving, 1 contact while opening, 2 contact while closing, 3 at target."""
        return self._get_var("OBJ")

    def move_and_wait(self, pos: int, timeout: float = 5.0) -> int:
        """Moves to `pos` and waits until it stops. Returns the OBJ status."""
        self.set_position(pos)
        time.sleep(0.1)
        t0 = time.time()
        obj = self.get_object()
        while obj == 0 and time.time() - t0 < timeout:
            time.sleep(0.05)
            obj = self.get_object()
        time.sleep(0.3)  # let the fingers settle
        return obj

    def close(self):
        try:
            self._sock.close()
        except Exception:
            pass


class UR7e:
    def __init__(
        self,
        addr: ArmAddress,
        frequency: float = 500.0,
        watchdog_min_frequency: float = 100.0,
        connect_timeout: float = 60.0,
        lookahead_time: float = 0.1,
        gain: float = 300.0,
    ):
        from rtde_control import RTDEControlInterface
        from rtde_receive import RTDEReceiveInterface

        self.addr = addr
        self.frequency = frequency
        self.dt = 1.0 / frequency
        self.lookahead_time = lookahead_time
        self.gain = gain
        self.r = RTDEReceiveInterface(addr.ip, frequency)
        print(
            f"Connecting RTDE control to {addr.ip} (ExternalControl port "
            f"{addr.ur_cap_port}). Press PLAY on the ExternalControl program."
        )
        deadline = time.monotonic() + connect_timeout
        while True:
            try:
                self.c = RTDEControlInterface(
                    addr.ip,
                    frequency,
                    RTDEControlInterface.FLAG_USE_EXT_UR_CAP,
                    addr.ur_cap_port,
                )
                break
            except RuntimeError as exc:
                if time.monotonic() > deadline:
                    raise TimeoutError(f"ExternalControl not reachable: {exc}") from exc
                print("  waiting for ExternalControl...")
                time.sleep(2.0)
        self._lock = threading.Lock()
        self._kick = False
        self._kick_thread = None
        self.c.setWatchdog(watchdog_min_frequency)
        self._kick_period = 0.5 / watchdog_min_frequency
        self.start_idle_watchdog()

    # ---- watchdog while idle --------------------------------------------------
    def _kick_loop(self):
        while self._kick:
            with self._lock:
                self.c.kickWatchdog()
            time.sleep(self._kick_period)

    def start_idle_watchdog(self):
        if self._kick_thread is None:
            self._kick = True
            self._kick_thread = threading.Thread(target=self._kick_loop, daemon=True)
            self._kick_thread.start()

    def stop_idle_watchdog(self):
        self._kick = False
        if self._kick_thread is not None:
            self._kick_thread.join()
            self._kick_thread = None

    # ---- state ----------------------------------------------------------------
    def q(self) -> np.ndarray:
        return np.asarray(self.r.getActualQ(), dtype=float)

    def tcp_offset(self) -> np.ndarray:
        with self._lock:
            return np.asarray(self.c.getTCPOffset(), dtype=float)

    def _sample(self):
        r = self.r
        return (
            r.getTimestamp(),
            r.getActualQ(),
            r.getFtRawWrench(),
            r.getActualTCPForce(),
            r.getActualTCPPose(),
        )

    # ---- motion ---------------------------------------------------------------
    def _servo(self, q):
        ok = self.c.servoJ(list(map(float, q)), 0.0, 0.0, self.dt, self.lookahead_time, self.gain)
        if not ok:
            raise RuntimeError(
                "servoJ returned False; is the ExternalControl program still running?"
            )

    def servo_approach(self, q_goal: np.ndarray, speed: float = 0.3):
        """Linear joint-space approach to q_goal streamed with servoJ."""
        q_start = self.q()
        q_goal = np.asarray(q_goal, dtype=float)
        n = int(np.ceil(max(np.max(np.abs(q_goal - q_start)) / speed, self.dt) / self.dt))
        self.stop_idle_watchdog()
        try:
            for k in range(1, n + 1):
                t0 = self.c.initPeriod()
                s = k / n
                s = s * s * (3 - 2 * s)  # smooth start/stop
                self._servo(q_start + s * (q_goal - q_start))
                self.c.waitPeriod(t0)
            for _ in range(int(0.5 / self.dt)):  # settle
                t0 = self.c.initPeriod()
                self._servo(q_goal)
                self.c.waitPeriod(t0)
            self.c.servoStop()
        finally:
            self.start_idle_watchdog()

    def run_trajectory(self, q_traj: np.ndarray, settle_s: float = 1.0) -> RunData:
        """Streams q_traj (sampled at 1/frequency) with servoJ and records state."""
        if np.max(np.abs(self.q() - q_traj[0])) > 0.02:
            raise RuntimeError("Arm is not at the trajectory start; call servo_approach.")
        n_settle = int(settle_s * self.frequency)
        q_full = np.concatenate(
            [np.repeat(q_traj[:1], n_settle, 0), q_traj, np.repeat(q_traj[-1:], n_settle, 0)]
        )
        log = []
        self.stop_idle_watchdog()
        try:
            for qk in q_full:
                t0 = self.c.initPeriod()
                self._servo(qk)
                log.append(self._sample())
                self.c.waitPeriod(t0)
            self.c.servoStop()
        finally:
            self.start_idle_watchdog()
        ts, q, ft, tcpf, tcpp = (np.asarray(x, dtype=float) for x in zip(*log))
        return RunData(
            t=ts - ts[0],
            q=q,
            ft_raw=ft,
            tcp_force=tcpf,
            tcp_pose=tcpp,
            tcp_offset=self.tcp_offset(),
        )

    def close(self):
        self.stop_idle_watchdog()
        for name in ("servoStop", "stopScript", "disconnect"):
            try:
                getattr(self.c, name)()
            except Exception as exc:
                print(f"Warning: {name}() failed: {exc}")
        try:
            self.r.disconnect()
        except Exception:
            pass
