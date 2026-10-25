"""Mirror the person's arms (rt/arm_sdk) and fingers (Inspire, Modbus) onto the G1.

    python body_mirror.py                      # dry run: print arm targets + finger values, publishes nothing
    python body_mirror.py --arms-live          # also drive the arms through rt/arm_sdk
    python body_mirror.py --fingers-live       # also drive the Inspire hands
    python body_mirror.py --arms-live --fingers-live --duration 30
    python body_mirror.py --backend lowcmd --arms-live    # robot seated in zero torque (FSM 0)

Arms: MediaPipe Pose -> shoulder pitch, shoulder roll, elbow. Shoulder yaw and wrists are held at 0.
Person's left arm -> robot's left arm, same as finger_mirror.py.
arm_sdk sequence follows g1_arm7_sdk_dds_example: weight ramps 0 -> 1 while holding the current pose,
then tracks with velocity + acceleration limits (gentler accel above FAST_VEL); on exit returns to the starting pose and ramps weight 1 -> 0.
Waist joints are held at their starting positions while arm_sdk is engaged.

--backend lowcmd is for a seated robot whose loco controller is in zero torque (arm_sdk is ignored there):
it releases the motion service (MotionSwitcher.ReleaseMode), publishes rt/lowcmd with arm and waist-hold stiffness ramped
0 -> KP, legs in light damping only, a tilt watchdog (MAX_TILT), and on exit restores the previous service with SelectMode.
"""
import argparse
import signal
import threading
import time

import numpy as np

import finger_mirror as fm
from rs_cam import DEFAULT_URL, HandTracker, PoseTracker, StreamCamera, largest_per_side

IFACE = "enx6c6e071c419f"
LEFT_ARM = list(range(15, 22))   # ShPitch ShRoll ShYaw Elbow WrRoll WrPitch WrYaw
RIGHT_ARM = list(range(22, 29))
WAIST = [12, 13, 14]
WEIGHT_IDX = 29

# Workspace limits (rad) per arm joint: conservative subset of the hardware range.
LIMITS = {
    "left": [(-1.57, 0.3), (-0.1, 1.8), (0, 0), (-0.9, 1.5), (0, 0), (0, 0), (0, 0)],
    "right": [(-1.57, 0.3), (-1.8, 0.1), (0, 0), (-0.9, 1.5), (0, 0), (0, 0), (0, 0)],
}
MAX_VEL = 1.0          # rad/s per joint
MAX_ACC = 6.0          # rad/s^2 per joint, used below FAST_VEL and always for braking
FAST_VEL = 0.5         # rad/s: above this, speeding up further is limited to FAST_ACC
FAST_ACC = 2.5         # rad/s^2
MEDIAN_N = 3           # arm target filter (ArmSmoother): median window, frames
EURO_MIN_CUTOFF = 0.4  # Hz, smoothing cutoff when the arm is still
EURO_BETA = 1.0        # cutoff grows by this many Hz per rad/s of arm speed
EURO_D_CUTOFF = 1.0    # Hz, cutoff for the speed estimate
STILL_SPEED = 0.3      # rad/s: below this filtered joint speed the arm counts as held still
DEADBAND = 0.1         # rad (~6 deg): while still, smaller drift is treated as pose noise
CONTROL_DT = 0.02      # 50 Hz, as in the SDK example
KP, KD = 60.0, 1.5
RAMP_IN_S, RAMP_OUT_S = 2.0, 1.5
LEG_KD = 1.0           # lowcmd backend: damping-only on legs (no stiffness)
WAIST_KP, WAIST_KD = 150.0, 3.0  # lowcmd backend: hold the waist at its start pose so arm motion can't swing the torso
MAX_TILT = np.radians(8)  # IMU roll/pitch drift from the start beyond this -> arms go home and the run ends
LOST_RETURN_S = 2.0    # person lost this long -> arms go back to the starting pose
POSE_ARMS = {"left": (11, 13, 15), "right": (12, 14, 16)}  # shoulder, elbow, wrist


def unit(v):
    return v / (np.linalg.norm(v) + 1e-9)


def pick_person(people, min_face_vis=0.5):
    """Largest torso among people whose face is in view. Someone right next to the robot (a spotter) fills
    the frame with their torso but their head is cut off, so they would otherwise win "largest box"."""
    faced = [p for p in people if p["vis"][0] >= min_face_vis]  # landmark 0 = nose
    return max(faced, key=lambda x: x["area"]) if faced else None


def retarget(world, vis, min_vis=0.5):
    """33x3 pose world landmarks (from the un-mirrored image) -> {side: 7 arm joint targets} for visible arms."""
    ls, rs, lh, rh = world[11], world[12], world[23], world[24]
    left = unit(ls - rs)
    up = (ls + rs) / 2 - (lh + rh) / 2
    up = unit(up - np.dot(up, left) * left)
    fwd = np.cross(left, up)
    out = {}
    for side, (s, e, w) in POSE_ARMS.items():
        if min(vis[s], vis[e], vis[w]) < min_vis:
            continue
        a = world[e] - world[s]
        ax, ay, az = unit(np.array([np.dot(a, fwd), np.dot(a, left), np.dot(a, up)]))
        roll = float(np.arcsin(np.clip(ay, -1, 1)))
        # Pitch is undefined with the arm straight out sideways; fade it to 0 as the arm approaches that.
        h = float(np.hypot(ax, az))
        pitch = float(np.arctan2(-ax, -az)) if h > 1e-3 else 0.0
        if pitch > np.pi / 2:
            pitch -= 2 * np.pi
        pitch *= float(np.clip((h - 0.15) / 0.25, 0.0, 1.0))
        b = world[w] - world[e]
        bend = float(np.arccos(np.clip(np.dot(unit(a), unit(b)), -1, 1)))  # 0 = straight arm
        elbow = np.pi / 2 - bend  # G1 elbow: q = 0 is a 90 deg bend, larger q straightens
        q = [pitch, roll, 0.0, elbow, 0.0, 0.0, 0.0]
        out[side] = [float(np.clip(v, lo, hi)) for v, (lo, hi) in zip(q, LIMITS[side])]
    return out


def profile_step(q, v, goal, dt=CONTROL_DT):
    """One tick of a velocity- and acceleration-limited move toward goal. Returns (q, v).
    The desired speed is the fastest we can still brake from at MAX_ACC. Speeding up past FAST_VEL
    is capped at FAST_ACC, so fast moves build up gently. Braking always gets the full MAX_ACC."""
    err = goal - q
    v_des = float(np.sign(err)) * min(MAX_VEL, float(np.sqrt(2 * MAX_ACC * abs(err))))
    speeding_up = abs(v_des) > abs(v) and v_des * v >= 0
    acc = FAST_ACC if speeding_up and abs(v) >= FAST_VEL else MAX_ACC
    v += float(np.clip(v_des - v, -acc * dt, acc * dt))
    q_next = q + v * dt
    if (goal - q_next) * err <= 0 and abs(v) <= MAX_ACC * dt:  # arrived: settle exactly, no dithering
        return goal, 0.0
    return q_next, v


class ArmController:
    """50 Hz arm_sdk publisher. States: ramp_in -> track -> return -> ramp_out -> done."""

    def __init__(self, live, backend="arm_sdk"):
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
        ChannelFactoryInitialize(0, IFACE)
        self.live = live
        self.backend = backend
        self.prev_mode = None
        self.low = None
        self.sub = ChannelSubscriber("rt/lowstate", LowState_)
        self.sub.Init(self._on_low, 10)
        t_end = time.monotonic() + 3
        while self.low is None and time.monotonic() < t_end:
            time.sleep(0.05)
        if self.low is None:
            raise RuntimeError("no rt/lowstate")
        self.start_q = {j: self.low.motor_state[j].q for j in LEFT_ARM + RIGHT_ARM + WAIST}
        self.cmd_q = dict(self.start_q)
        self.cmd_v = {j: 0.0 for j in self.start_q}
        self.start_rpy = list(self.low.imu_state.rpy)
        self.tripped = False
        self.target = {"left": [self.start_q[j] for j in LEFT_ARM], "right": [self.start_q[j] for j in RIGHT_ARM]}
        self.lock = threading.Lock()
        self.state, self.t_state, self.weight = "ramp_in", time.monotonic(), 0.0
        if live:
            from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
            from unitree_sdk2py.utils.crc import CRC
            from unitree_sdk2py.utils.thread import RecurrentThread
            self.crc = CRC()
            self.msg = unitree_hg_msg_dds__LowCmd_()
            if backend == "lowcmd":
                self._release_motion_service()
            self.pub = ChannelPublisher("rt/arm_sdk" if backend == "arm_sdk" else "rt/lowcmd", LowCmd_)
            self.pub.Init()
            self.t_state = time.monotonic()  # ramp-in starts now, not before the (slow) service release
            self.thread = RecurrentThread(interval=CONTROL_DT, target=self._tick, name=backend)
            self.thread.Start()

    def _release_motion_service(self):
        from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
        self.msc = MotionSwitcherClient()
        self.msc.SetTimeout(5.0)
        self.msc.Init()
        _, res = self.msc.CheckMode()
        self.prev_mode = res.get("name") if res else None
        print(f"motion service was '{self.prev_mode}', releasing", flush=True)
        for _ in range(10):
            _, res = self.msc.CheckMode()
            if not res or not res.get("name"):
                return
            self.msc.ReleaseMode()
            time.sleep(1.0)
        raise RuntimeError("motion service did not release")

    def _restore_motion_service(self):
        if self.prev_mode:
            code, _ = self.msc.SelectMode(self.prev_mode)
            time.sleep(1.0)
            print(f"restored motion service '{self.prev_mode}' (code {code}), now {self.msc.CheckMode()[1]}", flush=True)

    def _on_low(self, msg):
        self.low = msg

    def tilt(self):
        """(roll, pitch) drift from the starting IMU attitude, rad."""
        rpy = self.low.imu_state.rpy
        return rpy[0] - self.start_rpy[0], rpy[1] - self.start_rpy[1]

    def measured(self, side):
        return [self.low.motor_state[j].q for j in (LEFT_ARM if side == "left" else RIGHT_ARM)]

    def set_target(self, side, q):
        with self.lock:
            self.target[side] = list(q)

    def go_home(self):
        with self.lock:
            self.go_home_unlocked()

    def stop(self):
        """Begin return-to-start then release. Blocks until arm_sdk weight is 0."""
        if not self.live:
            return
        with self.lock:
            if self.state in ("ramp_in", "track"):
                self.go_home_unlocked()
                self.state, self.t_state = "return", time.monotonic()
        t_end = time.monotonic() + 10
        while self.state != "done" and time.monotonic() < t_end:
            time.sleep(0.05)
        if self.state != "done":
            print(f"WARNING: {self.backend} did not finish releasing (state={self.state}, weight={self.weight:.2f})")
        if self.backend == "lowcmd":
            self._restore_motion_service()

    def go_home_unlocked(self):
        self.target = {"left": [self.start_q[j] for j in LEFT_ARM], "right": [self.start_q[j] for j in RIGHT_ARM]}

    def _tick(self):
        now = time.monotonic()
        dt_state = now - self.t_state
        with self.lock:
            if self.state in ("ramp_in", "track") and max(map(abs, self.tilt())) > MAX_TILT:
                r, p = np.degrees(self.tilt())
                print(f"TILT TRIP: roll {r:+.1f} pitch {p:+.1f} deg from start, returning arms home", flush=True)
                self.tripped = True
                self.go_home_unlocked()
                self.state, self.t_state = "return", now
                dt_state = 0.0
            goal = dict(zip(LEFT_ARM, self.target["left"]))
            goal.update(zip(RIGHT_ARM, self.target["right"]))
            if self.state == "ramp_in":
                self.weight = min(1.0, dt_state / RAMP_IN_S)
                goal = {j: self.start_q[j] for j in LEFT_ARM + RIGHT_ARM}  # hold while taking over
                if self.weight >= 1.0:
                    self.state, self.t_state = "track", now
            elif self.state == "return":
                self.weight = 1.0
                if all(abs(self.cmd_q[j] - self.start_q[j]) < 0.02 for j in LEFT_ARM + RIGHT_ARM) or dt_state > 6:
                    self.state, self.t_state = "ramp_out", now
            elif self.state == "ramp_out":
                self.weight = max(0.0, 1.0 - dt_state / RAMP_OUT_S)
                if self.weight <= 0.0:
                    self.state = "done"
            elif self.state == "done":
                return
        for j, g in goal.items():
            self.cmd_q[j], self.cmd_v[j] = profile_step(self.cmd_q[j], self.cmd_v[j], g)
        m = self.msg
        if self.backend == "arm_sdk":
            m.motor_cmd[WEIGHT_IDX].q = self.weight
            for j in LEFT_ARM + RIGHT_ARM + WAIST:
                mc = m.motor_cmd[j]
                mc.q = self.cmd_q[j] if j not in WAIST else self.start_q[j]
                mc.dq, mc.tau, mc.kp, mc.kd = 0.0, 0.0, KP, KD
        else:
            # Full low-level command: arm stiffness scaled by the ramp, everything else damping only.
            m.mode_pr = 0
            m.mode_machine = self.low.mode_machine
            for j in range(29):
                mc = m.motor_cmd[j]
                mc.mode, mc.dq, mc.tau = 1, 0.0, 0.0
                if j in LEFT_ARM or j in RIGHT_ARM:
                    mc.q, mc.kp, mc.kd = self.cmd_q[j], KP * self.weight, KD
                elif j in WAIST:
                    mc.q, mc.kp, mc.kd = self.start_q[j], WAIST_KP * self.weight, max(WAIST_KD * self.weight, LEG_KD)
                else:
                    mc.q, mc.kp, mc.kd = self.low.motor_state[j].q, 0.0, LEG_KD
        m.crc = self.crc.Crc(m)
        self.pub.Write(m)


def lowpass_alpha(cutoff, dt):
    return 1.0 / (1.0 + 1.0 / (2 * np.pi * cutoff * dt))


class ArmSmoother:
    """Per-joint arm target filter, in radians: median of the last MEDIAN_N samples (drops one-frame spikes),
    then a One Euro filter (heavy smoothing when still, light when moving fast), then a still-hold: while the
    joint moves faster than STILL_SPEED the output follows, otherwise it stays put until the filtered value
    drifts more than DEADBAND from it (pose estimates drift slowly even when the person holds still).
    Resets after a tracking gap."""

    def __init__(self, reset_after=0.5):
        self.reset_after = reset_after
        self.t = None

    def __call__(self, x, now):
        x = np.asarray(x, dtype=float)
        if self.t is None or now - self.t > self.reset_after:
            self.buf, self.xf, self.dxf, self.out = [x], x, np.zeros_like(x), x.copy()
        else:
            dt = max(now - self.t, 1e-3)
            self.buf = (self.buf + [x])[-MEDIAN_N:]
            xm = np.median(self.buf, axis=0)
            self.dxf += lowpass_alpha(EURO_D_CUTOFF, dt) * ((xm - self.xf) / dt - self.dxf)
            a = lowpass_alpha(EURO_MIN_CUTOFF + EURO_BETA * np.abs(self.dxf), dt)
            self.xf = self.xf + a * (xm - self.xf)
            moved = (np.abs(self.dxf) > STILL_SPEED) | (np.abs(self.xf - self.out) > DEADBAND)
            self.out = np.where(moved, self.xf, self.out)
        self.t = now
        return self.out.tolist()


def deg(q):
    return " ".join(f"{np.degrees(v):5.0f}" for v in q)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--arms-live", action="store_true", help="publish arm commands")
    ap.add_argument("--backend", choices=["arm_sdk", "lowcmd"], default="arm_sdk",
                    help="arm_sdk needs the loco controller active; lowcmd for a seated robot in zero torque")
    ap.add_argument("--fingers-live", action="store_true", help="write to the Inspire hands")
    ap.add_argument("--hz", type=float, default=15.0)
    ap.add_argument("--alpha", type=float, default=0.4, help="finger EMA weight of the new sample")
    ap.add_argument("--duration", type=float, default=0)
    args = ap.parse_args()

    if args.fingers_live:
        import hands
    cam = StreamCamera(args.url)
    hand_tracker, pose_tracker = HandTracker(), PoseTracker()
    arms = ArmController(live=args.arms_live, backend=args.backend)
    cal = fm.load_calib()
    finger_smooth = {s: fm.Smoother(args.alpha) for s in ("left", "right")}
    arm_smooth = {s: ArmSmoother() for s in ("left", "right")}
    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())

    print(f"arms: {'LIVE' if args.arms_live else 'dry'} ({args.backend})  fingers: {'LIVE' if args.fingers_live else 'dry'}")
    print("arm cols (deg): ShPitch ShRoll Elbow | target vs [robot measured]")
    period, start, seq, last_seen = 1.0 / args.hz, time.monotonic(), None, time.monotonic()
    try:
        while not stop.is_set() and not arms.tripped and (not args.duration or time.monotonic() - start < args.duration):
            t0 = time.monotonic()
            bgr, _, seq = cam.read(seq)
            people = pose_tracker.detect(bgr[:, ::-1].copy())  # un-mirror for anatomical labels
            arm_q = {}
            p = pick_person(people)
            if p:
                for side, q in retarget(p["world"], p["vis"]).items():
                    arm_q[side] = arm_smooth[side](q, t0)
                    arms.set_target(side, arm_q[side])
            if arm_q:
                last_seen = t0
            elif t0 - last_seen > LOST_RETURN_S:
                arms.go_home()

            fingers = {}
            for side, hd in largest_per_side(hand_tracker.detect(bgr)).items():
                fingers[side] = finger_smooth[side](fm.flexion(hd["world"], cal[side]), t0)
                if args.fingers_live:
                    hands.set_fingers(side, fingers[side])

            parts = []
            for side in ("left", "right"):
                tgt = arm_q.get(side)
                meas = arms.measured(side)
                a = f"{deg([tgt[0], tgt[1], tgt[3]])}" if tgt else "    --    --    --"
                f = " ".join(f"{v:4d}" for v in fingers[side]) if side in fingers else "  --"
                parts.append(f"{side[0].upper()} arm {a} [{deg([meas[0], meas[1], meas[3]])}] fing {f}")
            r, p = np.degrees(arms.tilt())
            print(f"{t0 - start:5.1f}s {arms.state:8s} tilt {r:+4.1f} {p:+4.1f} | " + " | ".join(parts), flush=True)
            time.sleep(max(0.0, period - (time.monotonic() - t0)))
    finally:
        arms.stop()
        print(f"{args.backend} released" if args.arms_live else "dry run done")


if __name__ == "__main__":
    main()
