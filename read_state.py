"""Read-only G1 state monitor: arm joints + IMU from rt/lowstate, Inspire hand state. Publishes nothing."""
import sys
import time

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorStates_
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient

IFACE = sys.argv[1] if len(sys.argv) > 1 else "enx6c6e071c419f"
DURATION = 5

ARM_NAMES = ["ShPitch", "ShRoll", "ShYaw", "Elbow", "WrRoll", "WrPitch", "WrYaw"]
LEFT_ARM, RIGHT_ARM = range(15, 22), range(22, 29)
FINGERS = ["pinky", "ring", "middle", "index", "thumbBend", "thumbRot"]
RIGHT_HAND, LEFT_HAND = range(0, 6), range(6, 12)

ChannelFactoryInitialize(0, IFACE)
latest = {"low": None, "hand": None, "low_n": 0, "hand_n": 0}


def on(key):
    def cb(msg):
        latest[key] = msg
        latest[key + "_n"] += 1
    return cb


low = ChannelSubscriber("rt/lowstate", LowState_)
low.Init(on("low"), 10)
hand = ChannelSubscriber("rt/inspire/state", MotorStates_)
hand.Init(on("hand"), 10)

msc = MotionSwitcherClient()
msc.SetTimeout(3.0)
msc.Init()
code, mode = msc.CheckMode()
print(f"motion_switcher CheckMode -> code={code} result={mode}")


def fmt(vals):
    return " ".join(f"{v:+.3f}" for v in vals)


for t in range(DURATION):
    time.sleep(1.0)
    ls, hs = latest["low"], latest["hand"]
    print(f"\n--- t={t + 1}s --- msgs: lowstate={latest['low_n']} inspire={latest['hand_n']}")
    if ls is None:
        print("rt/lowstate: no message")
    else:
        imu = ls.imu_state
        print(f"mode_machine={ls.mode_machine} mode_pr={ls.mode_pr} tick={ls.tick}")
        print("L arm [" + " ".join(ARM_NAMES) + "]:", fmt(ls.motor_state[i].q for i in LEFT_ARM))
        print("R arm:", fmt(ls.motor_state[i].q for i in RIGHT_ARM))
        print(f"IMU rpy={fmt(imu.rpy)} gyro={fmt(imu.gyroscope)} acc={fmt(imu.accelerometer)} quat={fmt(imu.quaternion)}")
    if hs is None:
        print("rt/inspire/state: no message")
    else:
        print("L hand [" + " ".join(FINGERS) + "]:", fmt(hs.states[i].q for i in LEFT_HAND))
        print("R hand:", fmt(hs.states[i].q for i in RIGHT_HAND))
