"""Inspire FTP hand control over Modbus TCP (direct, no DDS bridge).

    from hands import set_fingers
    set_fingers("right", [1000] * 6)          # open
    set_fingers("192.168.123.210", [0] * 6)  # close, addressed by IP

Order: [pinky, ring, middle, index, thumbBend, thumbRot], 0 (closed) .. 1000 (open).
CLI:  python hands.py read
      python hands.py test 210 index [--to 500] [--hold 2]
"""
import argparse
import socket
import struct
import threading
import time

FINGERS = ["pinky", "ring", "middle", "index", "thumbBend", "thumbRot"]
# Unconfirmed: inspire_hand_ws itself disagrees on L/R. Fix after the visual test.
HANDS = {"right": "192.168.123.210", "left": "192.168.123.211"}
PORT = 6000

REG_ANGLE_SET = 1486
REG_FORCE_SET = 1498
REG_SPEED_SET = 1522
REG_ANGLE_ACT = 1546
REG_FORCE_ACT = 1582
REG_ERROR = 1606

FORCE_CAP = 1000     # per-finger force threshold written to FORCE_SET; hand stops pushing above it.
                     # FTP scale goes to ~3000, factory power-on default is 2000. Resting readings reach ~460.
SPEED = 600          # actuator speed (0..1000)
MAX_HZ = 30          # writes faster than this are coalesced (latest target wins)
MAX_STEP = 200       # max change per finger per write (slew limit)


class InspireHand:
    def __init__(self, ip, force_cap=FORCE_CAP, speed=SPEED, max_hz=MAX_HZ, max_step=MAX_STEP):
        self.ip = ip
        self.force_cap = max(0, min(3000, force_cap))
        self.speed = max(0, min(1000, speed))
        self.min_dt = 1.0 / max_hz
        self.max_step = max_step
        self._lock = threading.Lock()
        self._tid = 0
        self._sock = None
        self._last_write = 0.0
        self._last_cmd = None

    # --- Modbus primitives ---------------------------------------------
    def _connect(self):
        if self._sock is None:
            self._sock = socket.create_connection((self.ip, PORT), timeout=1.0)
            self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self._write_regs(REG_FORCE_SET, [self.force_cap] * 6)
            self._write_regs(REG_SPEED_SET, [self.speed] * 6)

    def _recv(self, n):
        buf = b""
        while len(buf) < n:
            chunk = self._sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("hand closed connection")
            buf += chunk
        return buf

    def _request(self, pdu):
        self._tid = (self._tid + 1) & 0xFFFF
        self._sock.sendall(struct.pack(">HHHB", self._tid, 0, len(pdu) + 1, 1) + pdu)
        _, _, length, _ = struct.unpack(">HHHB", self._recv(7))
        resp = self._recv(length - 1)
        if resp[0] & 0x80:
            raise RuntimeError(f"modbus exception {resp[1]} from {self.ip}")
        return resp

    def _read_regs(self, addr, count):
        resp = self._request(struct.pack(">BHH", 3, addr, count))
        return list(struct.unpack(f">{count}h", resp[2:2 + 2 * count]))

    def _write_regs(self, addr, values):
        data = struct.pack(f">{len(values)}h", *values)
        self._request(struct.pack(">BHHB", 16, addr, len(values), len(data)) + data)

    def _call(self, fn, *args):
        with self._lock:
            try:
                self._connect()
                return fn(*args)
            except (OSError, ConnectionError):
                self.close()
                raise

    # --- public API -----------------------------------------------------
    def read(self):
        return {
            "angle": self._call(self._read_regs, REG_ANGLE_ACT, 6),
            "force": self._call(self._read_regs, REG_FORCE_ACT, 6),
            "error": self._call(self._read_regs, REG_ERROR, 3),
        }

    def set(self, angles, force=False):
        """Command 6 angles. Returns the values actually sent, or None if rate-limited."""
        if len(angles) != 6:
            raise ValueError("need 6 values: " + ", ".join(FINGERS))
        target = [int(max(0, min(1000, a))) for a in angles]
        now = time.monotonic()
        if not force and now - self._last_write < self.min_dt:
            return None
        if self._last_cmd is None:
            self._last_cmd = self._call(self._read_regs, REG_ANGLE_ACT, 6)
        cmd = [prev + max(-self.max_step, min(self.max_step, t - prev))
               for prev, t in zip(self._last_cmd, target)]
        self._call(self._write_regs, REG_ANGLE_SET, cmd)
        self._last_cmd, self._last_write = cmd, now
        return cmd

    def move_to(self, angles, timeout=3.0):
        """Blocking: step through the slew limit until the target is commanded."""
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            cmd = self.set(angles)
            if cmd is not None and cmd == [int(max(0, min(1000, a))) for a in angles]:
                return cmd
            time.sleep(self.min_dt)
        return self._last_cmd

    def close(self):
        if self._sock is not None:
            try:
                self._sock.close()
            finally:
                self._sock = None


_hands = {}


def get_hand(hand):
    ip = HANDS.get(hand, hand)
    if ip.isdigit():
        ip = f"192.168.123.{ip}"
    if ip not in _hands:
        _hands[ip] = InspireHand(ip)
    return _hands[ip]


def set_fingers(hand, angles):
    """hand: 'left' | 'right' | IP | last octet. Non-blocking, rate-limited, slew-limited."""
    return get_hand(hand).set(angles)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("read")
    t = sub.add_parser("test", help="move one finger, hold, restore")
    t.add_argument("hand")
    t.add_argument("finger", choices=FINGERS)
    t.add_argument("--to", type=int, default=500)
    t.add_argument("--hold", type=float, default=2.0)
    args = ap.parse_args()

    if args.cmd == "read":
        for name, ip in HANDS.items():
            print(name, ip, get_hand(ip).read())
        return

    h = get_hand(args.hand)
    start = h.read()["angle"]
    goal = list(start)
    goal[FINGERS.index(args.finger)] = args.to
    print(f"{h.ip}: {args.finger} {start[FINGERS.index(args.finger)]} -> {args.to}")
    h.move_to(goal)
    time.sleep(args.hold)
    print("at:", h.read()["angle"])
    h.move_to(start)
    time.sleep(0.5)
    print("restored:", h.read()["angle"])


if __name__ == "__main__":
    main()
