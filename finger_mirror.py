"""Mirror the person's fingers onto the Inspire hands. Fingers only, no arm motion.

    python finger_mirror.py              # dry run (default): print 6 values per hand, no writes
    python finger_mirror.py --live       # write to the hands over Modbus via hands.py
    python finger_mirror.py --show       # also open a camera window

Values follow the Inspire scale: 0 = fully closed, 1000 = fully open,
order [pinky, ring, middle, index, thumbBend, thumbRot].
Person's left hand -> robot's left hand (image is mirrored so MediaPipe labels are the person's own).
Camera: Jetson cam_server.py MJPEG stream. With several people, the largest hand box per side wins.
"""
import argparse
import json
import os
import time

import cv2
import numpy as np

from rs_cam import DEFAULT_URL, HandTracker, StreamCamera, draw_hand, largest_per_side

# Landmark chains (wrist=0). Each finger: MCP, PIP, DIP, TIP.
CHAINS = {"pinky": [17, 18, 19, 20], "ring": [13, 14, 15, 16],
          "middle": [9, 10, 11, 12], "index": [5, 6, 7, 8]}
# Defaults if calib.json is missing: (open, closed) raw value per channel.
# Fingers/thumbBend: total bend in degrees. thumbRot: thumb tip-to-pinky MCP distance / palm length.
DEFAULT_CAL = [(25.0, 220.0)] * 4 + [(10.0, 80.0), (1.25, 0.55)]
CALIB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calib.json")


def load_calib():
    cal = {"left": list(DEFAULT_CAL), "right": list(DEFAULT_CAL)}
    if os.path.exists(CALIB_FILE):
        with open(CALIB_FILE) as f:
            cal.update(json.load(f))
    return cal


def bend(a, b, c):
    """Angle (deg) by which segment b->c deviates from a->b; 0 when straight."""
    u, v = b - a, c - b
    cos = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def raw_metrics(w):
    """21x3 world landmarks -> uncalibrated [pinky, ring, middle, index, thumbBend, thumbRot]."""
    out = []
    for name in ["pinky", "ring", "middle", "index"]:
        m, p, d, t = (w[i] for i in CHAINS[name])
        out.append(bend(w[0], m, p) + bend(m, p, d) + bend(p, d, t))
    out.append(bend(w[1], w[2], w[3]) + bend(w[2], w[3], w[4]))
    palm = np.linalg.norm(w[9] - w[0]) + 1e-9
    out.append(float(np.linalg.norm(w[4] - w[17]) / palm))
    return out


def flexion(w, cal=DEFAULT_CAL):
    """21x3 world landmarks -> [pinky, ring, middle, index, thumbBend, thumbRot] on 0 (closed)..1000 (open)."""
    return [float(np.clip(1000.0 * (x - closed) / (open_ - closed), 0, 1000))
            for x, (open_, closed) in zip(raw_metrics(w), cal)]


class Smoother:
    """Exponential moving average; resets after the hand has been lost for a while."""

    def __init__(self, alpha, reset_after=0.5):
        self.alpha, self.reset_after = alpha, reset_after
        self.v, self.t = None, 0.0

    def __call__(self, x, now):
        x = np.asarray(x, dtype=float)
        if self.v is None or now - self.t > self.reset_after:
            self.v = x
        else:
            self.v = self.alpha * x + (1 - self.alpha) * self.v
        self.t = now
        return self.v.round().astype(int).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=DEFAULT_URL, help="cam_server.py base URL")
    ap.add_argument("--live", action="store_true", help="write to the hands (default is dry run)")
    ap.add_argument("--show", action="store_true", help="show camera window")
    ap.add_argument("--hz", type=float, default=15.0)
    ap.add_argument("--alpha", type=float, default=0.4, help="EMA weight of the new sample")
    ap.add_argument("--duration", type=float, default=0, help="stop after N seconds (0 = forever)")
    args = ap.parse_args()

    if args.live:
        import hands
        hands.MAX_HZ = args.hz
    cam = StreamCamera(args.url)
    tracker = HandTracker()
    cal = load_calib()
    smooth = {"left": Smoother(args.alpha), "right": Smoother(args.alpha)}
    period, start, seq = 1.0 / args.hz, time.monotonic(), None
    print(f"mode: {'LIVE' if args.live else 'DRY RUN'}  order: pinky ring middle index thumbBend thumbRot")
    try:
        while not args.duration or time.monotonic() - start < args.duration:
            t0 = time.monotonic()
            bgr, t_cap, seq = cam.read(seq)
            seen = {}
            for side, hd in largest_per_side(tracker.detect(bgr)).items():
                seen[side] = smooth[side](flexion(hd["world"], cal[side]), t0)
                if args.live:
                    hands.set_fingers(side, seen[side])
                if args.show:
                    draw_hand(bgr, hd["px"])
            lat_ms = (time.time() - t_cap) * 1000
            line = "  ".join(f"{s[0].upper()}: {' '.join(f'{v:4d}' for v in seen[s])}"
                             if s in seen else f"{s[0].upper()}: {'--':>29}" for s in ("left", "right"))
            print(f"{t0 - start:6.1f}s  {line}  lat {lat_ms:3.0f}ms", flush=True)
            if args.show:
                cv2.imshow("finger_mirror", bgr)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            time.sleep(max(0.0, period - (time.monotonic() - t0)))
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
