"""Record raw finger metrics: OPEN hands for 5 s, transition 2 s, FISTS for 5 s. Writes calib.json.

Uses the per-channel median of each phase as the open/closed endpoints, per hand.
"""
import json
import sys
import time

import numpy as np

from finger_mirror import CALIB_FILE, raw_metrics
from rs_cam import DEFAULT_URL, HandTracker, StreamCamera, largest_per_side

NAMES = ["pinky", "ring", "middle", "index", "thumbBend", "thumbRot"]
PHASES = [("open", 5.0), ("transition", 2.0), ("fist", 5.0)]

cam = StreamCamera(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_URL)
tracker = HandTracker()
samples = {(ph, side): [] for ph, _ in PHASES for side in ("left", "right")}
seq = None
for phase, dur in PHASES:
    print(f">>> {phase.upper()} for {dur:.0f}s", flush=True)
    end = time.monotonic() + dur
    while time.monotonic() < end:
        bgr, _, seq = cam.read(seq)
        for side, hd in largest_per_side(tracker.detect(bgr)).items():
            samples[(phase, side)].append(raw_metrics(hd["world"]))

cal = {}
for side in ("left", "right"):
    o, c = np.array(samples[("open", side)]), np.array(samples[("fist", side)])
    print(f"\n{side}: {len(o)} open frames, {len(c)} fist frames")
    if len(o) < 10 or len(c) < 10:
        print("  not enough frames, keeping defaults for this hand")
        continue
    om, cm = np.median(o, 0), np.median(c, 0)
    for n, a, b, sa, sb in zip(NAMES, om, cm, o.std(0), c.std(0)):
        print(f"  {n:9s} open {a:7.2f} (sd {sa:5.2f})  fist {b:7.2f} (sd {sb:5.2f})  span {abs(a - b):6.2f}")
    cal[side] = [[float(a), float(b)] for a, b in zip(om, cm)]

if cal:
    with open(CALIB_FILE, "w") as f:
        json.dump(cal, f, indent=1)
    print("\nwrote", CALIB_FILE)
