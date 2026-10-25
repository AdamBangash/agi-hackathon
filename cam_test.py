"""Camera test: Jetson RGB stream with hand landmarks and camera->landmarks latency. q to quit.

    python cam_test.py [--url http://192.168.123.164:8081] [--no-window] [--duration 10]
"""
import argparse
import time

import cv2
import numpy as np

from rs_cam import DEFAULT_URL, HandTracker, StreamCamera, draw_hand, largest_per_side

ap = argparse.ArgumentParser()
ap.add_argument("--url", default=DEFAULT_URL)
ap.add_argument("--no-window", action="store_true")
ap.add_argument("--duration", type=float, default=0)
args = ap.parse_args()

cam = StreamCamera(args.url)
tracker = HandTracker()
print(f"clock offset jetson-laptop: {cam.offset * 1000:+.1f} ms")
lat, seq, start, t_print = [], None, time.monotonic(), time.monotonic()
while not args.duration or time.monotonic() - start < args.duration:
    bgr, t_cap, seq = cam.read(seq)
    hands = tracker.detect(bgr)
    lat.append((time.time() - t_cap) * 1000)
    chosen = largest_per_side(hands)
    for hd in hands:
        draw_hand(bgr, hd["px"], (0, 255, 0) if chosen.get(hd["side"]) is hd else (128, 128, 128))
    if time.monotonic() - t_print >= 1:
        recent = np.array(lat[-30:])
        print(f"hands={len(hands)} chosen={sorted(chosen)} latency ms: median {np.median(recent):.0f} "
              f"p95 {np.percentile(recent, 95):.0f}", flush=True)
        t_print = time.monotonic()
    if not args.no_window:
        cv2.putText(bgr, f"latency {lat[-1]:.0f} ms  hands {len(hands)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imshow("cam_test", bgr)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
if lat:
    a = np.array(lat[10:] or lat)
    print(f"TOTAL frames={len(lat)} latency ms: median {np.median(a):.0f} p95 {np.percentile(a, 95):.0f} max {a.max():.0f}")
