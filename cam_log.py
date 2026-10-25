"""Record what the arm mirror sees: annotated frames + a per-frame CSV of pose visibility and arm targets.

    python cam_log.py                    # 10 s -> logs/<timestamp>/
    python cam_log.py --duration 20 --every 5

Writes frames/NNNN.jpg (every --every-th frame, pose drawn, arm joints coloured by visibility),
log.csv (one row per frame) and contact.jpg (a grid of the saved frames). Read-only: publishes nothing.
"""
import argparse
import csv
import os
import time

import cv2
import numpy as np

from body_mirror import POSE_ARMS, pick_person, retarget
from rs_cam import DEFAULT_URL, PoseTracker, StreamCamera

BONES = [(11, 12), (11, 23), (12, 24), (23, 24), (11, 13), (13, 15), (12, 14), (14, 16)]
NAMES = {11: "lSh", 12: "rSh", 13: "lEl", 14: "rEl", 15: "lWr", 16: "rWr"}


def draw(img, p):
    h, w = img.shape[:2]
    # Pose ran on the un-mirrored image; the displayed image is mirrored, so flip x back.
    pt = {i: (int((1 - x) * w), int(y * h)) for i, (x, y) in enumerate(p["px"])}
    for a, b in BONES:
        cv2.line(img, pt[a], pt[b], (200, 200, 200), 2)
    for i, name in NAMES.items():
        v = p["vis"][i]
        color = (0, 200, 0) if v >= 0.5 else (0, 0, 255)  # green = usable, red = below the 0.5 cutoff
        cv2.circle(img, pt[i], 6, color, -1)
        cv2.putText(img, f"{name} {v:.2f}", (pt[i][0] + 8, pt[i][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--duration", type=float, default=10)
    ap.add_argument("--every", type=int, default=10, help="save every Nth frame as a jpg")
    ap.add_argument("--out", default=os.path.join("logs", time.strftime("%Y%m%d-%H%M%S")))
    args = ap.parse_args()

    os.makedirs(os.path.join(args.out, "frames"), exist_ok=True)
    cam, tracker = StreamCamera(args.url), PoseTracker()
    cols = ["t", "frame", "people"] + [f"vis_{NAMES[i]}" for i in sorted(NAMES)] + \
           [f"{s}_{j}" for s in ("left", "right") for j in ("pitch", "roll", "elbow")]
    saved, seq, n, start = [], None, 0, time.monotonic()
    with open(os.path.join(args.out, "log.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(cols)
        while time.monotonic() - start < args.duration:
            bgr, _, seq = cam.read(seq)
            t = time.monotonic() - start
            people = tracker.detect(bgr[:, ::-1].copy())
            row = [f"{t:.3f}", n, len(people)]
            img = bgr.copy()
            p = pick_person(people)
            if p:
                row += [f"{p['vis'][i]:.2f}" for i in sorted(NAMES)]
                q = retarget(p["world"], p["vis"])
                for s in ("left", "right"):
                    row += [f"{np.degrees(q[s][k]):.0f}" for k in (0, 1, 3)] if s in q else ["", "", ""]
                draw(img, p)
            else:
                row += [""] * (len(cols) - 3)
            wr.writerow(row)
            if n % args.every == 0:
                cv2.putText(img, f"{t:5.1f}s people={len(people)}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                path = os.path.join(args.out, "frames", f"{n:04d}.jpg")
                cv2.imwrite(path, img)
                saved.append(img)
            n += 1

    if saved:
        thumbs = [cv2.resize(s, (320, 240)) for s in saved[:12]]
        thumbs += [np.zeros_like(thumbs[0])] * (-len(thumbs) % 4)
        grid = np.vstack([np.hstack(thumbs[i:i + 4]) for i in range(0, len(thumbs), 4)])
        cv2.imwrite(os.path.join(args.out, "contact.jpg"), grid)
    print(f"{n} frames, {len(saved)} saved -> {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
