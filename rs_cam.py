"""Client for the Jetson's cam_server.py MJPEG stream, plus MediaPipe HandLandmarker (Tasks API)."""
import os
import threading
import time
import urllib.request

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "hand_landmarker.task")
HAND_EDGES = [(c.start, c.end) for c in vision.HandLandmarksConnections.HAND_CONNECTIONS]
DEFAULT_URL = "http://192.168.123.164:8081"


class StreamCamera:
    """Reads the MJPEG stream in a background thread and keeps only the newest frame."""

    def __init__(self, url=DEFAULT_URL):
        self.url = url.rstrip("/")
        self.offset = self._clock_offset()  # jetson_time - laptop_time
        self.cond = threading.Condition()
        self.jpeg, self.ts, self.seq = None, 0.0, 0
        threading.Thread(target=self._loop, daemon=True).start()

    def _clock_offset(self, n=10):
        best = None
        for _ in range(n):
            t0 = time.time()
            remote = float(urllib.request.urlopen(self.url + "/time", timeout=2).read())
            t1 = time.time()
            if best is None or t1 - t0 < best[0]:
                best = (t1 - t0, remote - (t0 + t1) / 2)
        return best[1]

    def _loop(self):
        while True:
            try:
                with urllib.request.urlopen(self.url + "/stream", timeout=5) as r:
                    while True:
                        headers = {}
                        line = r.readline()
                        while line in (b"\r\n", b"\n"):
                            line = r.readline()
                        if not line:
                            break
                        while True:  # boundary line already consumed; read part headers
                            line = r.readline().strip()
                            if not line:
                                break
                            k, _, v = line.partition(b":")
                            headers[k.strip().lower()] = v.strip()
                        jpeg = r.read(int(headers[b"content-length"]))
                        with self.cond:
                            self.jpeg, self.ts, self.seq = jpeg, float(headers[b"x-timestamp"]), self.seq + 1
                            self.cond.notify_all()
            except Exception as e:
                print("stream error, reconnecting:", e, flush=True)
                time.sleep(0.5)

    def read(self, last_seq=None, timeout=2.0):
        """Returns (bgr mirrored horizontally, capture time in laptop clock, seq). Waits for a new frame."""
        with self.cond:
            self.cond.wait_for(lambda: self.jpeg is not None and self.seq != last_seq, timeout)
            jpeg, ts, seq = self.jpeg, self.ts, self.seq
        img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        return cv2.flip(img, 1), ts - self.offset, seq


class HandTracker:
    def __init__(self, num_hands=4):
        opts = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.det = vision.HandLandmarker.create_from_options(opts)
        self.t0 = time.monotonic()

    def detect(self, bgr):
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        res = self.det.detect_for_video(img, int((time.monotonic() - self.t0) * 1000))
        hands = []
        for lm, wlm, hd in zip(res.hand_landmarks, res.hand_world_landmarks, res.handedness):
            px = np.array([[p.x, p.y] for p in lm])
            w, h = px.max(0) - px.min(0)
            hands.append({
                "side": hd[0].category_name.lower(),  # person's own left/right (image is mirrored)
                "score": hd[0].score,
                "px": px,                                          # normalized image coords
                "world": np.array([[p.x, p.y, p.z] for p in wlm]),  # metres, hand-centred
                "area": float(w * h),                              # normalized bbox area
            })
        return hands


def largest_per_side(hands):
    """Largest bounding box wins: at most one hand per side, the biggest (= usually nearest) one."""
    best = {}
    for hd in hands:
        if hd["side"] not in best or hd["area"] > best[hd["side"]]["area"]:
            best[hd["side"]] = hd
    return best


def draw_hand(img, px_norm, color=(0, 255, 0)):
    h, w = img.shape[:2]
    pts = (px_norm * [w, h]).astype(int)
    for a, b in HAND_EDGES:
        cv2.line(img, tuple(pts[a]), tuple(pts[b]), color, 2)
    for p in pts:
        cv2.circle(img, tuple(p), 3, (0, 0, 255), -1)


POSE_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "pose_landmarker_full.task")


class PoseTracker:
    """MediaPipe PoseLandmarker. Feed it the UN-mirrored image so left/right labels are anatomical."""

    def __init__(self, num_poses=3):
        opts = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=POSE_MODEL),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=num_poses,
            min_pose_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.det = vision.PoseLandmarker.create_from_options(opts)
        self.t0 = time.monotonic()

    def detect(self, bgr):
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        res = self.det.detect_for_video(img, int((time.monotonic() - self.t0) * 1000))
        people = []
        for lm, wlm in zip(res.pose_landmarks, res.pose_world_landmarks):
            px = np.array([[p.x, p.y] for p in lm])
            vis = np.array([p.visibility for p in lm])
            torso = px[[11, 12, 23, 24]]
            w, h = torso.max(0) - torso.min(0)
            people.append({
                "px": px,
                "vis": vis,
                "world": np.array([[p.x, p.y, p.z] for p in wlm]),  # metres, hip-centred
                "area": float(w * h),  # torso box, for "largest person wins"
            })
        return people
