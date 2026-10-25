"""Runs on the G1 Jetson: RealSense RGB (UVC) -> MJPEG over HTTP.

    python3 cam_server.py [--dev /dev/video4] [--port 8081] [--quality 80]

GET /stream  multipart MJPEG; each part carries X-Timestamp (Jetson time.time() at grab)
GET /frame   single JPEG
GET /time    Jetson time.time(), for clock-offset estimation on the client
Python 3.8 compatible.
"""
import argparse
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

BOUNDARY = b"frame"


class Camera:
    def __init__(self, dev, w, h, fps, quality):
        self.cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.cap.isOpened():
            raise RuntimeError("cannot open " + dev)
        self.params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        self.cond = threading.Condition()
        self.jpeg, self.ts, self.seq = None, 0.0, 0
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        n, t_report = 0, time.time()
        while True:
            ok, img = self.cap.read()
            ts = time.time()
            if not ok:
                time.sleep(0.01)
                continue
            ok, buf = cv2.imencode(".jpg", img, self.params)
            with self.cond:
                self.jpeg, self.ts, self.seq = buf.tobytes(), ts, self.seq + 1
                self.cond.notify_all()
            n += 1
            if ts - t_report >= 10:
                print("capture %.1f fps, %d KB/frame" % (n / (ts - t_report), len(buf) // 1024), flush=True)
                n, t_report = 0, ts

    def wait_next(self, last_seq, timeout=1.0):
        with self.cond:
            self.cond.wait_for(lambda: self.seq != last_seq, timeout)
            return self.jpeg, self.ts, self.seq


def make_handler(cam):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def _send(self, body, ctype):
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.startswith("/time"):
                self._send(repr(time.time()).encode(), "text/plain")
            elif self.path.startswith("/frame"):
                jpeg, _, _ = cam.wait_next(-1)
                self._send(jpeg, "image/jpeg")
            elif self.path.startswith("/stream"):
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=" + BOUNDARY.decode())
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()
                self.connection.setsockopt(6, 1, 1)  # TCP_NODELAY
                seq = -1
                try:
                    while True:
                        jpeg, ts, seq = cam.wait_next(seq)
                        if jpeg is None:
                            continue
                        self.wfile.write(b"--" + BOUNDARY + b"\r\nContent-Type: image/jpeg\r\n"
                                         + b"X-Timestamp: %.6f\r\nContent-Length: %d\r\n\r\n" % (ts, len(jpeg))
                                         + jpeg + b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    pass
            else:
                self.send_error(404)

    return Handler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", default="/dev/video4")
    ap.add_argument("--port", type=int, default=8081)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--quality", type=int, default=80)
    args = ap.parse_args()
    cam = Camera(args.dev, args.width, args.height, args.fps, args.quality)
    srv = ThreadingHTTPServer(("0.0.0.0", args.port), make_handler(cam))
    srv.daemon_threads = True
    print("serving http://0.0.0.0:%d/stream from %s" % (args.port, args.dev), flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
