"""Runs on the G1 Jetson next to cam_server.py: sends the current camera frame to Gemma (LiteRT-LM, local)
up to --hz times per second and streams what it says to the terminal.

    python3 brain.py                       # narrate: one short description per frame, streamed token by token
    python3 brain.py --tools               # instead let Gemma pick tools (printed only)
    python3 brain.py --inflight 2 --duration 60

Frames are sampled from cam_server every 1/--hz s. Each request takes the newest frame it hasn't sent yet,
so if Gemma is slower than --hz, stale frames are skipped (never queued). At most --inflight requests are
outstanding at once; with more than one, descriptions are printed whole instead of streamed so they don't interleave.
Nothing here moves the robot: body_mirror.py does the mirroring.
Python 3.8 compatible, stdlib only.
"""
import argparse
import base64
import json
import threading
import time
import urllib.request

TOOLS = [
    {"type": "function", "function": {
        "name": "perform_seal",
        "description": "Make the robot perform a named hand seal with its arms and fingers.",
        "parameters": {"type": "object", "properties": {
            "seal": {"type": "string", "enum": ["tiger", "dragon", "snake", "bird", "rat", "dog"]}},
            "required": ["seal"]}}},
    {"type": "function", "function": {
        "name": "say",
        "description": "Say a short sentence out loud.",
        "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}}},
    {"type": "function", "function": {
        "name": "mirror_on", "description": "Start copying the person's arm and finger movements.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "mirror_off", "description": "Stop copying the person's movements.",
        "parameters": {"type": "object", "properties": {}}}},
]
DESCRIBE = ("You are the eyes of a humanoid robot that is copying the person in front of it. "
            "In one short present-tense sentence, describe what the person is doing with their arms and hands.")
SYSTEM = ("You control a humanoid robot that sees through the camera image. If the person raises a hand, "
          "call perform_seal with seal='tiger'. If they wave, call mirror_on. Otherwise reply in one short sentence.")


class Latest:
    """Newest camera frame, refreshed every period by a background thread."""

    def __init__(self, url, period):
        self.url, self.period = url, period
        self.cond = threading.Condition()
        self.jpeg, self.ts, self.seq, self.errors = None, 0.0, 0, 0
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while True:
            t0 = time.monotonic()
            try:
                jpeg = urllib.request.urlopen(self.url + "/frame", timeout=2).read()
                with self.cond:
                    self.jpeg, self.ts, self.seq = jpeg, time.monotonic(), self.seq + 1
                    self.cond.notify_all()
            except OSError:
                self.errors += 1
            time.sleep(max(0.0, self.period - (time.monotonic() - t0)))

    def newer_than(self, seq):
        with self.cond:
            self.cond.wait_for(lambda: self.seq > seq, timeout=5)
            return self.jpeg, self.ts, self.seq


def ask(base, model, jpeg, max_tokens, timeout):
    body = {"model": model, "tools": TOOLS, "max_tokens": max_tokens, "messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": [
            {"type": "text", "text": "What is the person doing right now? Decide whether to act."},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()}}]}]}
    req = urllib.request.Request(base + "/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())["choices"][0]["message"]


def describe(base, model, jpeg, max_tokens, timeout, on_text):
    """Streams one description of the frame; calls on_text with each new chunk. Returns the full text."""
    body = {"model": model, "stream": True, "max_tokens": max_tokens, "messages": [
        {"role": "system", "content": DESCRIBE},
        {"role": "user", "content": [
            {"type": "text", "text": "What is the person doing right now?"},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()}}]}]}
    req = urllib.request.Request(base + "/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    text = []
    for line in urllib.request.urlopen(req, timeout=timeout):
        line = line.decode().strip()
        if line.startswith("event: response.error"):
            raise ValueError("gemma stream error")
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        chunk = json.loads(line[6:])
        if "error" in chunk:
            raise ValueError(chunk["error"])
        for c in chunk.get("choices", []):
            piece = (c.get("delta") or {}).get("content")
            if piece:
                text.append(piece)
                on_text(piece)
    return "".join(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", default="http://127.0.0.1:8081")
    ap.add_argument("--llm", default="http://127.0.0.1:9379/v1")
    ap.add_argument("--model", default="gemma4-e2b")
    ap.add_argument("--hz", type=float, default=10.0, help="frame sampling and max request rate")
    ap.add_argument("--inflight", type=int, default=1, help="max concurrent Gemma requests")
    ap.add_argument("--tools", action="store_true", help="let Gemma call tools instead of narrating")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--timeout", type=float, default=60)
    ap.add_argument("--duration", type=float, default=0)
    args = ap.parse_args()

    frames = Latest(args.cam, 1.0 / args.hz)
    lock = threading.Lock()
    stats = {"sent": 0, "done": 0, "lat": 0.0, "err": 0}
    next_send = [time.monotonic()]
    start = time.monotonic()

    def worker():
        seq = 0
        while True:
            with lock:  # pace all workers together to at most --hz requests per second
                wait = next_send[0] - time.monotonic()
                next_send[0] = max(next_send[0], time.monotonic()) + 1.0 / args.hz
            if wait > 0:
                time.sleep(wait)
            jpeg, ts, seq = frames.newer_than(seq)
            if jpeg is None:
                continue
            t0 = time.monotonic()
            with lock:
                stats["sent"] += 1
            stream = not args.tools and args.inflight == 1
            if stream:
                print("%6.1fs frame %d: " % (t0 - start, seq), end="", flush=True)
            try:
                if args.tools:
                    msg = ask(args.llm, args.model, jpeg, args.max_tokens, args.timeout)
                else:
                    text = describe(args.llm, args.model, jpeg, args.max_tokens, args.timeout,
                                    lambda piece: stream and print(piece, end="", flush=True))
            except (OSError, ValueError, KeyError) as e:
                if stream:
                    print()
                with lock:
                    stats["err"] += 1
                print("%6.1fs ERROR %s" % (t0 - start, e), flush=True)
                continue
            dt = time.monotonic() - t0
            with lock:
                stats["done"] += 1
                stats["lat"] += dt
            if stream:
                print("  (%.1fs)" % dt, flush=True)
                continue
            if args.tools:
                calls = [(c["function"]["name"], c["function"].get("arguments")) for c in msg.get("tool_calls") or []]
                out = "TOOLS %s" % calls if calls else repr((msg.get("content") or "").strip()[:160])
            else:
                out = text.strip()
            print("%6.1fs frame %d (age %.2fs) took %.2fs -> %s"
                  % (t0 - start, seq, t0 - ts + dt, dt, out), flush=True)

    for _ in range(args.inflight):
        threading.Thread(target=worker, daemon=True).start()
    print("brain (%s): %s %s, %.0f Hz target, %d in flight"
          % ("tools" if args.tools else "narrate", args.llm, args.model, args.hz, args.inflight), flush=True)
    try:
        while not args.duration or time.monotonic() - start < args.duration:
            time.sleep(5)
            if not args.tools and args.inflight == 1:
                continue  # stats would land mid-sentence in the streamed narration
            el = time.monotonic() - start
            with lock:
                s = dict(stats)
            print("-- %.0fs: %.2f req/s answered, mean latency %.2fs, %d errors, %d frame-grab errors"
                  % (el, s["done"] / el, s["lat"] / max(1, s["done"]), s["err"], frames.errors), flush=True)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
