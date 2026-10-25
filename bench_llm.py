"""Time N chat calls with one image + one tool against an OpenAI-compatible endpoint."""
import base64
import json
import sys
import time
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://192.168.4.227:9379/v1"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "gemma4-e2b"
IMG = sys.argv[3] if len(sys.argv) > 3 else None
N = 5

if IMG is None:
    IMG_BYTES = urllib.request.urlopen("http://192.168.123.164:8081/frame", timeout=3).read()
else:
    IMG_BYTES = open(IMG, "rb").read()
b64 = base64.b64encode(IMG_BYTES).decode()

TOOLS = [{"type": "function", "function": {
    "name": "perform_seal",
    "description": "Make the robot perform a named hand seal with its arms and fingers.",
    "parameters": {"type": "object", "properties": {
        "seal": {"type": "string", "enum": ["tiger", "dragon", "snake", "bird", "rat", "dog"]}},
        "required": ["seal"]}}}]

body = {"model": MODEL, "tools": TOOLS, "max_tokens": 128, "messages": [
    {"role": "system", "content": "You control a humanoid robot. If the person raises a hand, call perform_seal with seal='tiger'. Otherwise reply in one short sentence."},
    {"role": "user", "content": [
        {"type": "text", "text": "Describe the nearest person in one sentence and decide whether to act."},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64}}]}]}

times = []
for i in range(N):
    t0 = time.time()
    req = urllib.request.Request(BASE + "/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    resp = json.loads(urllib.request.urlopen(req, timeout=300).read())
    dt = time.time() - t0
    times.append(dt)
    msg = resp["choices"][0]["message"]
    out = msg.get("tool_calls") or (msg.get("content") or "")[:120]
    print(f"call {i + 1}: {dt:.2f}s usage={resp.get('usage')} -> {out}")
print(f"{MODEL}: mean {sum(times) / N:.2f}s  min {min(times):.2f}s  max {max(times):.2f}s  (first call includes model load)")
