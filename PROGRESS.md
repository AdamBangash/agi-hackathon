# hackAGI: G1 mirror + Gemma, progress log

Google DeepMind hackathon, 2026-10-24. Unitree G1 EDU with Inspire FTP hands, a RealSense D435i, and a Jetson Thor running Gemma.

## Status at a glance

| Piece | Status |
|---|---|
| Laptop ↔ G1 network, key-based SSH, project sync | Working |
| Robot state readout (arms, IMU, mode) | Working |
| Inspire hands: read and control over Modbus TCP | Working |
| RealSense on the robot Jetson → MJPEG stream to the laptop | Working, ~35 ms grab→landmarks |
| Finger mirroring (`finger_mirror.py --live`) | Working, demoed 3 × 30 s |
| Arm + finger mirroring, robot seated (`body_mirror.py --backend lowcmd`) | Working, demoed 1 × 30 s |
| Arm mirroring through `arm_sdk` (needs a standing robot) | Built, not usable while seated |
| Gemma 4 E2B via LiteRT-LM on the Thor (:9379) | Working, ~11 s/call (CPU) |
| Gemma 4 26B via llama.cpp on the Thor (:8000) | Built, not started yet |
| `brain.py`, `poses.yaml`, `config.yaml` | Not written yet |

## Machines and addresses

| Machine | Address | Login | Role |
|---|---|---|---|
| Laptop | `192.168.123.222` on `enx6c6e071c419f` (NetworkManager profile `g1-enx6c6e`) | — | Source of truth for code; runs vision + control |
| G1 dev Jetson (Orin NX) | `192.168.123.164` | `unitree@` (key auth) | Camera server, `~/hackAGI` mirror |
| G1 locomotion PC | `192.168.123.161` | SSH closed | Unitree's motion controller (not accessible) |
| Inspire hand (right*) | `192.168.123.210:6000` | Modbus TCP | Fingers |
| Inspire hand (left*) | `192.168.123.211:6000` | Modbus TCP | Fingers |
| Lidar (probably) | `192.168.123.120` | — | Not used |
| Jetson Thor | `192.168.4.227` (Wi-Fi network) | `jetson@` (key auth) | Gemma inference |

\* The left/right mapping is the factory default and is **still unconfirmed** (the vendor repo contradicts itself). Check by running `python hands.py test 210 index` and seeing which hand moves. If it's swapped, flip the two IPs in `HANDS` in `hands.py`.

### Robot Jetson facts
- JetPack 5.1.1 (L4T R35.3.1), Ubuntu 20.04, Python 3.8.10
- `unitree_sdk2py` preinstalled (`~/unitree_sdk2_python`)
- 1.7 TB disk free, 15 GB RAM
- `uv` + `litert-lm` 0.18.1 installed (uv-managed Python 3.11)
- tmux installed
- RealSense D435i enumerates as a UVC camera. **RGB = `/dev/video4`** (YUYV 640x480 at 30 fps); `/dev/video0` is depth (Z16).

### Robot state, as last observed
- Motion service `ai`, **loco FSM 0 = zero torque**. The robot is limp and slumped in a seat: pelvis pitched ~41° back, knees ~82°.
- That's why `arm_sdk` does nothing right now: the loco controller ignores it in zero torque.
- There's no gantry, so the team decided **not** to stand it up.

## Files (all in `~/gemma_thor`, mirrored to `unitree@192.168.123.164:~/hackAGI`)

| File | Runs on | What it does |
|---|---|---|
| `sync_to_robot.sh` | laptop | Polls every 2 s and rsyncs the project to the robot (`--delete`, laptop wins). Runs only while started. |
| `read_state.py` | laptop | Read-only: `rt/lowstate` arm joints + IMU at 1 Hz for 5 s, motion-switcher mode. Subscribes only. |
| `read_hands.py` | laptop | Read-only Modbus dump of both hands (angle, force, errors). |
| `hands.py` | laptop | **Hand control library + CLI**: `set_fingers(hand, [pinky, ring, middle, index, thumbBend, thumbRot])`, 0 = closed … 1000 = open. Has a rate limit, a slew limit and a force cap. |
| `cam_server.py` | robot Jetson | RealSense RGB (UVC) → MJPEG over HTTP on port 8081, with a capture timestamp per frame. |
| `rs_cam.py` | laptop | Stream client (latest frame only, clock-offset corrected), MediaPipe `HandTracker` + `PoseTracker`, and the "largest box wins" person selection. |
| `cam_test.py` | laptop | Camera window with hand landmarks and live latency (`--no-window` prints only). |
| `finger_mirror.py` | laptop | Fingers only. Dry run by default, `--live` writes to the hands. |
| `calibrate.py` | laptop | Per-hand finger calibration → `calib.json`. **The current version uses timed phases, which failed; it needs rework** (see next steps). |
| `body_mirror.py` | laptop | Arms (Pose) + fingers (Hands). Backends `arm_sdk` (robot standing) or `lowcmd` (robot seated). Dry run by default. |
| `bench_llm.py` | laptop | Times N chat calls with one camera frame + one tool against an OpenAI-compatible endpoint. |
| `thor_setup.sh` | Thor | Installs CUDA 13.2 + cmake, builds llama.cpp with CUDA (`~/llama.cpp/build/bin/llama-server`). Already run. |
| `start_gemma.sh` | Thor | Starts llama-server with the cached Gemma 4 26B-A4B q4_0 GGUF + mmproj on port 8000. Not run yet. |
| `models/` | laptop | `hand_landmarker.task`, `pose_landmarker_full.task` (MediaPipe Tasks models). |

## How it works

```
RealSense D435i ──USB──▶ G1 Jetson .164
                          cam_server.py (tmux "cam"): /dev/video4 → JPEG → http://…:8081/stream
                                │ Ethernet
                                ▼
                         Laptop
                          rs_cam.StreamCamera   newest frame only, mirrored horizontally
                          MediaPipe Hands       21 3-D landmarks per hand + handedness
                          MediaPipe Pose        33 3-D landmarks (fed the UN-mirrored image)
                          finger_mirror.flexion → 6 values per hand, 0..1000, EMA-smoothed
                          body_mirror.retarget  → shoulder pitch / roll, elbow per arm
                                │                                │
                   Modbus TCP (hands.py)            DDS (unitree_sdk2py)
                                ▼                                ▼
                  Inspire hands .210/.211         rt/arm_sdk  (loco controller active)
                                                  rt/lowcmd   (seated, service released)
```

### Fingers
- **Flexion.** For each finger, the bend angles at MCP, PIP and DIP are summed from MediaPipe's 3D world landmarks. That sum is mapped linearly from an (open, closed) pair to 1000…0.
  - Thumb bend uses the MCP and IP joints.
  - Thumb rotation is the distance from the thumb tip to the pinky knuckle divided by palm length.
  - The defaults in `DEFAULT_CAL` (fingers 25°→220°) work. Open hands read ~800–1000 and fists read ~0. Thumbs only reach ~100–300 when closed.
- **Mirroring.** The frame is flipped horizontally so MediaPipe's handedness matches the person's own hands. The person's left hand drives the robot's left hand.
- **Several people in view.** The largest hand bounding box per side wins.
- **Smoothing.** An EMA with α = 0.4 runs in a 15 Hz loop. If a hand disappears, the robot hand holds its last position.
- **`hands.py` safety.** These are written on connect:
  - `FORCE_SET` = 1000. The factory default is 2000. A cap of 500 was too low because resting force readings are ~460, so fingers stopped immediately.
  - `SPEED_SET` = 600.
  - Writes are coalesced to at most 30 Hz, and each finger moves at most 200 units per write.
- **Modbus registers used:**

| Register | Address |
|---|---|
| ANGLE_SET | 1486 |
| FORCE_SET | 1498 |
| SPEED_SET | 1522 |
| ANGLE_ACT | 1546 |
| FORCE_ACT | 1582 |
| ERROR | 1606 |
| STATUS | 1612 |
| DEFAULT_FORCE_SET (power-on) | 1044 |

### Arms
- **Retargeting.** A torso frame (forward, left, up) is built from the shoulders and hips. The upper-arm direction gives:
  - shoulder roll = `asin(left component)`
  - shoulder pitch = `atan2(-forward, -up)`, faded to 0 when the arm is straight out to the side, where pitch is undefined
  - elbow = the angle between the upper arm and the forearm
- **Not yet driven.** Shoulder yaw and all wrist joints are held at 0.
- **Validated on synthetic poses:**

| Pose | Result |
|---|---|
| Hanging | 0 / 0 / 0 |
| Forward | pitch −90° |
| Out to the side | roll ±90° |
| Curl | elbow 90° |
| Overhead | pitch clamped at −138° |

- **Workspace limits (rad):**

| Joint | Range |
|---|---|
| Shoulder pitch | −2.4 … 0.3 |
| Left shoulder roll | −0.1 … 1.8 |
| Right shoulder roll | −1.8 … 0.1 |
| Elbow | 0 … 1.9 |

- **Velocity clamp:** 1.5 rad/s per joint, 50 Hz control thread, kp 60, kd 1.5.
- **Person lost for more than 2 s:** the arms go back to the pose captured at start.
- **`arm_sdk` backend.** Follows `g1_arm7_sdk_dds_example`:
  - The weight (motor 29) ramps 0 → 1 over 2 s while holding the current pose.
  - The arms then track.
  - On exit they return to the start pose and the weight ramps 1 → 0 over 1.5 s.
  - The waist is held at its start position.
  - **Requires the loco controller to be active (standing).** It is ignored in FSM 0.
- **`lowcmd` backend** (seated robot):
  1. `MotionSwitcher.ReleaseMode()` turns off the `ai` service.
  2. `rt/lowcmd` is published with arm kp ramped 0 → 60. Legs and waist get kp 0, kd 1 (damping only).
  3. On exit, the arms return to the start pose and kp ramps to 0.
  4. `SelectMode('ai')` restores the service.
  - If the script crashes mid-run, the motors keep the last command and `ai` stays off. Restore with the snippet under Commands, or reboot.

### Latency
- **Grab on the Jetson → landmarks on the laptop:** median 33–35 ms, p95 ~43 ms (clock offset corrected through `/time`).
- **Inside the mirror loops:** 40–90 ms.
- **Not measured:** sensor exposure + USB, ~30–60 ms more. Estimated camera→command total is ~70–150 ms, under the 150 ms target.

### Gemma on the Thor
- **LiteRT-LM (working).**
  - `litert-lm serve --host 0.0.0.0 --port 9379` runs in tmux session `gemma`. Model id `gemma4-e2b` (2.4 GB, from `litert-community/gemma-4-E2B-it-litert-lm`).
  - Benchmark: 5 calls, each with one 640x480 camera frame (404 prompt tokens) and the `perform_seal` tool declared. **Mean 11.0 s** (9.5–13.7 s); answers were correct.
  - That's almost certainly the CPU backend.
  - LiteRT-LM currently supports only Gemma 4 E2B/E4B.
- **llama.cpp (built, not started).**
  - CUDA 13.2 and llama.cpp are built on the Thor. The 26B-A4B q4_0 GGUF + vision projector are cached in `~/.cache/huggingface`.
  - `start_gemma.sh` serves it on :8000 as an OpenAI-compatible API.

## Results log
1. **Robot read-out.** Arms and IMU read fine. Mode `ai`, `mode_machine=5`.
2. **Hand discovery.** They're not on USB (no ttyUSB on .164; .161 is closed). They're the Ethernet FTP model at .210/.211.
3. **One-finger test (.210 index → 500 → back).**
   - The first attempt didn't move: the force cap of 500 was below the resting force reading.
   - With the cap at 1000 it worked: 1000 → 499 → 966.
4. **Finger mirroring live, 3 × 30 s.** Clean open/close tracking, independent fingers (a two-finger sign), no hand errors.
5. **Calibration with timed open/fist phases: failed.** The fist phase was mostly open hands, so `calib.json` was deleted. Defaults are in use.
6. **`arm_sdk` live, 30 s.** No motion: the robot is in FSM 0, zero torque. Diagnosed with the loco GetFsmId RPC.
7. **`lowcmd` live, seated, 30 s.**
   - Arms tracked. Elbows were within a few degrees; shoulders sagged 5–15° under gravity.
   - The `ai` service was restored afterwards.
   - Tracking often dropped out with Pose + Hands running together.
   - The ramp-in was effectively skipped (timer bug, now fixed, not yet re-tested live).

## Commands

All laptop Python commands use the `tv` conda env: `~/miniconda/envs/tv/bin/python` (or `conda activate tv`). Run them from `~/gemma_thor`.

### Network and sync
```bash
nmcli con up g1-enx6c6e                       # static 192.168.123.222 on the USB Ethernet
ping 192.168.123.164
ssh unitree@192.168.123.164                    # key auth
./sync_to_robot.sh                             # keep running; mirrors to ~/hackAGI (use --once for a single sync)
```

### Robot state and hands (read-only)
```bash
python read_state.py enx6c6e071c419f           # arms + IMU + mode, 5 s
python hands.py read                           # both hands: angle / force / error
```

### Hand control
```bash
python hands.py test 210 index --to 500 --hold 2   # one finger, then restore
python -c "from hands import get_hand; get_hand('right').move_to([1000]*6)"   # open right hand
```

### Camera
```bash
# on the robot (already running in tmux session "cam")
ssh unitree@192.168.123.164 'tmux new -d -s cam "cd ~/hackAGI && python3 cam_server.py"'
ssh unitree@192.168.123.164 'tmux attach -t cam'          # view its log
curl -o frame.jpg http://192.168.123.164:8081/frame       # single frame
# on the laptop
python cam_test.py                      # window + landmarks + latency
python cam_test.py --no-window --duration 10
```

### Finger mirroring
```bash
python finger_mirror.py --duration 10            # dry run: prints 6 values per hand
python finger_mirror.py --live --duration 30     # drive the hands
python finger_mirror.py --live --show            # with camera window, Ctrl+C to stop
```

### Arm + finger mirroring
```bash
python body_mirror.py --duration 15                                         # dry run (reads lowstate only)
python body_mirror.py --backend lowcmd --arms-live --fingers-live --duration 30   # SEATED robot
python body_mirror.py --arms-live --fingers-live --duration 30              # STANDING robot (arm_sdk)
```
Before any arm run: the robot must be stable, nobody within ~0.6 m of the arms, and someone on the remote. Stop with Ctrl+C (graceful return + release). Don't kill it with `-9`.

Restore the motion service manually if a `lowcmd` run crashed:
```bash
python -c "
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
ChannelFactoryInitialize(0,'enx6c6e071c419f'); m=MotionSwitcherClient(); m.SetTimeout(5); m.Init()
print(m.SelectMode('ai')); print(m.CheckMode())"
```

Read the loco FSM (0 = zero torque):
```bash
python -c "
import json
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
ChannelFactoryInitialize(0,'enx6c6e071c419f'); c=LocoClient(); c.SetTimeout(3); c.Init()
print(c._Call(7001, json.dumps({})))"
```

### Gemma
```bash
# LiteRT-LM E2B (already running in tmux "gemma" on the Thor)
ssh jetson@192.168.4.227 'tmux new -d -s gemma "~/.local/bin/litert-lm serve --host 0.0.0.0 --port 9379"'
curl http://192.168.4.227:9379/v1/models
python3 bench_llm.py http://192.168.4.227:9379/v1 gemma4-e2b       # 5 timed calls, image + tool

# llama.cpp 26B (built, not started)
ssh jetson@192.168.4.227 'bash ~/start_gemma.sh; tail -f ~/gemma_server.log'
python3 bench_llm.py http://192.168.4.227:8000/v1 gemma-4-26b-a4b-it
```

## Next steps (recommended)

1. **Confirm the hand left/right mapping.** Run `python hands.py test 210 index`, watch which hand moves, and fix `HANDS` in `hands.py` if needed.
2. **Re-test seated arm mirroring with better framing.**
   - Sit him ~1.5 m back with his whole upper body in view, or tilt the camera down.
   - Do a dry run with big arm raises (overhead, out to the side) to check those angles on a real person.
   - Then run 30 s live again to confirm the ramp-in fix.
3. **Reduce tracking dropouts.** Options:
   - Run Pose every frame and Hands every other frame, or crop the hand regions from the Pose wrists.
   - Use `pose_landmarker_lite`.
   - Lower `min_tracking_confidence`.
4. **Shoulder yaw + wrists.** Compute yaw from the forearm direction relative to the upper arm, so "goalpost" arms point up instead of forward. Then add wrist roll from the hand orientation.
5. **Shoulder sag.** Add gravity-compensation feed-forward torque (or kp ~80 on the shoulder pitch joints only) so the shoulders stop sagging 5–15°.
6. **Rework finger calibration.** Make it untimed: record 10 s of repeated open/close and take the 5th/95th percentile per finger. Mainly to get full thumb range.
7. **Speed up Gemma.**
   - Try the LiteRT-LM GPU backend (model id `gemma4-e2b,gpu`, or the vision/backend flags of `litert-lm run`).
   - Start the 26B on llama.cpp (`start_gemma.sh`) and benchmark it with `bench_llm.py`.
   - Pick whichever is fast enough for a 1 Hz `brain.py`.
8. **Write `config.yaml`** with the chosen endpoint (base URL, model id, timeout).
9. **Write `brain.py` and `poses.yaml`.**
   - `brain.py` sends the current frame to the endpoint about once per second, with tools `perform_seal`, `say`, `mirror_on`, `mirror_off`.
   - `poses.yaml` holds 6 hand seals as arm joints + finger arrays, executed through `body_mirror`'s controller and `hands.py`.
10. **Standing / `arm_sdk` path.** Only once a gantry or straps are available: stand it with the remote, then use `--backend arm_sdk`.
11. **Housekeeping.**
    - The background `sync_to_robot.sh` stops when the Claude session ends, so restart it manually.
    - `hands.py` leaves the hands' force threshold at 1000 until a power cycle.
