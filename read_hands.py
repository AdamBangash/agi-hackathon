"""Read-only Inspire FTP hand state over Modbus TCP (function 0x03 only, no writes)."""
import socket
import struct

HANDS = {"right": "192.168.123.210", "left": "192.168.123.211"}  # factory default mapping
FINGERS = ["pinky", "ring", "middle", "index", "thumbBend", "thumbRot"]
REGS = {"angle_act": 1546, "force_act": 1582, "error": 1606}


def read_regs(sock, addr, count, tid=1):
    sock.sendall(struct.pack(">HHHBBHH", tid, 0, 6, 1, 3, addr, count))
    hdr = sock.recv(9)
    if hdr[7] != 3:
        raise RuntimeError(f"modbus exception {hdr[8]}")
    data = b""
    while len(data) < hdr[8]:
        data += sock.recv(hdr[8] - len(data))
    return struct.unpack(f">{count}h", data)


for side, ip in HANDS.items():
    try:
        with socket.create_connection((ip, 6000), timeout=2) as s:
            ang = read_regs(s, REGS["angle_act"], 6)
            frc = read_regs(s, REGS["force_act"], 6)
            err = [b for w in read_regs(s, REGS["error"], 3) for b in (w & 0xFF, w >> 8)]
        print(f"{side} ({ip}) angle 0-1000 (1000=open):", dict(zip(FINGERS, ang)))
        print(f"{side} force:", dict(zip(FINGERS, frc)), "errors:", err)
    except Exception as e:
        print(f"{side} ({ip}): {e}")
