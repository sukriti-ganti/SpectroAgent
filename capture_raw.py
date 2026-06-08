import serial
import re
import time
import numpy as np

PORT, BAUD = "COM5", 921600
DURATION = 30

ser = serial.Serial(PORT, BAUD, timeout=1)
ser.reset_input_buffer()
frames = []
t0 = time.time()
print(f"Capturing {DURATION}s of REAL CSI -- move around for variety...")
while time.time() - t0 < DURATION:
    line = ser.readline().decode("utf-8", "ignore").strip()
    if "CSI_DATA" not in line:
        continue
    m = re.search(r'"?\[([^\]]+)\]"?', line)
    if not m:
        continue
    try:
        vals = list(map(int, m.group(1).split(',')))
    except ValueError:
        continue
    frames.append(vals)

lengths = [len(f) for f in frames]
print(f"\nCaptured {len(frames)} real frames")
print(f"Subcarrier-value counts seen: min={min(lengths)} max={max(lengths)}")
np.save("captured_csi.npy", np.array(frames, dtype=object), allow_pickle=True)
print("Saved captured_csi.npy")