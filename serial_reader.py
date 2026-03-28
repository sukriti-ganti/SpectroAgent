import serial
import numpy as np
import re
from collections import deque

SERIAL_PORT = "COM5"
BAUD_RATE = 921600

def parse_csi_amplitude_variance(line):
    try:
        match = re.search(r'"?\[([^\]]+)\]"?', line)
        if not match:
            return None
        values = list(map(int, match.group(1).split(',')))
        if len(values) < 4:
            return None
        imag = values[0::2]
        real = values[1::2]
        amps = [np.sqrt(i**2 + r**2) for i, r in zip(imag, real)]
        nonzero = [a for a in amps if a > 0]
        if len(nonzero) < 5:
            return None
        return float(np.var(nonzero))
    except:
        return None

def read_stream():
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=2)
    ser.reset_input_buffer()
    print("🔌 Connected\n⏳ Collecting baseline — stay still...\n")

    baseline = []
    baseline_mean = None
    baseline_std = None

    while True:
        try:
            ser.reset_input_buffer()
            raw = ser.readline().decode("utf-8", errors="ignore").strip()
            if "CSI_DATA" not in raw:
                continue
            v = parse_csi_amplitude_variance(raw)
            if v is None:
                continue

            if baseline_mean is None:
                baseline.append(v)
                print(f"  {len(baseline)}/50", end="\r")
                if len(baseline) >= 50:
                    baseline_mean = np.mean(baseline)
                    baseline_std = np.std(baseline) + 1e-9
                    print(f"\n✅ Baseline locked — mean:{baseline_mean:.2f} std:{baseline_std:.2f}\n")
                continue

            z = (v - baseline_mean) / baseline_std
            yield v, z, baseline_mean, baseline_std

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
            continue