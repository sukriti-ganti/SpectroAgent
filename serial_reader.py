import serial
import numpy as np
import re

SERIAL_PORT = "COM5"
BAUD_RATE = 921600


def parse_amps(line):
    """Return per-subcarrier amplitude array, or None."""
    m = re.search(r'"?\[([^\]]+)\]"?', line)
    if not m:
        return None
    try:
        vals = list(map(int, m.group(1).split(',')))
    except ValueError:
        return None
    if len(vals) < 8:
        return None
    imag = np.array(vals[0::2], dtype=float)
    real = np.array(vals[1::2], dtype=float)
    return np.sqrt(imag ** 2 + real ** 2)


def variance_of(amps):
    nz = amps[amps > 0]
    return float(np.var(nz)) if len(nz) >= 5 else None


def read_stream(yield_amps=False):
    """
    Backward-compatible: yields (variance, z, mean, std).
    If yield_amps=True: yields (variance, z, mean, std, amps).
    """
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=2)
    ser.reset_input_buffer()
    print("Connected. Collecting baseline -- stay still...\n")

    baseline = []
    baseline_mean = baseline_std = None

    while True:
        try:
            ser.reset_input_buffer()
            raw = ser.readline().decode("utf-8", errors="ignore").strip()
            if "CSI_DATA" not in raw:
                continue
            amps = parse_amps(raw)
            if amps is None:
                continue
            v = variance_of(amps)
            if v is None:
                continue
            if baseline_mean is None:
                baseline.append(v)
                print(f"  {len(baseline)}/50", end="\r")
                if len(baseline) >= 50:
                    baseline_mean = np.mean(baseline)
                    baseline_std = np.std(baseline) + 1e-9
                    print("\nBaseline locked\n")
                continue
            z = (v - baseline_mean) / baseline_std
            if yield_amps:
                yield v, z, baseline_mean, baseline_std, amps
            else:
                yield v, z, baseline_mean, baseline_std
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
            continue