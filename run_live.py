import serial
import re
import time
import numpy as np

from perception_agent import PerceptionAgent
from reasoning_agent import ReasoningAgent
from nexus_jumeau import NexusJumeau
from resource_allocation_agent import ResourceAllocationAgent

PORT, BAUD = "COM5", 921600


def parse_variance(line):
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
    amps = np.sqrt(imag ** 2 + real ** 2)
    nz = amps[amps > 0]
    return float(np.var(nz)) if len(nz) >= 5 else None


perc = PerceptionAgent()
reas = ReasoningAgent()
nexus = NexusJumeau()
res = ResourceAllocationAgent()

ser = serial.Serial(PORT, BAUD, timeout=1)
try:
    print("Flushing buffer...")
    ser.reset_input_buffer()
    time.sleep(0.5)
    ser.reset_input_buffer()

    print("Warming up...")
    discarded = 0
    while discarded < 30:
        raw = ser.readline().decode("utf-8", "ignore").strip()
        if "CSI_DATA" in raw and parse_variance(raw) is not None:
            discarded += 1

    print("\n>>> STAY COMPLETELY STILL for baseline <<<\n")
    time.sleep(1)

    baseline = []
    base_mean = base_std = None

    print(f"{'Variance':>12}  {'Z':>6}  {'Raw':>9}  {'Risk':>5}  {'Next':>8}  State")
    print("-" * 70)

    while True:
        ser.reset_input_buffer()          # jump to newest data, drop backlog
        raw = ser.readline().decode("utf-8", "ignore").strip()
        if "CSI_DATA" not in raw:
            continue
        v = parse_variance(raw)
        if v is None:
            continue

        if base_mean is None:
            baseline.append(v)
            print(f"  baseline {len(baseline)}/50", end="\r")
            if len(baseline) >= 50:
                base_mean = np.mean(baseline)
                base_std = np.std(baseline) + 1e-9
                cv = base_std / (base_mean + 1e-9)
                if cv > 0.5:
                    print(f"\nBaseline too noisy (CV={cv:.2f}) -- "
                          f"you moved during calibration. Restarting...\n")
                    baseline = []
                    base_mean = base_std = None
                    continue
                print(f"\nBaseline locked  (mean={base_mean:.0f}, "
                      f"std={base_std:.0f}, CV={cv:.2f})\n")
            continue

        z = (v - base_mean) / base_std
        raw_class = perc.classify(v, z)
        reas.add(raw_class)
        state, risk = reas.get_confirmed_state()
        nexus.update(state)
        pred, conf = nexus.predict_next(state)

        print(f"{v:>12,.0f}  {z:>6.2f}  {raw_class:>9}  "
              f"{risk:>5.2f}  {pred:>8}  {state}")

except KeyboardInterrupt:
    print("\nStopped by user")
finally:
    ser.close()
    print("Port COM5 released")