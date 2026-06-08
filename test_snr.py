import serial
import re
import time
import numpy as np
from snr_agent import SNRAgent

PORT, BAUD = "COM5", 921600
ser = serial.Serial(PORT, BAUD, timeout=1)
ser.reset_input_buffer()
snr = SNRAgent(window=8)


def next_amps():
    while True:
        raw = ser.readline().decode("utf-8", "ignore").strip()
        if "CSI_DATA" not in raw:
            continue
        m = re.search(r'"?\[([^\]]+)\]"?', raw)
        if not m:
            continue
        try:
            v = list(map(int, m.group(1).split(',')))
        except ValueError:
            continue
        if len(v) < 8:
            continue
        imag, real = np.array(v[0::2], float), np.array(v[1::2], float)
        return np.sqrt(imag ** 2 + real ** 2)


def collect(seconds, sink_raw, sink_cond, calib=False):
    t0 = time.time()
    while time.time() - t0 < seconds:
        amps = next_amps()
        if calib:
            snr.calibrate(amps)
            continue
        snr.push(amps)
        f, _ = snr.feature(risk=0.0)
        sink_cond.append(f)
        sink_raw.append(snr.raw_feature(amps))


print("Phase 1/3  STAY STILL (calibration) - 8s")
collect(8, [], [], calib=True)
snr.finish_calibration()

still_raw, still_cond = [], []
print("Phase 2/3  STAY STILL (baseline capture) - 8s")
collect(8, still_raw, still_cond)

evt_raw, evt_cond = [], []
print("Phase 3/3  MOVE / wave your hand - 8s")
collect(8, evt_raw, evt_cond)


def out_snr_db(still, evt):
    still, evt = np.array(still), np.array(evt)
    sep = evt.mean() - still.mean()
    noise = still.std() + 1e-9
    d = sep / noise
    return 10 * np.log10(d ** 2 + 1e-9), d


snr_raw_db, d_raw = out_snr_db(still_raw, evt_raw)
snr_cond_db, d_cond = out_snr_db(still_cond, evt_cond)

print("\n================  SNR PROOF  ================")
print(f"Raw  (1 snapshot, equal weight): SNR={snr_raw_db:6.2f} dB  d'={d_raw:5.2f}")
print(f"SNR agent (MRC + integration):   SNR={snr_cond_db:6.2f} dB  d'={d_cond:5.2f}")
print(f"GAIN:                           +{snr_cond_db - snr_raw_db:5.2f} dB")
np.savez("snr_proof.npz", still_raw=still_raw, still_cond=still_cond,
         evt_raw=evt_raw, evt_cond=evt_cond, snr_db=per := snr.per_subcarrier_snr())
print("Saved snr_proof.npz")