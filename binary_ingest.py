import numpy as np
import struct
import re
import time

frames = list(np.load("captured_csi.npy", allow_pickle=True))

# auto-detect the dominant frame length (most common value count)
lengths = [len(f) for f in frames]
N_VALS = max(set(lengths), key=lengths.count)
N_VALS -= N_VALS % 2          # force even (I/Q pairs)
print(f"Locked frame size: {N_VALS} int16 values  ->  {N_VALS*2} bytes/frame")

# detect int width from value range
flat = np.concatenate([np.array(f, dtype=np.int64) for f in frames])
maxabs = np.abs(flat).max()
code = "b" if maxabs < 128 else "h"
width = 1 if code == "b" else 2
print(f"Max |value| = {maxabs}  ->  using int{width*8}  "
      f"({N_VALS*width} bytes/frame)")

fmt = "<" + code * N_VALS
FRAME = struct.calcsize(fmt)


def fit(vals):
    v = list(vals)[:N_VALS]
    v += [0] * (N_VALS - len(v))
    return v


def variance_from_iq(iq):
    iq = np.asarray(iq, dtype=np.float64)
    imag, real = iq[0::2], iq[1::2]
    amps = np.sqrt(imag ** 2 + real ** 2)
    nz = amps[amps > 0]
    return float(np.var(nz)) if len(nz) >= 5 else 0.0


# STEP 2: pack real frames as little-endian binary
blob = b"".join(struct.pack(fmt, *fit(f)) for f in frames)
with open("csi_stream.bin", "wb") as fh:
    fh.write(blob)
print(f"\nPacked {len(frames)} frames -> {len(blob)} bytes "
      f"({FRAME}-byte frames)")

# STEP 3a: BINARY ingestion (production path)
t0 = time.perf_counter()
bin_vars = []
with open("csi_stream.bin", "rb") as fh:
    raw = fh.read()
for off in range(0, len(raw), FRAME):
    chunk = raw[off:off + FRAME]
    if len(chunk) < FRAME:
        break
    iq = struct.unpack(fmt, chunk)
    bin_vars.append(variance_from_iq(iq))
t_bin = time.perf_counter() - t0

# STEP 3b: ASCII ingestion (legacy path) for comparison
ascii_lines = ['CSI_DATA [' + ','.join(map(str, fit(f))) + ']' for f in frames]
t0 = time.perf_counter()
asc_vars = []
for line in ascii_lines:
    m = re.search(r'\[([^\]]+)\]', line)
    iq = list(map(int, m.group(1).split(',')))
    asc_vars.append(variance_from_iq(iq))
t_asc = time.perf_counter() - t0

n = len(bin_vars)
print("\n================  INGESTION PROOF  ================")
print(f"Frames processed : {n}")
print(f"ASCII  parse     : {t_asc*1e3:8.2f} ms  ({t_asc/n*1e6:6.1f} us/frame)")
print(f"BINARY parse     : {t_bin*1e3:8.2f} ms  ({t_bin/n*1e6:6.1f} us/frame)")
print(f"Speedup          : {t_asc/t_bin:5.1f}x")
print(f"Output identical : {np.allclose(bin_vars, asc_vars)}")
print(f"Frame layout     : {FRAME} bytes, little-endian int{width*8}, "
      f"{N_VALS//2} IQ pairs")