import numpy as np
from collections import deque


class SNRAgent:
    """
    Signal-conditioning agent (AMPLIFIER). Runs BEFORE perception.
    Hampel outlier rejection -> per-subcarrier SNR -> maximal-ratio
    combining -> adaptive coherent integration (window shrinks with risk).
    """

    def __init__(self, window=8, hampel_k=3.0):
        self.window = window
        self.hampel_k = hampel_k
        self.buf = deque(maxlen=window)
        self.noise_floor = None
        self.N = None
        self.calib = []

    def _fit(self, amps):
        amps = np.asarray(amps, dtype=float)
        if self.N is None:
            self.N = len(amps)
        if len(amps) >= self.N:
            return amps[:self.N]
        return None

    def hampel(self, amps):
        med = np.median(amps)
        mad = np.median(np.abs(amps - med)) + 1e-9
        thr = self.hampel_k * 1.4826 * mad
        out = amps.copy()
        out[np.abs(amps - med) > thr] = med
        return out

    def calibrate(self, amps):
        a = self._fit(amps)
        if a is not None:
            self.calib.append(self.hampel(a))

    def finish_calibration(self):
        M = np.vstack(self.calib)
        self.noise_floor = np.var(M, axis=0) + 1e-9
        self.calib = []

    def push(self, amps):
        a = self._fit(amps)
        if a is not None:
            self.buf.append(self.hampel(a))

    def feature(self, risk=0.0):
        eff = max(2, int(self.window * (1.0 - 0.6 * risk)))
        if len(self.buf) < 2 or self.noise_floor is None:
            return 0.0, 0.0
        M = np.vstack(list(self.buf)[-eff:])
        sig = np.var(M, axis=0)
        snr_k = sig / self.noise_floor
        total = snr_k.sum() + 1e-9
        w = snr_k / total
        feature = float(np.sum(w * sig))
        snr_out_db = 10.0 * np.log10(total)
        return feature, snr_out_db

    def raw_feature(self, amps):
        a = np.asarray(amps, dtype=float)
        nz = a[a > 0]
        return float(np.var(nz)) if len(nz) >= 5 else 0.0

    def per_subcarrier_snr(self):
        if self.noise_floor is None or len(self.buf) < 2:
            return None
        sig = np.var(np.vstack(self.buf), axis=0)
        return sig / self.noise_floor