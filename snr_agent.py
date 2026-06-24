import numpy as np
from collections import deque


class SNRAgent:
    """
    Signal-conditioning agent. Runs BEFORE perception.
    Hampel outlier rejection -> per-subcarrier SNR weighting (maximal-ratio
    combining) -> adaptive integration (window shrinks as risk rises).

    IMPORTANT: the underlying feature is sum-of-squares ENERGY per subcarrier
    (I^2 + Q^2), summed across subcarriers -- the SAME quantity the FPGA
    computes. The agent conditions that energy; it does not change its meaning.
    """

    def __init__(self, window=8, hampel_k=3.0):
        self.window = window
        self.hampel_k = hampel_k
        self.buf = deque(maxlen=window)      # holds recent per-subcarrier energy
        self.noise_floor = None
        self.N = None
        self.calib = []

    def _fit(self, amps):
        """amps = per-subcarrier amplitude array. Returns fixed-length or None."""
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

    @staticmethod
    def _energy(amps):
        """Per-subcarrier energy = amplitude^2 = I^2 + Q^2."""
        return np.asarray(amps, dtype=float) ** 2

    def calibrate(self, amps):
        a = self._fit(amps)
        if a is not None:
            self.calib.append(self._energy(self.hampel(a)))

    def finish_calibration(self):
        M = np.vstack(self.calib)               # rows = still-frame energies
        # noise floor = how much each subcarrier's energy wobbles when still
        self.noise_floor = np.var(M, axis=0) + 1e-9
        self.calib = []

    def push(self, amps):
        a = self._fit(amps)
        if a is not None:
            self.buf.append(self._energy(self.hampel(a)))

    def feature(self, risk=0.0):
        """SNR-conditioned scalar + combined SNR in dB.
        Window shrinks as risk rises (faster response under threat)."""
        eff = max(2, int(self.window * (1.0 - 0.6 * risk)))
        if len(self.buf) < 2 or self.noise_floor is None:
            return 0.0, 0.0
        M = np.vstack(list(self.buf)[-eff:])    # recent energies
        sig = np.mean(M, axis=0)                 # mean energy per subcarrier
        snr_k = sig / self.noise_floor           # per-subcarrier SNR
        total = snr_k.sum() + 1e-9
        w = snr_k / total                        # MRC weights
        feature = float(np.sum(w * sig))         # SNR-weighted energy
        snr_out_db = 10.0 * np.log10(total)
        return feature, snr_out_db

    def raw_feature(self, amps):
        """Unconditioned sum-of-squares energy -- Sigma x^2. Matches the FPGA."""
        a = np.asarray(amps, dtype=float)
        return float(np.sum(a ** 2))

    def per_subcarrier_snr(self):
        if self.noise_floor is None or len(self.buf) < 2:
            return None
        sig = np.mean(np.vstack(self.buf), axis=0)
        return sig / self.noise_floor
