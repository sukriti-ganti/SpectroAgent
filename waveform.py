"""
SpectroAgent  -  FULL FIVE-AGENT DASHBOARD  (deeper-session version)
====================================================================
All five agents live: SNR conditioning, Perception, Reasoning,
Nexus Jumeau (Markov), Resource Allocation. Threaded serial read so the
plot never lags. Grey = raw sum-of-squares energy (matches FPGA),
pink = SNR-conditioned, green = risk score.

RUN:  python full_dashboard.py
"""

import sys, re, threading
import numpy as np
import serial
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

from snr_agent import SNRAgent
from perception_agent import PerceptionAgent
from reasoning_agent import ReasoningAgent
from nexus_jumeau import NexusJumeau
from resource_allocation_agent import ResourceAllocationAgent

# ---------------- CONFIG ----------------
PORT, BAUD  = "COM5", 921600
N           = 240
CALIB_TARGET = 50      # SNR calibration frames
BASE_TARGET  = 20      # baseline (z-score) frames
# ----------------------------------------

# shared snapshot between reader thread and GUI
_data = {"raw": 0.0, "cond": 0.0, "risk": 0.0,
         "state": "CLEAR", "perc": "CLEAR", "pred": "CLEAR",
         "conf": 0.0, "mode": "LOW POWER", "snr": 0.0,
         "phase": "calib_snr", "calib": 0, "base": 0}
_lock = threading.Lock()
_stop = threading.Event()


def parse_amps(line):
    """One CSI line -> per-subcarrier amplitude array, or None."""
    if "CSI_DATA" not in line:
        return None
    m = re.search(r'"?\[([^\]]+)\]"?', line)
    if not m:
        return None
    try:
        v = list(map(int, m.group(1).split(',')))
    except ValueError:
        return None
    if len(v) < 8:
        return None
    imag = np.array(v[0::2], dtype=float)
    real = np.array(v[1::2], dtype=float)
    return np.sqrt(imag ** 2 + real ** 2)


def reader():
    """Background thread: read + run agents. GUI just displays the snapshot."""
    try:
        ser = serial.Serial(PORT, BAUD, timeout=1)
    except Exception as e:
        print(f"Could not open {PORT}: {e}")
        _stop.set()
        return
    ser.reset_input_buffer()

    snr   = SNRAgent(window=8)
    perc  = PerceptionAgent()
    reas  = ReasoningAgent()
    nexus = NexusJumeau()
    res   = ResourceAllocationAgent()

    calib_n = 0
    base_feats = []
    base_mean = base_std = None
    last_risk = 0.0

    while not _stop.is_set():
        try:
            line = ser.readline().decode("utf-8", "ignore")
        except Exception:
            continue
        amps = parse_amps(line)
        if amps is None:
            continue

        # phase 1: SNR calibration (still)
        if snr.noise_floor is None:
            snr.calibrate(amps)
            calib_n += 1
            with _lock:
                _data["phase"] = "calib_snr"; _data["calib"] = calib_n
            if calib_n >= CALIB_TARGET:
                snr.finish_calibration()
                snr.buf.clear()
            continue

        snr.push(amps)
        feat, snr_db = snr.feature(risk=last_risk)

        # phase 2: baseline for z-score (still)
        if base_mean is None:
            base_feats.append(feat)
            with _lock:
                _data["phase"] = "calib_base"; _data["base"] = len(base_feats)
            if len(base_feats) >= BASE_TARGET:
                base_mean = float(np.mean(base_feats))
                base_std = float(np.std(base_feats)) + 1e-9
            continue

        # phase 3: live
        z = (feat - base_mean) / base_std
        perc_class = perc.classify(feat, z)
        reas.add(perc_class)
        state, risk = reas.get_confirmed_state()
        nexus.update(state)
        pred, conf = nexus.predict_next(state)
        alloc = res.allocate(risk)
        last_risk = risk

        with _lock:
            _data.update(phase="live",
                         raw=snr.raw_feature(amps),    # true Sigma x^2
                         cond=feat, risk=risk, state=state,
                         perc=perc_class, pred=pred, conf=conf,
                         mode=alloc["mode"], snr=snr_db)
    ser.close()


# ---------------- GUI ----------------
pg.setConfigOption('background', '#0a0a0a')
pg.setConfigOption('foreground', '#9980a0')

app = QtWidgets.QApplication(sys.argv)
win = pg.GraphicsLayoutWidget(title="SpectroAgent - Live Waveform")
win.resize(1100, 720)

status = win.addLabel("starting...", row=0, col=0, size="11pt", color="#d946a8")

p1 = win.addPlot(row=1, col=0,
                 title="CSI energy  -  raw (grey)  vs  SNR-conditioned (pink)")
p1.showGrid(x=False, y=True, alpha=0.15); p1.setMenuEnabled(False)
raw_buf  = np.zeros(N); cond_buf = np.zeros(N); risk_buf = np.zeros(N)
craw  = p1.plot(raw_buf,  pen=pg.mkPen("#5a4a5a", width=1))
ccond = p1.plot(cond_buf, pen=pg.mkPen("#d946a8", width=2))

p2 = win.addPlot(row=2, col=0, title="Risk score")
p2.setYRange(0, 1.05); p2.showGrid(x=False, y=True, alpha=0.15); p2.setMenuEnabled(False)
crisk = p2.plot(risk_buf, pen=pg.mkPen("#22c55e", width=2),
                fillLevel=0, brush=(34, 197, 94, 40))


def roll(b, val):
    b[:-1] = b[1:]; b[-1] = val; return b


def update():
    global raw_buf, cond_buf, risk_buf
    with _lock:
        d = dict(_data)

    if d["phase"] == "calib_snr":
        status.setText(f"<span style='color:#eab308'>CALIBRATING "
                       f"{min(d['calib'], CALIB_TARGET)}/{CALIB_TARGET}  -  STAY STILL</span>")
        return
    if d["phase"] == "calib_base":
        status.setText(f"<span style='color:#eab308'>STABILISING BASELINE "
                       f"{d['base']}/{BASE_TARGET}  -  STAY STILL</span>")
        return

    raw_buf  = roll(raw_buf,  d["raw"])
    cond_buf = roll(cond_buf, d["cond"])
    risk_buf = roll(risk_buf, d["risk"])
    craw.setData(raw_buf); ccond.setData(cond_buf); crisk.setData(risk_buf)

    col = {"CLEAR": "#22c55e", "HEIGHTENED": "#eab308",
           "PRE_ESCALATION": "#f59e0b", "EMERGENCY": "#f97316",
           "CRITICAL": "#ef4444"}.get(d["state"], "#9980a0")
    status.setText(
        f"<span style='color:{col}'>{d['state']}</span>"
        f"  -  risk {d['risk']:.2f}"
        f"  -  perception {d['perc']}"
        f"  -  next {d['pred']} ({d['conf']:.0%})"
        f"  -  {d['mode']}"
        f"  -  combined SNR {d['snr']:.1f} dB")


timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(50)

if __name__ == "__main__":
    t = threading.Thread(target=reader, daemon=True)
    t.start()
    win.show()
    try:
        sys.exit(app.exec_())
    finally:
        _stop.set()
