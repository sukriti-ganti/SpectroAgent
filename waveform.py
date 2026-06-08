import sys
import re
import numpy as np
import serial
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

from snr_agent import SNRAgent
from perception_agent import PerceptionAgent
from reasoning_agent import ReasoningAgent
from nexus_jumeau import NexusJumeau
from resource_allocation_agent import ResourceAllocationAgent

PORT, BAUD = "COM5", 921600
N = 240
CALIB_TARGET = 50

ser = serial.Serial(PORT, BAUD, timeout=0)
ser.reset_input_buffer()

snr = SNRAgent(window=8)
perc = PerceptionAgent()
reas = ReasoningAgent()
nexus = NexusJumeau()
res = ResourceAllocationAgent()

raw_buf = np.zeros(N)
cond_buf = np.zeros(N)
risk_buf = np.zeros(N)
calib_feats = []
base_mean = base_std = None
calibrated = False
calib_count = 0
last = {"state": "CLEAR", "risk": 0.0, "raw": "CLEAR", "pred": "CLEAR",
        "conf": 0.0, "mode": "LOW POWER", "snr": 0.0}


def parse_line(raw):
    if "CSI_DATA" not in raw:
        return None
    m = re.search(r'"?\[([^\]]+)\]"?', raw)
    if not m:
        return None
    try:
        v = list(map(int, m.group(1).split(',')))
    except ValueError:
        return None
    if len(v) < 8:
        return None
    imag, real = np.array(v[0::2], float), np.array(v[1::2], float)
    return np.sqrt(imag ** 2 + real ** 2)


pg.setConfigOption('background', '#0a0a0a')
pg.setConfigOption('foreground', '#9980a0')

app = QtWidgets.QApplication(sys.argv)
win = pg.GraphicsLayoutWidget(title="SpectroAgent - Live Waveform")
win.resize(1100, 720)

status = win.addLabel("SpectroAgent  -  calibrating...", row=0, col=0,
                      size="11pt", color="#d946a8")
p1 = win.addPlot(row=1, col=0,
                 title="CSI feature  -  raw (grey)  vs  SNR-conditioned (pink)")
p1.showGrid(x=False, y=True, alpha=0.15)
p1.setMenuEnabled(False)
craw = p1.plot(raw_buf, pen=pg.mkPen("#5a4a5a", width=1))
ccond = p1.plot(cond_buf, pen=pg.mkPen("#d946a8", width=2))

p2 = win.addPlot(row=2, col=0, title="Risk score")
p2.setYRange(0, 1.05)
p2.showGrid(x=False, y=True, alpha=0.15)
p2.setMenuEnabled(False)
crisk = p2.plot(risk_buf, pen=pg.mkPen("#22c55e", width=2),
                fillLevel=0, brush=(34, 197, 94, 40))


def roll(buf, val):
    buf[:-1] = buf[1:]
    buf[-1] = val
    return buf


def update():
    global raw_buf, cond_buf, risk_buf, base_mean, base_std
    global calibrated, calib_count
    try:
        chunk = ser.read(4096).decode("utf-8", "ignore")
    except Exception:
        return
    if not chunk:
        return
    for line in chunk.split("\n"):
        amps = parse_line(line.strip())
        if amps is None:
            continue
        if not calibrated:
            snr.calibrate(amps)
            calib_count += 1
            status.setText(f"SpectroAgent  -  calibrating "
                           f"{min(calib_count, CALIB_TARGET)}/{CALIB_TARGET}"
                           f"  -  stay still")
            if calib_count >= CALIB_TARGET:
                snr.finish_calibration()
                snr.buf.clear()
                calibrated = True
            continue
        snr.push(amps)
        feat, snr_db = snr.feature(risk=last["risk"])
        if base_mean is None:
            calib_feats.append(feat)
            status.setText("SpectroAgent  -  stabilising baseline  -  stay still")
            if len(calib_feats) >= 20:
                base_mean = float(np.mean(calib_feats))
                base_std = float(np.std(calib_feats)) + 1e-9
            continue
        z = (feat - base_mean) / base_std
        raw_class = perc.classify(feat, z)
        reas.add(raw_class)
        state, risk = reas.get_confirmed_state()
        nexus.update(state)
        pred, conf = nexus.predict_next(state)
        alloc = res.allocate(risk)
        raw_buf = roll(raw_buf, snr.raw_feature(amps))
        cond_buf = roll(cond_buf, feat)
        risk_buf = roll(risk_buf, risk)
        last.update(state=state, risk=risk, raw=raw_class, pred=pred,
                    conf=conf, mode=alloc["mode"], snr=snr_db)

    craw.setData(raw_buf)
    ccond.setData(cond_buf)
    crisk.setData(risk_buf)
    if base_mean is None:
        return
    col = {"CLEAR": "#22c55e", "HEIGHTENED": "#eab308",
           "PRE_ESCALATION": "#f59e0b", "EMERGENCY": "#f97316",
           "CRITICAL": "#ef4444"}.get(last["state"], "#9980a0")
    status.setText(
        f"<span style='color:{col}'>{last['state']}</span>"
        f"  -  risk {last['risk']:.2f}"
        f"  -  perception {last['raw']}"
        f"  -  next {last['pred']} ({last['conf']:.0%})"
        f"  -  {last['mode']}"
        f"  -  combined SNR {last['snr']:.1f} dB")


timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(40)

if __name__ == "__main__":
    win.show()
    sys.exit(app.exec_())