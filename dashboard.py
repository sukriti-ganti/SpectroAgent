import streamlit as st
import serial
import numpy as np
import re
import time
from collections import deque
from perception_agent import PerceptionAgent
from reasoning_agent import ReasoningAgent
from nexus_jumeau import NexusJumeau
from resource_allocation_agent import ResourceAllocationAgent

st.set_page_config(page_title="SpectroAgent", layout="wide")

st.markdown("""
<style>
body, .stApp { background-color: #0a0a0a; }
.block-container { padding-top: 0.5rem; padding-bottom: 0.5rem; }
.metric-card {
    background: #110d11;
    border: 0.5px solid #2a1a2a;
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 8px;
}
.metric-label {
    font-size: 10px;
    color: #9980a0;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 4px;
}
.metric-value {
    font-size: 22px;
    font-weight: 500;
    color: #e8d8e8;
}
.agent-card {
    background: #110d11;
    border: 0.5px solid #2a1a2a;
    border-radius: 8px;
    padding: 10px 12px;
    height: 90px;
}
.agent-label {
    font-size: 9px;
    color: #d946a8;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 6px;
}
.agent-value {
    font-size: 14px;
    font-weight: 500;
}
.agent-sub {
    font-size: 9px;
    color: #9980a0;
    margin-top: 4px;
}
.section-label {
    font-size: 10px;
    color: #9980a0;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 6px;
    margin-top: 6px;
}
.alert-box {
    background: #3d0a0a;
    border: 1px solid #ef4444;
    border-radius: 8px;
    padding: 14px;
    margin-bottom: 12px;
    font-size: 14px;
    color: #f09595;
    font-weight: 500;
}
</style>
""", unsafe_allow_html=True)

SERIAL_PORT = "COM5"
BAUD_RATE = 921600

if "ser" not in st.session_state:
    try:
        st.session_state.ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        st.session_state.ser.reset_input_buffer()
        st.session_state.connected = True
    except Exception as e:
        st.session_state.connected = False
        st.session_state.error = str(e)

if not st.session_state.connected:
    st.error(f"Cannot connect to ESP32: {st.session_state.error}")
    st.stop()

if "baseline_mean" not in st.session_state:
    st.session_state.baseline_samples = []
    st.session_state.baseline_mean = None
    st.session_state.baseline_std = None
    st.session_state.variance_history = deque(maxlen=60)
    st.session_state.risk_history = deque(maxlen=60)
    st.session_state.alert_count = 0
    st.session_state.last_state = "CLEAR"
    st.session_state.last_risk = 0.0
    st.session_state.last_predicted = "CLEAR"
    st.session_state.last_confidence = 0.0
    st.session_state.last_allocation = {"mode": "LOW POWER", "interval": 0.5}
    st.session_state.last_raw = "CLEAR"
    st.session_state.last_variance = 0.0
    st.session_state.last_z = 0.0
    st.session_state.perception = PerceptionAgent()
    st.session_state.reasoning = ReasoningAgent()
    st.session_state.nexus = NexusJumeau()
    st.session_state.resource = ResourceAllocationAgent()

def get_readings(n=3):
    ser = st.session_state.ser
    ser.reset_input_buffer()
    readings = []
    attempts = 0
    while len(readings) < n and attempts < 30:
        attempts += 1
        try:
            raw = ser.readline().decode("utf-8", errors="ignore").strip()
            if "CSI_DATA" not in raw:
                continue
            match = re.search(r'"?\[([^\]]+)\]"?', raw)
            if not match:
                continue
            values = list(map(int, match.group(1).split(',')))
            if len(values) < 4:
                continue
            imag = values[0::2]
            real = values[1::2]
            amps = [np.sqrt(i**2 + r**2) for i, r in zip(imag, real)]
            nonzero = [a for a in amps if a > 0]
            if len(nonzero) < 5:
                continue
            readings.append(float(np.var(nonzero)))
        except:
            continue
    return readings

def state_color(s):
    return {
        "CLEAR": "#22c55e",
        "HEIGHTENED": "#eab308",
        "PRE_ESCALATION": "#f59e0b",
        "EMERGENCY": "#f97316",
        "CRITICAL": "#ef4444"
    }.get(s, "#9980a0")

readings = get_readings(3)

if not readings:
    st.warning("Waiting for ESP32 data...")
    time.sleep(0.3)
    st.rerun()

if st.session_state.baseline_mean is None:
    for v in readings:
        st.session_state.baseline_samples.append(v)
    n = len(st.session_state.baseline_samples)
    st.markdown("<h3 style='color:#d946a8;'>SpectroAgent — Calibrating</h3>",
                unsafe_allow_html=True)
    st.progress(min(n / 50, 1.0))
    st.markdown(f"<p style='color:#9980a0;'>Stay completely still... {min(n,50)}/50</p>",
                unsafe_allow_html=True)
    if n >= 50:
        st.session_state.baseline_mean = np.mean(
            st.session_state.baseline_samples[:50])
        st.session_state.baseline_std = np.std(
            st.session_state.baseline_samples[:50]) + 1e-9
    time.sleep(0.1)
    st.rerun()

for variance in readings:
    z = (variance - st.session_state.baseline_mean) / st.session_state.baseline_std
    raw_class = st.session_state.perception.classify(variance, z)
    st.session_state.reasoning.add(raw_class)
    state, risk = st.session_state.reasoning.get_confirmed_state()
    st.session_state.nexus.update(state)
    predicted, confidence = st.session_state.nexus.predict_next(state)
    allocation = st.session_state.resource.allocate(risk)
    st.session_state.variance_history.append(round(variance, 2))
    st.session_state.risk_history.append(round(risk, 3))
    if state in ["CRITICAL", "EMERGENCY"]:
        st.session_state.alert_count += 1
    st.session_state.last_state = state
    st.session_state.last_risk = risk
    st.session_state.last_raw = raw_class
    st.session_state.last_predicted = predicted
    st.session_state.last_confidence = confidence
    st.session_state.last_allocation = allocation
    st.session_state.last_variance = variance
    st.session_state.last_z = z

state = st.session_state.last_state
risk = st.session_state.last_risk
raw_class = st.session_state.last_raw
predicted = st.session_state.last_predicted
confidence = st.session_state.last_confidence
allocation = st.session_state.last_allocation
variance = st.session_state.last_variance
z = st.session_state.last_z
sc = state_color(state)
var_list = list(st.session_state.variance_history)
risk_list = list(st.session_state.risk_history)

st.markdown("<h2 style='color:#d946a8;font-size:15px;margin-bottom:0;'>SpectroAgent — Live Industrial Safety Monitor</h2>",
            unsafe_allow_html=True)
st.markdown("<p style='color:#9980a0;font-size:10px;margin-top:2px;margin-bottom:8px;'>ESP32 WiFi CSI · 5-agent pipeline · real-time zone monitoring</p>",
            unsafe_allow_html=True)

if state in ["CRITICAL", "EMERGENCY"]:
    st.markdown(f"""
    <div class='alert-box'>
    ANOMALY CONFIRMED — {state} &nbsp;·&nbsp; Risk Score: {risk:.2f} &nbsp;·&nbsp; Buzzer Triggered &nbsp;·&nbsp; Alert Dispatched
    </div>""", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"""<div class='metric-card'>
    <div class='metric-label'>CSI Variance</div>
    <div class='metric-value'>{variance:.1f}</div>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""<div class='metric-card'>
    <div class='metric-label'>Z-Score</div>
    <div class='metric-value'>{z:.2f}</div>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""<div class='metric-card'>
    <div class='metric-label'>Risk Score</div>
    <div class='metric-value' style='color:{sc};'>{risk:.2f}</div>
    </div>""", unsafe_allow_html=True)
with c4:
    st.markdown(f"""<div class='metric-card'>
    <div class='metric-label'>Total Alerts</div>
    <div class='metric-value'>{st.session_state.alert_count}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("---")

col_a, col_b = st.columns(2)
with col_a:
    st.markdown("<div class='section-label'>CSI Variance — this session</div>",
                unsafe_allow_html=True)
    if var_list:
        st.markdown(f"""
        <div class='metric-card'>
        <div class='metric-label'>Latest reading</div>
        <div class='metric-value'>{var_list[-1]:.1f}</div>
        </div>
        <div class='metric-card'>
        <div class='metric-label'>Session minimum</div>
        <div class='metric-value'>{min(var_list):.1f}</div>
        </div>
        <div class='metric-card'>
        <div class='metric-label'>Session maximum</div>
        <div class='metric-value'>{max(var_list):.1f}</div>
        </div>
        """, unsafe_allow_html=True)

with col_b:
    st.markdown("<div class='section-label'>Risk Score — this session</div>",
                unsafe_allow_html=True)
    if risk_list:
        st.markdown(f"""
        <div class='metric-card'>
        <div class='metric-label'>Current risk</div>
        <div class='metric-value' style='color:{sc};'>{risk_list[-1]:.2f}</div>
        </div>
        <div class='metric-card'>
        <div class='metric-label'>Peak risk</div>
        <div class='metric-value'>{max(risk_list):.2f}</div>
        </div>
        <div class='metric-card'>
        <div class='metric-label'>Average risk</div>
        <div class='metric-value'>{sum(risk_list)/len(risk_list):.2f}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

bar_pct = int(risk * 100)
st.markdown(f"""
<div class='section-label'>Risk Score Meter</div>
<div style='background:#1a101a;border-radius:6px;height:14px;overflow:hidden;margin-bottom:6px;'>
    <div style='height:100%;width:{bar_pct}%;background:{sc};border-radius:6px;transition:width 0.3s;'></div>
</div>
<div style='display:flex;justify-content:space-between;font-size:9px;color:#9980a0;margin-bottom:12px;'>
    <span>0.0 CLEAR</span><span>0.2 HEIGHTENED</span><span>0.4 EMERGENCY</span><span>0.6 CRITICAL</span><span>1.0</span>
</div>""", unsafe_allow_html=True)

st.markdown("<div class='section-label'>Agent Pipeline — live states</div>",
            unsafe_allow_html=True)
a1, a2, a3, a4, a5 = st.columns(5)

pc = state_color(raw_class)
comms = "ALERT FIRED" if state in ["CRITICAL","EMERGENCY"] else "STANDBY"
comms_color = "#ef4444" if state in ["CRITICAL","EMERGENCY"] else "#9980a0"

with a1:
    st.markdown(f"""<div class='agent-card'>
    <div class='agent-label'>Perception</div>
    <div class='agent-value' style='color:{pc};'>{raw_class}</div>
    <div class='agent-sub'>SVM classifier</div>
    </div>""", unsafe_allow_html=True)
with a2:
    st.markdown(f"""<div class='agent-card'>
    <div class='agent-label'>Reasoning</div>
    <div class='agent-value' style='color:{sc};'>{state}</div>
    <div class='agent-sub'>risk window · {risk:.2f}</div>
    </div>""", unsafe_allow_html=True)
with a3:
    st.markdown(f"""<div class='agent-card'>
    <div class='agent-label'>Nexus Jumeau</div>
    <div class='agent-value' style='color:#e8d8e8;'>{predicted}</div>
    <div class='agent-sub'>next state · {confidence:.0%} conf</div>
    </div>""", unsafe_allow_html=True)
with a4:
    st.markdown(f"""<div class='agent-card'>
    <div class='agent-label'>Resources</div>
    <div class='agent-value' style='color:#e8d8e8;'>{allocation['mode']}</div>
    <div class='agent-sub'>{int(allocation['interval']*1000)}ms sampling</div>
    </div>""", unsafe_allow_html=True)
with a5:
    st.markdown(f"""<div class='agent-card'>
    <div class='agent-label'>Comms</div>
    <div class='agent-value' style='color:{comms_color};'>{comms}</div>
    <div class='agent-sub'>buzzer + Telegram</div>
    </div>""", unsafe_allow_html=True)

time.sleep(0.1)
st.rerun()
