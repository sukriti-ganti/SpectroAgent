# SpectroAgent + Nexus Jumeau

**Agentic AI-Enabled Predictive Maintenance and Safety Intelligence for Industrial IoT Systems using WiFi Channel State Information**

> GENATHON 2026 · KL University Hyderabad · Track 1

---

## What It Does

SpectroAgent uses a single ESP32 microcontroller to passively listen to WiFi signals already present in any industrial facility. When something sudden happens in the zone — an impact, an equipment failure, an anomalous event — the WiFi signal gets disturbed in a specific pattern. Five AI agents process this disturbance in real time and raise an alert in under 3 seconds.

**No cameras. No wearables. No new infrastructure. Just WiFi.**

---

## Why It Matters

| Problem | SpectroAgent's Answer |
|---|---|
| Cameras fail in smoke, dust, darkness | Works in any atmospheric condition |
| Wearables require worker compliance | Completely passive — workers do nothing |
| Dedicated sensors need installation | Uses existing WiFi router |
| Current systems are reactive | Nexus Jumeau predicts 500ms ahead |
| Shadow IoT — devices unmonitored | Zero new devices added to network |

---

## System Architecture

```
ESP32 (WiFi CSI sniffer)
        ↓
Baseline Calibration (locked in 20 samples)
        ↓
Perception Agent — SVM classifier
→ CLEAR / MOTION / ANOMALY
        ↓
Reasoning Agent — sliding window risk scorer
→ Risk score 0.0 to 1.0
        ↓
     ┌──┴──────────────────┐
     ↓                     ↓
Nexus Jumeau          Resource Allocation
Markov chain          ISAC-inspired dynamic
digital twin          sensing rate
     └──────┬─────────────┘
            ↓
    Communication Agent
    → Buzzer + Telegram alert
            ↓
    Streamlit Dashboard
```

---

## The 5 Agents

### 1. Perception Agent
- **Type:** Support Vector Machine classifier
- **Input:** Z-score, recent max Z, rate of change
- **Output:** CLEAR / MOTION / ANOMALY
- **How:** Computes Z-score relative to a locked baseline mean and std. Uses rule-based decision boundary tuned for industrial noise environments.

### 2. Reasoning Agent
- **Type:** Sliding window ensemble voting
- **Input:** Last 3 classifications from Perception Agent
- **Output:** Risk score 0.0 → 1.0 + confirmed state
- **How:** Weighted average — CLEAR=0.0, MOTION=0.3, ANOMALY=1.0. Single noisy reading is ignored. Sustained pattern triggers alert.

### 3. Nexus Jumeau (Digital Twin)
- **Type:** Markov chain — unsupervised online learning
- **Input:** Sequence of confirmed zone states
- **Output:** Predicted next state + confidence %
- **How:** Maintains 3×3 transition probability matrix updated live with every observation. Predicts next zone state 500ms ahead.

### 4. Resource Allocation Agent
- **Type:** ISAC-inspired deterministic policy
- **Input:** Current risk score
- **Output:** Sampling rate + resource mode
- **How:** As risk rises, sensing intensity increases. LOW POWER at 500ms sampling when CLEAR. CRITICAL at 25ms sampling. Software analogue of ETSI Work Item 8 RF tradeoff.

| Risk Score | Mode | Sampling Rate |
|---|---|---|
| < 0.10 | LOW POWER | 500ms |
| 0.10–0.30 | HEIGHTENED | 100ms |
| 0.30–0.60 | PRE-ESCALATION | 75ms |
| 0.60–0.85 | EMERGENCY | 50ms |
| > 0.85 | CRITICAL | 25ms |

### 5. Communication Agent
- **Type:** Threshold trigger with cooldown
- **Input:** Confirmed state from Reasoning Agent
- **Output:** Buzzer trigger + Telegram alert
- **How:** Fires when state crosses EMERGENCY or CRITICAL. 30-second cooldown prevents alert flooding.

---

## Hardware

| Component | Detail |
|---|---|
| Microcontroller | ESP32 (dual core 240MHz, 520KB SRAM) |
| Firmware | ESP-IDF 5.5 + esp-csi library |
| Mode | Promiscuous sniffer — no transmitter needed |
| Connection | USB-UART (CP210x) → COM5 |
| Baud rate | 921600 |
| Data field | CSI amplitude array — interleaved imaginary/real pairs |
| Cost | Rs. 400 |

---

## Key Technical Discoveries

- **Field parsing:** CSI array contains interleaved imaginary/real pairs. Amplitude = √(imag² + real²). Most zero values must be filtered before computing variance.
- **Baud rate:** 921600 required — 115200 returns garbage values.
- **Baseline:** Fixed calibration window fails in busy environments. EMA (alpha=0.05) self-adapts to any noise floor.
- **Serial lag:** `reset_input_buffer()` must only be called once at startup, not inside the read loop.
- **Detection physics:** WiFi at 2.4GHz (wavelength 12.5cm) is more sensitive to rapid mechanical impulses than slow movements — exactly matching industrial anomaly signatures.

---

## Software Stack

| Component | Technology |
|---|---|
| Language | Python |
| Firmware | ESP-IDF 5.5 |
| ML classifier | scikit-learn (rule-based SVM) |
| Digital twin | NumPy Markov chain |
| Dashboard | Streamlit |
| Serial comms | pyserial |
| Alerts | Telegram Bot API |

---

## File Structure

```
SpectroAgent/
├── serial_reader.py          # ESP32 serial + baseline calibration
├── perception_agent.py       # SVM classifier
├── reasoning_agent.py        # Sliding window risk scorer
├── nexus_jumeau.py           # Markov chain digital twin
├── resource_allocation_agent.py  # ISAC-inspired dynamic sensing
├── communication_agent.py    # Buzzer + Telegram alerts
└── dashboard.py              # Streamlit live dashboard
```

---

## Setup & Run

### Requirements
```
pip install pyserial numpy streamlit plotly requests
```

### Flash ESP32
```bash
# In ESP-IDF CMD terminal
cd C:\path\to\esp-csi\examples\get-started\csi_recv
idf.py flash
# Hold BOOT button while flashing
```

### Run Dashboard
```bash
python -m streamlit run dashboard.py
```

### Run Terminal Only (no dashboard)
```bash
python serial_reader.py
```

---

## Demo Sequence

1. **Calibration** — System collects 20 baseline samples. Stay still.
2. **Normal movement** — Walk past ESP32. Dashboard shows HEIGHTENED, no alert.
3. **Impact event** — Bang table hard 3 times fast, freeze. Risk score climbs → CRITICAL → alert fires.
4. **Through-wall reveal** — Move ESP32 behind wall. Repeat impact. SpectroAgent detects through the barrier.
5. **Dashboard walkthrough** — All 5 agent states updating live in real time.

---

## Research Anchor

| Reference | Relevance |
|---|---|
| ETSI Work Item 8 (2025) | Open research call for AI/ML-aided ISAC implementations — SpectroAgent is a direct response |
| ISAC (6G core concept) | Integrated Sensing and Communication — same WiFi used for internet also senses physical events |
| Halperin et al. (2009) | Foundational CSI sensing research |
| IEEE ISAC 2026 | Target conference for publication |

---

## Future Scope

- **Production scale:** Replace ESP32 with enterprise WiFi APs (Cisco Catalyst 9100) — same pipeline, 50–100m coverage per node
- **6G ISAC:** Deploy dedicated SDR nodes per ETSI Work Item 8
- **SomnoAgent:** Sleep monitoring via bedroom WiFi CSI
- **Multi-node mesh:** Full factory floor coverage
- **SCADA integration:** OPC-UA protocol connection to existing plant management systems

---

## Closing

> "One chip. 400 rupees. Ambient WiFi. Five AI agents. Through walls. Under 3 seconds. This is SpectroAgent."

---

*Built at GENATHON 2026 · KL University Hyderabad*
