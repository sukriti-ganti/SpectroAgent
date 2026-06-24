SpectroAgent - CLEAN CODE SET
=============================
Everything consistent on sum-of-squares ENERGY (Sigma x^2) -- matches the FPGA.
Threaded serial read so the dashboard never lags.

WHICH FILE TO RUN
-----------------
demo_dashboard.py   -> FILM THIS. Minimal, bulletproof: flat baseline -> hand-wave
                       spike -> settles. One plot, instant response, says STILL / MOTION.
full_dashboard.py   -> The deeper-session version. All five agents, three traces,
                       agent status line. Also threaded (no lag).

AGENT MODULES (imported by full_dashboard.py)
---------------------------------------------
snr_agent.py                  SNR conditioning. raw_feature = Sigma x^2 (FPGA-matching).
perception_agent.py           Linear classifier: CLEAR / MOTION / ANOMALY.
reasoning_agent.py            Window voting -> risk score -> state.
nexus_jumeau.py               First-order Markov, Laplace smoothing, 500 ms lookahead.
resource_allocation_agent.py  Risk -> sensing rate / CIC factor R.

BEFORE YOU RUN
--------------
1. pip install pyqtgraph pyserial numpy PyQt5
2. Set PORT at the top of the dashboard if not COM5.
3. Router + ESP32 in the SAME room, ~1-2 m apart, clear line between them.
   (Through-wall = weak, jittery baseline = false fires. Same room fixes it.)

FILMING demo_dashboard.py
-------------------------
1. Start it. STAY STILL through "CALIBRATING 60/60" (baseline locks).
2. Stay still 5-8 s on camera -> flat line = proof it doesn't false-fire.
3. Wave your hand in the GAP between router and ESP, across their line of sight.
4. Stop -> it settles back to flat.
5. Frame so router, ESP32, laptop screen AND your hand are all visible.

NOTE ON THE FEATURE
-------------------
Old code computed np.var() of amplitudes ("variance"). That did NOT match your
silicon. These files compute Sigma x^2 (sum of squares of I and Q across
subcarriers) everywhere -- so "bit-exact algorithm-to-silicon" is literally true,
and you say "sum-of-squares energy", never "variance".
