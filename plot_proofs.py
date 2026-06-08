import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "font.size": 12, "axes.spines.top": False, "axes.spines.right": False,
})
TEAL, GREY = "#0d9488", "#9ca3af"

d = np.load("snr_proof.npz", allow_pickle=True)

# Figure 1: raw vs conditioned, still then event
fig, ax = plt.subplots(figsize=(8, 4))
sr, sc = list(d["still_raw"]), list(d["still_cond"])
er, ec = list(d["evt_raw"]), list(d["evt_cond"])
raw_series = sr + er
cond_series = sc + ec
split = len(sr)
ax.plot(raw_series, color=GREY, lw=1, label="Raw (equal-weight variance)")
ax.plot(cond_series, color=TEAL, lw=2, label="SNR-conditioned (MRC)")
ax.axvline(split, color="#ef4444", ls="--", lw=1)
ax.text(split, ax.get_ylim()[1]*0.9, " motion starts", color="#ef4444")
ax.set_xlabel("sample"); ax.set_ylabel("feature")
ax.set_title("SNR agent: raw vs conditioned feature")
ax.legend(frameon=False)
fig.tight_layout(); fig.savefig("fig_snr_trace.png", dpi=200)

# Figure 2: per-subcarrier SNR map
snr_map = d["snr_db"]
if snr_map is not None and np.ndim(snr_map) == 1:
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.bar(range(len(snr_map)), 10*np.log10(snr_map + 1e-9), color=TEAL)
    ax.set_xlabel("subcarrier index"); ax.set_ylabel("SNR (dB)")
    ax.set_title("Per-subcarrier SNR -- justifies MRC weighting")
    fig.tight_layout(); fig.savefig("fig_subcarrier_snr.png", dpi=200)

print("Saved fig_snr_trace.png and fig_subcarrier_snr.png")