from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne
from scipy import signal as sp_signal
from scipy.stats import percentileofscore
import csv

PROJECT_DIR = Path(__file__).resolve().parent
sfreq       = 300

# =========================================================
# CHANNEL DEFINITIONS
# =========================================================

CH_MAP = {
    "S1:CZ":    "Cz",
    "S2:CP2":   "CP2",
    "S3:CP3":   "CP3",
    "S4:FC2":   "FC2",
    "S5:FC3":   "FC3",
    "S6:vEOGt": "vEOGt",
    "S7:vEOGb": "vEOGb",
}

EEG_CHANNELS        = ["S1:CZ", "S2:CP2", "S3:CP3", "S4:FC2", "S5:FC3"]
EOG_CHANNELS        = ["S6:vEOGt", "S7:vEOGb"]
ALL_SIGNAL_CHANNELS = EEG_CHANNELS + EOG_CHANNELS

# =========================================================
# CONFIGURATION
# =========================================================

SUBJECT          = "Sub2_data"
EXCLUDE_SESSION  = 5          # removed entirely from all analysis
COMPARE_SESSIONS = [6, 12]    # examined independently vs the reference group
# Reference group = all loaded sessions not in COMPARE_SESSIONS

APPLY_ICA        = True
ICA_THRESHOLD    = 0.3

BANDS = {
    "Delta":  (1,   4),
    "Theta":  (4,   8),
    "Alpha":  (8,  13),
    "Beta":   (14, 30),
    "Gamma":  (30, 45),
}

# Colours for the two comparison sessions
COLORS = {6: "#E53935", 12: "#1565C0"}

# =========================================================
# HELPER: LOAD RESTING-STATE EEG (trigger-2 windows only)
# =========================================================

def load_resting_state(file_path):
    """Extract EEG during all contiguous trigger-2 windows, concatenate them.
    Returns (eeg_data, total_duration_s) or (None, None) on failure."""
    df = pd.read_csv(file_path, comment="#", skipinitialspace=True)

    trigger_col = None
    for candidate in ["Manual trigger", "Trigger"]:
        if candidate in df.columns:
            temp = pd.to_numeric(df[candidate], errors="coerce").fillna(0).astype(int)
            if temp.sum() > 0:
                trigger_col = candidate
                df[candidate] = temp
                break

    if trigger_col is None:
        return None, None

    trigger       = df[trigger_col].astype(int)
    trigger2_mask = (trigger == 2).values
    if trigger2_mask.sum() == 0:
        return None, None

    data_cols = ALL_SIGNAL_CHANNELS + [trigger_col]
    ch_names  = [CH_MAP[c] for c in ALL_SIGNAL_CHANNELS] + ["STI"]
    ch_types  = ["eeg"] * 5 + ["eog"] * 2 + ["stim"]

    data = df[data_cols].values.T.astype(float)
    data[:-1] *= 1e-6
    data = np.nan_to_num(data)

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw  = mne.io.RawArray(data, info, verbose=False)

    montage = mne.channels.make_standard_montage("standard_1020")
    raw.set_montage(montage, on_missing="ignore")

    raw.filter(l_freq=1.0, h_freq=45.0, method="fir",
               fir_window="hamming", verbose=False)

    if APPLY_ICA:
        ica = mne.preprocessing.ICA(
            n_components=len(EEG_CHANNELS), random_state=42,
            max_iter="auto", verbose=False,
        )
        ica.fit(raw, picks="eeg", verbose=False)

        sources   = ica.get_sources(raw).get_data()
        eog_data  = raw.get_data(picks=["vEOGt", "vEOGb"])
        veog_diff = eog_data[0] - eog_data[1]

        corrs = np.array([
            np.abs(np.corrcoef(sources[i], veog_diff)[0, 1])
            for i in range(sources.shape[0])
        ])
        best = int(np.argmax(corrs))
        if corrs[best] > ICA_THRESHOLD:
            ica.exclude = [best]
            ica.apply(raw, verbose=False)

    picks    = mne.pick_types(raw.info, eeg=True, eog=False, stim=False)
    eeg_full = raw.get_data(picks=picks)

    changes = np.diff(trigger2_mask.astype(int), prepend=0, append=0)
    starts  = np.where(changes ==  1)[0]
    ends    = np.where(changes == -1)[0]

    segments = [
        eeg_full[:, s:e]
        for s, e in zip(starts, ends)
        if (e - s) >= sfreq
    ]

    if not segments:
        return None, None

    eeg_data  = np.concatenate(segments, axis=1)
    total_dur = eeg_data.shape[1] / sfreq

    if total_dur < 5.0:
        return None, None

    return eeg_data, total_dur


# =========================================================
# FEATURE EXTRACTION
# =========================================================

def compute_features(eeg_data):
    """Compute resting-state spectral features from (n_ch, n_times) array."""
    nperseg = min(int(sfreq * 4), eeg_data.shape[1] // 2)
    freqs, psd = sp_signal.welch(eeg_data, fs=sfreq, nperseg=nperseg)

    band_powers = {}
    for band, (fmin, fmax) in BANDS.items():
        mask = (freqs >= fmin) & (freqs <= fmax)
        band_powers[band] = float(np.mean(psd[:, mask]))

    total_power = sum(band_powers.values())

    feats = {}
    for band, power in band_powers.items():
        feats[f"Rel {band}"] = power / total_power if total_power > 0 else 0.0

    feats["Alpha/Beta"] = (
        band_powers["Alpha"] / band_powers["Beta"]
        if band_powers["Beta"] > 0 else 0.0
    )
    feats["RMS (µV)"] = float(np.sqrt(np.mean(eeg_data ** 2)) * 1e6)

    psd_avg  = psd.mean(axis=0)
    psd_norm = psd_avg / (psd_avg.sum() + 1e-30)
    feats["Spectral Entropy"] = float(-np.sum(psd_norm * np.log(psd_norm + 1e-30)))

    return feats


# =========================================================
# MAIN: LOAD SESSIONS  (skip session 5 entirely)
# =========================================================

subject_dir  = PROJECT_DIR / SUBJECT
filtered_dir = subject_dir / "Filtered_data"

session_files = sorted(
    filtered_dir.glob("*.csv"),
    key=lambda f: int(m.group(1)) if (m := re.search(r"Ses(\d+)", f.stem, re.IGNORECASE)) else 0,
)

print(f"Subject: {SUBJECT}  —  {len(session_files)} files found\n")

records = []   # list of (ses_num, ses_name, feats_dict)

for f in session_files:
    ses_name = f.stem.replace("_filtered", "")
    m_ses    = re.search(r"Ses(\d+)", ses_name, re.IGNORECASE)
    ses_num  = int(m_ses.group(1)) if m_ses else 0

    if ses_num == EXCLUDE_SESSION:
        print(f"  {ses_name} → excluded (session {EXCLUDE_SESSION})")
        continue

    print(f"  {ses_name} ...", end=" ", flush=True)
    eeg_data, dur = load_resting_state(f)
    if eeg_data is None:
        print("skipped (insufficient trigger-2 data)")
        continue

    feats = compute_features(eeg_data)
    records.append((ses_num, ses_name, feats))
    print(f"{dur:.1f} s  →  OK")

if not records:
    raise RuntimeError("No sessions with usable resting-state data found.")

feat_names = list(records[0][2].keys())
all_feats  = {fn: np.array([r[2][fn] for r in records]) for fn in feat_names}

# Indices into records
ref_mask  = np.array([r[0] not in COMPARE_SESSIONS for r in records])
ref_nums  = [r[0] for r in records if r[0] not in COMPARE_SESSIONS]
comp_idx  = {ses: next((i for i, r in enumerate(records) if r[0] == ses), None)
             for ses in COMPARE_SESSIONS}
ref_vals  = {fn: all_feats[fn][ref_mask] for fn in feat_names}

n_ref = int(ref_mask.sum())

# =========================================================
# STATISTICAL SUMMARY  (printed to console)
# =========================================================

def session_stats(ses_num, fn):
    idx = comp_idx[ses_num]
    val = float(all_feats[fn][idx])
    ref = ref_vals[fn]
    mu  = float(ref.mean())
    sd  = float(ref.std())
    z   = (val - mu) / sd if sd > 0 else 0.0
    pct = float(percentileofscore(ref, val, kind="rank"))
    return val, mu, sd, z, pct

print("\n" + "=" * 75)
print(f"  RESTING-STATE COMPARISON  (reference n={n_ref} sessions, "
      f"excl. {EXCLUDE_SESSION} + comparison sessions)")
print("=" * 75)

for ses in COMPARE_SESSIONS:
    if comp_idx[ses] is None:
        print(f"\n  Session {ses}: not found in loaded data.")
        continue
    print(f"\n  ── Session {ses} vs reference ──")
    print(f"  {'Feature':<22}  {'Ses val':>9}  {'Ref µ':>9}  {'Ref σ':>8}  "
          f"{'Z-score':>8}  {'Percentile':>10}")
    print("  " + "-" * 73)
    for fn in feat_names:
        val, mu, sd, z, pct = session_stats(ses, fn)
        flag = "  ***" if abs(z) >= 2 else ("  *" if abs(z) >= 1 else "")
        print(f"  {fn:<22}  {val:>9.4f}  {mu:>9.4f}  {sd:>8.4f}  "
              f"{z:>+8.2f}  {pct:>9.1f}%{flag}")

print("\n  (* |z|≥1,  *** |z|≥2  relative to reference distribution)")
print("=" * 75)

# =========================================================
# FIGURES: one strip plot + one z-score chart per comparison session
# =========================================================

out_dir = PROJECT_DIR / "outputs" / "results"
out_dir.mkdir(parents=True, exist_ok=True)

n_feats   = len(feat_names)
n_cols    = 3
n_rows    = -(-n_feats // n_cols)
saved_figs = []

np.random.seed(42)
# Pre-compute jitter once so both sessions see the same reference layout
ref_jitter = np.random.normal(0, 0.08, size=n_ref)

for ses in COMPARE_SESSIONS:
    if comp_idx[ses] is None:
        print(f"Session {ses} not found — plots skipped.")
        continue

    color = COLORS[ses]

    # ----------------------------------------------------------
    # Strip plot: reference distribution + this session's value
    # ----------------------------------------------------------
    fig_s, axes_s = plt.subplots(
        n_rows, n_cols,
        figsize=(6 * n_cols, 4.5 * n_rows),
        squeeze=False,
    )

    for fi, fn in enumerate(feat_names):
        ax  = axes_s[fi // n_cols][fi % n_cols]
        ref = ref_vals[fn]
        mu  = ref.mean()
        sd  = ref.std()

        # Reference strip
        ax.scatter(np.zeros(n_ref) + ref_jitter, ref,
                   color="#78909C", alpha=0.65, s=40, zorder=3)

        # Reference mean ± 1 SD
        ax.errorbar(0, mu, yerr=sd, fmt="D", color="#37474F",
                    markersize=7, linewidth=2, capsize=6, zorder=5)

        # ±2 SD band
        ax.axhspan(mu - 2 * sd, mu + 2 * sd, color="gray", alpha=0.06, zorder=1)
        ax.axhline(mu, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

        # This session's value
        val, _, _, z, pct = session_stats(ses, fn)
        ax.scatter(1.5, val, color=color, s=150, marker="*", zorder=6)
        ax.annotate(
            f"z = {z:+.2f}\n{pct:.0f}th pct",
            xy=(1.5, val),
            xytext=(1.75, val),
            va="center", ha="left",
            fontsize=8, color=color, fontweight="bold",
        )

        ax.set_xticks([0, 1.5])
        ax.set_xticklabels([f"Reference (n={n_ref})", f"Session {ses}"], fontsize=9)
        ax.set_xlim(-0.6, 2.8)
        ax.set_ylabel(fn, fontsize=9)
        ax.set_title(fn, fontsize=10, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

    for fi in range(n_feats, n_rows * n_cols):
        axes_s[fi // n_cols][fi % n_cols].set_visible(False)

    legend_handles_s = [
        plt.Line2D([0], [0], marker="o", color="#78909C", linestyle="none",
                   markersize=7, label=f"Reference sessions (n={n_ref})"),
        plt.Line2D([0], [0], marker="D", color="#37474F", linestyle="none",
                   markersize=7, label="Ref mean ± 1 SD"),
        plt.Line2D([0], [0], marker="*", color=color, linestyle="none",
                   markersize=12, label=f"Session {ses}"),
    ]
    fig_s.legend(handles=legend_handles_s, loc="lower center",
                 ncol=3, fontsize=9, framealpha=0.9,
                 bbox_to_anchor=(0.5, -0.01))
    fig_s.suptitle(
        f"{SUBJECT}  —  Session {ses} vs reference  |  Trigger-2 resting-state EEG\n"
        f"Annotations: z-score and percentile rank",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0.04, 1, 1])

    path_s = out_dir / f"resting_state_ses{ses}_strip.png"
    fig_s.savefig(path_s, dpi=150, bbox_inches="tight")
    saved_figs.append(path_s)

    # ----------------------------------------------------------
    # Z-score chart: horizontal bars for this session only
    # ----------------------------------------------------------
    fig_z, ax_z = plt.subplots(figsize=(8, 5))

    y_pos    = np.arange(n_feats)
    z_scores = [session_stats(ses, fn)[3] for fn in feat_names]
    pct_vals = [session_stats(ses, fn)[4] for fn in feat_names]

    bar_colors = [color if abs(z) < 2 else "#B71C1C" if color == "#E53935"
                  else "#0D47A1" for z in z_scores]

    bars = ax_z.barh(y_pos, z_scores, height=0.55,
                     color=bar_colors, alpha=0.85, zorder=3)

    for i, (z, pct, bar) in enumerate(zip(z_scores, pct_vals, bars)):
        x_text = z + (0.08 if z >= 0 else -0.08)
        ha     = "left" if z >= 0 else "right"
        ax_z.text(x_text, bar.get_y() + bar.get_height() / 2,
                  f"z={z:+.2f}  ({pct:.0f}th pct)",
                  va="center", ha=ha,
                  fontsize=8.5, color=color, fontweight="bold")

    ax_z.axvline(0,  color="black", linewidth=1.2, zorder=4)
    ax_z.axvline( 2, color="gray",  linewidth=1,   linestyle="--", alpha=0.6)
    ax_z.axvline(-2, color="gray",  linewidth=1,   linestyle="--", alpha=0.6,
                 label="z = ±2  (~95th / 5th pct)")
    ax_z.axvline( 1, color="gray",  linewidth=0.7, linestyle=":",  alpha=0.4)
    ax_z.axvline(-1, color="gray",  linewidth=0.7, linestyle=":",  alpha=0.4,
                 label="z = ±1  (~84th / 16th pct)")

    ax_z.set_yticks(y_pos)
    ax_z.set_yticklabels(feat_names, fontsize=10)
    ax_z.set_xlabel("Z-score relative to reference sessions", fontsize=10)
    ax_z.set_title(
        f"{SUBJECT}  —  Session {ses}: z-scores vs reference (n={n_ref})\n"
        f"(Reference = all sessions except {EXCLUDE_SESSION}, "
        + ", ".join(str(s) for s in COMPARE_SESSIONS) + ")",
        fontsize=11, fontweight="bold",
    )
    ax_z.legend(fontsize=9, loc="lower right")
    ax_z.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    path_z = out_dir / f"resting_state_ses{ses}_zscores.png"
    fig_z.savefig(path_z, dpi=150, bbox_inches="tight")
    saved_figs.append(path_z)

plt.show()
print("\nFigures saved to:")
for p in saved_figs:
    print(f"  {p}")

# =========================================================
# SAVE CSV
# =========================================================

out_csv = out_dir / "resting_state_features.csv"
with open(out_csv, "w", newline="") as fh:
    writer = csv.writer(fh)
    z_cols  = [f"z_{fn.replace(' ','_').replace('/','_').replace('(','').replace(')','').replace('µ','u')}"
               for fn in feat_names]
    pct_cols = [f"pct_{fn.replace(' ','_').replace('/','_').replace('(','').replace(')','').replace('µ','u')}"
                for fn in feat_names]
    writer.writerow(
        ["subject", "session", "session_num", "group"] +
        feat_names + z_cols + pct_cols
    )

    ref_mu_vec = np.array([ref_vals[fn].mean() for fn in feat_names])
    ref_sd_vec = np.array([ref_vals[fn].std()  for fn in feat_names])

    for r in records:
        ses_num, ses_name, feats = r
        if ses_num in COMPARE_SESSIONS:
            group = f"comparison_{ses_num}"
        else:
            group = "reference"

        feat_vals = np.array([feats[fn] for fn in feat_names])
        z_vals    = np.where(ref_sd_vec > 0,
                             (feat_vals - ref_mu_vec) / ref_sd_vec, 0.0)
        pct_vals  = np.array([
            percentileofscore(ref_vals[fn], feats[fn], kind="rank")
            for fn in feat_names
        ])

        writer.writerow(
            [SUBJECT, ses_name, ses_num, group] +
            [f"{v:.6f}" for v in feat_vals] +
            [f"{z:.4f}"  for z in z_vals] +
            [f"{p:.1f}"  for p in pct_vals]
        )

print(f"Features saved to:\n  {out_csv}")
