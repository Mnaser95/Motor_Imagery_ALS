"""
08_questionnaire_resting_eeg.py

For each session:
  1. Load filtered EEG, apply BPF (1-50 Hz) + ICA.
  2. Load preprocessed resting-state EEG saved by 00_manual_epoch_review.py.
  3. Compute spectral features from the resting-state EEG.
  4. Load the matching pre-session questionnaire (Q/<session_num>.csv).
  5. Encode choice-type answers ordinally.

Outputs (in outputs/09_questionnaire/):
  - questionnaire_resting_eeg.csv       : one row per session
  - resting_eeg_correlations_spearman.csv
  - questionnaire_resting_eeg_heatmap.png
"""

from pathlib import Path
import re
import numpy as np
import pandas as pd
import mne
from scipy import signal, stats
import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLM
import matplotlib
matplotlib.use("Agg")          # non-interactive backend; change to "TkAgg" for pop-up
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# =========================================================
# CHANNEL DEFINITIONS
# =========================================================

PROJECT_DIR = Path(__file__).resolve().parent
sfreq       = 300

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
EEG_CH_NAMES        = ["Cz", "CP2", "CP3", "FC2", "FC3"]

# =========================================================
# CONFIGURATION — edit this section before running
# =========================================================

SUBJECTS      = ["Sub1_data", "Sub2_data", "Sub3_data"]

# ALSFRS-R scores (out of 40) per subject folder name
ALSFRS_SCORES = {
    "Sub1_data": 35,
    "Sub2_data": 20,
    "Sub3_data": 29,
    "Sub4_data": 24,
    "Sub5_data": 39,
    "Sub6_data": 27,
}

MIN_REST_SEC  = 5.0    # skip session if resting-state segment < this many seconds

# Frequency bands (Hz)
BANDS = {
    "delta": (1,  4),
    "theta": (4,  8),
    "alpha": (8,  13),
    "beta":  (13, 30),
    "gamma": (30, 50),
}

# ERD/ERS computation bands and epoch window
ERD_ERS_BANDS = {
    "alpha": (8, 13),
    "beta":  (14, 30),
}
MI_TMIN, MI_TMAX = 0, 4

# EEG_CH_NAMES = ["Cz", "CP2", "CP3", "FC2", "FC3"]
# idx:               0      1      2      3      4
RIGHT_CH_IDX = [1, 3]   # CP2, FC2  — right hemisphere
LEFT_CH_IDX  = [2, 4]   # CP3, FC3  — left hemisphere

# =========================================================
# ORDINAL ENCODING — extend / adjust if new responses appear
# =========================================================
# Key = lowercase substring of the question text (unambiguous prefix).
# Value = dict mapping lowercase response → integer score.
# Unknown responses are silently mapped to NaN.

QUESTION_ENCODINGS: dict[str, dict[str, float]] = {
    "time of day": {
        "morning": 0, "afternoon": 1, "evening": 2,
    },
    "how many hours did you sleep": {
        "extremely less than normal": 0,
        "very less than normal":      0,
        "somewhat less than normal":  1,
        "less than normal":           1,
        "normal":                     2,
        "somewhat more than normal":  3,
        "more than normal":           3,
        "extremely more than normal": 4,
        "very more than normal":      4,
    },
    "sleep quality": {
        "poor": 0, "fair": 1, "good": 2, "very good": 3, "excellent": 4,
    },
    "physically tired": {
        "not at all":       0,
        "slightly tired":   1,
        "moderately tired": 2,
        "very tired":       3,
        "extremely tired":  4,
    },
    "mentally fatigued": {
        "not fatigued":       0,
        "not at all":         0,
        "slightly fatigued":  1,
        "moderately fatigued":2,
        "very fatigued":      3,
        "extremely fatigued": 4,
    },
    "overall mood": {
        "very negative": 0,
        "negative":      1,
        "neutral":       2,
        "positive":      3,
        "very positive": 4,
    },
    "stressed": {
        "not at all":        0,
        "slightly":          1,
        "slightly stressed": 1,
        "moderately":        2,
        "moderately stressed":2,
        "very stressed":     3,
        "extremely stressed":4,
    },
    "caffeine": {
        "none":    0,
        "1 cup":   1,
        "2 cups":  2,
        "3+ cups": 3,
        "3 cups":  3,
        "low":     1,
        "moderate":2,
        "high":    3,
    },
    "alcohol in the last 24": {
        "no": 0, "yes": 1,
    },
    "muscle weakness": {
        "much better than usual":  0,
        "better than usual":       1,
        "about the same":          2,
        "worse than usual":        3,
        "much worse than usual":   4,
    },
    "spasticity": {
        "none":     0,
        "mild":     1,
        "moderate": 2,
        "severe":   3,
    },
}

# Display labels for each key (same order as QUESTION_ENCODINGS)
Q_LABELS: dict[str, str] = {
    "time of day":                "Time of Day",
    "how many hours did you sleep":"Sleep Hours",
    "sleep quality":              "Sleep Quality",
    "physically tired":           "Physical Fatigue",
    "mentally fatigued":          "Mental Fatigue",
    "overall mood":               "Mood",
    "stressed":                   "Stress",
    "caffeine":                   "Caffeine",
    "alcohol in the last 24":     "Alcohol",
    "muscle weakness":            "Muscle Weakness",
    "spasticity":                 "Spasticity",
}

# =========================================================
# HELPER: LOAD PREPROCESSED DATA FROM 00_manual_epoch_review
# =========================================================

def load_rest_data(subject, ses_name):
    """Load resting-state numpy array saved by 00_manual_epoch_review.py."""
    rest_file = (PROJECT_DIR / "outputs" / "00_manual_epoch_review"
                 / "resting_state" / subject / f"{ses_name}_rest.npy")
    if not rest_file.exists():
        return None, 0
    eeg_data  = np.load(str(rest_file))
    rest_dur  = eeg_data.shape[1] / sfreq
    if rest_dur < MIN_REST_SEC:
        return None, 0
    return eeg_data, rest_dur


def load_mi_epochs(subject, ses_name):
    """Load MI epochs .fif saved by 00_manual_epoch_review.py. Returns None if missing."""
    epo_file = (PROJECT_DIR / "outputs" / "00_manual_epoch_review"
                / "preprocessed_epochs" / subject / f"{ses_name}_epo.fif")
    if not epo_file.exists():
        return None
    return mne.read_epochs(str(epo_file), verbose=False)


def load_bad_channels(subject, ses_name):
    """Return bad EEG channel names saved by 00_manual_epoch_review.py."""
    import json
    json_file = (PROJECT_DIR / "outputs" / "00_manual_epoch_review"
                 / "epoch_rejection" / subject / f"{ses_name}_bad_epochs.json")
    if not json_file.exists():
        return []
    with open(json_file) as fh:
        d = json.load(fh)
    return [ch for ch in d.get("bad_channels", []) if ch in EEG_CH_NAMES]

# =========================================================
# HELPER: SAMPLE ENTROPY
# =========================================================

def _sample_entropy(x, m=2, r_factor=0.2, max_n=500):
    """Sample entropy of 1-D signal x (vectorised inner loop)."""
    x = np.asarray(x[:max_n], dtype=float)
    r = r_factor * x.std(ddof=1)
    N = len(x)
    if N < m + 2 or r == 0:
        return np.nan
    B, A = 0, 0
    for i in range(N - m - 1):
        js    = np.arange(i + 1, N - m)
        c_m   = x[js[:, None] + np.arange(m)]
        c_m1  = x[js[:, None] + np.arange(m + 1)]
        hit_m  = np.max(np.abs(c_m  - x[i : i + m]),     axis=1) <= r
        hit_m1 = np.max(np.abs(c_m1 - x[i : i + m + 1]), axis=1) <= r
        B += int(hit_m.sum())
        A += int((hit_m & hit_m1).sum())
    if B == 0:
        return np.nan
    return float(-np.log(A / B)) if A > 0 else float(np.inf)

# =========================================================
# HELPER: COMPUTE SPECTRAL FEATURES
# =========================================================

def compute_features(eeg: np.ndarray, ch_names=None) -> dict[str, float]:
    """
    eeg: (n_channels, n_samples) array.
    ch_names: list of channel names in the same order as eeg rows (defaults to EEG_CH_NAMES).
    Returns a flat dict of feature_name → float value.
    """
    if ch_names is None:
        ch_names = EEG_CH_NAMES
    n_ch, n_samp = eeg.shape
    nperseg = min(sfreq * 4, n_samp)        # 4-s segments if enough data
    freqs, psd = signal.welch(eeg, fs=sfreq, nperseg=nperseg)
    # psd: (n_ch, n_freqs)

    feats: dict[str, float] = {}

    # --- Relative band power: each band / sum-of-all-bands, per channel and mean ---
    band_powers = {}
    for band, (flo, fhi) in BANDS.items():
        mask = (freqs >= flo) & (freqs < fhi)
        band_powers[band] = psd[:, mask].mean(axis=1)  # (n_ch,)

    total_pow = sum(band_powers.values())               # (n_ch,), element-wise sum
    total_pow = np.where(total_pow == 0, 1e-30, total_pow)

    for band, bp in band_powers.items():
        rel = bp / total_pow
        for ci, ch in enumerate(ch_names):
            feats[f"{ch}_{band}_power"] = float(rel[ci])
        feats[f"mean_{band}_power"] = float(rel.mean())

    # --- Per-band asymmetry: mean of FC (FC2−FC3) and CP (CP2−CP3) log-ratio ---
    idx = {ch: i for i, ch in enumerate(ch_names)}
    eps = 1e-30
    for band, (flo, fhi) in BANDS.items():
        mask = (freqs >= flo) & (freqs < fhi)
        raw_pow = psd[:, mask].mean(axis=1)
        fc_asym = (float(np.log(raw_pow[idx["FC2"]] + eps) - np.log(raw_pow[idx["FC3"]] + eps))
                   if "FC2" in idx and "FC3" in idx else np.nan)
        cp_asym = (float(np.log(raw_pow[idx["CP2"]] + eps) - np.log(raw_pow[idx["CP3"]] + eps))
                   if "CP2" in idx and "CP3" in idx else np.nan)
        parts = [v for v in [fc_asym, cp_asym] if not np.isnan(v)]
        feats[f"mean_{band}_asymmetry"] = float(np.mean(parts)) if parts else np.nan

    # --- Spectral entropy: per channel and mean ---
    ents = []
    for ci in range(n_ch):
        p = psd[ci] / (psd[ci].sum() + eps)
        se = float(-np.sum(p * np.log(p + eps)))
        feats[f"{ch_names[ci]}_spectral_entropy"] = se
        ents.append(se)
    feats["mean_spectral_entropy"] = float(np.mean(ents))

    # --- Spectral ratios (channel-mean raw band powers) ---
    mean_raw = {b: float(bp.mean()) for b, bp in band_powers.items()}
    feats["theta_alpha_ratio"] = mean_raw["theta"] / (mean_raw["alpha"] + eps)
    feats["alpha_beta_ratio"]  = mean_raw["alpha"] / (mean_raw["beta"]  + eps)
    feats["theta_beta_ratio"]  = mean_raw["theta"] / (mean_raw["beta"]  + eps)

    # --- Individual Alpha Frequency (IAF) ---
    psd_mean   = psd.mean(axis=0)
    alpha_mask = (freqs >= 8) & (freqs <= 13)
    feats["IAF"] = (float(freqs[alpha_mask][np.argmax(psd_mean[alpha_mask])])
                    if alpha_mask.sum() > 0 else np.nan)

    # --- Hjorth parameters: per channel and mean ---
    h_act, h_mob, h_cplx = [], [], []
    for ci in range(n_ch):
        x   = eeg[ci]
        dx  = np.diff(x);  ddx = np.diff(dx)
        v_x  = np.var(x);  v_dx = np.var(dx);  v_ddx = np.var(ddx)
        act  = v_x
        mob  = float(np.sqrt(v_dx  / v_x)   if v_x  > 0 else 0.0)
        cplx = float(np.sqrt(v_ddx / v_dx) / mob if (v_dx > 0 and mob > 0) else 0.0)
        feats[f"{ch_names[ci]}_hjorth_activity"]   = float(act)
        feats[f"{ch_names[ci]}_hjorth_mobility"]   = mob
        feats[f"{ch_names[ci]}_hjorth_complexity"] = cplx
        h_act.append(act);  h_mob.append(mob);  h_cplx.append(cplx)
    feats["mean_hjorth_activity"]   = float(np.mean(h_act))
    feats["mean_hjorth_mobility"]   = float(np.mean(h_mob))
    feats["mean_hjorth_complexity"] = float(np.mean(h_cplx))

    # --- Sample entropy: per channel and mean ---
    samp_ents = []
    for ci in range(n_ch):
        se = _sample_entropy(eeg[ci])
        feats[f"{ch_names[ci]}_sample_entropy"] = se
        samp_ents.append(se)
    feats["mean_sample_entropy"] = float(np.nanmean(samp_ents))

    return feats

# =========================================================
# HELPER: ERD / ERS FROM MI EPOCHS
# =========================================================

def compute_erd_ers(eeg_rest: np.ndarray, epochs_fif, ch_names=None) -> dict[str, float]:
    """
    Compute average ERD (contralateral) and ERS (ipsilateral) for MI_Left
    and MI_Right epochs.  Baseline = resting-state numpy array.
    Formula: (A - R) / R * 100  (negative = desynchronisation, positive = sync).

    ch_names: active channel list for eeg_rest (defaults to EEG_CH_NAMES).
    Bad channels must already be excluded from eeg_rest and listed in ch_names.
    The same channels are excluded from the epoch PSD computation so baseline
    and activation power are always computed over the same set.

    Returns a dict with 8 keys (ERD/ERS_alpha/beta_MI_Left/Right),
    or an empty dict if baseline or epochs are unavailable.
    """
    if ch_names is None:
        ch_names = EEG_CH_NAMES
    if eeg_rest is None or eeg_rest.shape[1] < sfreq:
        return {}

    # Hemisphere channels restricted to whichever are still active
    right_active = [ch for ch in ["CP2", "FC2"] if ch in ch_names]
    left_active  = [ch for ch in ["CP3", "FC3"] if ch in ch_names]
    if not right_active or not left_active:
        return {}

    # Indices into eeg_rest (active channels only)
    right_rest_idx = [ch_names.index(ch)     for ch in right_active]
    left_rest_idx  = [ch_names.index(ch)     for ch in left_active]
    # Indices into epoch data (always full 5-ch from .fif) — same channel subset
    right_epo_idx  = [EEG_CH_NAMES.index(ch) for ch in right_active]
    left_epo_idx   = [EEG_CH_NAMES.index(ch) for ch in left_active]

    # --- Baseline PSD from resting-state array ---
    nperseg_rest = min(sfreq * 4, eeg_rest.shape[1])
    freqs_r, psd_rest = signal.welch(eeg_rest, fs=sfreq, nperseg=nperseg_rest)

    baseline: dict[tuple, float] = {}
    for band, (flo, fhi) in ERD_ERS_BANDS.items():
        bm = (freqs_r >= flo) & (freqs_r < fhi)
        baseline[("right", band)] = float(psd_rest[right_rest_idx, :][:, bm].mean())
        baseline[("left",  band)] = float(psd_rest[left_rest_idx,  :][:, bm].mean())

    if epochs_fif is None:
        return {}

    picks_eeg = mne.pick_types(epochs_fif.info, eeg=True, eog=False, stim=False)
    epochs = epochs_fif

    result: dict[str, float] = {}

    # MI_Left:  contralateral = right hemisphere,  ipsilateral = left
    # MI_Right: contralateral = left  hemisphere,  ipsilateral = right
    task_cfg = [
        ("MI_Left",  "right", "left",  right_epo_idx, left_epo_idx),
        ("MI_Right", "left",  "right", left_epo_idx,  right_epo_idx),
    ]

    for task, contra_hemi, ipsi_hemi, contra_idx, ipsi_idx in task_cfg:
        try:
            X = epochs[task].get_data(picks=picks_eeg)  # (n_epochs, 5, n_times)
        except KeyError:
            X = np.empty((0,))

        if X.ndim < 3 or len(X) == 0:
            for band in ERD_ERS_BANDS:
                result[f"ERD_{band}_{task}"] = np.nan
                result[f"ERS_{band}_{task}"] = np.nan
            continue

        nperseg_mi = min(sfreq * 4, X.shape[2])

        for band, (flo, fhi) in ERD_ERS_BANDS.items():
            r_contra = baseline[(contra_hemi, band)]
            r_ipsi   = baseline[(ipsi_hemi,   band)]
            erd_vals, ers_vals = [], []

            for ep in X:
                freqs_m, psd_ep = signal.welch(ep, fs=sfreq, nperseg=nperseg_mi)
                bm = (freqs_m >= flo) & (freqs_m < fhi)
                a_contra = float(psd_ep[contra_idx, :][:, bm].mean())
                a_ipsi   = float(psd_ep[ipsi_idx,  :][:, bm].mean())
                if r_contra > 0:
                    erd_vals.append((a_contra - r_contra) / r_contra * 100)
                if r_ipsi > 0:
                    ers_vals.append((a_ipsi - r_ipsi) / r_ipsi * 100)

            result[f"ERD_{band}_{task}"] = float(np.mean(erd_vals)) if erd_vals else np.nan
            result[f"ERS_{band}_{task}"] = float(np.mean(ers_vals)) if ers_vals else np.nan

    return result


# =========================================================
# HELPER: PARSE PRE-SESSION QUESTIONNAIRE
# =========================================================

def parse_questionnaire(q_file: Path) -> dict[str, float]:
    """
    Returns {Q_LABELS[key]: score} for each recognisable choice-type
    pre-session item. Unknown responses yield NaN (not dropped).
    """
    try:
        df = pd.read_csv(q_file, skipinitialspace=True, keep_default_na=False)
    except Exception:
        return {}

    if "form.itemText" not in df.columns or "form.response" not in df.columns:
        return {}

    # Pre-session rows only (form.type == "choice")
    rows = df[df["form.type"].fillna("") == "choice"][
        ["form.itemText", "form.response"]
    ].dropna(subset=["form.itemText"])

    scores: dict[str, float] = {}
    for _, row in rows.iterrows():
        q_text  = str(row["form.itemText"]).lower().strip()
        resp    = str(row["form.response"]).lower().strip()

        for key, enc in QUESTION_ENCODINGS.items():
            if key in q_text:
                label = Q_LABELS[key]
                scores[label] = enc.get(resp, np.nan)
                break

    return scores

# =========================================================
# MAIN — collect all sessions
# =========================================================

# Load per-session CSP+LDA accuracy produced by 06b
_acc_csv = PROJECT_DIR / "outputs" / "06_classify_sinlge_session" / "standard_csp_results_per_session.csv"
_acc_lookup: dict[tuple, float] = {}
if _acc_csv.exists():
    _df_acc = pd.read_csv(_acc_csv)
    for _, _row in _df_acc.iterrows():
        _acc_lookup[(_row["subject"], _row["session"])] = float(_row["lda_mean_%"])
else:
    print(f"[WARNING] {_acc_csv.name} not found — CSP+LDA accuracy will be missing.")

records: list[dict] = []

for subject in SUBJECTS:
    subject_dir  = PROJECT_DIR / subject
    filtered_dir = subject_dir / "Filtered_data"
    q_dir        = subject_dir / "Q"

    session_files = sorted(
        filtered_dir.glob("*.csv"),
        key=lambda f: int(m.group(1)) if (m := re.search(r"Ses(\d+)", f.stem, re.IGNORECASE)) else 0,
    )

    if not session_files:
        print(f"[{subject}] No filtered CSV files — skipping.")
        continue

    print(f"\n{'=' * 55}")
    print(f"  {subject}  —  {len(session_files)} sessions")
    print(f"{'=' * 55}")

    for f in session_files:
        ses_name = f.stem.replace("_filtered", "")
        m        = re.search(r"Ses(\d+)", ses_name, re.IGNORECASE)
        ses_num  = int(m.group(1)) if m else None
        q_file   = (q_dir / f"{ses_num}.csv") if ses_num is not None else None

        print(f"  {ses_name} ...", end=" ", flush=True)

        # --- Load preprocessed data from 00_manual_epoch_review ---
        eeg_rest, rest_dur = load_rest_data(subject, ses_name)
        if eeg_rest is None:
            print("skipped (run 00_manual_epoch_review.py first)")
            continue

        bad_ch = load_bad_channels(subject, ses_name)
        active_ch_names = [ch for ch in EEG_CH_NAMES if ch not in bad_ch]
        if bad_ch:
            keep     = [i for i, ch in enumerate(EEG_CH_NAMES) if ch not in bad_ch]
            eeg_rest = eeg_rest[keep, :]
            print(f"[bad ch dropped: {bad_ch}] ", end="")

        epochs_fif = load_mi_epochs(subject, ses_name)

        feats = compute_features(eeg_rest, ch_names=active_ch_names)
        feats.update(compute_erd_ers(eeg_rest, epochs_fif, ch_names=active_ch_names))
        feats["csp_lda_accuracy_%"] = _acc_lookup.get((subject, ses_name), np.nan)

        # --- Questionnaire ---
        if q_file is None or not q_file.exists():
            print(f"skipped (no Q/{ses_num}.csv)")
            continue

        q_scores = parse_questionnaire(q_file)
        if not q_scores:
            print("skipped (questionnaire parse failed)")
            continue

        record: dict = {"subject": subject, "session": ses_name,
                        "rest_duration_s": round(rest_dur, 1)}
        record.update(q_scores)
        record.update(feats)
        records.append(record)

        n_q_known = sum(1 for v in q_scores.values() if not np.isnan(v))
        print(f"OK  ({rest_dur:.1f} s rest,  {n_q_known}/{len(q_scores)} Q items encoded)")

if not records:
    print("\nNo records collected — nothing to analyse.")
    raise SystemExit

df_all = pd.DataFrame(records)

out_dir = PROJECT_DIR / "outputs" / Path(__file__).stem
out_dir.mkdir(parents=True, exist_ok=True)

# Build (column_name, display_label) pairs; filter to cols present in df_all.
_col_label_pairs = (
    [(f"mean_{b}_power",      f"{b.capitalize()} Power")     for b in BANDS]
    + [(f"mean_{b}_asymmetry",  f"{b.capitalize()} Asymmetry") for b in BANDS]
    + [
        ("mean_spectral_entropy",   "Spec. Entropy"),
        ("theta_alpha_ratio",        "θ/α Ratio"),
        ("alpha_beta_ratio",         "α/β Ratio"),
        ("theta_beta_ratio",         "θ/β Ratio"),
        ("IAF",                      "IAF (Hz)"),
        ("mean_hjorth_activity",     "Hjorth Activity"),
        ("mean_hjorth_mobility",     "Hjorth Mobility"),
        ("mean_hjorth_complexity",   "Hjorth Complexity"),
        ("mean_sample_entropy",      "Sample Entropy"),
        # ERD/ERS features (MI_Left then MI_Right; alpha then beta)
        ("ERD_alpha_MI_Left",        "ERD α (MI Left)"),
        ("ERS_alpha_MI_Left",        "ERS α (MI Left)"),
        ("ERD_beta_MI_Left",         "ERD β (MI Left)"),
        ("ERS_beta_MI_Left",         "ERS β (MI Left)"),
        ("ERD_alpha_MI_Right",       "ERD α (MI Right)"),
        ("ERS_alpha_MI_Right",       "ERS α (MI Right)"),
        ("ERD_beta_MI_Right",        "ERD β (MI Right)"),
        ("ERS_beta_MI_Right",        "ERS β (MI Right)"),
        ("csp_lda_accuracy_%",       "CSP+LDA Acc. (%)"),
    ]
)
_col_label_pairs = [(c, l) for c, l in _col_label_pairs if c in df_all.columns]
if _col_label_pairs:
    eeg_summary_cols, eeg_labels = map(list, zip(*_col_label_pairs))
else:
    eeg_summary_cols, eeg_labels = [], []

# =========================================================
# HEATMAP HELPER
# =========================================================

def draw_heatmap(ax, r_mat, p_mat, ne, nq, q_cols, title,
                 blocked_row_indices=(), cbar_ax=None):
    cmap = plt.cm.RdBu_r
    norm = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    im   = ax.imshow(r_mat, cmap=cmap, norm=norm, aspect="auto")

    for i in range(r_mat.shape[0]):
        if i in blocked_row_indices:
            continue
        for j in range(r_mat.shape[1]):
            if np.isnan(r_mat[i, j]):
                continue
            stars = ("***" if p_mat[i, j] < 0.001 else
                     "**"  if p_mat[i, j] < 0.01  else
                     "*"   if p_mat[i, j] < 0.05  else "")
            txt   = f"{r_mat[i, j]:.2f}{stars}"
            color = "white" if abs(r_mat[i, j]) > 0.55 else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7.5,
                    color=color, fontweight="bold" if stars else "normal")

    for i in blocked_row_indices:
        ax.add_patch(plt.Rectangle((-0.5, i - 0.5), ne, 1,
                                   color="black", zorder=3))

    ax.set_xticks(range(ne))
    ax.set_xticklabels(eeg_labels, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(nq))
    ax.set_yticklabels(q_cols, fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    for x in np.arange(-0.5, ne, 1):
        ax.axvline(x, color="white", linewidth=0.6)
    for y in np.arange(-0.5, nq, 1):
        ax.axhline(y, color="white", linewidth=0.6)
    if cbar_ax is not None:
        plt.colorbar(im, cax=cbar_ax, label="Correlation coefficient")

# =========================================================
# PER-SUBJECT HEATMAPS
# =========================================================

for subject in SUBJECTS:
    df_sub = df_all[df_all["subject"] == subject].copy()
    if df_sub.empty:
        print(f"\n[{subject}] No data — skipping heatmap.")
        continue

    min_changes = max(1, 0.20 * len(df_sub))
    q_cols = []
    blocked_names = set()
    for v in Q_LABELS.values():
        if v not in df_sub.columns:
            continue
        col = pd.to_numeric(df_sub[v], errors="coerce").dropna()
        if len(col) < 2:
            continue
        q_cols.append(v)
        mode_val = col.mode().iloc[0]
        if (col != mode_val).sum() < min_changes:
            blocked_names.add(v)

    if not q_cols or not eeg_summary_cols:
        print(f"\n[{subject}] Too few data points — skipping heatmap.")
        continue

    nq = len(q_cols)
    ne = len(eeg_summary_cols)
    blocked_row_indices = {qi for qi, v in enumerate(q_cols) if v in blocked_names}

    df_num = df_sub[q_cols + eeg_summary_cols].apply(pd.to_numeric, errors="coerce")

    spearman_r = np.full((nq, ne), np.nan)
    spearman_p = np.full((nq, ne), np.nan)

    for qi, q in enumerate(q_cols):
        if qi in blocked_row_indices:
            continue
        for ei, e in enumerate(eeg_summary_cols):
            valid = df_num[[q, e]].dropna()
            if len(valid) < 5:
                continue
            sr, sp = stats.spearmanr(valid[q], valid[e])
            spearman_r[qi, ei] = sr;  spearman_p[qi, ei] = sp

    fig_h = max(5, nq * 0.55 + 2.5)
    fig_w = max(8, ne * 1.0 + 3)
    fig   = plt.figure(figsize=(fig_w, fig_h))
    gs    = fig.add_gridspec(1, 2, width_ratios=[ne, 0.6],
                             wspace=0.08, left=0.18, right=0.93,
                             top=0.88, bottom=0.22)
    ax1     = fig.add_subplot(gs[0])
    cbar_ax = fig.add_subplot(gs[1])

    draw_heatmap(ax1, spearman_r, spearman_p, ne, nq, q_cols, "Spearman ρ",
                 blocked_row_indices=blocked_row_indices, cbar_ax=cbar_ax)

    n_ses = len(df_sub)
    fig.suptitle(
        f"{subject}  —  Pre-session questionnaire vs. resting-state EEG\n"
        f"({n_ses} sessions)     *p<0.05   **p<0.01   ***p<0.001",
        fontsize=11, y=0.97,
    )

    out_fig = out_dir / f"{subject}_questionnaire_resting_eeg_heatmap.png"
    fig.savefig(out_fig, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[{subject}] Heatmap saved to:\n  {out_fig}")

# =========================================================
# CROSS-SUBJECT: MIXED-EFFECTS MODEL  (feature ~ ALSFRS)
# =========================================================
# Each session is one observation. ALSFRS is a subject-level
# fixed effect; a random intercept per subject absorbs
# between-subject baseline differences unrelated to ALSFRS.
# Features are z-scored so β coefficients are comparable.
#
# Model: feature_z ~ ALSFRS_z + (1 | subject)

# Map ALSFRS score onto every session row
df_all["ALSFRS"] = df_all["subject"].map(ALSFRS_SCORES)

# Only keep subjects that have an ALSFRS score
df_lme = df_all.dropna(subset=["ALSFRS"]).copy()
n_subj_lme = df_lme["subject"].nunique()

if n_subj_lme < 3:
    print("\n[LME] Fewer than 3 subjects with ALSFRS scores — skipping.")
else:
    feat_cols = [c for c in eeg_summary_cols if c in df_lme.columns]
    feat_lbls = [l for c, l in zip(eeg_summary_cols, eeg_labels) if c in feat_cols]

    # Standardise ALSFRS across the dataset (one value per subject but repeated)
    als_mu  = df_lme["ALSFRS"].mean()
    als_sig = df_lme["ALSFRS"].std()
    als_sig = als_sig if als_sig > 0 else 1.0
    df_lme["ALSFRS_z"] = (df_lme["ALSFRS"] - als_mu) / als_sig

    lme_results = []   # (label, col, beta, ci_lo, ci_hi, pval, converged)

    for col, lbl in zip(feat_cols, feat_lbls):
        subset = df_lme[["subject", "ALSFRS_z", col]].copy()
        subset[col] = pd.to_numeric(subset[col], errors="coerce")
        subset = subset.dropna()

        # Need at least 2 subjects with ≥2 sessions each to fit the random effect
        counts = subset.groupby("subject").size()
        if (counts >= 2).sum() < 2 or subset["subject"].nunique() < 3:
            lme_results.append((lbl, col, np.nan, np.nan, np.nan, np.nan, False))
            continue

        # Z-score the feature using training-set statistics
        mu_f  = subset[col].mean()
        sig_f = subset[col].std()
        sig_f = sig_f if sig_f > 0 else 1.0
        subset["feat_z"] = (subset[col] - mu_f) / sig_f

        exog   = sm.add_constant(subset[["ALSFRS_z"]])
        groups = subset["subject"]

        try:
            mdl    = MixedLM(subset["feat_z"], exog, groups=groups)
            result = mdl.fit(reml=True, disp=False)
            beta   = float(result.fe_params["ALSFRS_z"])
            ci     = result.conf_int().loc["ALSFRS_z"]
            pval   = float(result.pvalues["ALSFRS_z"])
            lme_results.append((lbl, col, beta, float(ci.iloc[0]), float(ci.iloc[1]),
                                pval, True))
        except Exception:
            lme_results.append((lbl, col, np.nan, np.nan, np.nan, np.nan, False))

    # --- Save CSV ---
    df_lme_out = pd.DataFrame(lme_results,
                              columns=["feature", "col", "beta_z", "ci_lo", "ci_hi",
                                       "p_value", "converged"])
    lme_csv = out_dir / "alsfrs_lme_results.csv"
    df_lme_out.drop(columns=["col"]).to_csv(lme_csv, index=False)
    print(f"\n[LME] Results saved to:\n  {lme_csv}")

    # --- Horizontal bar chart: β with 95 % CI ---
    valid_res = [(lbl, b, lo, hi, p)
                 for lbl, col, b, lo, hi, p, ok in lme_results
                 if ok and not np.isnan(b)]

    if valid_res:
        lbls, betas, ci_los, ci_his, pvals = zip(*valid_res)
        betas   = np.array(betas)
        ci_los  = np.array(ci_los)
        ci_hi_s = np.array(ci_his)
        pvals   = np.array(pvals)

        y = np.arange(len(lbls))
        err_lo = betas - ci_los
        err_hi = ci_hi_s - betas

        colors = []
        for p in pvals:
            if p < 0.001:   colors.append("#B71C1C")
            elif p < 0.01:  colors.append("#E53935")
            elif p < 0.05:  colors.append("#FF7043")
            else:           colors.append("#90A4AE")

        fig, ax = plt.subplots(figsize=(7, max(5, len(lbls) * 0.38)))
        ax.barh(y, betas, xerr=[err_lo, err_hi],
                color=colors, alpha=0.85, capsize=3, ecolor="black",
                error_kw={"linewidth": 1.0})
        ax.axvline(0, color="black", linewidth=0.9)

        for yi, (b, p) in enumerate(zip(betas, pvals)):
            stars = ("***" if p < 0.001 else "**" if p < 0.01
                     else "*" if p < 0.05 else "")
            if stars:
                ax.text(b + (0.02 if b >= 0 else -0.02), yi, stars,
                        va="center", ha="left" if b >= 0 else "right",
                        fontsize=9, color="black")

        ax.set_yticks(y)
        ax.set_yticklabels(lbls, fontsize=8)
        ax.set_xlabel("Standardised β  (ALSFRS_z → feature_z)  ± 95% CI",
                      fontsize=10)
        ax.grid(axis="x", alpha=0.3)

        from matplotlib.patches import Patch
        legend_items = [
            Patch(color="#B71C1C", label="p < 0.001"),
            Patch(color="#E53935", label="p < 0.01"),
            Patch(color="#FF7043", label="p < 0.05"),
            Patch(color="#90A4AE", label="n.s."),
        ]
        ax.legend(handles=legend_items, fontsize=8, loc="lower right")
        ax.set_title(
            f"Mixed-effects model: EEG features ~ ALSFRS\n"
            f"({n_subj_lme} subjects, all sessions, random intercept per subject)",
            fontsize=10, fontweight="bold"
        )
        fig.tight_layout()

        out_fig_lme = out_dir / "alsfrs_lme_barchart.png"
        fig.savefig(out_fig_lme, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[LME] Bar chart saved to:\n  {out_fig_lme}")
