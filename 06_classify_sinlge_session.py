from pathlib import Path
import json
import re
import math
import numpy as np
import pandas as pd
import scipy.linalg
import matplotlib.pyplot as plt
import mne
from mne.decoding import CSP
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import csv
from collections import defaultdict

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

MI_TMIN, MI_TMAX = 0, 4

# =========================================================
# CONFIGURATION — edit this section before running
# =========================================================

SUBJECT  = "Sub3_data"   # ← change this before running
SUBJECTS = [SUBJECT]

N_SPLITS    = 5
TRAIN_RATIO = 0.75

APPLY_ICA     = True
ICA_THRESHOLD = 0.3

CSP_COMPONENTS = 4

GROUP_SIZE         = 5
GROUP_SIZE_SUBJECT = {"Sub3_data": 4}
N_GROUPS_SUBJECT   = {"Sub2_data": 2}

EEG_CH_NAMES = ["Cz", "CP2", "CP3", "FC2", "FC3"]

# Exponent for quality score: q_i = (1/mean_Z_i)^k
#   Higher k → larger spread between best and worst epochs
QUALITY_EXPONENT = 3

# Sessions shown in black and excluded from group mean calculations.
MARKED_SESSIONS = {
    "Sub1_data": {5, 6, 12},
    "Sub2_data": {8},
    "Sub3_data": {14, 15, 19},
}

# =========================================================
# WEIGHTED CSP
# =========================================================

class WeightedCSP:
    """
    CSP with per-epoch quality-weighted class covariance.
    Each epoch i contributes q_i * C_i to its class covariance matrix.
    Epochs with high mean impedance (low q_i) count less toward the
    spatial filters that CSP learns.
    """

    def __init__(self, n_components=4):
        self.n_components = n_components
        self.filters_     = None

    def fit(self, X, y, sample_weight=None):
        n_epochs, n_ch, n_times = X.shape
        if sample_weight is None:
            sample_weight = np.ones(n_epochs)

        covs = []
        for cls in [0, 1]:
            idx = (y == cls)
            w   = sample_weight[idx].copy()
            w  /= w.sum()
            C   = np.zeros((n_ch, n_ch))
            for wi, xi in zip(w, X[idx]):
                C += wi * (xi @ xi.T) / n_times
            covs.append(C)

        evals, evecs = scipy.linalg.eigh(covs[1], covs[0] + covs[1])

        n_low  = self.n_components // 2
        n_high = self.n_components - n_low
        idx_s  = np.argsort(evals)
        sel    = np.concatenate([idx_s[:n_low], idx_s[-n_high:]])
        self.filters_ = evecs[:, sel]
        return self

    def transform(self, X):
        return np.array([np.var(self.filters_.T @ x, axis=1) for x in X])

    def fit_transform(self, X, y, sample_weight=None):
        return self.fit(X, y, sample_weight).transform(X)


# =========================================================
# HELPER: LOAD IMPEDANCE TABLE
# =========================================================

def load_impedance_table(z_path):
    """
    Returns a DataFrame with integer 'Time' (seconds) and one column per
    EEG_CHANNEL. 'Clipping' entries and other non-numeric values → NaN.
    """
    if not z_path.exists():
        return None
    try:
        df = pd.read_csv(z_path, comment="#", skipinitialspace=True)
    except Exception:
        return None

    required = ["Time"] + EEG_CHANNELS
    if any(c not in df.columns for c in required):
        return None

    df["Time"] = pd.to_numeric(df["Time"], errors="coerce")
    for col in EEG_CHANNELS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df[required].dropna(subset=["Time"]).reset_index(drop=True)


def epoch_quality(z_table, onset_sec, active_ch_cols):
    """
    Scalar quality score for one epoch: 1 / mean_impedance at onset time.
    Uses only the active (non-dropped) EEG channel columns.
    Returns 1.0 (neutral) if no valid impedance data.
    """
    if z_table is None or not active_ch_cols:
        return 1.0

    idx   = (z_table["Time"] - onset_sec).abs().idxmin()
    z_row = z_table.loc[idx, active_ch_cols].values.astype(float)
    valid = z_row[np.isfinite(z_row) & (z_row > 0)]

    if len(valid) == 0:
        return 1.0

    return float(np.mean(valid)) ** (-QUALITY_EXPONENT)


# =========================================================
# HELPER: LOAD SESSION → MI EPOCHS + ONSET TIMES
# =========================================================

def load_session(file_path):
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
        return None, None, None

    trigger = df[trigger_col].astype(int)

    baseline_rows   = trigger[trigger.isin([1, 2, 3, 4, 5])].index
    mi_onset_cutoff = int(baseline_rows[-1]) if len(baseline_rows) > 0 else 0

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

    raw.filter(l_freq=6.0, h_freq=45.0, method="fir",
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

    all_events = mne.find_events(raw, stim_channel="STI",
                                 consecutive=True, min_duration=0.01,
                                 verbose=False)

    mi_events = all_events[
        np.isin(all_events[:, 2], [8, 9]) &
        (all_events[:, 0] >= mi_onset_cutoff)
    ]

    if len(mi_events) == 0:
        return None, None, None

    picks = mne.pick_types(raw.info, eeg=True, eog=False, stim=False)
    epochs = mne.Epochs(
        raw, mi_events,
        event_id={"MI_Left": 8, "MI_Right": 9},
        tmin=MI_TMIN, tmax=MI_TMAX,
        baseline=None, picks=picks,
        preload=True, verbose=False,
    )

    X         = epochs.get_data()
    y         = (epochs.events[:, 2] == 9).astype(int)
    onset_sec = epochs.events[:, 0] / sfreq
    return X, y, onset_sec


# =========================================================
# CLASSIFIERS
# =========================================================

def train_weighted_csp_lda(X_tr, y_tr, X_te, y_te, q_tr):
    n_csp = min(CSP_COMPONENTS, X_tr.shape[1] - 1)
    csp   = WeightedCSP(n_components=n_csp)
    lda   = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")

    X_tr_feat = np.nan_to_num(csp.fit_transform(X_tr, y_tr, sample_weight=q_tr))
    X_te_feat = np.nan_to_num(csp.transform(X_te))

    lda.fit(X_tr_feat, y_tr)
    return (lda.predict(X_te_feat) == y_te).mean() * 100


def train_standard_csp_lda(X_tr, y_tr, X_te, y_te):
    n_csp = min(CSP_COMPONENTS, X_tr.shape[1] - 1)
    csp   = CSP(n_components=n_csp, reg=None, log=False, norm_trace=False)
    lda   = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")

    X_tr_feat = np.nan_to_num(csp.fit_transform(X_tr, y_tr))
    X_te_feat = np.nan_to_num(csp.transform(X_te))

    lda.fit(X_tr_feat, y_tr)
    return (lda.predict(X_te_feat) == y_te).mean() * 100


# =========================================================
# MAIN
# =========================================================

all_results_weighted = []
all_results_standard = []

for subject in SUBJECTS:

    subject_dir  = PROJECT_DIR / subject
    filtered_dir = subject_dir / "Filtered_data"
    z_dir        = subject_dir / "Z"
    marked       = MARKED_SESSIONS.get(subject, set())

    session_files = sorted(
        filtered_dir.glob("*.csv"),
        key=lambda f: int(m.group(1)) if (m := re.search(r"Ses(\d+)", f.stem, re.IGNORECASE)) else 0
    )

    if len(session_files) == 0:
        print(f"[{subject}] No filtered CSV files found — skipping.\n")
        continue

    print(f"{'=' * 60}")
    print(f"  Subject: {subject}  —  {len(session_files)} files found")
    print(f"{'=' * 60}")

    for f in session_files:
        ses_name = f.stem.replace("_filtered", "")
        print(f"\n  Session: {ses_name}", flush=True)

        X, y, onset_sec = load_session(f)
        if X is None:
            print("    skipped (no MI epochs)")
            continue

        # --- Epoch rejection ---
        rejection_file = (PROJECT_DIR / "outputs" / "00_manual_epoch_review" / "epoch_rejection"
                          / subject / f"{ses_name}_bad_epochs.json")
        bad_ch = []
        if rejection_file.exists():
            with open(rejection_file) as fh:
                bad = json.load(fh)
            good_mask = np.ones(len(X), dtype=bool)
            good_mask[bad["bad_indices"]] = False
            X, y, onset_sec = X[good_mask], y[good_mask], onset_sec[good_mask]
            print(f"    Rejection file loaded: {bad['n_bad']} epochs dropped")
            bad_ch = bad.get("bad_channels", [])

        # --- Drop bad channels ---
        active_eeg_cols = [c for c, name in zip(EEG_CHANNELS, EEG_CH_NAMES)
                           if name not in bad_ch]
        if bad_ch:
            ch_idx = [EEG_CH_NAMES.index(ch) for ch in bad_ch
                      if ch in EEG_CH_NAMES]
            X = np.delete(X, ch_idx, axis=1)
            n_remaining = X.shape[1]
            print(f"    Bad channels dropped: {bad_ch}  "
                  f"({n_remaining} channels remaining, "
                  f"CSP capped at {min(CSP_COMPONENTS, n_remaining-1)} components)")
            if n_remaining < 2:
                print("    skipped (fewer than 2 EEG channels remaining)")
                continue

        # --- Load impedance table and compute per-epoch quality scores ---
        ses_match = re.search(r"Ses(\d+)", ses_name, re.IGNORECASE)
        z_table   = None
        if ses_match:
            z_path  = z_dir / f"{int(ses_match.group(1))}.csv"
            z_table = load_impedance_table(z_path)

        if z_table is not None:
            q = np.array([
                epoch_quality(z_table, t, active_eeg_cols)
                for t in onset_sec
            ])
            q_min, q_max = q.min(), q.max()
            print(f"    Impedance quality scores: "
                  f"min={q_min:.3f}  max={q_max:.3f}  "
                  f"ratio={q_max/max(q_min,1e-9):.2f}x")
        else:
            q = np.ones(len(X))
            print("    Impedance file not found — uniform quality scores")

        n_trials = len(X)
        print(f"    {n_trials} epochs  (L={int((y==0).sum())}  R={int((y==1).sum())})")

        weighted_accs = []
        standard_accs = []

        for split_idx in range(N_SPLITS):
            seed = split_idx * 17 + 3

            idx_all = np.arange(len(X))
            idx_tr, idx_te = train_test_split(
                idx_all, test_size=(1.0 - TRAIN_RATIO),
                random_state=seed, stratify=y,
            )
            X_tr, y_tr, q_tr = X[idx_tr], y[idx_tr], q[idx_tr]
            X_te, y_te        = X[idx_te], y[idx_te]

            weighted_accs.append(train_weighted_csp_lda(X_tr, y_tr, X_te, y_te, q_tr))
            standard_accs.append(train_standard_csp_lda(X_tr, y_tr, X_te, y_te))

        all_results_weighted.append((subject, ses_name, weighted_accs))
        all_results_standard.append((subject, ses_name, standard_accs))


# =========================================================
# PLOT + CSV HELPER
# =========================================================

def plot_and_save(all_results, title_suffix, fig_filename, csv_filename, csv_detail_filename):

    subject_sessions: dict = defaultdict(list)
    for subject, ses_name, accs in all_results:
        subject_sessions[subject].append((ses_name, accs))

    COLOR_LDA = "#2196F3"
    COLOR_AVG = "#E65100"

    n_subj = len(subject_sessions)
    fig, axes = plt.subplots(
        n_subj, 1,
        figsize=(max(10, len(all_results) * 1.4), 5 * n_subj),
        squeeze=False,
    )

    group_rows = []

    for ax_idx, (subject, sessions) in enumerate(subject_sessions.items()):
        ax     = axes[ax_idx][0]
        marked = MARKED_SESSIONS.get(subject, set())

        if subject in N_GROUPS_SUBJECT:
            grp = math.ceil(len(sessions) / N_GROUPS_SUBJECT[subject])
        else:
            grp = GROUP_SIZE_SUBJECT.get(subject, GROUP_SIZE)

        n        = len(sessions)
        lda_data = [s[1] for s in sessions]
        centers  = np.arange(n, dtype=float)

        ses_nums = []
        for sn, _ in sessions:
            m = re.search(r"Ses(\d+)", sn, re.IGNORECASE)
            ses_nums.append(int(m.group(1)) if m else sn)

        for i, (data_i, num) in enumerate(zip(lda_data, ses_nums)):
            color = "black" if num in marked else COLOR_LDA
            ax.boxplot(
                [data_i], positions=[centers[i]], widths=0.6,
                patch_artist=True,
                medianprops=dict(color="yellow", linewidth=2),
                boxprops=dict(facecolor=color, alpha=0.85),
                whiskerprops=dict(color=color, linewidth=1.2),
                capprops=dict(color=color, linewidth=1.2),
                flierprops=dict(marker="o", markerfacecolor=color,
                                markersize=4, alpha=0.5, linestyle="none"),
            )

        ax.axhline(50, color="gray", linestyle="--", linewidth=1, alpha=0.6)

        grp_x        = []
        grp_mean_all = []
        grp_mean_act = []

        for g_idx, start in enumerate(range(0, n, grp)):
            end = min(start + grp, n)

            if start > 0:
                ax.axvline(centers[start] - 0.5, color="black",
                           linestyle=":", linewidth=1, alpha=0.4)

            grp_x.append(float(np.mean(centers[start:end])))

            all_means = [float(np.mean(lda_data[i])) for i in range(start, end)]
            grp_mean_all.append(float(np.mean(all_means)))

            active_means = [
                float(np.mean(lda_data[i]))
                for i in range(start, end)
                if ses_nums[i] not in marked
            ]
            grp_mean_act.append(
                float(np.mean(active_means)) if active_means else float(np.mean(all_means))
            )

            group = sessions[start:end]
            lda_means_csv = [
                float(np.mean(la)) for sn, la in group
                if (m := re.search(r"Ses(\d+)", sn, re.IGNORECASE))
                and int(m.group(1)) not in marked
            ]
            if lda_means_csv:
                group_rows.append((
                    subject, f"Group {g_idx + 1}", sessions[start][0], sessions[end - 1][0],
                    float(np.mean(lda_means_csv)), float(np.std(lda_means_csv)),
                ))

        ax.plot(grp_x, grp_mean_all, "o-",  color=COLOR_AVG,  linewidth=2,
                markersize=7, zorder=6, label="Group mean (all)")
        ax.plot(grp_x, grp_mean_act, "s--", color="#4CAF50",   linewidth=2,
                markersize=7, zorder=6, label="Group mean (excl. marked)")

        for x, va, ve in zip(grp_x, grp_mean_all, grp_mean_act):
            ax.text(x, va + 1.2, f"{va:.1f}%", ha="center", fontsize=7,
                    color=COLOR_AVG, fontweight="bold")
            ax.text(x, ve - 2.5, f"{ve:.1f}%", ha="center", fontsize=7,
                    color="#4CAF50", fontweight="bold")

        ax.set_xticks(centers)
        ax.set_xticklabels([str(s) for s in ses_nums], fontsize=9)
        ax.set_xlim(centers[0] - 0.5, centers[-1] + 0.5)
        ax.set_ylim(0, 110)
        ax.set_xlabel("Session", fontsize=10)
        ax.set_ylabel("Accuracy (%)", fontsize=10)
        ax.set_title(
            f"{subject}  —  {title_suffix}",
            fontsize=11, fontweight="bold",
        )
        ax.grid(axis="y", alpha=0.3)
        ax.legend(
            handles=[
                plt.Line2D([0], [0], color="gray", linestyle="--",
                           linewidth=1, label="Chance (50%)"),
                plt.Line2D([0], [0], color=COLOR_AVG, marker="o",
                           linewidth=2, label="Group mean (all)"),
                plt.Line2D([0], [0], color="#4CAF50", marker="s",
                           linestyle="--", linewidth=2, label="Group mean (excl. marked)"),
            ],
            fontsize=9, loc="upper right",
        )

    fig.tight_layout()

    out_dir = PROJECT_DIR / "outputs" / Path(__file__).stem
    out_dir.mkdir(parents=True, exist_ok=True)

    out_fig = out_dir / fig_filename
    fig.savefig(out_fig, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Figure saved to:\n  {out_fig}")

    out_csv = out_dir / csv_filename
    with open(out_csv, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["subject", "group", "first_session", "last_session",
                         "lda_mean_%", "lda_std_%"])
        for row in group_rows:
            subj, lbl, first, last, lm, ls = row
            writer.writerow([subj, lbl, first, last, f"{lm:.2f}", f"{ls:.2f}"])

    out_csv_detail = out_dir / csv_detail_filename
    with open(out_csv_detail, "w", newline="") as fh:
        writer = csv.writer(fh)
        split_headers = [f"split_{i+1}_%" for i in range(N_SPLITS)]
        writer.writerow(
            ["subject", "session", "lda_mean_%", "lda_std_%"]
            + [f"lda_{h}" for h in split_headers]
        )
        for subject, ses_name, accs in all_results:
            writer.writerow(
                [subject, ses_name,
                 f"{np.mean(accs):.2f}", f"{np.std(accs):.2f}"]
                + [f"{a:.2f}" for a in accs]
            )

    print(f"\nResults saved to:\n  {out_csv}\n  {out_csv_detail}")


# =========================================================
# PLOT AND SAVE BOTH METHODS
# =========================================================

plot_and_save(
    all_results_weighted,
    title_suffix=f"impedance-weighted CSP+LDA  ({N_SPLITS} splits, exponent={QUALITY_EXPONENT})",
    fig_filename=f"{SUBJECT}_impedance_weighted_boxplot.png",
    csv_filename=f"{SUBJECT}_impedance_weighted_results.csv",
    csv_detail_filename=f"{SUBJECT}_impedance_weighted_results_per_session.csv",
)

plot_and_save(
    all_results_standard,
    title_suffix=f"standard CSP+LDA  ({N_SPLITS} splits)",
    fig_filename=f"{SUBJECT}_standard_csp_boxplot.png",
    csv_filename=f"{SUBJECT}_standard_csp_results.csv",
    csv_detail_filename=f"{SUBJECT}_standard_csp_results_per_session.csv",
)
