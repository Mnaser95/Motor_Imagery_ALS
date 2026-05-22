from pathlib import Path
import json
import re
import numpy as np
import pandas as pd
import scipy.linalg
import mne
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt
import joblib
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

MI_TMIN, MI_TMAX = 0, 4

# =========================================================
# CONFIGURATION — edit this section before running
# =========================================================

SUBJECT  = "Sub1_data"   # ← change this before running
SUBJECTS = [SUBJECT]


CSP_COMPONENTS = 4

EEG_CH_NAMES = ["Cz", "CP2", "CP3", "FC2", "FC3"]

# Exponent for quality score: q_i = (1/mean_Z_i)^k
QUALITY_EXPONENT = 3

# Sessions excluded entirely from training (same as EXCLUDE_SESSIONS in script 07)
EXCLUDE_SESSIONS = {
    "Sub1_data": {5, 6, 12},
    "Sub2_data": {8},
    "Sub3_data": {14, 15, 19},
}

# =========================================================
# WEIGHTED CSP
# =========================================================

class WeightedCSP:
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
               fir_window="hamming", picks="eeg", verbose=False)
    raw.filter(l_freq=0.5, h_freq=45.0, method="fir",
               fir_window="hamming", picks="eog", verbose=False)

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
# MAIN
# =========================================================

for subject in SUBJECTS:

    subject_dir  = PROJECT_DIR / subject
    filtered_dir = subject_dir / "Filtered_data"
    z_dir        = subject_dir / "Z"
    excluded     = EXCLUDE_SESSIONS.get(subject, set())

    session_files = sorted(
        filtered_dir.glob("*.csv"),
        key=lambda f: int(m.group(1)) if (m := re.search(r"Ses(\d+)", f.stem, re.IGNORECASE)) else 0,
    )

    if not session_files:
        print(f"[{subject}] No filtered CSV files found — skipping.\n")
        continue

    print(f"\n{'=' * 60}")
    print(f"  Subject: {subject}  —  {len(session_files)} files found")
    print(f"{'=' * 60}")

    X_all, y_all, q_all, ses_label_all, trial_num_all = [], [], [], [], []
    session_log = []   # for the training summary CSV

    for f in session_files:
        ses_name = f.stem.replace("_filtered", "")
        ses_m    = re.search(r"Ses(\d+)", ses_name, re.IGNORECASE)
        ses_num  = int(ses_m.group(1)) if ses_m else None

        if ses_num in excluded:
            print(f"\n  Session: {ses_name}  — excluded")
            continue

        print(f"\n  Session: {ses_name}", flush=True)

        preproc_file = (PROJECT_DIR / "outputs" / "00_manual_epoch_review"
                        / "preprocessed_epochs" / subject / f"{ses_name}_epo.fif")
        if preproc_file.exists():
            print(f"    Loading preprocessed epochs: {preproc_file.name}")
            _epo      = mne.read_epochs(preproc_file, verbose=False)
            eeg_picks = mne.pick_types(_epo.info, eeg=True, eog=False, stim=False)
            X         = _epo.get_data(picks=eeg_picks)
            y         = (_epo.events[:, 2] == 9).astype(int)
            onset_sec = _epo.events[:, 0] / sfreq
        else:
            print("    No preprocessed file found — running filter+ICA")
            X, y, onset_sec = load_session(f)
        if X is None:
            print("    skipped (no MI epochs)")
            continue

        # --- Epoch rejection ---
        rej_file = (PROJECT_DIR / "outputs" / "00_manual_epoch_review" / "epoch_rejection"
                    / subject / f"{ses_name}_bad_epochs.json")
        bad_ch = []
        if rej_file.exists():
            with open(rej_file) as fh:
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
            ch_idx = [EEG_CH_NAMES.index(ch) for ch in bad_ch if ch in EEG_CH_NAMES]
            X = np.delete(X, ch_idx, axis=1)
            if X.shape[1] < 2:
                print("    skipped (fewer than 2 EEG channels remaining)")
                continue
            print(f"    Bad channels dropped: {bad_ch}")

        # --- Impedance quality scores ---
        z_table = None
        if ses_num is not None:
            z_path  = z_dir / f"{ses_num}.csv"
            z_table = load_impedance_table(z_path)

        if z_table is not None:
            q = np.array([epoch_quality(z_table, t, active_eeg_cols) for t in onset_sec])
            print(f"    Impedance quality: min={q.min():.3f}  max={q.max():.3f}")
        else:
            q = np.ones(len(X))
            print("    Impedance file not found — uniform quality scores")

        n_left  = int((y == 0).sum())
        n_right = int((y == 1).sum())
        print(f"    {len(X)} epochs  (L={n_left}  R={n_right})")

        X_all.append(X)
        y_all.append(y)
        q_all.append(q)
        ses_label_all.append(np.full(len(X), ses_name, dtype=object))
        trial_num_all.append(np.arange(0, len(X)))
        session_log.append({
            "session": ses_name,
            "n_epochs": len(X),
            "n_left": n_left,
            "n_right": n_right,
        })

    if not X_all:
        print(f"\n  No usable data for {subject} — skipping.\n")
        continue

    # Pad to common channel count (in case bad channels were dropped unevenly)
    max_ch = max(x.shape[1] for x in X_all)
    X_all_padded = []
    for x in X_all:
        if x.shape[1] < max_ch:
            pad = np.zeros((x.shape[0], max_ch - x.shape[1], x.shape[2]))
            x   = np.concatenate([x, pad], axis=1)
        X_all_padded.append(x)

    X_train    = np.concatenate(X_all_padded, axis=0)
    y_train    = np.concatenate(y_all,        axis=0)
    q_train    = np.concatenate(q_all,        axis=0)
    ses_labels  = np.concatenate(ses_label_all, axis=0)
    trial_nums  = np.concatenate(trial_num_all, axis=0)
    unique_ses  = list(dict.fromkeys(ses_labels))   # ordered, unique session names

    n_total = len(X_train)
    print(f"\n  Total training epochs: {n_total}  "
          f"(L={int((y_train==0).sum())}  R={int((y_train==1).sum())})")

    out_dir = PROJECT_DIR / "outputs" / Path(__file__).stem / subject
    out_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------
    # Train: impedance-weighted CSP + LDA
    # -------------------------------------------------------
    n_csp = min(CSP_COMPONENTS, X_train.shape[1] - 1)

    w_csp = WeightedCSP(n_components=n_csp)
    w_lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    X_feat_w = np.nan_to_num(w_csp.fit_transform(X_train, y_train, sample_weight=q_train))
    w_lda.fit(X_feat_w, y_train)

    w_model_path = out_dir / f"{subject}_weighted_csp_lda.pkl"
    joblib.dump({"csp": w_csp, "lda": w_lda, "n_csp": n_csp,
                 "subject": subject, "n_epochs": n_total}, w_model_path)
    print(f"\n  Weighted CSP+LDA trained and saved:\n    {w_model_path}")

    # -------------------------------------------------------
    # Train: standard CSP + LDA
    # -------------------------------------------------------
    s_csp = CSP(n_components=n_csp, reg=None, log=False, norm_trace=False)
    s_lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    X_feat_s = np.nan_to_num(s_csp.fit_transform(X_train, y_train))
    s_lda.fit(X_feat_s, y_train)

    s_model_path = out_dir / f"{subject}_standard_csp_lda.pkl"
    joblib.dump({"csp": s_csp, "lda": s_lda, "n_csp": n_csp,
                 "subject": subject, "n_epochs": n_total}, s_model_path)
    print(f"  Standard CSP+LDA  trained and saved:\n    {s_model_path}")

    # -------------------------------------------------------
    # CSP feature-space plots (training data only)
    # -------------------------------------------------------
    TRAIN_COLORS  = ["#E53935", "#1565C0", "#F9A825", "#AB47BC", "#FF7043",
                     "#00897B", "#6D4C41", "#546E7A", "#43A047", "#FB8C00"]
    CLASS_MARKERS = {0: "o", 1: "^"}
    CLASS_LABELS  = {0: "MI Left", 1: "MI Right"}

    color_map = {s: TRAIN_COLORS[i % len(TRAIN_COLORS)]
                 for i, s in enumerate(unique_ses)}

    CLASS_COLORS = {0: "#1565C0", 1: "#E53935"}   # blue = MI Left, red = MI Right

    for method_label, csp_obj, is_mne in [
        ("Impedance-Weighted CSP", w_csp, False),
        ("Standard CSP",           s_csp, True),
    ]:
        tag = "weighted" if not is_mne else "standard"

        # Transform all training epochs once with the fitted model
        if is_mne:
            csp_viz = CSP(n_components=min(2, n_csp), reg=None, log=True, norm_trace=False)
            csp_viz.fit(X_train, y_train)
            feats = np.nan_to_num(csp_viz.transform(X_train))
        else:
            filters2 = csp_obj.filters_[:, :min(2, n_csp)]
            feats = np.array([
                np.log(np.var(filters2.T @ x, axis=1) + 1e-30)
                for x in X_train
            ])

        if feats.shape[1] < 2:
            print(f"  (CSP scatter skipped for {method_label} — fewer than 2 components)")
            continue

        for ses in unique_ses:
            mask_ses  = (ses_labels == ses)
            feats_ses = feats[mask_ses]
            y_ses     = y_train[mask_ses]
            tnum_ses  = trial_nums[mask_ses]

            fig, ax = plt.subplots(figsize=(6, 5))

            for cls in [0, 1]:
                mask = (y_ses == cls)
                if not mask.any():
                    continue
                ax.scatter(feats_ses[mask, 0], feats_ses[mask, 1],
                           color=CLASS_COLORS[cls], marker=CLASS_MARKERS[cls],
                           label=CLASS_LABELS[cls],
                           alpha=0.70, s=50,
                           edgecolors="black" if cls == 0 else "none",
                           linewidths=0.5)
                mx, my = feats_ses[mask, 0].mean(), feats_ses[mask, 1].mean()
                ax.scatter(mx, my, marker="*", s=300, color="white",
                           edgecolors=CLASS_COLORS[cls], linewidths=1.8, zorder=5)
                ax.scatter(mx, my, marker=CLASS_MARKERS[cls], s=120,
                           color=CLASS_COLORS[cls], edgecolors="black",
                           linewidths=1.0, zorder=6)

            # Trial number labels
            for i in range(len(feats_ses)):
                ax.text(feats_ses[i, 0], feats_ses[i, 1],
                        str(tnum_ses[i]),
                        fontsize=6, ha="center", va="bottom",
                        color="black", alpha=0.75)

            ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
            ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")
            ax.set_xlabel("CSP Component 1 (log-var)", fontsize=10)
            ax.set_ylabel("CSP Component 2 (log-var)", fontsize=10)
            ax.legend(fontsize=9, loc="upper right", framealpha=0.8)
            fig.suptitle(
                f"{subject}  |  {ses}  |  {method_label}\n"
                f"CSP trained on all sessions  ({mask_ses.sum()} epochs shown)",
                fontsize=10, fontweight="bold",
            )
            fig.tight_layout()

            plot_path = out_dir / f"{subject}_{tag}_csp_scatter_{ses}.png"
            fig.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  CSP scatter saved: {plot_path.name}")

    # -------------------------------------------------------
    # Training summary CSV
    # -------------------------------------------------------
    summary_path = out_dir / f"{subject}_training_summary.csv"
    with open(summary_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["session", "n_epochs", "n_left", "n_right"])
        writer.writeheader()
        writer.writerows(session_log)
        writer.writerow({"session": "TOTAL", "n_epochs": n_total,
                         "n_left": int((y_train==0).sum()),
                         "n_right": int((y_train==1).sum())})
    print(f"  Training summary saved:\n    {summary_path}\n")

print("Done.")
