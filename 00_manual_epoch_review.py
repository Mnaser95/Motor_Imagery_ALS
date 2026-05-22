"""
Manual epoch review — run this BEFORE classification scripts.

For each session the MNE epoch browser opens showing all EEG + EOG channels.
  • Click on an epoch to mark it bad (it turns grey).
  • Click again to unmark.
  • Close the window when done.

Rejected indices are saved to:
  outputs/epoch_rejection/<subject>/<session>_bad_epochs.json

06_classify_single_session.py loads these files automatically if they exist.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import mne

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

MI_TMIN, MI_TMAX = 0.0, 4.0

# =========================================================
# CONFIGURATION — edit this section before running
# =========================================================

SUBJECT   = "Sub1_data"

# EOG artifact removal method applied before saving preprocessed epochs:
#   "ica"        — ICA fitted on a 1 Hz HPF copy, applied to 6 Hz filtered data
#   "regression" — linear regression of the VEOG differential from each EEG channel
#   "none"       — no removal
EOG_REMOVAL   = "none"
ICA_THRESHOLD = 0.5   # correlation threshold; only used when EOG_REMOVAL == "ica"

APPLY_SPIKE_REPLACEMENT = True   # replace samples > mean + SPIKE_STD_THRESHOLD*std with mean
SPIKE_STD_THRESHOLD     = 2.0

# Set to False to skip the interactive browser entirely (preprocessed .fif files
# are still saved and existing bad-epoch decisions are preserved).
# Useful when experimenting with ICA settings without stepping through sessions.
SHOW_EPOCH_BROWSER = False

# =========================================================
# SPIKE REPLACEMENT
# =========================================================

def _replace_spikes(raw_obj):
    """Per EEG channel: samples > mean + SPIKE_STD_THRESHOLD*std → mean."""
    eeg_idx = mne.pick_types(raw_obj.info, eeg=True, eog=False, stim=False)
    n_replaced = 0
    for ch in eeg_idx:
        d    = raw_obj._data[ch]
        mu   = d.mean()
        mask = d > mu + SPIKE_STD_THRESHOLD * d.std()
        d[mask] = mu
        n_replaced += mask.sum()
    if n_replaced:
        print(f"    Spike replacement: {n_replaced} samples clipped to channel mean")

# =========================================================
# HELPER: LOAD ONE SESSION → MI EPOCHS (EEG + EOG)
# =========================================================

def load_session_for_review(file_path):
    """
    Same preprocessing as classification scripts but keeps EOG channels
    in the epochs for visual review. Returns (epochs, session_key).
    """
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

    # "none" mode: keep a 1 Hz HPF copy — browser shows blinks, .fif stays 6 Hz
    if EOG_REMOVAL == "none":
        raw_for_display = raw.copy()
        raw_for_display.filter(l_freq=1.0, h_freq=50.0, method="fir",
                               fir_window="hamming", picks="eeg", verbose=False)
        raw_for_display.filter(l_freq=0.5, h_freq=50.0, method="fir",
                               fir_window="hamming", picks="eog", verbose=False)
        if APPLY_SPIKE_REPLACEMENT: _replace_spikes(raw_for_display)

    # ICA needs a 1 Hz HPF copy so blink energy is intact during fitting
    if EOG_REMOVAL == "ica":
        raw_for_ica = raw.copy()
        raw_for_ica.filter(l_freq=1.0, h_freq=50.0, method="fir",
                           fir_window="hamming", picks="eeg", verbose=False)
        raw_for_ica.filter(l_freq=0.5, h_freq=50.0, method="fir",
                           fir_window="hamming", picks="eog", verbose=False)
        if APPLY_SPIKE_REPLACEMENT: _replace_spikes(raw_for_ica)

    # Analysis filter applied to raw (6 Hz — used for .fif and classification)
    raw.filter(l_freq=6.0, h_freq=50.0, method="fir",
               fir_window="hamming", picks="eeg", verbose=False)
    raw.filter(l_freq=0.5, h_freq=50.0, method="fir",
               fir_window="hamming", picks="eog", verbose=False)
    if APPLY_SPIKE_REPLACEMENT: _replace_spikes(raw)

    if EOG_REMOVAL == "ica":
        ica = mne.preprocessing.ICA(
            n_components=len(EEG_CHANNELS), random_state=42,
            max_iter="auto", verbose=False,
        )
        ica.fit(raw_for_ica, picks="eeg", verbose=False)

        sources   = ica.get_sources(raw_for_ica).get_data()
        eog_data  = raw_for_ica.get_data(picks=["vEOGt", "vEOGb"])
        veog_diff = eog_data[0] - eog_data[1]

        corrs = np.array([
            np.abs(np.corrcoef(sources[i], veog_diff)[0, 1])
            for i in range(sources.shape[0])
        ])
        best = int(np.argmax(corrs))
        print(f"    ICA: best component IC{best}  corr={corrs[best]:.3f}  "
              f"(threshold={ICA_THRESHOLD})", end="")
        if corrs[best] > ICA_THRESHOLD:
            ica.exclude = [best]
            ica.apply(raw, verbose=False)
            print(f"  → IC{best} removed")
        else:
            print("  → none removed")

    elif EOG_REMOVAL == "regression":
        eeg_idx  = mne.pick_types(raw.info, eeg=True, eog=False, stim=False)
        eog_data = raw.get_data(picks=["vEOGt", "vEOGb"])
        veog     = (eog_data[0] - eog_data[1]).reshape(1, -1)
        eeg_data = raw.get_data(picks=eeg_idx)
        # Fit β per EEG channel: EEG = intercept + β × VEOG, subtract EOG term
        design         = np.vstack([np.ones((1, veog.shape[1])), veog])
        betas, _, _, _ = np.linalg.lstsq(design.T, eeg_data.T, rcond=None)
        raw._data[eeg_idx] = eeg_data - betas[1:].T @ veog
        print("    EOG regression applied")

    else:
        print("    No EOG artifact removal")

    all_events = mne.find_events(raw, stim_channel="STI",
                                 consecutive=True, min_duration=0.01,
                                 verbose=False)

    mi_events = all_events[
        np.isin(all_events[:, 2], [8, 9]) &
        (all_events[:, 0] >= mi_onset_cutoff)
    ]

    if len(mi_events) == 0:
        return None, None, None

    all_ch = ["Cz", "CP2", "CP3", "FC2", "FC3", "vEOGt", "vEOGb"]
    picks  = mne.pick_channels(raw.info["ch_names"], include=all_ch, ordered=True)

    # 6 Hz epochs — always saved to .fif for classification scripts
    epochs = mne.Epochs(
        raw, mi_events,
        event_id={"MI_Left": 8, "MI_Right": 9},
        tmin=MI_TMIN, tmax=MI_TMAX,
        baseline=None, picks=picks,
        preload=True, verbose=False,
    )

    # Display epochs: 1 Hz HPF when EOG_REMOVAL=="none" so blinks are visible;
    # otherwise same object as epochs (no extra memory)
    if EOG_REMOVAL == "none":
        picks_d        = mne.pick_channels(raw_for_display.info["ch_names"],
                                           include=all_ch, ordered=True)
        epochs_display = mne.Epochs(
            raw_for_display, mi_events,
            event_id={"MI_Left": 8, "MI_Right": 9},
            tmin=MI_TMIN, tmax=MI_TMAX,
            baseline=None, picks=picks_d,
            preload=True, verbose=False,
        )
    else:
        epochs_display = epochs

    print(f"    Channels in epochs: {epochs.ch_names}")

    session_key = file_path.stem.replace("_filtered", "").replace("_raw", "")
    return epochs, epochs_display, session_key

# =========================================================
# MAIN — iterate sessions, open browser, save rejections
# =========================================================

subject_dir   = PROJECT_DIR / SUBJECT
filtered_dir  = subject_dir / "Filtered_data"
session_files = sorted(filtered_dir.glob("*.csv"))

if len(session_files) == 0:
    raise RuntimeError(f"No filtered CSV files found in {filtered_dir}")

out_dir = PROJECT_DIR / "outputs" / Path(__file__).stem / "epoch_rejection" / SUBJECT
out_dir.mkdir(parents=True, exist_ok=True)

print(f"Subject : {SUBJECT}  —  {len(session_files)} sessions\n")
print("Controls in the epoch browser:")
print("  • Click an epoch to mark it bad (grey = rejected)")
print("  • Click again to unmark")
print("  • Close the window to proceed to the next session\n")

for f in session_files:
    print(f"Loading {f.name} ...", end=" ", flush=True)
    epochs, epochs_display, session_key = load_session_for_review(f)

    if epochs is None:
        print("skipped (no MI epochs)")
        continue

    n_epochs = len(epochs)
    print(f"{n_epochs} epochs  "
          f"(L={int((epochs.events[:,2]==8).sum())}  "
          f"R={int((epochs.events[:,2]==9).sum())})")

    # Save preprocessed epochs (post-filter, post-ICA, ALL epochs) so
    # classification scripts can load them instead of re-running preprocessing.
    preproc_dir = PROJECT_DIR / "outputs" / "00_manual_epoch_review" / "preprocessed_epochs" / SUBJECT
    preproc_dir.mkdir(parents=True, exist_ok=True)
    preproc_file = preproc_dir / f"{session_key}_epo.fif"
    epochs.save(str(preproc_file), overwrite=True, verbose=False)
    print(f"    Preprocessed epochs saved → {preproc_file.name}")

    out_json = out_dir / f"{session_key}_bad_epochs.json"

    # Load previously rejected indices
    prev_bad = set()
    prev_bad_channels = []
    if out_json.exists():
        with open(out_json) as fh:
            prev = json.load(fh)
        prev_bad = set(prev["bad_indices"])
        prev_bad_channels = prev.get("bad_channels", [])
        if prev_bad:
            print(f"  Previously rejected: {sorted(prev_bad)}")

    if SHOW_EPOCH_BROWSER:
        disp_label = "(1 Hz HPF — blinks visible)" if EOG_REMOVAL == "none" else ""
        plot_picks = mne.pick_channels(
            epochs_display.info["ch_names"],
            include=["Cz", "CP2", "CP3", "FC2", "FC3", "vEOGt", "vEOGb"],
            ordered=True,
        )
        print(f"    Plot picks: {[epochs_display.ch_names[i] for i in plot_picks]}  {disp_label}")

        epochs_display.plot(
            picks=plot_picks,
            block=True,
            n_epochs=5,
            n_channels=len(plot_picks),
            title=f"{SUBJECT}  |  {session_key}  ({n_epochs} epochs)  {disp_label}",
            scalings={"eeg": 50e-6, "eog": 150e-6},
            show_scrollbars=True,
        )

        n_after_browser = len(epochs_display)
        print(f"    After browser: {n_after_browser} epochs remain "
              f"({n_epochs - n_after_browser} newly dropped by browser)")
        nonempty_logs = [(i, list(log)) for i, log in enumerate(epochs_display.drop_log) if log]
        print(f"    drop_log non-empty entries: {nonempty_logs}")

        newly_marked = {
            i for i, log in enumerate(epochs_display.drop_log)
            if any(r in ("USER", "user") for r in log)
        }
        print(f"    Newly marked indices: {sorted(newly_marked)}")

        bad_indices  = sorted(prev_bad | newly_marked)
        bad_channels = sorted(epochs_display.info["bads"])
    else:
        print("    Browser skipped (SHOW_EPOCH_BROWSER=False) — existing decisions preserved")
        bad_indices  = sorted(prev_bad)
        bad_channels = prev_bad_channels

    payload = {
        "subject":      SUBJECT,
        "session":      session_key,
        "total_epochs": n_epochs,
        "bad_indices":  bad_indices,
        "n_bad":        len(bad_indices),
        "bad_channels": bad_channels,
    }

    with open(out_json, "w") as fh:
        json.dump(payload, fh, indent=2)

    pct = len(bad_indices) / n_epochs * 100
    ch_str = f"  bad channels: {bad_channels}" if bad_channels else ""
    print(f"  Saved {len(bad_indices)}/{n_epochs} bad epochs "
          f"({pct:.0f}%){ch_str}  →  {out_json.name}\n")

print("Review complete.")
