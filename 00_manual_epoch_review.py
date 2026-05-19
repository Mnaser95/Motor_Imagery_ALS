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

SUBJECT   = "Sub3_data"

APPLY_ICA     = True
ICA_THRESHOLD = 0.3

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

    raw.filter(l_freq=6.0, h_freq=50.0, method="fir",
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
        return None, None

    # Include EOG channels so they appear in the browser
    picks = mne.pick_types(raw.info, eeg=True, eog=True, stim=False)

    epochs = mne.Epochs(
        raw, mi_events,
        event_id={"MI_Left": 8, "MI_Right": 9},
        tmin=MI_TMIN, tmax=MI_TMAX,
        baseline=None, picks=picks,
        preload=True, verbose=False,
    )

    session_key = file_path.stem.replace("_filtered", "").replace("_raw", "")
    return epochs, session_key

# =========================================================
# MAIN — iterate sessions, open browser, save rejections
# =========================================================

subject_dir   = PROJECT_DIR / SUBJECT
filtered_dir  = subject_dir / "Filtered_data"
session_files = sorted(filtered_dir.glob("*.csv"))

if len(session_files) == 0:
    raise RuntimeError(f"No filtered CSV files found in {filtered_dir}")

out_dir = PROJECT_DIR / "outputs" / "epoch_rejection" / SUBJECT
out_dir.mkdir(parents=True, exist_ok=True)

print(f"Subject : {SUBJECT}  —  {len(session_files)} sessions\n")
print("Controls in the epoch browser:")
print("  • Click an epoch to mark it bad (grey = rejected)")
print("  • Click again to unmark")
print("  • Close the window to proceed to the next session\n")

for f in session_files:
    print(f"Loading {f.name} ...", end=" ", flush=True)
    epochs, session_key = load_session_for_review(f)

    if epochs is None:
        print("skipped (no MI epochs)")
        continue

    n_epochs = len(epochs)
    print(f"{n_epochs} epochs  "
          f"(L={int((epochs.events[:,2]==8).sum())}  "
          f"R={int((epochs.events[:,2]==9).sum())})")

    out_json = out_dir / f"{session_key}_bad_epochs.json"

    # Load previously rejected indices (kept even if user doesn't re-mark them)
    prev_bad = set()
    if out_json.exists():
        with open(out_json) as fh:
            prev = json.load(fh)
        prev_bad = set(prev["bad_indices"])
        if prev_bad:
            print(f"  Previously rejected: {sorted(prev_bad)}  "
                  f"(preserved automatically — re-mark to keep, skip to keep, "
                  f"they will NOT appear pre-marked in the browser)")

    # Open interactive browser — block until window is closed
    # All epochs shown with full waveforms; mark bad ones by clicking
    epochs.plot(
        block=True,
        n_epochs=5,
        title=f"{SUBJECT}  |  {session_key}  ({n_epochs} epochs)",
        scalings={"eeg": 50e-6, "eog": 150e-6},
        show_scrollbars=True,
    )

    # Indices newly marked bad inside the browser this session
    newly_marked = {
        i for i, log in enumerate(epochs.drop_log)
        if any(r in ("USER", "user") for r in log)
    }

    # Union: preserve past decisions + add any new ones.
    # To UN-reject a previously bad epoch, edit the JSON directly.
    bad_indices = sorted(prev_bad | newly_marked)

    # Channels marked bad by clicking their name in the browser
    bad_channels = sorted(epochs.info["bads"])

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
