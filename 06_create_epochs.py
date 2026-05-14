from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mne

# =========================================================
# PROJECT DIRECTORY
# =========================================================

PROJECT_DIR = Path(__file__).resolve().parent

sfreq = 300

# =========================================================
# CHANNEL DEFINITIONS
# =========================================================

ch_map = {
    "S1:CZ": "Cz",
    "S2:CP2": "CP2",
    "S3:CP3": "CP3",
    "S4:FC2": "FC2",
    "S5:FC3": "FC3",
    "S6:vEOGt": "vEOGt",
    "S7:vEOGb": "vEOGb"
}

eeg_channels = [
    "S1:CZ",
    "S2:CP2",
    "S3:CP3",
    "S4:FC2",
    "S5:FC3"
]

eog_channels = [
    "S6:vEOGt",
    "S7:vEOGb"
]

all_signal_channels = eeg_channels + eog_channels

# =========================================================
# SELECT SUBJECT
# =========================================================

subject_folders = sorted(PROJECT_DIR.glob("Sub*_data"))

print("\nAvailable subjects:")
for i, s in enumerate(subject_folders):
    print(f"{i + 1}. {s.name}")

subject_idx = int(input("\nSelect subject number: ")) - 1
subject_dir = subject_folders[subject_idx]

# =========================================================
# SELECT FILTERED SESSION
# =========================================================

filtered_dir = subject_dir / "Filtered_data"

filtered_files = sorted(filtered_dir.glob("*.csv"))

print(f"\nAvailable filtered sessions for {subject_dir.name}:")
for i, f in enumerate(filtered_files):
    print(f"{i + 1}. {f.name}")

session_idx = int(input("\nSelect filtered session number: ")) - 1

file_path = filtered_files[session_idx]

print(f"\nSelected file:\n{file_path.name}")

# =========================================================
# LOAD CSV
# =========================================================

df = pd.read_csv(
    file_path,
    comment="#",
    skipinitialspace=True
)

# =========================================================
# DETECT TRIGGER COLUMN
# =========================================================

trigger_col = None

for candidate in ["Manual trigger", "Trigger"]:

    if candidate in df.columns:

        temp = pd.to_numeric(
            df[candidate],
            errors="coerce"
        ).fillna(0).astype(int)

        if temp.sum() > 0:

            trigger_col = candidate

            df[candidate] = temp

            break

if trigger_col is None:
    raise RuntimeError("No active trigger column found.")

print(f"\nUsing trigger column: {trigger_col}")

# =========================================================
# CREATE MNE RAW OBJECT
# =========================================================

data_cols = all_signal_channels + [trigger_col]

ch_names = [ch_map[c] for c in all_signal_channels] + ["STI"]

ch_types = [
    "eeg",
    "eeg",
    "eeg",
    "eeg",
    "eeg",
    "eog",
    "eog",
    "stim"
]

data = df[data_cols].values.T

# Convert EEG/EOG microvolts → volts
data[:-1] = data[:-1] * 1e-6

data = np.nan_to_num(data)

info = mne.create_info(
    ch_names=ch_names,
    sfreq=sfreq,
    ch_types=ch_types
)

raw = mne.io.RawArray(data, info)

# =========================================================
# APPLY MONTAGE
# =========================================================

montage = mne.channels.make_standard_montage("standard_1020")

raw.set_montage(
    montage,
    on_missing="ignore"
)

print("\nRaw object created:")
print(raw)

# =========================================================
# FIND EVENTS
# =========================================================

events = mne.find_events(
    raw,
    stim_channel="STI",
    consecutive=True,
    min_duration=0.01,
    verbose=False
)

event_dict = {
    "MI_Left": 8,
    "MI_Right": 9
}

mi_events = events[
    np.isin(events[:, 2], [8, 9])
]

print("\n===================================")
print("EVENT SUMMARY")
print("===================================")

print(f"\nTotal MI events: {len(mi_events)}")

print(f"MI_Left:  {np.sum(mi_events[:, 2] == 8)}")
print(f"MI_Right: {np.sum(mi_events[:, 2] == 9)}")

if len(mi_events) == 0:
    raise RuntimeError("No MI events found.")

# =========================================================
# EPOCH SETTINGS
# =========================================================

tmin = -2.0
tmax = 4.0

print("\nEpoch window:")
print(f"{tmin} sec → {tmax} sec")

# =========================================================
# CREATE EPOCHS
# =========================================================

picks = mne.pick_types(
    raw.info,
    eeg=True,
    eog=True,
    stim=False
)

epochs = mne.Epochs(
    raw,
    mi_events,
    event_id=event_dict,
    tmin=tmin,
    tmax=tmax,
    baseline=(-2.0, -0.1),
    picks=picks,
    preload=True,
    verbose=True
)

print("\n===================================")
print("EPOCH SUMMARY")
print("===================================")

print(epochs)

print(f"\nLeft epochs:  {len(epochs['MI_Left'])}")
print(f"Right epochs: {len(epochs['MI_Right'])}")

# =========================================================
# OPTIONAL ARTIFACT REJECTION
# =========================================================

reject_criteria = dict(
    eeg=150e-6
)

epochs_clean = epochs.copy()

epochs_clean.drop_bad(
    reject=reject_criteria
)

print("\n===================================")
print("ARTIFACT REJECTION")
print("===================================")

print(f"\nBefore rejection: {len(epochs)}")
print(f"After rejection:  {len(epochs_clean)}")
print(f"Dropped epochs:   {len(epochs) - len(epochs_clean)}")

# =========================================================
# AVERAGED RESPONSES
# =========================================================

evoked_left = epochs_clean["MI_Left"].average()

evoked_right = epochs_clean["MI_Right"].average()

# =========================================================
# OUTPUT DIRECTORY
# =========================================================

output_dir = PROJECT_DIR / "outputs" / "figures"
output_dir.mkdir(parents=True, exist_ok=True)

session_key = file_path.stem.replace("_filtered", "")

# =========================================================
# PLOT LEFT AVERAGE
# =========================================================

fig_left = evoked_left.plot(
    spatial_colors=True,
    show=False
)

left_path = output_dir / f"{subject_dir.name}_{session_key}_MI_Left_average.png"

fig_left.savefig(left_path, dpi=300)

print(f"\nSaved:\n{left_path}")

# =========================================================
# PLOT RIGHT AVERAGE
# =========================================================

fig_right = evoked_right.plot(
    spatial_colors=True,
    show=False
)

right_path = output_dir / f"{subject_dir.name}_{session_key}_MI_Right_average.png"

fig_right.savefig(right_path, dpi=300)

print(f"\nSaved:\n{right_path}")

# =========================================================
# SAVE EPOCH COUNTS
# =========================================================

summary = pd.DataFrame([{
    "subject": subject_dir.name,
    "session": session_key,
    "left_epochs_before": len(epochs["MI_Left"]),
    "right_epochs_before": len(epochs["MI_Right"]),
    "left_epochs_after": len(epochs_clean["MI_Left"]),
    "right_epochs_after": len(epochs_clean["MI_Right"]),
    "dropped_epochs": len(epochs) - len(epochs_clean)
}])

summary_path = (
    PROJECT_DIR
    / "outputs"
    / f"{subject_dir.name}_{session_key}_epoch_summary.csv"
)

summary.to_csv(summary_path, index=False)

print(f"\nSaved:\n{summary_path}")

print("\n===================================")
print("EPOCHING COMPLETE")
print("===================================")
