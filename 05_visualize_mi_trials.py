from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).resolve().parent

CHANNELS_TO_PLOT = [
    "S1:CZ",
    "S2:CP2",
    "S3:CP3",
    "S4:FC2",
    "S5:FC3",
    "S6:vEOGt",
    "S7:vEOGb",
]

sfreq = 300

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

print(f"\nSelected file: {file_path.name}")

df = pd.read_csv(file_path, comment="#", skipinitialspace=True)

# =========================================================
# DETECT TRIGGER COLUMN
# =========================================================

trigger_col = None

for candidate in ["Manual trigger", "Trigger"]:
    if candidate in df.columns:
        temp = pd.to_numeric(df[candidate], errors="coerce").fillna(0).astype(int)
        if temp.sum() > 0:
            trigger_col = candidate
            df[candidate] = temp
            break

if trigger_col is None:
    raise RuntimeError("No active trigger column found.")

trigger = df[trigger_col].astype(int)

# =========================================================
# DETECT EVENT ONSETS
# =========================================================

onset_mask = (trigger != trigger.shift(1)) & (trigger != 0)

events = pd.DataFrame({
    "sample": trigger.index[onset_mask],
    "trigger": trigger[onset_mask].values
})

mi_events = events[events["trigger"].isin([8, 9])].reset_index(drop=True)

print("\nMI events found:")
print(f"MI_Left  trigger 8: {(mi_events['trigger'] == 8).sum()}")
print(f"MI_Right trigger 9: {(mi_events['trigger'] == 9).sum()}")

if len(mi_events) == 0:
    raise RuntimeError("No MI events found.")

# =========================================================
# SELECT MI EVENT
# =========================================================

print("\nAvailable MI trials:")
for i, row in mi_events.iterrows():
    label = "Left" if row["trigger"] == 8 else "Right"
    time_sec = row["sample"] / sfreq
    print(f"{i + 1}. {label} | trigger {row['trigger']} | time {time_sec:.2f} sec")

trial_idx = int(input("\nSelect MI trial number to visualize: ")) - 1

if trial_idx < 0 or trial_idx >= len(mi_events):
    raise ValueError("Invalid MI trial selection.")

event_sample = int(mi_events.loc[trial_idx, "sample"])
event_trigger = int(mi_events.loc[trial_idx, "trigger"])
event_label = "MI_Left" if event_trigger == 8 else "MI_Right"

# =========================================================
# EPOCH WINDOW
# =========================================================

tmin = float(input("\nEnter start time relative to trigger, example -2: "))
tmax = float(input("Enter end time relative to trigger, example 4: "))

start_sample = event_sample + int(tmin * sfreq)
end_sample = event_sample + int(tmax * sfreq)

if start_sample < 0:
    raise ValueError("Window starts before recording begins.")

if end_sample > len(df):
    raise ValueError("Window ends after recording finishes.")

time = [(i - event_sample) / sfreq for i in range(start_sample, end_sample)]

# =========================================================
# OUTPUT FOLDER
# =========================================================

output_dir = PROJECT_DIR / "outputs" / "figures"
output_dir.mkdir(parents=True, exist_ok=True)

session_key = file_path.stem.replace("_filtered", "")

# =========================================================
# PLOT CHANNELS AROUND MI EVENT
# =========================================================

plt.figure(figsize=(15, 9))

offset = 0

for ch in CHANNELS_TO_PLOT:
    if ch not in df.columns:
        print(f"Missing channel skipped: {ch}")
        continue

    signal = pd.to_numeric(df[ch], errors="coerce").fillna(0).values[start_sample:end_sample]

    plt.plot(time, signal + offset, label=ch)

    offset += 1200

plt.axvline(0, linestyle="--", linewidth=2, label="MI trigger onset")

plt.title(
    f"{event_label} Trial Visualization - {subject_dir.name} - {session_key}"
)
plt.xlabel("Time relative to MI trigger (seconds)")
plt.ylabel("Amplitude + vertical offset")
plt.legend(loc="upper right")
plt.tight_layout()

save_path = output_dir / f"{subject_dir.name}_{session_key}_{event_label}_trial_{trial_idx + 1}.png"

plt.savefig(save_path, dpi=300)
plt.show()

print(f"\nSaved figure:\n{save_path}")
print("\nMI trial visualization complete.")