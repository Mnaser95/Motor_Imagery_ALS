from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# =========================================================
# PROJECT DIRECTORY
# =========================================================

PROJECT_DIR = Path(__file__).resolve().parent

CHANNELS_TO_PLOT = [
    "S1:CZ",
    "S2:CP2",
    "S3:CP3",
    "S4:FC2",
    "S5:FC3",
    "S6:vEOGt",
    "S7:vEOGb"
]

sfreq = 300

# =========================================================
# USER SELECT SUBJECT
# =========================================================

subject_folders = sorted(PROJECT_DIR.glob("Sub*_data"))

if len(subject_folders) == 0:
    raise RuntimeError("No subject folders found. Make sure this script is inside EEG_Study.")

print("\nAvailable subjects:")
for i, s in enumerate(subject_folders):
    print(f"{i + 1}. {s.name}")

subject_idx = int(input("\nSelect subject number: ")) - 1

if subject_idx < 0 or subject_idx >= len(subject_folders):
    raise ValueError("Invalid subject selection.")

subject_dir = subject_folders[subject_idx]

# =========================================================
# USER SELECT SESSION
# =========================================================

raw_dir = subject_dir / "Raw_data"
filtered_dir = subject_dir / "Filtered_data"

raw_files = sorted(raw_dir.glob("*.csv"))

if len(raw_files) == 0:
    raise RuntimeError(f"No raw CSV files found in:\n{raw_dir}")

print(f"\nAvailable raw sessions for {subject_dir.name}:")
for i, f in enumerate(raw_files):
    print(f"{i + 1}. {f.name}")

session_idx = int(input("\nSelect raw session number: ")) - 1

if session_idx < 0 or session_idx >= len(raw_files):
    raise ValueError("Invalid raw session selection.")

raw_file = raw_files[session_idx]

# =========================================================
# FIND MATCHING FILTERED FILE SAFELY
# =========================================================

filtered_files = sorted(filtered_dir.glob("*.csv"))

if len(filtered_files) == 0:
    raise RuntimeError(f"No filtered CSV files found in:\n{filtered_dir}")

raw_stem = raw_file.stem.replace("_raw", "")
clean_raw_stem = raw_stem.replace("_0001", "").replace("_0002", "")

exact_filtered_name = raw_file.name.replace("_raw.csv", "_filtered.csv")
filtered_file = filtered_dir / exact_filtered_name

if not filtered_file.exists():

    possible_matches = [
        f for f in filtered_files
        if clean_raw_stem in f.stem
    ]

    if len(possible_matches) == 1:
        filtered_file = possible_matches[0]

    else:
        print("\nCould not automatically find matching filtered file.")
        print("\nAvailable filtered sessions:")

        for i, f in enumerate(filtered_files):
            print(f"{i + 1}. {f.name}")

        filtered_idx = int(input("\nSelect matching filtered session number: ")) - 1

        if filtered_idx < 0 or filtered_idx >= len(filtered_files):
            raise ValueError("Invalid filtered session selection.")

        filtered_file = filtered_files[filtered_idx]

print("\nSelected files:")
print(f"RAW:      {raw_file.name}")
print(f"FILTERED: {filtered_file.name}")

# =========================================================
# LOAD DATA
# =========================================================

raw_df = pd.read_csv(raw_file, comment="#", skipinitialspace=True)
filtered_df = pd.read_csv(filtered_file, comment="#", skipinitialspace=True)

# =========================================================
# USER-SELECTED TIME WINDOW
# =========================================================

total_duration_sec = min(len(raw_df), len(filtered_df)) / sfreq

print(f"\nTotal comparable recording duration: {total_duration_sec:.2f} seconds")

start_sec = float(input("\nEnter START time in seconds, example 100: "))
duration_sec = float(input("Enter duration to visualize in seconds, example 20: "))

start_sample = int(start_sec * sfreq)
end_sample = int((start_sec + duration_sec) * sfreq)

if start_sample < 0:
    raise ValueError("Start time cannot be negative.")

if start_sample >= min(len(raw_df), len(filtered_df)):
    raise ValueError("Start time is beyond the recording length.")

end_sample = min(end_sample, len(raw_df), len(filtered_df))

time = [i / sfreq for i in range(start_sample, end_sample)]

# =========================================================
# OUTPUT DIRECTORY
# =========================================================

output_dir = PROJECT_DIR / "outputs" / "figures"
output_dir.mkdir(parents=True, exist_ok=True)

session_key = raw_file.stem.replace("_raw", "")

# =========================================================
# OFFSET PLOT FUNCTION
# =========================================================

def plot_offset_channels(df, title, output_name):

    plt.figure(figsize=(15, 9))

    offset = 0

    for ch in CHANNELS_TO_PLOT:

        if ch not in df.columns:
            print(f"Missing channel skipped: {ch}")
            continue

        signal = pd.to_numeric(
            df[ch],
            errors="coerce"
        ).fillna(0).values[start_sample:end_sample]

        plt.plot(
            time,
            signal + offset,
            label=ch
        )

        offset += 1200

    plt.title(title)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude + vertical offset")
    plt.legend(loc="upper right")
    plt.tight_layout()

    save_path = output_dir / output_name
    plt.savefig(save_path, dpi=300)
    plt.show()

    print(f"Saved: {save_path}")

# =========================================================
# RAW CHANNEL PLOT
# =========================================================

plot_offset_channels(
    raw_df,
    f"Raw EEG + EOG Channels - {subject_dir.name} - {session_key}",
    f"{subject_dir.name}_{session_key}_raw_all_channels_offset.png"
)

# =========================================================
# FILTERED CHANNEL PLOT
# =========================================================

plot_offset_channels(
    filtered_df,
    f"Filtered EEG + EOG Channels - {subject_dir.name} - {session_key}",
    f"{subject_dir.name}_{session_key}_filtered_all_channels_offset.png"
)

# =========================================================
# TRIGGER TIMELINE
# =========================================================

trigger_col = None

for candidate in ["Manual trigger", "Trigger"]:
    if candidate in filtered_df.columns:
        trigger_col = candidate
        break

if trigger_col is not None:

    trigger = pd.to_numeric(
        filtered_df[trigger_col],
        errors="coerce"
    ).fillna(0).values[start_sample:end_sample]

    plt.figure(figsize=(15, 3))
    plt.plot(time, trigger)

    plt.title(f"Trigger Timeline - {subject_dir.name} - {session_key}")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Trigger Value")
    plt.tight_layout()

    save_path = output_dir / f"{subject_dir.name}_{session_key}_trigger_timeline.png"
    plt.savefig(save_path, dpi=300)
    plt.show()

    print(f"Saved: {save_path}")

else:
    print("No trigger column found.")

print("\nVisualization complete.")