from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
INPUT_FILE = PROJECT_DIR / "outputs" / "signal_quality_summary.csv"
OUTPUT_FILE = PROJECT_DIR / "outputs" / "raw_vs_filtered_quality_comparison.csv"

df = pd.read_csv(INPUT_FILE)

# Keep only successful rows
df = df[df["status"] == "OK"].copy()

# Separate raw and filtered
raw = df[df["data_type"] == "Raw_data"].copy()
filtered = df[df["data_type"] == "Filtered_data"].copy()

# Create matching session name
raw["session_key"] = raw["file"].str.replace("_raw.csv", "", regex=False)
filtered["session_key"] = filtered["file"].str.replace("_filtered.csv", "", regex=False)

# Merge raw and filtered by subject, session, channel
merged = pd.merge(
    raw,
    filtered,
    on=["subject", "session_key", "channel"],
    suffixes=("_raw", "_filtered")
)

# Compute improvement ratios
merged["std_reduction_percent"] = (
    (merged["std_amplitude_raw"] - merged["std_amplitude_filtered"])
    / merged["std_amplitude_raw"]
) * 100

merged["rms_reduction_percent"] = (
    (merged["rms_raw"] - merged["rms_filtered"])
    / merged["rms_raw"]
) * 100

merged["peak_to_peak_reduction_percent"] = (
    (merged["peak_to_peak_raw"] - merged["peak_to_peak_filtered"])
    / merged["peak_to_peak_raw"]
) * 100

merged["drift_reduction_percent"] = (
    (abs(merged["drift_raw"]) - abs(merged["drift_filtered"]))
    / abs(merged["drift_raw"])
) * 100

# Select useful columns
comparison = merged[
    [
        "subject",
        "session_key",
        "channel",

        "std_amplitude_raw",
        "std_amplitude_filtered",
        "std_reduction_percent",

        "rms_raw",
        "rms_filtered",
        "rms_reduction_percent",

        "peak_to_peak_raw",
        "peak_to_peak_filtered",
        "peak_to_peak_reduction_percent",

        "drift_raw",
        "drift_filtered",
        "drift_reduction_percent",

        "high_amp_samples_raw",
        "high_amp_samples_filtered",
    ]
]

comparison.to_csv(OUTPUT_FILE, index=False)

print("\n================================================")
print("RAW VS FILTERED COMPARISON COMPLETE")
print("================================================")

print(comparison.head())

print(f"\nTotal matched rows: {len(comparison)}")
print(f"\nSaved to:\n{OUTPUT_FILE}")