from pathlib import Path
import pandas as pd
import numpy as np

# =========================================================
# PROJECT DIRECTORY
# =========================================================

PROJECT_DIR = Path(__file__).resolve().parent

# =========================================================
# EEG CHANNELS
# =========================================================

EEG_CHANNELS = [
    "S1:CZ",
    "S2:CP2",
    "S3:CP3",
    "S4:FC2",
    "S5:FC3"
]

# =========================================================
# STORAGE
# =========================================================

results = []

# =========================================================
# FIND SUBJECTS
# =========================================================

subject_folders = sorted(PROJECT_DIR.glob("Sub*_data"))

print("\nSubjects found:")
print([s.name for s in subject_folders])

# =========================================================
# LOOP THROUGH SUBJECTS
# =========================================================

for subject_dir in subject_folders:

    subject_id = subject_dir.name

    # -----------------------------------------------------
    # RAW + FILTERED
    # -----------------------------------------------------

    for data_type in ["Raw_data", "Filtered_data"]:

        data_dir = subject_dir / data_type

        if not data_dir.exists():
            continue

        csv_files = sorted(data_dir.glob("*.csv"))

        print(f"\n{subject_id} | {data_type} | files: {len(csv_files)}")

        # -------------------------------------------------
        # LOOP THROUGH FILES
        # -------------------------------------------------

        for file_path in csv_files:

            print(f"Checking: {file_path.name}")

            try:

                # =========================================
                # LOAD CSV
                # =========================================

                df = pd.read_csv(
                    file_path,
                    comment="#",
                    skipinitialspace=True
                )

                # =========================================
                # CHECK CHANNELS
                # =========================================

                missing = [
                    ch for ch in EEG_CHANNELS
                    if ch not in df.columns
                ]

                if len(missing) > 0:

                    results.append({
                        "subject": subject_id,
                        "data_type": data_type,
                        "file": file_path.name,
                        "status": f"Missing columns: {missing}"
                    })

                    continue

                # =========================================
                # ANALYZE CHANNELS
                # =========================================

                for ch in EEG_CHANNELS:

                    signal = pd.to_numeric(
                        df[ch],
                        errors="coerce"
                    ).values

                    signal = np.nan_to_num(signal)

                    # -------------------------------------
                    # BASIC METRICS
                    # -------------------------------------

                    mean_amp = np.mean(signal)

                    std_amp = np.std(signal)

                    rms = np.sqrt(np.mean(signal**2))

                    peak_to_peak = np.ptp(signal)

                    # -------------------------------------
                    # DRIFT
                    # -------------------------------------

                    drift = np.mean(signal[-1000:]) - np.mean(signal[:1000])

                    # -------------------------------------
                    # HIGH AMPLITUDE BURSTS
                    # -------------------------------------

                    high_amp_samples = np.sum(
                        np.abs(signal) > 100
                    )

                    # =====================================
                    # SAVE
                    # =====================================

                    results.append({

                        "subject": subject_id,
                        "data_type": data_type,
                        "file": file_path.name,
                        "channel": ch,

                        "mean_amplitude": mean_amp,
                        "std_amplitude": std_amp,
                        "rms": rms,
                        "peak_to_peak": peak_to_peak,
                        "drift": drift,
                        "high_amp_samples": int(high_amp_samples),

                        "status": "OK"
                    })

            except Exception as e:

                results.append({

                    "subject": subject_id,
                    "data_type": data_type,
                    "file": file_path.name,
                    "channel": None,

                    "mean_amplitude": None,
                    "std_amplitude": None,
                    "rms": None,
                    "peak_to_peak": None,
                    "drift": None,
                    "high_amp_samples": None,

                    "status": f"ERROR: {e}"
                })

# =========================================================
# CREATE DATAFRAME
# =========================================================

results_df = pd.DataFrame(results)

# =========================================================
# SAVE OUTPUT
# =========================================================

output_dir = PROJECT_DIR / "outputs"
output_dir.mkdir(exist_ok=True)

output_file = output_dir / "signal_quality_summary.csv"

results_df.to_csv(output_file, index=False)

# =========================================================
# FINAL OUTPUT
# =========================================================

print("\n================================================")
print("SIGNAL QUALITY ANALYSIS COMPLETE")
print("================================================")

print(results_df.head())

print(f"\nTotal rows: {len(results_df)}")

print(f"\nSaved to:\n{output_file}")