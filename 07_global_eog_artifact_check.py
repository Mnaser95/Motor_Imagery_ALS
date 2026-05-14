from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent

EEG_CHANNELS = [
    "S1:CZ",
    "S2:CP2",
    "S3:CP3",
    "S4:FC2",
    "S5:FC3"
]

EOG_CHANNELS = [
    "S6:vEOGt",
    "S7:vEOGb"
]

results = []

subject_folders = sorted(PROJECT_DIR.glob("Sub*_data"))

for subject_dir in subject_folders:

    subject_id = subject_dir.name

    for data_type in ["Raw_data", "Filtered_data"]:

        data_dir = subject_dir / data_type

        if not data_dir.exists():
            continue

        csv_files = sorted(data_dir.glob("*.csv"))

        print(f"\nProcessing {subject_id} | {data_type} | {len(csv_files)} files")

        for file_path in csv_files:

            try:
                df = pd.read_csv(
                    file_path,
                    comment="#",
                    skipinitialspace=True
                )

                missing_cols = [
                    ch for ch in EEG_CHANNELS + EOG_CHANNELS
                    if ch not in df.columns
                ]

                if missing_cols:
                    results.append({
                        "subject": subject_id,
                        "data_type": data_type,
                        "file": file_path.name,
                        "channel": None,
                        "corr_vEOGt": None,
                        "corr_vEOGb": None,
                        "corr_vEOG_diff": None,
                        "max_abs_eog_corr": None,
                        "eog_risk": "Missing columns",
                        "status": f"Missing: {missing_cols}"
                    })
                    continue

                veogt = pd.to_numeric(df["S6:vEOGt"], errors="coerce").fillna(0).values
                veogb = pd.to_numeric(df["S7:vEOGb"], errors="coerce").fillna(0).values
                veog_diff = veogt - veogb

                for eeg_ch in EEG_CHANNELS:

                    eeg = pd.to_numeric(df[eeg_ch], errors="coerce").fillna(0).values

                    corr_vEOGt = np.corrcoef(eeg, veogt)[0, 1]
                    corr_vEOGb = np.corrcoef(eeg, veogb)[0, 1]
                    corr_vEOG_diff = np.corrcoef(eeg, veog_diff)[0, 1]

                    max_abs_corr = max(
                        abs(corr_vEOGt),
                        abs(corr_vEOGb),
                        abs(corr_vEOG_diff)
                    )

                    if max_abs_corr >= 0.50:
                        eog_risk = "High"
                    elif max_abs_corr >= 0.30:
                        eog_risk = "Moderate"
                    else:
                        eog_risk = "Low"

                    results.append({
                        "subject": subject_id,
                        "data_type": data_type,
                        "file": file_path.name,
                        "channel": eeg_ch,
                        "corr_vEOGt": corr_vEOGt,
                        "corr_vEOGb": corr_vEOGb,
                        "corr_vEOG_diff": corr_vEOG_diff,
                        "max_abs_eog_corr": max_abs_corr,
                        "eog_risk": eog_risk,
                        "status": "OK"
                    })

            except Exception as e:
                results.append({
                    "subject": subject_id,
                    "data_type": data_type,
                    "file": file_path.name,
                    "channel": None,
                    "corr_vEOGt": None,
                    "corr_vEOGb": None,
                    "corr_vEOG_diff": None,
                    "max_abs_eog_corr": None,
                    "eog_risk": "ERROR",
                    "status": f"ERROR: {e}"
                })

results_df = pd.DataFrame(results)

output_dir = PROJECT_DIR / "outputs"
output_dir.mkdir(exist_ok=True)

output_file = output_dir / "global_eog_artifact_summary.csv"
results_df.to_csv(output_file, index=False)

print("\n======================================")
print("GLOBAL EOG ARTIFACT CHECK COMPLETE")
print("======================================")

print(results_df.head())

print(f"\nTotal rows: {len(results_df)}")
print(f"\nSaved to:\n{output_file}")

print("\nEOG risk counts:")
print(results_df["eog_risk"].value_counts())
