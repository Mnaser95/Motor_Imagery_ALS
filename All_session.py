from pathlib import Path
import pandas as pd

# =========================================================
# PROJECT DIRECTORY
# =========================================================

PROJECT_DIR = Path(__file__).resolve().parent

print(f"\nProject directory:\n{PROJECT_DIR}")

# =========================================================
# FIND SUBJECT FOLDERS
# =========================================================

subject_folders = sorted(PROJECT_DIR.glob("Sub*_data"))

print("\nSubject folders found:")
print([p.name for p in subject_folders])

if len(subject_folders) == 0:
    raise RuntimeError("No subject folders found. Make sure this script is inside EEG_Study.")

# =========================================================
# SUMMARY STORAGE
# =========================================================

summary_rows = []

# =========================================================
# LOOP THROUGH SUBJECTS
# =========================================================

for subject_dir in subject_folders:

    subject_id = subject_dir.name

    for data_type in ["Raw_data", "Filtered_data"]:

        data_dir = subject_dir / data_type

        if not data_dir.exists():
            print(f"Missing folder skipped: {data_dir}")
            continue

        csv_files = sorted(data_dir.glob("*.csv"))

        print(f"\n{subject_id} | {data_type} | files found: {len(csv_files)}")

        for file_path in csv_files:

            try:
                df = pd.read_csv(
                    file_path,
                    comment="#",
                    skipinitialspace=True
                )

                # =========================================
                # DETECT TRIGGER COLUMN
                # =========================================

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

                # =========================================
                # HANDLE NO ACTIVE TRIGGER
                # =========================================

                if trigger_col is None:

                    summary_rows.append({
                        "subject": subject_id,
                        "data_type": data_type,
                        "file": file_path.name,
                        "rows": df.shape[0],
                        "columns": df.shape[1],
                        "trigger_col": None,
                        "unique_triggers": None,
                        "file_role": "No_active_trigger",
                        "MI_Left_samples": None,
                        "MI_Right_samples": None,
                        "MI_Left_trials": None,
                        "MI_Right_trials": None,
                        "Left_sec_per_trial": None,
                        "Right_sec_per_trial": None,
                        "total_events": None,
                        "status": "No active trigger"
                    })

                    continue

                # =========================================
                # TRIGGER ANALYSIS
                # =========================================

                trigger = df[trigger_col].astype(int)

                unique_triggers = sorted(trigger.unique().tolist())

                # -----------------------------------------
                # FILE ROLE CLASSIFICATION
                # -----------------------------------------

                has_left = 8 in unique_triggers
                has_right = 9 in unique_triggers

                has_baseline = any(
                    t in unique_triggers
                    for t in [1, 2, 3, 4, 5, 6, 7]
                )

                if has_left and has_right and has_baseline:
                    file_role = "Full_session"

                elif has_left and has_right:
                    file_role = "MI_task"

                elif has_baseline:
                    file_role = "Baseline_only"

                else:
                    file_role = "Unknown_check"

                # -----------------------------------------
                # SAMPLE COUNTS
                # -----------------------------------------

                mi_left_samples = int((trigger == 8).sum())
                mi_right_samples = int((trigger == 9).sum())

                # -----------------------------------------
                # EVENT ONSETS ONLY
                # -----------------------------------------

                onsets = trigger[
                    (trigger != trigger.shift(1)) &
                    (trigger != 0)
                ]

                mi_left_trials = int((onsets == 8).sum())
                mi_right_trials = int((onsets == 9).sum())
                total_events = int(len(onsets))

                # -----------------------------------------
                # ESTIMATED DURATION PER TRIAL
                # -----------------------------------------

                sfreq = 300

                left_sec_per_trial = (
                    mi_left_samples / mi_left_trials / sfreq
                    if mi_left_trials > 0 else None
                )

                right_sec_per_trial = (
                    mi_right_samples / mi_right_trials / sfreq
                    if mi_right_trials > 0 else None
                )

                # =========================================
                # SAVE SUMMARY
                # =========================================

                summary_rows.append({
                    "subject": subject_id,
                    "data_type": data_type,
                    "file": file_path.name,
                    "rows": df.shape[0],
                    "columns": df.shape[1],
                    "trigger_col": trigger_col,
                    "unique_triggers": str(unique_triggers),
                    "file_role": file_role,
                    "MI_Left_samples": mi_left_samples,
                    "MI_Right_samples": mi_right_samples,
                    "MI_Left_trials": mi_left_trials,
                    "MI_Right_trials": mi_right_trials,
                    "Left_sec_per_trial": left_sec_per_trial,
                    "Right_sec_per_trial": right_sec_per_trial,
                    "total_events": total_events,
                    "status": "OK"
                })

            except Exception as e:

                summary_rows.append({
                    "subject": subject_id,
                    "data_type": data_type,
                    "file": file_path.name,
                    "rows": None,
                    "columns": None,
                    "trigger_col": None,
                    "unique_triggers": None,
                    "file_role": "ERROR",
                    "MI_Left_samples": None,
                    "MI_Right_samples": None,
                    "MI_Left_trials": None,
                    "MI_Right_trials": None,
                    "Left_sec_per_trial": None,
                    "Right_sec_per_trial": None,
                    "total_events": None,
                    "status": f"ERROR: {e}"
                })

# =========================================================
# SAVE SUMMARY
# =========================================================

summary = pd.DataFrame(summary_rows)

output_dir = PROJECT_DIR / "outputs"
output_dir.mkdir(exist_ok=True)

output_file = output_dir / "all_sessions_summary.csv"

summary.to_csv(output_file, index=False)

print("\n================================================")
print("DONE")
print("================================================")

print(summary)

print(f"\nTotal files analyzed: {len(summary)}")

print("\nFile role counts:")
print(summary["file_role"].value_counts(dropna=False))

# =========================================================
# SHOW BASELINE-ONLY FILES
# =========================================================

baseline_only = summary[
    summary["file_role"] == "Baseline_only"
]

print("\n================================================")
print("BASELINE-ONLY FILES")
print("================================================")

if len(baseline_only) == 0:

    print("\nNo baseline-only files found.")

else:

    for idx, row in baseline_only.iterrows():

        print(
            f"\nSubject: {row['subject']}"
            f"\nData type: {row['data_type']}"
            f"\nFile: {row['file']}"
            f"\nTriggers: {row['unique_triggers']}"
        )

print("\n================================================")
print(f"Total baseline-only files: {len(baseline_only)}")
print("================================================")

print("\n================================================")
print(f"Total baseline-only files: {len(baseline_only)}")
print("================================================")

# =========================================================
# SHOW FILTERED FILES WITH NO MI EVENTS
# =========================================================

filtered_no_mi = summary[
    (summary["data_type"] == "Filtered_data") &
    (
        (summary["MI_Left_trials"].fillna(0) == 0) |
        (summary["MI_Right_trials"].fillna(0) == 0)
    )
]

print("\n================================================")
print("FILTERED FILES WITH NO / LOW MI EVENTS")
print("================================================")

if len(filtered_no_mi) == 0:

    print("\nNo filtered files with missing MI events found.")

else:

    for idx, row in filtered_no_mi.iterrows():

        print(
            f"\nSubject: {row['subject']}"
            f"\nFile: {row['file']}"
            f"\nRole: {row['file_role']}"
            f"\nMI_Left_trials: {row['MI_Left_trials']}"
            f"\nMI_Right_trials: {row['MI_Right_trials']}"
            f"\nTriggers: {row['unique_triggers']}"
        )

print("\n================================================")
print(f"Total filtered files with missing MI: {len(filtered_no_mi)}")
print("================================================")


print(f"\nSaved to:\n{output_file}")