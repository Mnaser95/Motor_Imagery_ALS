from pathlib import Path
import pandas as pd
from datetime import datetime

# =========================================================
# PROJECT DIRECTORY
# =========================================================

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "outputs"
SUMMARY_FILE = OUTPUT_DIR / "pipeline_summary.txt"

OUTPUT_DIR.mkdir(exist_ok=True)

# =========================================================
# SCRIPT DESCRIPTIONS
# =========================================================

scripts = [
    {
        "script": "All_session.py",
        "purpose": "Scans all subject folders, detects trigger columns, counts MI trials, classifies files as Full_session or Baseline_only.",
        "main_output": "outputs/all_sessions_summary.csv"
    },
    {
        "script": "02_signal_quality_summary.py",
        "purpose": "Computes signal quality metrics for each EEG channel across all subjects and sessions.",
        "main_output": "outputs/signal_quality_summary.csv"
    },
    {
        "script": "03_compare_raw_filtered_quality.py",
        "purpose": "Compares raw and filtered data quality using STD, RMS, peak-to-peak, drift, and high-amplitude samples.",
        "main_output": "outputs/raw_vs_filtered_quality_comparison.csv"
    },
    {
        "script": "04_visualize_session.py",
        "purpose": "Visualizes selected raw and filtered EEG/EOG channels for a user-selected subject, session, and time window.",
        "main_output": "outputs/figures/*.png"
    },
    {
        "script": "05_visualize_mi_trials.py",
        "purpose": "Visualizes EEG/EOG signals around selected MI_Left or MI_Right trigger-centered trials.",
        "main_output": "outputs/figures/*.png"
    },
    {
        "script": "06_create_epochs.py",
        "purpose": "Creates MNE epochs around MI_Left and MI_Right events using filtered EEG data.",
        "main_output": "outputs/*_epoch_summary.csv and outputs/figures/*.png"
    },
    {
        "script": "07_global_eog_artifact_check.py",
        "purpose": "Computes correlation between EEG channels and EOG channels to screen for eye-artifact contamination.",
        "main_output": "outputs/global_eog_artifact_summary.csv"
    },
    {
        "script": "08_trigger_sequence_check.py",
        "purpose": "Checks trigger transition sequences across all files to detect unusual or unexpected transitions.",
        "main_output": "outputs/trigger_sequence_check.csv"
    },
    {
        "script": "09_trigger_protocol_analysis.py",
        "purpose": "Analyzes trigger values, transition frequencies, and trigger durations to understand the experiment protocol.",
        "main_output": "outputs/trigger_protocol_analysis.csv and outputs/trigger_transition_counts.csv"
    }
]

# =========================================================
# DATASET STRUCTURE SUMMARY
# =========================================================

subject_folders = sorted(PROJECT_DIR.glob("Sub*_data"))

dataset_rows = []

for subject_dir in subject_folders:

    subject_id = subject_dir.name

    for data_type in ["Raw_data", "Filtered_data"]:

        data_dir = subject_dir / data_type

        if data_dir.exists():
            csv_count = len(list(data_dir.glob("*.csv")))
        else:
            csv_count = 0

        dataset_rows.append({
            "subject": subject_id,
            "data_type": data_type,
            "csv_files": csv_count
        })

dataset_df = pd.DataFrame(dataset_rows)

# =========================================================
# OUTPUT FILE SUMMARY
# =========================================================

output_files = []

if OUTPUT_DIR.exists():

    for file_path in sorted(OUTPUT_DIR.rglob("*")):

        if file_path.is_file():

            output_files.append({
                "file": str(file_path.relative_to(PROJECT_DIR)),
                "size_kb": round(file_path.stat().st_size / 1024, 2)
            })

output_df = pd.DataFrame(output_files)

# =========================================================
# READ IMPORTANT RESULT FILES IF AVAILABLE
# =========================================================

all_sessions_path = OUTPUT_DIR / "all_sessions_summary.csv"
signal_quality_path = OUTPUT_DIR / "signal_quality_summary.csv"
raw_filtered_path = OUTPUT_DIR / "raw_vs_filtered_quality_comparison.csv"
eog_path = OUTPUT_DIR / "global_eog_artifact_summary.csv"
transition_path = OUTPUT_DIR / "trigger_transition_counts.csv"

summary_lines = []

summary_lines.append("EEG STUDY PIPELINE SUMMARY")
summary_lines.append("=" * 80)
summary_lines.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
summary_lines.append(f"Project directory: {PROJECT_DIR}")
summary_lines.append("")

# =========================================================
# SECTION 1: FOLDER STRUCTURE
# =========================================================

summary_lines.append("1. DATASET FOLDER STRUCTURE")
summary_lines.append("-" * 80)

if len(dataset_df) > 0:
    summary_lines.append(dataset_df.to_string(index=False))
else:
    summary_lines.append("No subject folders found.")

summary_lines.append("")

# =========================================================
# SECTION 2: SCRIPT INVENTORY
# =========================================================

summary_lines.append("2. SCRIPT INVENTORY")
summary_lines.append("-" * 80)

for item in scripts:
    script_path = PROJECT_DIR / item["script"]

    exists_text = "FOUND" if script_path.exists() else "MISSING"

    summary_lines.append(f"Script: {item['script']} [{exists_text}]")
    summary_lines.append(f"Purpose: {item['purpose']}")
    summary_lines.append(f"Main output: {item['main_output']}")
    summary_lines.append("")

# =========================================================
# SECTION 3: ALL SESSION SUMMARY
# =========================================================

summary_lines.append("3. DATASET / SESSION SUMMARY")
summary_lines.append("-" * 80)

if all_sessions_path.exists():

    df = pd.read_csv(all_sessions_path)

    summary_lines.append(f"Total files indexed: {len(df)}")

    if "file_role" in df.columns:
        summary_lines.append("")
        summary_lines.append("File role counts:")
        summary_lines.append(df["file_role"].value_counts(dropna=False).to_string())

    if "data_type" in df.columns:
        summary_lines.append("")
        summary_lines.append("Raw/Filtered counts:")
        summary_lines.append(df["data_type"].value_counts(dropna=False).to_string())

    if "MI_Left_trials" in df.columns and "MI_Right_trials" in df.columns:
        summary_lines.append("")
        summary_lines.append("MI trial count summary:")
        summary_lines.append(
            df[
                ["MI_Left_trials", "MI_Right_trials"]
            ].describe().to_string()
        )

    baseline_only = df[df.get("file_role", "") == "Baseline_only"]

    summary_lines.append("")
    summary_lines.append(f"Baseline-only files: {len(baseline_only)}")

    if len(baseline_only) > 0:

        for _, row in baseline_only.iterrows():

            summary_lines.append(
                f"- {row['subject']} | {row['data_type']} | {row['file']} | triggers={row['unique_triggers']}"
            )

else:
    summary_lines.append("all_sessions_summary.csv not found.")

summary_lines.append("")

# =========================================================
# SECTION 4: SIGNAL QUALITY SUMMARY
# =========================================================

summary_lines.append("4. SIGNAL QUALITY SUMMARY")
summary_lines.append("-" * 80)

if signal_quality_path.exists():

    sq = pd.read_csv(signal_quality_path)

    summary_lines.append(f"Total signal-quality rows: {len(sq)}")

    if "data_type" in sq.columns:
        summary_lines.append("")
        summary_lines.append("Rows by data type:")
        summary_lines.append(sq["data_type"].value_counts(dropna=False).to_string())

    quality_cols = [
        "std_amplitude",
        "rms",
        "peak_to_peak",
        "drift",
        "high_amp_samples"
    ]

    available_quality_cols = [
        c for c in quality_cols
        if c in sq.columns
    ]

    if available_quality_cols:
        summary_lines.append("")
        summary_lines.append("Signal-quality metric summary:")
        summary_lines.append(
            sq[available_quality_cols].describe().to_string()
        )

else:
    summary_lines.append("signal_quality_summary.csv not found.")

summary_lines.append("")

# =========================================================
# SECTION 5: RAW VS FILTERED SUMMARY
# =========================================================

summary_lines.append("5. RAW VS FILTERED QUALITY COMPARISON")
summary_lines.append("-" * 80)

if raw_filtered_path.exists():

    rf = pd.read_csv(raw_filtered_path)

    summary_lines.append(f"Matched raw-filtered channel rows: {len(rf)}")

    reduction_cols = [
        "std_reduction_percent",
        "rms_reduction_percent",
        "peak_to_peak_reduction_percent",
        "drift_reduction_percent"
    ]

    available_reduction_cols = [
        c for c in reduction_cols
        if c in rf.columns
    ]

    if available_reduction_cols:
        summary_lines.append("")
        summary_lines.append("Filtering reduction summary:")
        summary_lines.append(
            rf[available_reduction_cols].describe().to_string()
        )

else:
    summary_lines.append("raw_vs_filtered_quality_comparison.csv not found.")

summary_lines.append("")

# =========================================================
# SECTION 6: EOG ARTIFACT SUMMARY
# =========================================================

summary_lines.append("6. EOG ARTIFACT SUMMARY")
summary_lines.append("-" * 80)

if eog_path.exists():

    eog = pd.read_csv(eog_path)

    summary_lines.append(f"Total EOG screening rows: {len(eog)}")

    if "eog_risk" in eog.columns:
        summary_lines.append("")
        summary_lines.append("EOG risk counts:")
        summary_lines.append(eog["eog_risk"].value_counts(dropna=False).to_string())

    if "max_abs_eog_corr" in eog.columns:
        summary_lines.append("")
        summary_lines.append("Max absolute EOG correlation summary:")
        summary_lines.append(
            eog["max_abs_eog_corr"].describe().to_string()
        )

else:
    summary_lines.append("global_eog_artifact_summary.csv not found.")

summary_lines.append("")

# =========================================================
# SECTION 7: TRIGGER PROTOCOL SUMMARY
# =========================================================

summary_lines.append("7. TRIGGER PROTOCOL SUMMARY")
summary_lines.append("-" * 80)

if transition_path.exists():

    tr = pd.read_csv(transition_path)

    summary_lines.append("Most common trigger transitions:")
    summary_lines.append(
        tr.head(20).to_string(index=False)
    )

else:
    summary_lines.append("trigger_transition_counts.csv not found.")

summary_lines.append("")

# =========================================================
# SECTION 8: OUTPUT INVENTORY
# =========================================================

summary_lines.append("8. OUTPUT FILE INVENTORY")
summary_lines.append("-" * 80)

if len(output_df) > 0:
    summary_lines.append(output_df.to_string(index=False))
else:
    summary_lines.append("No output files found.")

summary_lines.append("")

# =========================================================
# SECTION 9: CURRENT STATUS
# =========================================================

summary_lines.append("9. CURRENT PIPELINE STATUS")
summary_lines.append("-" * 80)

summary_lines.append(
    "The current EEG_Study folder supports dataset indexing, trigger verification, "
    "file-role classification, raw-versus-filtered quality comparison, signal-quality "
    "metrics, visualization, epoch creation, EOG artifact screening, and trigger protocol analysis."
)

summary_lines.append("")
summary_lines.append("This stage is dataset-understanding and quality-control preparation.")
summary_lines.append("It is not yet the final MLSP feature-engineering or machine-learning stage.")
summary_lines.append("")
summary_lines.append("Recommended next stage:")
summary_lines.append("- Create a clean dataset manifest for usable MI files.")
summary_lines.append("- Then proceed to PSD / Mu-Beta feature extraction.")
summary_lines.append("- Then proceed to ERD/ERS and ML classification.")

# =========================================================
# WRITE SUMMARY TXT
# =========================================================

with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(summary_lines))

# =========================================================
# ALSO SAVE SCRIPT INVENTORY AS CSV
# =========================================================

script_df = pd.DataFrame(scripts)
script_df["exists"] = script_df["script"].apply(
    lambda x: (PROJECT_DIR / x).exists()
)

script_inventory_file = OUTPUT_DIR / "pipeline_script_inventory.csv"
script_df.to_csv(script_inventory_file, index=False)

# =========================================================
# PRINT TO TERMINAL
# =========================================================

print("\n================================================")
print("PIPELINE SUMMARY GENERATED")
print("================================================")

print(f"\nSaved text summary to:\n{SUMMARY_FILE}")
print(f"\nSaved script inventory to:\n{script_inventory_file}")

print("\nPreview:")
print("\n".join(summary_lines[:40]))