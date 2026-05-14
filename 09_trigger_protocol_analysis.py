from pathlib import Path
import pandas as pd
import numpy as np
from collections import Counter

PROJECT_DIR = Path(__file__).resolve().parent

sfreq = 300

results = []
transition_counter = Counter()

subject_folders = sorted(PROJECT_DIR.glob("Sub*_data"))

# =========================================================
# LOOP THROUGH SUBJECTS
# =========================================================

for subject_dir in subject_folders:

    subject_id = subject_dir.name

    for data_type in ["Raw_data", "Filtered_data"]:

        data_dir = subject_dir / data_type

        if not data_dir.exists():
            continue

        csv_files = sorted(data_dir.glob("*.csv"))

        print(f"\nProcessing {subject_id} | {data_type}")

        for file_path in csv_files:

            try:

                # =========================================
                # LOAD FILE
                # =========================================

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

                if trigger_col is None:
                    continue

                trigger = df[trigger_col].astype(int)

                # =========================================
                # UNIQUE TRIGGERS
                # =========================================

                unique_triggers = sorted(trigger.unique())

                # =========================================
                # DETECT TRANSITIONS
                # =========================================

                prev = trigger.iloc[0]

                transitions = []

                for curr in trigger.iloc[1:]:

                    if curr != prev:

                        transition = (int(prev), int(curr))

                        transitions.append(transition)

                        transition_counter[transition] += 1

                        prev = curr

                # =========================================
                # TRIGGER DURATIONS
                # =========================================

                trigger_blocks = []

                current_trigger = trigger.iloc[0]
                start_idx = 0

                for idx in range(1, len(trigger)):

                    if trigger.iloc[idx] != current_trigger:

                        duration_samples = idx - start_idx

                        duration_sec = duration_samples / sfreq

                        trigger_blocks.append({
                            "trigger": int(current_trigger),
                            "duration_sec": duration_sec
                        })

                        current_trigger = trigger.iloc[idx]
                        start_idx = idx

                block_df = pd.DataFrame(trigger_blocks)

                # =========================================
                # SAVE PER-TRIGGER STATS
                # =========================================

                for trig in sorted(block_df["trigger"].unique()):

                    trig_df = block_df[
                        block_df["trigger"] == trig
                    ]

                    results.append({

                        "subject": subject_id,
                        "data_type": data_type,
                        "file": file_path.name,

                        "trigger_value": trig,

                        "n_blocks": len(trig_df),

                        "mean_duration_sec":
                            trig_df["duration_sec"].mean(),

                        "std_duration_sec":
                            trig_df["duration_sec"].std(),

                        "min_duration_sec":
                            trig_df["duration_sec"].min(),

                        "max_duration_sec":
                            trig_df["duration_sec"].max(),

                        "unique_triggers":
                            str(unique_triggers),

                        "n_unique_transitions":
                            len(set(transitions))
                    })

            except Exception as e:

                results.append({

                    "subject": subject_id,
                    "data_type": data_type,
                    "file": file_path.name,

                    "trigger_value": None,

                    "n_blocks": None,

                    "mean_duration_sec": None,
                    "std_duration_sec": None,
                    "min_duration_sec": None,
                    "max_duration_sec": None,

                    "unique_triggers": None,

                    "n_unique_transitions": None,

                    "error": str(e)
                })

# =========================================================
# SAVE MAIN RESULTS
# =========================================================

results_df = pd.DataFrame(results)

output_dir = PROJECT_DIR / "outputs"
output_dir.mkdir(exist_ok=True)

results_file = output_dir / "trigger_protocol_analysis.csv"

results_df.to_csv(results_file, index=False)

# =========================================================
# SAVE TRANSITION COUNTS
# =========================================================

transition_rows = []

for transition, count in transition_counter.items():

    transition_rows.append({
        "from_trigger": transition[0],
        "to_trigger": transition[1],
        "count": count
    })

transition_df = pd.DataFrame(transition_rows)

transition_df = transition_df.sort_values(
    by="count",
    ascending=False
)

transition_file = output_dir / "trigger_transition_counts.csv"

transition_df.to_csv(transition_file, index=False)

# =========================================================
# FINAL SUMMARY
# =========================================================

print("\n======================================")
print("TRIGGER PROTOCOL ANALYSIS COMPLETE")
print("======================================")

print("\nMost common transitions:")
print(transition_df.head(20))

print("\nUnique trigger values found:")
print(sorted(results_df["trigger_value"].dropna().unique()))

print(f"\nSaved main analysis to:\n{results_file}")

print(f"\nSaved transition counts to:\n{transition_file}")