from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent

results = []

subject_folders = sorted(PROJECT_DIR.glob("Sub*_data"))

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

                df = pd.read_csv(
                    file_path,
                    comment="#",
                    skipinitialspace=True
                )

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

                    results.append({
                        "subject": subject_id,
                        "data_type": data_type,
                        "file": file_path.name,
                        "status": "No active trigger",
                        "n_transitions": None,
                        "unique_transitions": None,
                        "unexpected_transitions": None
                    })

                    continue

                trigger = df[trigger_col].astype(int)

                # -----------------------------------------
                # TRANSITIONS
                # -----------------------------------------

                transitions = []

                prev = trigger.iloc[0]

                for curr in trigger.iloc[1:]:

                    if curr != prev:

                        transitions.append((prev, curr))

                        prev = curr

                unique_transitions = sorted(list(set(transitions)))

                # -----------------------------------------
                # EXPECTED TRANSITIONS
                # -----------------------------------------

                expected = {
                    (0,1),
                    (1,2),
                    (2,3),
                    (3,4),
                    (4,5),
                    (5,6),
                    (6,7),
                    (7,8),
                    (7,9),
                    (8,0),
                    (9,0),
                    (0,7),
                    (0,8),
                    (0,9)
                }

                unexpected = [
                    t for t in unique_transitions
                    if t not in expected
                ]

                results.append({
                    "subject": subject_id,
                    "data_type": data_type,
                    "file": file_path.name,
                    "status": "OK",
                    "n_transitions": len(transitions),
                    "unique_transitions": str(unique_transitions),
                    "unexpected_transitions": str(unexpected)
                })

            except Exception as e:

                results.append({
                    "subject": subject_id,
                    "data_type": data_type,
                    "file": file_path.name,
                    "status": f"ERROR: {e}",
                    "n_transitions": None,
                    "unique_transitions": None,
                    "unexpected_transitions": None
                })

# =========================================================
# SAVE
# =========================================================

results_df = pd.DataFrame(results)

output_dir = PROJECT_DIR / "outputs"
output_dir.mkdir(exist_ok=True)

output_file = output_dir / "trigger_sequence_check.csv"

results_df.to_csv(output_file, index=False)

# =========================================================
# SUMMARY
# =========================================================

print("\n======================================")
print("TRIGGER SEQUENCE CHECK COMPLETE")
print("======================================")

print(results_df.head())

problematic = results_df[
    results_df["unexpected_transitions"] != "[]"
]

print("\n======================================")
print("FILES WITH UNEXPECTED TRANSITIONS")
print("======================================")

if len(problematic) == 0:

    print("\nNo unexpected trigger transitions found.")

else:

    print(problematic[
        [
            "subject",
            "data_type",
            "file",
            "unexpected_transitions"
        ]
    ])

print(f"\nSaved to:\n{output_file}")