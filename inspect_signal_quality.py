from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne


PROJECT_DIR = Path(__file__).resolve().parent
SFREQ = 300

DEFAULT_SUBJECT = "Sub1"
DEFAULT_SESSION = "001"
DEFAULT_DATA_TYPE = "filtered"  # raw or filtered


CHANNEL_ALIASES = {
    "S1:CZ": "Cz",
    "S1:Cz": "Cz",
    "S2:C2": "C2",
    "S2:CP2": "C2",
    "S3:C3": "C3",
    "S3:CP3": "C3",
    "S4:C4": "C4",
    "S4:FC2": "C4",
    "S5:C5": "C5",
    "S5:FC3": "C5",
    "S6:hEOG": "hEOG",
    "S6:vEOGt": "hEOG",
    "S7:vEOG": "vEOG",
    "S7:vEOGb": "vEOG",
}

EEG_CHANNELS = ["Cz", "C2", "C3", "C4", "C5"]
EOG_CHANNELS = ["hEOG", "vEOG"]
ALL_SIGNAL_CHANNELS = EEG_CHANNELS + EOG_CHANNELS

EVENT_DICT = {
    "Init": 1,
    "RestingState_NoMovement": 2,
    "EyeMovement": 3,
    "JawFacialMovement": 4,
    "HeadMovement": 5,
    "Baseline": 6,
    "Cue_Cross": 7,
    "MI_LeftHand": 8,
    "MI_RightHand": 9,
}


def find_session_file(subject, session, data_type):
    subject_num = subject.replace("Sub", "")
    session_num = session.zfill(3)
    subject_dir = PROJECT_DIR / f"Sub{subject_num}_data"

    if data_type == "filtered":
        search_dir = subject_dir / "Filtered_data"
        patterns = [
            f"Sub{subject_num.zfill(3)}Ses{session_num}_filtered.csv",
            f"Sub001Ses{session_num}_filtered.csv",
            f"*Ses{session_num}*filtered*.csv",
        ]
    else:
        search_dir = subject_dir / "Raw_data"
        patterns = [
            f"Sub{subject_num.zfill(3)}Ses{session_num}_raw.csv",
            f"Sub001Ses{session_num}_raw.csv",
            f"*Ses{session_num}*raw*.csv",
        ]

    if not search_dir.exists():
        raise FileNotFoundError(f"Folder not found: {search_dir}")

    for pattern in patterns:
        matches = sorted(search_dir.glob(pattern))
        if matches:
            return matches[0]

    raise FileNotFoundError(f"No file found in {search_dir}")


def load_csv(file_path):
    print("\n--- Loading CSV ---")
    print(file_path)
    df = pd.read_csv(file_path, comment="#", skipinitialspace=True)
    print(f"Shape: {df.shape}")
    print("Columns:")
    print(df.columns.tolist())
    return df


def prepare_channels(df):
    selected_cols = []
    ch_names = []
    ch_types = []

    print("\n--- Channel mapping ---")

    used_mne_names = set()

    for csv_col, mne_name in CHANNEL_ALIASES.items():
        if csv_col in df.columns and mne_name not in used_mne_names:
            selected_cols.append(csv_col)
            ch_names.append(mne_name)
            ch_types.append("eog" if mne_name in EOG_CHANNELS else "eeg")
            used_mne_names.add(mne_name)
            print(f"{csv_col} -> {mne_name}")

    trigger_col = None

    if "Manual trigger" in df.columns:
        manual = pd.to_numeric(df["Manual trigger"], errors="coerce").fillna(0)
        if manual.sum() > 0:
            trigger_col = "Manual trigger"

    if trigger_col is None and "Trigger" in df.columns:
        trig = pd.to_numeric(df["Trigger"], errors="coerce").fillna(0)
        if trig.sum() > 0:
            trigger_col = "Trigger"

    if trigger_col is not None:
        selected_cols.append(trigger_col)
        ch_names.append("STI")
        ch_types.append("stim")
        print(f"Trigger column -> {trigger_col}")
    else:
        print("No active trigger column found.")

    return selected_cols, ch_names, ch_types, trigger_col


def create_raw(df, selected_cols, ch_names, ch_types, trigger_col):
    clean = df[selected_cols].copy()

    for col in selected_cols:
        clean[col] = pd.to_numeric(clean[col], errors="coerce")
        clean[col] = clean[col].replace([np.inf, -np.inf], np.nan)
        clean[col] = clean[col].fillna(0)

    if trigger_col is not None:
        clean[trigger_col] = clean[trigger_col].astype(int)

    data = clean.values.T

    if trigger_col is not None:
        data[:-1] *= 1e-6
    else:
        data *= 1e-6

    info = mne.create_info(ch_names=ch_names, sfreq=SFREQ, ch_types=ch_types)
    raw = mne.io.RawArray(data, info)

    montage = mne.channels.make_standard_montage("standard_1020")
    raw.set_montage(montage, on_missing="ignore")

    print("\n--- MNE Raw Created ---")
    print(raw)
    print(raw.ch_names)

    return raw


def analyze_events(raw, output_dir):
    if "STI" not in raw.ch_names:
        print("No STI channel. Skipping events.")
        return None, None

    print("\n--- Event extraction ---")

    events = mne.find_events(
        raw,
        stim_channel="STI",
        consecutive=True,
        min_duration=0.01,
        verbose=True,
    )

    rows = []
    for name, val in EVENT_DICT.items():
        count = int(np.sum(events[:, 2] == val))
        rows.append({"event_name": name, "trigger_value": val, "count": count})
        print(f"{val:>2} | {name:<25} | {count}")

    event_counts = pd.DataFrame(rows)
    event_counts.to_csv(output_dir / "event_counts.csv", index=False)

    if len(events) > 0:
        fig = mne.viz.plot_events(
            events,
            sfreq=raw.info["sfreq"],
            first_samp=raw.first_samp,
            event_id=EVENT_DICT,
        )
        fig.savefig(output_dir / "event_timeline.png", dpi=300)
        plt.close(fig)

    return events, event_counts


def compute_channel_quality(raw, output_dir):
    rows = []

    for ch in raw.ch_names:
        if ch == "STI":
            continue

        x = raw.get_data(picks=[ch])[0] * 1e6

        n_10s = int(10 * SFREQ)
        if len(x) > 2 * n_10s:
            drift = np.mean(x[-n_10s:]) - np.mean(x[:n_10s])
        else:
            drift = np.nan

        rows.append({
            "channel": ch,
            "mean_uV": np.mean(x),
            "std_uV": np.std(x),
            "min_uV": np.min(x),
            "max_uV": np.max(x),
            "peak_to_peak_uV": np.ptp(x),
            "rms_uV": np.sqrt(np.mean(x ** 2)),
            "drift_first10s_vs_last10s_uV": drift,
            "samples_over_150uV": int(np.sum(np.abs(x) > 150)),
            "percent_over_150uV": 100 * np.mean(np.abs(x) > 150),
            "samples_over_300uV": int(np.sum(np.abs(x) > 300)),
            "percent_over_300uV": 100 * np.mean(np.abs(x) > 300),
            "zero_percent": 100 * np.mean(np.isclose(x, 0, atol=1e-12)),
            "comment": classify_channel(x, drift),
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "channel_quality_metrics.csv", index=False)
    print("\n--- Channel quality ---")
    print(df)
    return df


def classify_channel(x, drift):
    ptp = np.ptp(x)
    std = np.std(x)

    if std < 1e-6:
        return "flat/dead channel"
    if ptp > 1000:
        return "extreme artifact somewhere in session"
    if ptp > 500:
        return "large artifact somewhere in session"
    if not np.isnan(drift) and abs(drift) > 100:
        return "large drift"
    if 100 * np.mean(np.abs(x) > 150) > 5:
        return "many high-amplitude samples"
    return "ok"


def compute_eog_correlation(raw, output_dir):
    rows = []

    for eeg_ch in EEG_CHANNELS:
        if eeg_ch not in raw.ch_names:
            continue

        eeg = raw.get_data(picks=[eeg_ch])[0] * 1e6

        corr_h = np.nan
        corr_v = np.nan

        if "hEOG" in raw.ch_names:
            h = raw.get_data(picks=["hEOG"])[0] * 1e6
            corr_h = np.corrcoef(eeg, h)[0, 1]

        if "vEOG" in raw.ch_names:
            v = raw.get_data(picks=["vEOG"])[0] * 1e6
            corr_v = np.corrcoef(eeg, v)[0, 1]

        max_corr = np.nanmax([abs(corr_h), abs(corr_v)])

        rows.append({
            "eeg_channel": eeg_ch,
            "corr_with_hEOG": corr_h,
            "corr_with_vEOG": corr_v,
            "max_abs_eog_corr": max_corr,
            "comment": "possible EOG contamination" if max_corr > 0.30 else "low/moderate EOG relation",
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "eog_correlation_metrics.csv", index=False)
    print("\n--- EOG correlation ---")
    print(df)
    return df


def plot_event_context_all_channels(raw, events, output_dir, title_prefix):
    """
    This is more useful than first 60 seconds.
    It plots all EEG/EOG channels around meaningful event blocks.
    """

    if events is None or len(events) == 0:
        return

    context_events = {
        "EyeMovement": 3,
        "JawFacialMovement": 4,
        "HeadMovement": 5,
        "MI_LeftHand": 8,
        "MI_RightHand": 9,
    }

    tmin = -4
    tmax = 8

    for event_name, code in context_events.items():
        event_rows = events[events[:, 2] == code]

        if len(event_rows) == 0:
            continue

        for trial_i, ev in enumerate(event_rows[:5], start=1):
            start = int(ev[0] + tmin * SFREQ)
            end = int(ev[0] + tmax * SFREQ)

            if start < 0 or end > raw.n_times:
                continue

            time = np.arange(end - start) / SFREQ + tmin

            plt.figure(figsize=(16, 9))

            offset = 0
            spacing = 300

            for ch in ALL_SIGNAL_CHANNELS:
                if ch not in raw.ch_names:
                    continue

                x = raw.get_data(picks=[ch])[0][start:end] * 1e6
                plt.plot(time, x + offset, label=ch)
                offset += spacing

            plt.axvline(0, linestyle="--", linewidth=2, label=f"{event_name} trigger")

            if event_name.startswith("MI"):
                plt.axvspan(-2, 0, alpha=0.12, label="cue/context")
                plt.axvspan(0, 4, alpha=0.10, label="MI period")

            plt.title(f"{title_prefix} | {event_name} | Trial {trial_i} | All channels")
            plt.xlabel("Time relative to trigger (seconds)")
            plt.ylabel("Amplitude + offset (uV)")
            plt.legend(loc="upper right")
            plt.tight_layout()

            out = output_dir / f"ALL_CHANNELS_{event_name}_trial_{trial_i}.png"
            plt.savefig(out, dpi=300)
            plt.close()


def plot_mi_average_and_trials(raw, events, output_dir, title_prefix):
    """
    Plots each channel for MI events:
    individual trials + average trial.
    Window is wider than before so it is not too early.
    """

    if events is None or len(events) == 0:
        return

    mi_events = {
        "MI_LeftHand": 8,
        "MI_RightHand": 9,
    }

    tmin = -4
    tmax = 8

    for event_name, code in mi_events.items():
        event_rows = events[events[:, 2] == code]

        if len(event_rows) == 0:
            continue

        for ch in ALL_SIGNAL_CHANNELS:
            if ch not in raw.ch_names:
                continue

            trials = []

            for ev in event_rows:
                start = int(ev[0] + tmin * SFREQ)
                end = int(ev[0] + tmax * SFREQ)

                if start < 0 or end > raw.n_times:
                    continue

                x = raw.get_data(picks=[ch])[0][start:end] * 1e6
                trials.append(x)

            if len(trials) == 0:
                continue

            trials = np.array(trials)
            time = np.arange(trials.shape[1]) / SFREQ + tmin
            avg = np.mean(trials, axis=0)

            plt.figure(figsize=(16, 6))

            for tr in trials:
                plt.plot(time, tr, alpha=0.20)

            plt.plot(time, avg, linewidth=3, label="Average")

            plt.axvline(0, linestyle="--", linewidth=2, label=f"{event_name} trigger")
            plt.axvspan(-2, 0, alpha=0.12, label="cue/context")
            plt.axvspan(0, 4, alpha=0.10, label="MI period")

            plt.title(f"{title_prefix} | {ch} aligned to {event_name}")
            plt.xlabel("Time relative to MI trigger (seconds)")
            plt.ylabel("Amplitude (uV)")
            plt.legend()
            plt.tight_layout()

            out = output_dir / f"{ch}_MI_trials_average_{event_name}.png"
            plt.savefig(out, dpi=300)
            plt.close()


def plot_artifact_average_and_trials(raw, events, output_dir, title_prefix):
    """
    Plots artifact blocks separately:
    EyeMovement, JawFacialMovement, HeadMovement.
    """

    if events is None or len(events) == 0:
        return

    artifact_events = {
        "EyeMovement": 3,
        "JawFacialMovement": 4,
        "HeadMovement": 5,
    }

    tmin = -2
    tmax = 10

    for event_name, code in artifact_events.items():
        event_rows = events[events[:, 2] == code]

        if len(event_rows) == 0:
            continue

        for ch in ALL_SIGNAL_CHANNELS:
            if ch not in raw.ch_names:
                continue

            trials = []

            for ev in event_rows:
                start = int(ev[0] + tmin * SFREQ)
                end = int(ev[0] + tmax * SFREQ)

                if start < 0 or end > raw.n_times:
                    continue

                x = raw.get_data(picks=[ch])[0][start:end] * 1e6
                trials.append(x)

            if len(trials) == 0:
                continue

            trials = np.array(trials)
            time = np.arange(trials.shape[1]) / SFREQ + tmin
            avg = np.mean(trials, axis=0)

            plt.figure(figsize=(16, 6))

            for tr in trials:
                plt.plot(time, tr, alpha=0.35)

            plt.plot(time, avg, linewidth=3, label="Average")
            plt.axvline(0, linestyle="--", linewidth=2, label=f"{event_name} trigger")

            plt.title(f"{title_prefix} | {ch} aligned to {event_name}")
            plt.xlabel("Time relative to artifact trigger (seconds)")
            plt.ylabel("Amplitude (uV)")
            plt.legend()
            plt.tight_layout()

            out = output_dir / f"{ch}_ARTIFACT_trials_average_{event_name}.png"
            plt.savefig(out, dpi=300)
            plt.close()


def plot_eeg_eog_event_overlay(raw, events, output_dir, title_prefix):
    """
    For each meaningful event, overlay each EEG channel with hEOG/vEOG.
    This directly answers: is EOG leaking into EEG?
    """

    if events is None or len(events) == 0:
        return

    event_codes = {
        "EyeMovement": 3,
        "JawFacialMovement": 4,
        "HeadMovement": 5,
        "MI_LeftHand": 8,
        "MI_RightHand": 9,
    }

    tmin = -4
    tmax = 8

    for event_name, code in event_codes.items():
        event_rows = events[events[:, 2] == code]

        if len(event_rows) == 0:
            continue

        for eeg_ch in EEG_CHANNELS:
            if eeg_ch not in raw.ch_names:
                continue

            for trial_i, ev in enumerate(event_rows[:5], start=1):
                start = int(ev[0] + tmin * SFREQ)
                end = int(ev[0] + tmax * SFREQ)

                if start < 0 or end > raw.n_times:
                    continue

                time = np.arange(end - start) / SFREQ + tmin
                eeg = raw.get_data(picks=[eeg_ch])[0][start:end] * 1e6

                plt.figure(figsize=(16, 5))
                plt.plot(time, eeg, label=eeg_ch, linewidth=1.5)

                if "hEOG" in raw.ch_names:
                    h = raw.get_data(picks=["hEOG"])[0][start:end] * 1e6
                    plt.plot(time, h, label="hEOG", alpha=0.7)

                if "vEOG" in raw.ch_names:
                    v = raw.get_data(picks=["vEOG"])[0][start:end] * 1e6
                    plt.plot(time, v, label="vEOG", alpha=0.7)

                plt.axvline(0, linestyle="--", linewidth=2)

                if event_name.startswith("MI"):
                    plt.axvspan(-2, 0, alpha=0.12, label="cue/context")
                    plt.axvspan(0, 4, alpha=0.10, label="MI period")

                plt.title(f"{title_prefix} | {eeg_ch} + EOG | {event_name} trial {trial_i}")
                plt.xlabel("Time relative to trigger (seconds)")
                plt.ylabel("Amplitude (uV)")
                plt.legend()
                plt.tight_layout()

                out = output_dir / f"{eeg_ch}_EOG_OVERLAY_{event_name}_trial_{trial_i}.png"
                plt.savefig(out, dpi=300)
                plt.close()


def plot_mi_cleanliness_summary(raw, events, output_dir, title_prefix):
    """
    Makes a simple table-like CSV:
    For every MI trial, compute max amplitude in EEG and EOG.
    This helps identify bad MI trials.
    """

    if events is None or len(events) == 0:
        return None

    mi_events = {
        "MI_LeftHand": 8,
        "MI_RightHand": 9,
    }

    tmin = 0
    tmax = 4

    rows = []

    for event_name, code in mi_events.items():
        event_rows = events[events[:, 2] == code]

        for trial_idx, ev in enumerate(event_rows, start=1):
            start = int(ev[0] + tmin * SFREQ)
            end = int(ev[0] + tmax * SFREQ)

            if start < 0 or end > raw.n_times:
                continue

            row = {
                "event": event_name,
                "trial": trial_idx,
                "trigger_sample": int(ev[0]),
                "trigger_time_sec": ev[0] / SFREQ,
            }

            max_eeg = []
            max_eog = []

            for ch in EEG_CHANNELS:
                if ch in raw.ch_names:
                    x = raw.get_data(picks=[ch])[0][start:end] * 1e6
                    val = np.max(np.abs(x))
                    row[f"{ch}_max_abs_uV"] = val
                    max_eeg.append(val)

            for ch in EOG_CHANNELS:
                if ch in raw.ch_names:
                    x = raw.get_data(picks=[ch])[0][start:end] * 1e6
                    val = np.max(np.abs(x))
                    row[f"{ch}_max_abs_uV"] = val
                    max_eog.append(val)

            row["max_EEG_abs_uV"] = np.max(max_eeg) if max_eeg else np.nan
            row["max_EOG_abs_uV"] = np.max(max_eog) if max_eog else np.nan

            if row["max_EEG_abs_uV"] > 300:
                row["comment"] = "bad/high EEG amplitude during MI"
            elif row["max_EOG_abs_uV"] > 300:
                row["comment"] = "possible EOG artifact during MI"
            else:
                row["comment"] = "looks acceptable by amplitude threshold"

            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "mi_trial_cleanliness_summary.csv", index=False)

    print("\n--- MI trial cleanliness summary ---")
    print(df[["event", "trial", "trigger_time_sec", "max_EEG_abs_uV", "max_EOG_abs_uV", "comment"]])

    return df


def save_summary(output_dir, file_path, raw, event_counts, metrics_df, eog_df, mi_clean_df):
    with open(output_dir / "inspection_summary.txt", "w") as f:
        f.write("Signal Inspection Summary\n")
        f.write("=========================\n\n")
        f.write(f"File: {file_path}\n")
        f.write(f"Sampling frequency: {SFREQ} Hz\n")
        f.write(f"Duration: {raw.times[-1]:.2f} seconds\n")
        f.write(f"Channels: {raw.ch_names}\n\n")

        f.write("Event Counts\n")
        f.write("------------\n")
        f.write(event_counts.to_string(index=False) if event_counts is not None else "No events")
        f.write("\n\n")

        f.write("Channel Quality Metrics\n")
        f.write("-----------------------\n")
        f.write(metrics_df.to_string(index=False))
        f.write("\n\n")

        f.write("EOG Correlation Metrics\n")
        f.write("-----------------------\n")
        f.write(eog_df.to_string(index=False))
        f.write("\n\n")

        f.write("MI Trial Cleanliness\n")
        f.write("--------------------\n")
        if mi_clean_df is not None:
            f.write(mi_clean_df.to_string(index=False))
        else:
            f.write("Not available")


def run(subject, session, data_type):
    file_path = find_session_file(subject, session, data_type)

    subject_num = subject.replace("Sub", "")
    session_num = session.zfill(3)

    output_dir = PROJECT_DIR / "outputs" / "signal_quality_v2" / f"Sub{subject_num}_Ses{session_num}_{data_type}"
    output_dir.mkdir(parents=True, exist_ok=True)

    title_prefix = f"Sub{subject_num} Ses{session_num} {data_type.upper()}"

    df = load_csv(file_path)
    processed_data_path = output_dir / "processed_input_data.csv"
    df.to_csv(processed_data_path, index=False)
    print(f"Saved loaded input data to: {processed_data_path}")

    selected_cols, ch_names, ch_types, trigger_col = prepare_channels(df)
    raw = create_raw(df, selected_cols, ch_names, ch_types, trigger_col)

    events, event_counts = analyze_events(raw, output_dir)
    metrics_df = compute_channel_quality(raw, output_dir)
    eog_df = compute_eog_correlation(raw, output_dir)

    # Main useful plots
    plot_event_context_all_channels(raw, events, output_dir, title_prefix)
    plot_artifact_average_and_trials(raw, events, output_dir, title_prefix)
    plot_mi_average_and_trials(raw, events, output_dir, title_prefix)
    plot_eeg_eog_event_overlay(raw, events, output_dir, title_prefix)
    mi_clean_df = plot_mi_cleanliness_summary(raw, events, output_dir, title_prefix)

    save_summary(output_dir, file_path, raw, event_counts, metrics_df, eog_df, mi_clean_df)

    print("\nDONE")
    print(f"Output folder: {output_dir}")
    print("\nStart by checking:")
    print("1. event_counts.csv")
    print("2. mi_trial_cleanliness_summary.csv")
    print("3. processed_input_data.csv")
    print("4. ALL_CHANNELS_MI_LeftHand_trial_1.png")
    print("5. ALL_CHANNELS_MI_RightHand_trial_1.png")
    print("6. C2_EOG_OVERLAY_MI_LeftHand_trial_1.png")
    print("7. C3_EOG_OVERLAY_MI_RightHand_trial_1.png")
    print("8. hEOG_ARTIFACT_trials_average_EyeMovement.png")
    print("9. vEOG_ARTIFACT_trials_average_EyeMovement.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", default=DEFAULT_SUBJECT)
    parser.add_argument("--session", default=DEFAULT_SESSION)
    parser.add_argument("--data-type", choices=["raw", "filtered"], default=DEFAULT_DATA_TYPE)
    args = parser.parse_args()

    run(args.subject, args.session, args.data_type)
