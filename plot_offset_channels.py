from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
SFREQ = 300

CSV_PATH = PROJECT_DIR / "Sub1_data" / "Filtered_data" / "Sub001Ses001_filtered.csv"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "signal_quality" / "Sub1_Ses001_filtered_offset_plots"

CHANNEL_MAP = {
    "S1:CZ": "Cz",
    "S2:CP2": "C2",
    "S3:CP3": "C3",
    "S4:FC2": "C4",
    "S5:FC3": "C5",
    "S6:vEOGt": "hEOG",
    "S7:vEOGb": "vEOG",
}


def load_channel_data():
    df = pd.read_csv(CSV_PATH, comment="#", skipinitialspace=True)
    cols = [col for col in CHANNEL_MAP if col in df.columns]
    names = [CHANNEL_MAP[col] for col in cols]
    data_uV = df[cols].to_numpy(dtype=float).T
    time = np.arange(data_uV.shape[1]) / SFREQ
    return time, data_uV, names


def plot_offset_full_session(time, data_uV, names, spacing=500, clip_uV=400):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    step = max(1, int(SFREQ / 10))
    time_ds = time[::step]
    data_ds = data_uV[:, ::step]

    plt.figure(figsize=(18, 8))
    for idx, (name, signal) in enumerate(zip(names, data_ds)):
        offset = idx * spacing
        signal = np.clip(signal, -clip_uV, clip_uV)
        plt.plot(time_ds, signal + offset, linewidth=0.8, alpha=0.9, label=name)
        plt.text(time_ds[-1] + 5, offset, name, va="center", fontsize=9)

    plt.title("Sub1 Ses001 FILTERED - Full session - All channels with offsets")
    plt.xlabel("Time (seconds)")
    plt.ylabel(f"Amplitude + vertical offset (uV), clipped at +/-{clip_uV} uV")
    plt.xlim(time_ds[0], time_ds[-1] + 35)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "all_channels_full_session_offset.png", dpi=300)
    plt.close()


def plot_offset_intervals(time, data_uV, names, start_sec=200, stop_sec=800, interval_sec=100, spacing=500, clip_uV=400):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    intervals = []
    current = start_sec
    while current < stop_sec:
        intervals.append((current, min(current + interval_sec, stop_sec)))
        current += interval_sec

    step = max(1, int(SFREQ / 10))
    fig, axes = plt.subplots(
        len(intervals),
        1,
        figsize=(18, 3.2 * len(intervals)),
        sharey=True
    )

    for ax, (start, stop) in zip(axes, intervals):
        start_sample = int(start * SFREQ)
        stop_sample = int(stop * SFREQ)
        time_window = time[start_sample:stop_sample:step]
        data_window = data_uV[:, start_sample:stop_sample:step]

        for idx, (name, signal) in enumerate(zip(names, data_window)):
            offset = idx * spacing
            signal = np.clip(signal, -clip_uV, clip_uV)
            ax.plot(time_window, signal + offset, linewidth=0.8, alpha=0.9, label=name)

        ax.set_title(f"{start}-{stop} seconds")
        ax.set_xlim(start, stop)
        ax.set_ylabel("Channel")
        ax.set_yticks(np.arange(len(names)) * spacing)
        ax.set_yticklabels(names)
        ax.grid(True, linewidth=0.4, alpha=0.3)

    axes[-1].set_xlabel("Time (seconds)")
    axes[0].legend(loc="upper right", ncol=4)
    fig.suptitle(
        "Sub1 Ses001 FILTERED - All channels with offsets - 200 to 800 seconds in 100s intervals",
        y=0.995
    )
    fig.text(
        0.005,
        0.5,
        f"Amplitude + vertical offset (uV), clipped at +/-{clip_uV} uV",
        rotation="vertical",
        va="center"
    )
    fig.tight_layout(rect=[0.025, 0, 1, 0.985])
    fig.savefig(OUTPUT_DIR / "all_channels_offset_200_800s_100s_intervals.png", dpi=300)
    plt.close(fig)


def plot_frequency_amplitude_offset_intervals(
    time,
    data_uV,
    names,
    start_sec=200,
    stop_sec=800,
    interval_sec=100,
    spacing=80,
    fmin=0.5,
    fmax=45,
):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    intervals = []
    current = start_sec
    while current < stop_sec:
        intervals.append((current, min(current + interval_sec, stop_sec)))
        current += interval_sec

    fig, axes = plt.subplots(
        len(intervals),
        1,
        figsize=(18, 3.2 * len(intervals)),
        sharex=True,
        sharey=True
    )

    for ax, (start, stop) in zip(axes, intervals):
        start_sample = int(start * SFREQ)
        stop_sample = int(stop * SFREQ)
        data_window = data_uV[:, start_sample:stop_sample]
        data_window = data_window - np.mean(data_window, axis=1, keepdims=True)

        window = np.hanning(data_window.shape[1])
        freqs = np.fft.rfftfreq(data_window.shape[1], d=1 / SFREQ)
        fft_values = np.fft.rfft(data_window * window, axis=1)
        amplitude = 2 * np.abs(fft_values) / np.sum(window)

        freq_mask = (freqs >= fmin) & (freqs <= fmax)
        freqs = freqs[freq_mask]
        amplitude = amplitude[:, freq_mask]

        display_limit = np.nanpercentile(amplitude, 98)
        if display_limit == 0 or np.isnan(display_limit):
            display_limit = 1

        for idx, (name, amp) in enumerate(zip(names, amplitude)):
            offset = idx * spacing
            amp = np.clip(amp, 0, display_limit)
            ax.plot(freqs, amp + offset, linewidth=1.0, alpha=0.9, label=name)

        ax.set_title(f"{start}-{stop} seconds")
        ax.set_ylabel("Channel")
        ax.set_yticks(np.arange(len(names)) * spacing)
        ax.set_yticklabels(names)
        ax.grid(True, linewidth=0.4, alpha=0.3)

    axes[-1].set_xlabel("Frequency (Hz)")
    axes[-1].set_xlim(fmin, fmax)
    axes[0].legend(loc="upper right", ncol=4)
    fig.suptitle(
        "Sub1 Ses001 FILTERED - EEG/EOG frequency amplitude with channel offsets - "
        "200 to 800 seconds in 100s intervals",
        y=0.995
    )
    fig.text(
        0.005,
        0.5,
        "Frequency amplitude + channel offset (uV)",
        rotation="vertical",
        va="center"
    )
    fig.tight_layout(rect=[0.025, 0, 1, 0.985])
    fig.savefig(
        OUTPUT_DIR / "all_channels_frequency_amplitude_offset_200_800s_100s_intervals.png",
        dpi=300
    )
    plt.close(fig)


def main():
    time, data_uV, names = load_channel_data()
    plot_offset_full_session(time, data_uV, names)
    plot_offset_intervals(time, data_uV, names)
    plot_frequency_amplitude_offset_intervals(time, data_uV, names)
    print(f"Saved offset plots to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
