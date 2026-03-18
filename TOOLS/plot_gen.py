
from datetime import datetime
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from TOOLS.logger import logger
from TOOLS.scales import Scales


def build_filename( base_name: str, folder: str, file_extension: str = "mid" ) -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    return folder / f"{base_name}_{timestamp}.{file_extension}"

def save_sync_plot_and_csv(phase_sync_history,base_name: str, folder: str = "metrics/sync"):

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    if not phase_sync_history:
        return

    data = np.array(phase_sync_history, dtype=float)
    t = data[:, 0]
    R = data[:, 1]
    min_conf = data[:, 2]
    mean_conf = data[:, 3] 
    max_conf = data[:, 4]

    # CSV
    csv_path = build_filename(f"{base_name}_kuramoto_R", folder, file_extension="csv")
    np.savetxt(csv_path, data, delimiter=",", header="time_s,R,kuramoto_conf_min,kuramoto_conf_mean,kuramoto_conf_max", comments="")

    # Plot
    plt.figure()
    plt.plot(t, R, label="Phase Sync R")
    plt.plot(t, min_conf, label="Kuramoto Confidence Min", linestyle='--')
    plt.plot(t, mean_conf, label="Kuramoto Confidence Mean", linestyle='--')
    plt.plot(t, max_conf, label="Kuramoto Confidence Max", linestyle='--')
    plt.xlabel("Simulation time (s)")
    plt.ylabel("Kuramoto order parameter R")
    plt.title("Synchronization over time")
    plt.legend()
    png_path = build_filename(f"{base_name}_kuramoto_R", folder, file_extension="png")
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.savefig('metrics/last/last_Kuramoto_sync.png', dpi=150, bbox_inches="tight")
    plt.close()

    logger.log("INFO", f"Saved sync metrics: {csv_path} and {png_path}")

def save_harmonic_scale_plot(notes_history, base_name: str, folder: str = "metrics/harmonic_scales"):
    """scan each scale and count how many notes each second belong to it,
    then plot a bar chart of the counts for the best scale"""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    if not notes_history:
        return

    data = np.array(notes_history, dtype=float)

    if data.ndim != 2 or data.shape[1] < 2:
        logger.log("WARNING", "notes_history must contain rows like (time_s, midi_note)")
        return

    
    # sort by time and exclude 5 first seconds to avoid initialisation
    data = data[np.argsort(data[:, 0])]
    data = data[data[:, 0] >= 5]

    time_interval = 3.0
    max_time = data[-1, 0]
    time_bins = np.arange(0, np.floor(max_time) + 1, time_interval)

    dominant_scores = []
    dominant_scale_indices = []

    # score every scale for every 3-second interval
    for t in time_bins:
        notes_in_interval = data[(data[:, 0] >= t) & (data[:, 0] < t + time_interval), 1].astype(int)

        if len(notes_in_interval) == 0:
            dominant_scores.append(0.0)
            dominant_scale_indices.append(-1)  # no dominant scale
            continue

        scale_percentages = []
        for scale in Scales:
            count_in_scale = sum((note % 12) in scale.notes for note in notes_in_interval)
            percent_in_scale = 100.0 * count_in_scale / len(notes_in_interval)
            scale_percentages.append(percent_in_scale)

        scale_percentages = np.array(scale_percentages)
        best_idx = int(np.argmax(scale_percentages))
        best_score = float(scale_percentages[best_idx])

        dominant_scores.append(best_score)
        dominant_scale_indices.append(best_idx)

    dominant_scores = np.array(dominant_scores)
    dominant_scale_indices = np.array(dominant_scale_indices)

    # color list for segments
    cmap = plt.get_cmap("tab20")
    scale_colors = {i: cmap(i % 20) for i in range(len(Scales))}
    scale_colors[-1] = "gray"  # intervals with no notes

    plt.figure(figsize=(12, 6))

    # plot by continuous segments of same dominant scale
    start_idx = 0
    for i in range(1, len(time_bins) + 1):
        segment_end = (i == len(time_bins)) or (dominant_scale_indices[i] != dominant_scale_indices[start_idx])

        if segment_end:
            scale_idx = dominant_scale_indices[start_idx]
            end = min(i+1, len(time_bins))
            x_segment = time_bins[start_idx:end]
            y_segment = dominant_scores[start_idx:end]

            color = scale_colors[scale_idx]

            plt.step(x_segment, y_segment,where = "post", color=color, linewidth=2.5, marker="o", markersize=3)

            start_idx = i

    plt.xlabel("Time (s)")
    plt.ylabel("Dominant scale score (%)")
    plt.title("Evolution of the dominant harmonic scale over time")
    plt.ylim(0, 105)
    plt.grid(True, alpha=0.3)

    # custom legend scales used
    used_scales = sorted(set(dominant_scale_indices))
    handles = []
    labels = []
    for idx in used_scales:
        color = scale_colors[idx]
        name = "No notes" if idx == -1 else Scales[idx].name
        handles.append(plt.Line2D([0], [0], color=color, lw=3))
        labels.append(name)

    if labels:
        plt.legend(handles, labels, title="Dominant scale", bbox_to_anchor=(1.02, 1), loc="upper left")

    png_path = build_filename(f"{base_name}_harmonic_scale", folder, file_extension="png")
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.savefig("metrics/last/last_harmonic_scale.png", dpi=150, bbox_inches="tight")
    plt.close()

    logger.log("INFO", f"Saved harmonic scale metrics: {png_path}")

def save_beat_played_plot(beat_played_history, base_name: str, folder: str = "metrics/beat_played", window_s = 2.0):
    """plot a step chart of the beat played over time
    
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    if not beat_played_history:
        return

    data = np.array(beat_played_history, dtype=float)

    if data.ndim != 2 or data.shape[1] < 2:
        logger.log("WARNING", "beat_played_history must contain rows like (time_s, beat_number)")
        return

    data = data[np.argsort(data[:, 0])]

    times = data[:, 0]
    beats = data[:, 1].astype(int)

    N = 4 # number of beats
    sigma2_best = 0.0
    sigma2_worst = np.var([1.0, 0.0, 0.0, 0.0])  # 0.1875

    evenness_times = []
    evenness_values = []
    beat_counts_history = []

    for i, t in enumerate(times):
        t_min = t - window_s

        # events inside sliding window [t - window_s, t]
        mask = (times >= t_min) & (times <= t)
        window_beats = beats[mask]

        # keep only valid beats and warn if some are out of range
        if np.any((window_beats < 0) | (window_beats >= N)):
            logger.log("WARNING", f"Found beat numbers out of range [0, {N-1}] in beat_played_history. They will be ignored in evenness calculation.")
        window_beats = window_beats[(window_beats >= 0) & (window_beats < N)]
        

        if len(window_beats) == 0:
            continue

        counts = np.zeros(N, dtype=int)
        for b in window_beats:
            counts[b] += 1

        proportions = counts / len(window_beats)
        sigma2 = np.var(proportions)

        evenness = (sigma2 - sigma2_best) / (sigma2_worst - sigma2_best)
        evenness = float(np.clip(evenness, 0.0, 1.0))

        evenness_times.append(t)
        evenness_values.append(evenness)
        beat_counts_history.append(counts.copy())

    if not evenness_times:
        return

    evenness_times = np.array(evenness_times)
    evenness_values = np.array(evenness_values)
    beat_counts_history = np.array(beat_counts_history)

    # --- Plot 1: evenness over time ---
    plt.figure(figsize=(10, 5))
    plt.step(evenness_times, evenness_values, where="post")
    plt.xlabel("Time (s)")
    plt.ylabel("Evenness ε(t)")
    plt.title(f"Beat Distribution Evenness (sliding window = {window_s:.2f}s)")
    plt.ylim(-0.02, 1.02)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    name = build_filename(f"{base_name}", folder, file_extension="png")
    plt.savefig(f"{name}_evenness.png", dpi=200)
    plt.savefig("metrics/last/last_evenness.png", dpi=200)
    plt.close()

    # --- Plot 2: raw beat event counts in the sliding window ---
    plt.figure(figsize=(10, 5))
    for beat_idx in range(N):
        plt.step(
            evenness_times,
            beat_counts_history[:, beat_idx],
            where="post",
            label=f"Beat {beat_idx}"
        )

    plt.xlabel("Time (s)")
    plt.ylabel(f"Number of events in last {window_s:.2f}s")
    plt.title("Beat Event Distribution Over Time")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{name}_beat_counts.png", dpi=200)
    plt.savefig("metrics/last/last_beat_count.png",dpi = 200)
    plt.close() 
