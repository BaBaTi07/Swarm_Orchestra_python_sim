
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

def interpolate_runs(all_runs, value_col: int, time_grid: np.ndarray):

    interpolated = []

    for run in all_runs:
        if not run:
            continue

        data = np.array(run, dtype=float)
        if data.ndim != 2 or data.shape[1] <= value_col:
            continue

        # sort by time
        data = data[np.argsort(data[:, 0])]
        t = data[:, 0]
        y = data[:, value_col]

        # remove duplicate times if any
        uniq_t, uniq_idx = np.unique(t, return_index=True)
        y = y[uniq_idx]
        t = uniq_t

        if len(t) < 2:
            continue

        # interpolate only inside valid interval, NaN outside
        yi = np.interp(time_grid, t, y)
        yi[time_grid < t[0]] = np.nan
        yi[time_grid > t[-1]] = np.nan

        interpolated.append(yi)

    if not interpolated:
        return np.empty((0, len(time_grid)))

    return np.array(interpolated, dtype=float)

def save_sync_plot(phase_sync_history, base_name: str, folder: str = "metrics/sync"):

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    if not phase_sync_history:
        return
    
    # determine common max time from all runs
    t_max  = phase_sync_history[0][-1][0] if phase_sync_history[0] else 0.0
    time_grid = np.arange(0.0, t_max + 1e-9, 1.0)

    # interpolate metrics
    R_mat        = interpolate_runs(phase_sync_history, value_col=1, time_grid=time_grid)
    min_conf_mat = interpolate_runs(phase_sync_history, value_col=2, time_grid=time_grid)
    mean_conf_mat= interpolate_runs(phase_sync_history, value_col=3, time_grid=time_grid)
    max_conf_mat = interpolate_runs(phase_sync_history, value_col=4, time_grid=time_grid)

    def stats(mat):
        if mat.size == 0:
            return {
                "mean": np.array([]),
                "median": np.array([]),
                "q25": np.array([]),
                "q75": np.array([]),
            }

        mean = np.full(mat.shape[1], np.nan)
        median = np.full(mat.shape[1], np.nan)
        q25 = np.full(mat.shape[1], np.nan)
        q75 = np.full(mat.shape[1], np.nan)

        for i in range(mat.shape[1]):
            col = mat[:, i]
            col = col[~np.isnan(col)]
            if len(col) == 0:
                continue

            mean[i] = np.mean(col)
            median[i] = np.median(col)
            q25[i] = np.percentile(col, 25)
            q75[i] = np.percentile(col, 75)

        return {
            "mean": mean,
            "median": median,
            "q25": q25,
            "q75": q75,
        }

    R_stats = stats(R_mat)

    # Plot
    plt.figure()

    # R: middle 50% + median + mean
    plt.fill_between(time_grid, R_stats["q25"], R_stats["q75"], alpha=0.20, label="Phase Sync R - middle 50%")
    plt.plot(time_grid, R_stats["median"], linewidth=2.5, label="Phase Sync R median")
    plt.plot(time_grid, R_stats["mean"], linestyle="--", linewidth=1.8, label="Phase Sync R mean")

    # sparse boxplots for R
    box_times = np.arange(0.0, t_max + 1e-9, 20.0)
    box_idx = [np.argmin(np.abs(time_grid - bt)) for bt in box_times]

    box_data = []
    box_positions = []
    for idx in box_idx:
        vals = R_mat[:, idx]
        vals = vals[~np.isnan(vals)]
        if len(vals) > 0:
            box_data.append(vals)
            box_positions.append(time_grid[idx])

    if box_data:
        plt.boxplot(
            box_data,
            positions=box_positions,
            widths=20.0 * 0.35,
            manage_ticks=False,
            patch_artist=True,
            boxprops=dict(alpha=0.25),
            medianprops=dict(linewidth=1.5),
            whiskerprops=dict(linewidth=1.0),
            capprops=dict(linewidth=1.0),
            flierprops=dict(marker='o', markersize=3, alpha=0.5),
        )

    plt.xlabel("Simulation time (s)")
    plt.ylabel("Value")
    plt.title("Synchronization over time across multiple runs")
    plt.ylim(-0.02, 1.05)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()

    png_path = build_filename(f"{base_name}_sync_aggregate", folder, file_extension="png")
    plt.savefig(png_path, dpi=180, bbox_inches="tight")
    plt.close()

    logger.log("INFO", f"Saved sync metrics:{png_path}")

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
