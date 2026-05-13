
from datetime import datetime
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tools.logger import logger
from tools.scales import Scales, CHORD_PATTERNS


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

    # global mean over all runs and all time steps
    global_sync_mean = np.nan
    valid_R = R_mat[~np.isnan(R_mat)]
    if valid_R.size > 0:
        global_sync_mean = np.mean(valid_R)

    #global mean after 200 seconds to focus on steady state
    global_sync_mean_steady = np.nan
    steady_state_data = R_mat[:, time_grid >= 200]
    valid_steady_state_data = steady_state_data[~np.isnan(steady_state_data)]
    if valid_steady_state_data.size > 0:
        global_sync_mean_steady = np.mean(valid_steady_state_data)

    # Plot
    plt.figure()

    # R: middle 50% + median + mean
    plt.fill_between(time_grid, R_stats["q25"], R_stats["q75"], alpha=0.20, label="Phase Sync R - middle 50%")
    plt.plot(time_grid, R_stats["median"], linewidth=2.5, label="Phase Sync R median")
    plt.plot(time_grid, R_stats["mean"], linestyle="--", linewidth=1.8, label="Phase Sync R mean")
    if not np.isnan(global_sync_mean):
        plt.text(
            0.01, 0.01,
            f"Global mean sync = {global_sync_mean:.3f}",
            transform=plt.gca().transAxes,
            fontsize=10,
            verticalalignment='bottom'
        )
    if not np.isnan(global_sync_mean_steady):
        plt.text(
            0.01, 0.06,
            f"Global mean after 200s = {global_sync_mean_steady:.3f}",
            transform=plt.gca().transAxes,
            fontsize=10,
            verticalalignment='bottom'
        )
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
    plt.title(f"Synchronization : {base_name}")
    plt.ylim(-0.02, 1.05)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()

    png_path = build_filename(f"{base_name}_sync_aggregate", folder, file_extension="png")
    plt.savefig(png_path, dpi=180, bbox_inches="tight")
    plt.savefig("metrics/last/mult_last_sync_aggregate.png", dpi=180, bbox_inches="tight")
    plt.close()

    logger.log("INFO", f"Saved sync metrics:{png_path}")

def save_harmonic_scale_plot(notes_history, base_name: str, folder: str = "metrics/harmonic_scales"):
    """scan each scale and count how many notes each second belong to it,
    then plot a bar chart of the counts for the best scale"""

    #temporary here for test purpose
    save_chord_count_plot(notes_history, base_name)

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

def save_chord_count_plot( notes_history, base_name: str, folder: str = "metrics/chords/new", time_interval: float = 2.0, default_note_duration_s: float = 0.5, min_overlap_ratio: float = 0.8):
    """
    Plot the proportion of notes that belong to a detected chord per interval.

    Two curves:
    - chord_notes_any_beat: % of notes belonging to a chord in the interval, ignoring temporal overlap
    - chord_notes_same_beat: % of notes belonging to a chord with sufficient temporal overlap

    notes_history can contain:
    - (time_s, midi_note)
    - (time_s, midi_note, duration_s)
    """

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    if not notes_history:
        return

    data = np.array(notes_history, dtype=float)

    if data.ndim != 2 or data.shape[1] < 2:
        logger.log("WARNING", "notes_history must contain rows like (time_s, midi_note) or (time_s, midi_note, duration_s)")
        return

    data = data[np.argsort(data[:, 0])]
    data = data[data[:, 0] >= 5]

    if len(data) == 0:
        logger.log("WARNING", "No notes after initialisation period.")
        return

    has_duration = data.shape[1] >= 3

    max_time = data[-1, 0]
    time_bins = np.arange(0, np.floor(max_time) + time_interval, time_interval)

    chord_notes_any_beat = []
    chord_notes_same_beat = []

    def all_possible_chords():
        chords = []
        for root in range(12):
            for name, intervals in CHORD_PATTERNS.items():
                notes = {(root + i) % 12 for i in intervals}
                chords.append((root, name, notes))
        return chords

    possible_chords = all_possible_chords()

    for t0 in time_bins:
        t1 = t0 + time_interval

        interval_notes = data[(data[:, 0] >= t0) & (data[:, 0] < t1)]

        if len(interval_notes) == 0:
            chord_notes_any_beat.append(0.0)
            chord_notes_same_beat.append(0.0)
            continue

        total_notes = len(interval_notes)
        pitch_classes = interval_notes[:, 1].astype(int) % 12

        # ------------------------------------------------------------
        # 1. Chord notes ignoring temporal overlap
        # ------------------------------------------------------------
        chord_pitch_classes_any = set()

        unique_pcs = set(pitch_classes)

        for root, name, chord_notes in possible_chords:
            if chord_notes.issubset(unique_pcs):
                chord_pitch_classes_any.update(chord_notes)

        notes_in_any_chord = sum(pc in chord_pitch_classes_any for pc in pitch_classes)
        proportion_any = notes_in_any_chord / total_notes * 100.0
        chord_notes_any_beat.append(proportion_any)

        # ------------------------------------------------------------
        # 2. Chord notes with temporal overlap
        # ------------------------------------------------------------
        chord_pitch_classes_same = set()

        if has_duration:
            starts = interval_notes[:, 0]
            durations = interval_notes[:, 2]
        else:
            starts = interval_notes[:, 0]
            durations = np.full(len(interval_notes), default_note_duration_s)

        ends = starts + durations

        for root, name, chord_notes in possible_chords:
            matching_indices = [
                i for i, pc in enumerate(pitch_classes)
                if pc in chord_notes
            ]

            matching_pcs = {pitch_classes[i] for i in matching_indices}

            if not chord_notes.issubset(matching_pcs):
                continue

            chord_start = max(starts[i] for i in matching_indices)
            chord_end = min(ends[i] for i in matching_indices)

            overlap_duration = chord_end - chord_start

            if overlap_duration <= 0:
                continue

            min_duration = min(durations[i] for i in matching_indices)
            overlap_ratio = overlap_duration / min_duration

            if overlap_ratio >= min_overlap_ratio:
                chord_pitch_classes_same.update(chord_notes)

        notes_in_same_chord = sum(pc in chord_pitch_classes_same for pc in pitch_classes)
        proportion_same = notes_in_same_chord / total_notes * 100.0
        chord_notes_same_beat.append(proportion_same)

    # ------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------
    plt.figure(figsize=(10, 5))

    plt.plot(time_bins, chord_notes_any_beat, label="Chord notes, any timing")
    plt.plot(time_bins, chord_notes_same_beat, label="Chord notes, temporal overlap")

    plt.xlabel("Time (s)")
    plt.ylabel("Notes belonging to a chord (%)")
    plt.title("Proportion of chord-related notes over time")
    plt.ylim(0, 100)
    plt.grid(True)
    plt.legend()

    png_path = build_filename(f"{base_name}_chord_count", folder, file_extension="png")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig("metrics/last/last_chord_count.png", dpi=300, bbox_inches="tight")
    plt.close()

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

def generate_multiple_execution_harmonic_graph(all_notes_history, base_name: str, folder: str = "metrics/harmonic_scales", time_interval: float = 3.0, min_time: float = 5.0):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    if not all_notes_history:
        return
    
    #temporary here for test purpose
    generate_multiple_execution_chord_graph(all_notes_history, base_name)

    processed_runs = []

    for run in all_notes_history:
        if not run:
            continue

        data = np.array(run, dtype=float)
        if data.ndim != 2 or data.shape[1] < 2:
            continue

        data = data[np.argsort(data[:, 0])]
        data = data[data[:, 0] >= min_time]

        if len(data) == 0:
            continue

        max_time = data[-1, 0]
        time_bins = np.arange(min_time, np.floor(max_time) + 1e-9, time_interval)

        scores = []
        for t in time_bins:
            notes_in_interval = data[(data[:, 0] >= t) & (data[:, 0] < t + time_interval), 1].astype(int)

            if len(notes_in_interval) == 0:
                scores.append(np.nan)
                continue

            best_score = 0.0
            for scale in Scales:
                count_in_scale = sum((note % 12) in scale.notes for note in notes_in_interval)
                percent_in_scale = 100.0 * count_in_scale / len(notes_in_interval)
                best_score = max(best_score, percent_in_scale)

            scores.append(best_score)

        if len(time_bins) > 0:
            processed_runs.append((time_bins, np.array(scores, dtype=float)))

    if not processed_runs:
        logger.log("WARNING", "No valid notes_history found for harmonic aggregate plot")
        return

    t_max = max(t[-1] for t, _ in processed_runs)
    time_grid = np.arange(min_time, t_max + 1e-9, time_interval)

    mat = []
    for t, y in processed_runs:
        yi = np.interp(time_grid, t, y)
        yi[time_grid < t[0]] = np.nan
        yi[time_grid > t[-1]] = np.nan
        mat.append(yi)

    mat = np.array(mat, dtype=float)

    mean = np.nanmean(mat, axis=0)
    median = np.nanmedian(mat, axis=0)
    q25 = np.nanpercentile(mat, 25, axis=0)
    q75 = np.nanpercentile(mat, 75, axis=0)

    plt.figure(figsize=(10, 5))
    plt.fill_between(time_grid, q25, q75, alpha=0.20, label="middle 50%")
    plt.plot(time_grid, median, linewidth=2.5, label="median")
    plt.plot(time_grid, mean, linestyle="--", linewidth=1.8, label="mean")

    plt.xlabel("Time (s)")
    plt.ylabel("% harmonically synchronized")
    plt.title("Harmonic synchronization across multiple runs")
    plt.ylim(0, 105)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    png_path = build_filename(f"{base_name}_harmonic_sync_aggregate", folder, file_extension="png")
    plt.savefig(png_path, dpi=180, bbox_inches="tight")
    plt.savefig("metrics/last/mult_last_harmonic_aggregate.png", dpi=180, bbox_inches="tight")
    plt.close()

    logger.log("INFO", f"Saved harmonic aggregate graph: {png_path}")

def generate_multiple_execution_chord_graph(all_notes_history, base_name: str, folder: str = "metrics/chords/multiple", time_interval: float = 2.0, default_note_duration_s: float = 0.5, min_overlap_ratio: float = 0.8, min_time: float = 5.0):
    """
    Generate an aggregate graph over multiple executions showing the proportion
    of notes belonging to a detected chord.

    Two aggregate curves:
    - chord notes, any timing
    - chord notes, with temporal overlap
    """

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    if not all_notes_history:
        return

    def all_possible_chords():
        chords = []
        for root in range(12):
            for name, intervals in CHORD_PATTERNS.items():
                notes = {(root + i) % 12 for i in intervals}
                chords.append((root, name, notes))
        return chords

    possible_chords = all_possible_chords()
    processed_any_runs = []
    processed_same_runs = []

    for run in all_notes_history:
        if not run:
            continue

        data = np.array(run, dtype=float)

        if data.ndim != 2 or data.shape[1] < 2:
            continue

        data = data[np.argsort(data[:, 0])]
        data = data[data[:, 0] >= min_time]

        if len(data) == 0:
            continue

        has_duration = data.shape[1] >= 3

        max_time = data[-1, 0]
        time_bins = np.arange(min_time, np.floor(max_time) + 1e-9, time_interval)

        chord_notes_any_beat = []
        chord_notes_same_beat = []

        for t0 in time_bins:
            t1 = t0 + time_interval

            interval_notes = data[(data[:, 0] >= t0) & (data[:, 0] < t1)]

            if len(interval_notes) == 0:
                chord_notes_any_beat.append(np.nan)
                chord_notes_same_beat.append(np.nan)
                continue

            total_notes = len(interval_notes)
            pitch_classes = interval_notes[:, 1].astype(int) % 12

            # ------------------------------------------------------------
            # 1. Chord notes ignoring temporal overlap
            # ------------------------------------------------------------
            chord_pitch_classes_any = set()
            unique_pcs = set(pitch_classes)

            for root, name, chord_notes in possible_chords:
                if chord_notes.issubset(unique_pcs):
                    chord_pitch_classes_any.update(chord_notes)

            notes_in_any_chord = sum(pc in chord_pitch_classes_any for pc in pitch_classes)
            proportion_any = notes_in_any_chord / total_notes * 100.0
            chord_notes_any_beat.append(proportion_any)

            # ------------------------------------------------------------
            # 2. Chord notes with temporal overlap
            # ------------------------------------------------------------
            chord_pitch_classes_same = set()

            if has_duration:
                starts = interval_notes[:, 0]
                durations = interval_notes[:, 2]
            else:
                starts = interval_notes[:, 0]
                durations = np.full(len(interval_notes), default_note_duration_s)

            ends = starts + durations

            for root, name, chord_notes in possible_chords:
                matching_indices = [
                    i for i, pc in enumerate(pitch_classes)
                    if pc in chord_notes
                ]

                matching_pcs = {pitch_classes[i] for i in matching_indices}

                if not chord_notes.issubset(matching_pcs):
                    continue

                chord_start = max(starts[i] for i in matching_indices)
                chord_end = min(ends[i] for i in matching_indices)

                overlap_duration = chord_end - chord_start

                if overlap_duration <= 0:
                    continue

                min_duration = min(durations[i] for i in matching_indices)
                overlap_ratio = overlap_duration / min_duration

                if overlap_ratio >= min_overlap_ratio:
                    chord_pitch_classes_same.update(chord_notes)

            notes_in_same_chord = sum(pc in chord_pitch_classes_same for pc in pitch_classes)
            proportion_same = notes_in_same_chord / total_notes * 100.0
            chord_notes_same_beat.append(proportion_same)

        if len(time_bins) > 1:
            processed_any_runs.append((time_bins, np.array(chord_notes_any_beat, dtype=float)))
            processed_same_runs.append((time_bins, np.array(chord_notes_same_beat, dtype=float)))

    if not processed_any_runs:
        logger.log("WARNING", "No valid notes_history found for aggregate chord plot")
        return

    t_max = max(t[-1] for t, _ in processed_any_runs)
    time_grid = np.arange(min_time, t_max + 1e-9, time_interval)

    def build_matrix(processed_runs):
        mat = []

        for t, y in processed_runs:
            valid = ~np.isnan(y)

            if np.sum(valid) < 2:
                continue

            t_valid = t[valid]
            y_valid = y[valid]

            yi = np.interp(time_grid, t_valid, y_valid)
            yi[time_grid < t_valid[0]] = np.nan
            yi[time_grid > t_valid[-1]] = np.nan

            mat.append(yi)

        if not mat:
            return np.empty((0, len(time_grid)))

        return np.array(mat, dtype=float)

    any_mat = build_matrix(processed_any_runs)
    same_mat = build_matrix(processed_same_runs)

    if any_mat.size == 0 and same_mat.size == 0:
        logger.log("WARNING", "No valid chord data after interpolation")
        return

    def compute_stats(mat):
        if mat.size == 0:
            return None

        return {
            "mean": np.nanmean(mat, axis=0),
            "median": np.nanmedian(mat, axis=0),
            "q25": np.nanpercentile(mat, 25, axis=0),
            "q75": np.nanpercentile(mat, 75, axis=0),
        }

    any_stats = compute_stats(any_mat)
    same_stats = compute_stats(same_mat)

    plt.figure(figsize=(10, 5))

    if any_stats is not None:
        plt.fill_between(
            time_grid,
            any_stats["q25"],
            any_stats["q75"],
            alpha=0.15,
            label="Any timing - middle 50%"
        )
        plt.plot(
            time_grid,
            any_stats["median"],
            linewidth=2.5,
            label="Any timing - median"
        )
        plt.plot(
            time_grid,
            any_stats["mean"],
            linestyle="--",
            linewidth=1.8,
            label="Any timing - mean"
        )

    if same_stats is not None:
        plt.fill_between(
            time_grid,
            same_stats["q25"],
            same_stats["q75"],
            alpha=0.15,
            label="Temporal overlap - middle 50%"
        )
        plt.plot(
            time_grid,
            same_stats["median"],
            linewidth=2.5,
            label="Temporal overlap - median"
        )
        plt.plot(
            time_grid,
            same_stats["mean"],
            linestyle="--",
            linewidth=1.8,
            label="Temporal overlap - mean"
        )

    plt.xlabel("Time (s)")
    plt.ylabel("Notes belonging to a chord (%)")
    plt.title("Chord-related notes across multiple runs")
    plt.ylim(0, 105)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    png_path = build_filename(f"{base_name}_chord_aggregate", folder, file_extension="png")
    plt.savefig(png_path, dpi=180, bbox_inches="tight")
    plt.savefig("metrics/last/mult_last_chord_aggregate.png", dpi=180, bbox_inches="tight")
    plt.close()

    logger.log("INFO", f"Saved aggregate chord graph: {png_path}")

def generate_multiple_execution_beat_evenness_graph(all_beat_played_history, base_name: str, folder: str = "metrics/beat_played", window_s: float = 2.0, n_beats: int = 4):

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    if not all_beat_played_history:
        return

    sigma2_best = 0.0
    sigma2_worst = np.var([1.0] + [0.0] * (n_beats - 1))

    processed_runs = []

    for run in all_beat_played_history:
        if not run:
            continue

        data = np.array(run, dtype=float)
        if data.ndim != 2 or data.shape[1] < 2:
            continue

        data = data[np.argsort(data[:, 0])]
        times = data[:, 0]
        beats = data[:, 1].astype(int)

        evenness_times = []
        evenness_values = []

        for t in times:
            t_min = t - window_s

            mask = (times >= t_min) & (times <= t)
            window_beats = beats[mask]

            if np.any((window_beats < 0) | (window_beats >= n_beats)):
                logger.log(
                    "WARNING",
                    f"Found beat numbers out of range [0, {n_beats-1}] in beat_played_history. They will be ignored."
                )

            window_beats = window_beats[(window_beats >= 0) & (window_beats < n_beats)]

            if len(window_beats) == 0:
                continue

            counts = np.zeros(n_beats, dtype=int)
            for b in window_beats:
                counts[b] += 1

            proportions = counts / len(window_beats)
            sigma2 = np.var(proportions)

            evenness = (sigma2 - sigma2_best) / (sigma2_worst - sigma2_best)
            evenness = float(np.clip(evenness, 0.0, 1.0))

            evenness_times.append(t)
            evenness_values.append(evenness)

        if len(evenness_times) > 1:
            processed_runs.append((
                np.array(evenness_times, dtype=float),
                np.array(evenness_values, dtype=float)
            ))

    if not processed_runs:
        logger.log("WARNING", "No valid beat_played_history found for aggregate evenness plot")
        return

    t_max = max(t[-1] for t, _ in processed_runs)
    time_grid = np.arange(0.0, t_max + 1e-9, 1.0)

    mat = []
    for t, y in processed_runs:
        yi = np.interp(time_grid, t, y)
        yi[time_grid < t[0]] = np.nan
        yi[time_grid > t[-1]] = np.nan
        mat.append(yi)

    mat = np.array(mat, dtype=float)

    mean = np.nanmean(mat, axis=0)
    median = np.nanmedian(mat, axis=0)
    q25 = np.nanpercentile(mat, 25, axis=0)
    q75 = np.nanpercentile(mat, 75, axis=0)

    plt.figure(figsize=(10, 5))
    plt.fill_between(time_grid, q25, q75, alpha=0.20, label="middle 50%")
    plt.plot(time_grid, median, linewidth=2.5, label="median")
    plt.plot(time_grid, mean, linestyle="--", linewidth=1.8, label="mean")

    plt.xlabel("Time (s)")
    plt.ylabel("Beat distribution evenness ε(t)")
    plt.title(f"Beat Distribution Evenness across multiple runs (window = {window_s:.2f}s)")
    plt.ylim(-0.02, 1.02)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    png_path = build_filename(f"{base_name}_beat_evenness_aggregate", folder, file_extension="png")
    plt.savefig(png_path, dpi=180, bbox_inches="tight")
    plt.savefig("metrics/last/mult_last_beat_evenness_aggregate.png", dpi=180, bbox_inches="tight")
    plt.close()

    logger.log("INFO", f"Saved aggregate beat evenness graph: {png_path}")