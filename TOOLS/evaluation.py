from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from CONTROL.harmony_algo import CHORD_PATTERNS
from TOOLS.scales import Scales
from TOOLS.logger import logger
from TOOLS.plot_gen import build_filename

# Evaluationof the swarm music production, in terms of musical quality
# Criteria of evaluation are the following:
#
# - Coherence and stability
#    - synchronisation
#    - harmony (Scale, chords and no disonant notes)
#
# - Diversity
#    - Note diversity 
#    - Rhythm distribution


def evaluate_musical_quality(
    phase_sync_history,
    notes_history,
    beat_played_history,
    base_name: str = "musical_quality",
    folder: str = "metrics/quality",
    min_time: float = 200.0,
    time_interval: float = 2.0,
    diversity_window_s: float = 60.0,
    default_note_duration_s: float = 0.5,
    min_overlap_ratio: float = 0.8,
    beat_match_tolerance_s: float = 0.05,
    weights: dict | None = None,
    plot: bool = True,
):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    if weights is None:
        weights = {
            "phase_sync": 0.35,
            "scale_harmony": 0.30,
            "chord_harmony": 0.10,
            "beat_evenness": 0.10,
            "note_diversity_by_beat": 0.15,
        }

    weights = normalize_weights(weights)

    notes_data = prepare_notes_data(notes_history, min_time)

    if notes_data is None:
        return None

    time_bins = build_time_bins(notes_data, min_time, time_interval)

    phase_sync_score = compute_phase_sync_score(
        phase_sync_history,
        min_time=min_time
    )

    scale_harmony_score = compute_scale_harmony_score(
        notes_data,
        time_bins,
        time_interval=time_interval
    )

    chord_harmony_score = compute_chord_harmony_score(
        notes_data,
        time_bins,
        time_interval=time_interval,
        default_note_duration_s=default_note_duration_s,
        min_overlap_ratio=min_overlap_ratio
    )

    chord_harmony_score_regardless_of_beats = compute_chord_harmony_score_regardless_of_beats(
        notes_data,
        time_bins,
        time_interval=time_interval,
    )

    beat_evenness_score = compute_beat_evenness_score(
        beat_played_history,
        min_time=min_time,
        window_s=time_interval,
        n_beats=4
    )

    note_diversity_by_beat_score = compute_note_diversity_by_beat_score(
        notes_data,
        beat_played_history,
        time_bins,
        min_time=min_time,
        diversity_window_s=diversity_window_s,
        beat_match_tolerance_s=beat_match_tolerance_s,
        n_beats=4
    )

    chord_harmony_combined = compute_weighted_final_score(
    {
        "same_beat": chord_harmony_score,
        "global": chord_harmony_score_regardless_of_beats,
    },
    {
        "same_beat": 0.5,
        "global": 0.5,
    }
)
    display_scores = {
        "phase_sync": phase_sync_score,
        "scale_harmony": scale_harmony_score,
        "chord_harmony": chord_harmony_combined,
        "beat_evenness": beat_evenness_score,
        "note_diversity_by_beat": note_diversity_by_beat_score,
    }

    scoring_scores = {
        "phase_sync": phase_sync_score,

        # Scale harmony is permissive, so only the part above 70% counts.
        "scale_harmony": rescale_score_above_baseline(
            scale_harmony_score,
            baseline=0.70
        ),

        "chord_harmony": chord_harmony_combined,
        "beat_evenness": beat_evenness_score,
        "note_diversity_by_beat": note_diversity_by_beat_score,
    }

    final_score = compute_weighted_final_score(scoring_scores, weights)

    results = {
        "final_score": final_score,
        "final_score_percent": final_score * 100.0 if not np.isnan(final_score) else np.nan,
        "display_scores": display_scores,
        "scoring_scores": scoring_scores,
        "weights": weights,
    }

    if plot:
        plot_musical_quality_scores(
            display_scores,
            final_score,
            base_name,
            folder
        )

    logger.log(
        "INFO",
        f"Musical quality score: {results['final_score_percent']:.2f}% | details: {display_scores}"
    )

    return results



# ============================================================
# Generic helpers
# ============================================================

def safe_mean(values):
    values = np.array(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return np.nan
    return float(np.mean(values))


def normalize_weights(weights: dict) -> dict:
    weight_sum = sum(weights.values())
    if weight_sum == 0:
        raise ValueError("The sum of weights cannot be zero.")
    return {k: v / weight_sum for k, v in weights.items()}


def normalized_entropy(values, n_classes: int):
    if len(values) == 0:
        return np.nan

    counts = np.zeros(n_classes, dtype=float)

    for v in values:
        if 0 <= v < n_classes:
            counts[v] += 1

    total = np.sum(counts)
    if total == 0:
        return np.nan

    probs = counts / total
    probs = probs[probs > 0]

    entropy = -np.sum(probs * np.log(probs))
    max_entropy = np.log(n_classes)

    if max_entropy == 0:
        return 0.0

    return float(entropy / max_entropy)


def get_possible_chords():
    chords = []

    for root in range(12):
        for name, intervals in CHORD_PATTERNS.items():
            notes = {(root + i) % 12 for i in intervals}
            chords.append((root, name, notes))

    return chords

def rescale_score_above_baseline(score, baseline: float = 0.70):
    """
    Rescale a score so that:
    - score <= baseline gives 0
    - score = 1 gives 1

    Example:
    baseline = 0.70
    score = 0.95 -> (0.95 - 0.70) / 0.30 = 0.833
    score = 0.76 -> (0.76 - 0.70) / 0.30 = 0.20
    """

    if np.isnan(score):
        return np.nan

    if score <= baseline:
        return 0.0

    return float(np.clip((score - baseline) / (1.0 - baseline), 0.0, 1.0))

# ============================================================
# Data preparation
# ============================================================

def prepare_notes_data(notes_history, min_time: float = 5.0):
    if not notes_history:
        return None

    data = np.array(notes_history, dtype=float)

    if data.ndim != 2 or data.shape[1] < 2:
        logger.log("WARNING", "notes_history must contain at least (time_s, midi_note)")
        return None

    data = data[np.argsort(data[:, 0])]
    data = data[data[:, 0] >= min_time]

    if len(data) == 0:
        logger.log("WARNING", "No notes after initialisation period.")
        return None

    return data


def build_time_bins(notes_data, min_time: float, time_interval: float):
    max_time = notes_data[-1, 0]
    return np.arange(min_time, max_time + 1e-9, time_interval)

# ============================================================
# Metric 1: phase synchronization
# ============================================================

def compute_phase_sync_score(phase_sync_history, min_time: float = 5.0):
    if not phase_sync_history:
        return np.nan

    data = np.array(phase_sync_history, dtype=float)

    if data.ndim != 2 or data.shape[1] < 2:
        return np.nan

    data = data[np.argsort(data[:, 0])]
    data = data[data[:, 0] >= min_time]

    if len(data) == 0:
        return np.nan

    return safe_mean(data[:, 1])


# ============================================================
# Metric 2: harmonic scale coherence
# ============================================================

def compute_scale_harmony_score(notes_data, time_bins, time_interval: float):
    scores = []

    for t0 in time_bins:
        t1 = t0 + time_interval

        interval_notes = notes_data[
            (notes_data[:, 0] >= t0) &
            (notes_data[:, 0] < t1)
        ]

        if len(interval_notes) == 0:
            scores.append(np.nan)
            continue

        midi_notes = interval_notes[:, 1].astype(int)

        best_score = 0.0

        for scale in Scales:
            count_in_scale = sum((note % 12) in scale.notes for note in midi_notes)
            score = count_in_scale / len(midi_notes)
            best_score = max(best_score, score)

        scores.append(best_score)

    return safe_mean(scores)

# ============================================================
# Metric 3: chord harmony
# ============================================================
def compute_chord_harmony_score_regardless_of_beats(notes_data, time_bins, time_interval: float):
    # check if the a chord can be form in a short time window, without considering the beat information.
    # This is a more permissive metric, as it allows chords to be formed even if the notes are not perfectly aligned in time.
    # It is useful to evaluate the harmony of the swarm because every chord can not be form to avoid beat crowding
    possible_chords = get_possible_chords()
    chord_scores = []

    for t0 in time_bins:
        t1 = t0 + time_interval

        interval_notes = notes_data[
            (notes_data[:, 0] >= t0) &
            (notes_data[:, 0] < t1)
        ]

        if len(interval_notes) == 0:
            chord_scores.append(np.nan)
            continue

        pitch_classes = interval_notes[:, 1].astype(int) % 12
        total_notes = len(interval_notes)

        chord_pitch_classes = set()

        for root, name, chord_notes in possible_chords:
            if chord_notes.issubset(set(pitch_classes)):
                chord_pitch_classes.update(chord_notes)

        notes_in_chord = sum(pc in chord_pitch_classes for pc in pitch_classes)
        score = notes_in_chord / total_notes
        chord_scores.append(score)

    return safe_mean(chord_scores)


def compute_chord_harmony_score(
    notes_data,
    time_bins,
    time_interval: float,
    default_note_duration_s: float = 0.5,
    min_overlap_ratio: float = 0.8,
):
    possible_chords = get_possible_chords()
    has_duration = notes_data.shape[1] >= 3

    chord_scores_any = []
    chord_scores_same = []

    for t0 in time_bins:
        any_score, same_score = compute_chord_scores_for_interval(
            notes_data,
            t0,
            time_interval,
            possible_chords,
            has_duration,
            default_note_duration_s,
            min_overlap_ratio
        )

        chord_scores_any.append(any_score)
        chord_scores_same.append(same_score)

    chord_harmony_score = safe_mean(chord_scores_same)

    if np.isnan(chord_harmony_score):
        chord_harmony_score = safe_mean(chord_scores_any)

    return chord_harmony_score


def compute_chord_scores_for_interval(
    notes_data,
    t0: float,
    time_interval: float,
    possible_chords,
    has_duration: bool,
    default_note_duration_s: float,
    min_overlap_ratio: float,
):
    t1 = t0 + time_interval

    interval_notes = notes_data[
        (notes_data[:, 0] >= t0) &
        (notes_data[:, 0] < t1)
    ]

    if len(interval_notes) == 0:
        return np.nan, np.nan

    total_notes = len(interval_notes)
    pitch_classes = interval_notes[:, 1].astype(int) % 12

    any_score = compute_chord_score_any_timing(
        pitch_classes,
        possible_chords,
        total_notes
    )

    same_score = compute_chord_score_temporal_overlap(
        interval_notes,
        pitch_classes,
        possible_chords,
        has_duration,
        default_note_duration_s,
        min_overlap_ratio,
        total_notes
    )

    return any_score, same_score


def compute_chord_score_any_timing(pitch_classes, possible_chords, total_notes: int):
    chord_pitch_classes = set()
    unique_pcs = set(pitch_classes)

    for root, name, chord_notes in possible_chords:
        if chord_notes.issubset(unique_pcs):
            chord_pitch_classes.update(chord_notes)

    notes_in_chord = sum(pc in chord_pitch_classes for pc in pitch_classes)
    return notes_in_chord / total_notes


def compute_chord_score_temporal_overlap(
    interval_notes,
    pitch_classes,
    possible_chords,
    has_duration: bool,
    default_note_duration_s: float,
    min_overlap_ratio: float,
    total_notes: int,
):
    chord_pitch_classes = set()

    starts = interval_notes[:, 0]

    if has_duration:
        durations = interval_notes[:, 2]
    else:
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
            chord_pitch_classes.update(chord_notes)

    notes_in_chord = sum(pc in chord_pitch_classes for pc in pitch_classes)
    return notes_in_chord / total_notes

# ============================================================
# Metric 4: beat evenness
# ============================================================

def compute_beat_evenness_score(
    beat_played_history,
    min_time: float = 5.0,
    window_s: float = 2.0,
    n_beats: int = 4,
):
    if not beat_played_history:
        return np.nan

    data = np.array(beat_played_history, dtype=float)

    if data.ndim != 2 or data.shape[1] < 2:
        return np.nan

    data = data[np.argsort(data[:, 0])]
    data = data[data[:, 0] >= min_time]

    if len(data) == 0:
        return np.nan

    times = data[:, 0]
    beats = data[:, 1].astype(int)

    sigma2_best = 0.0
    sigma2_worst = np.var([1.0] + [0.0] * (n_beats - 1))

    evenness_scores = []

    for t in times:
        t_min = t - window_s

        mask = (times >= t_min) & (times <= t)
        window_beats = beats[mask]
        window_beats = window_beats[
            (window_beats >= 0) &
            (window_beats < n_beats)
        ]

        if len(window_beats) == 0:
            continue

        counts = np.zeros(n_beats, dtype=int)

        for b in window_beats:
            counts[b] += 1

        proportions = counts / len(window_beats)
        sigma2 = np.var(proportions)

        unevenness = (sigma2 - sigma2_best) / (sigma2_worst - sigma2_best)
        unevenness = float(np.clip(unevenness, 0.0, 1.0))

        evenness = 1.0 - unevenness
        evenness_scores.append(evenness)

    return safe_mean(evenness_scores)

# ============================================================
# Metric 5: note diversity by beat
# ============================================================

def compute_note_diversity_by_beat_score(
    notes_data,
    beat_played_history,
    time_bins,
    min_time: float = 5.0,
    diversity_window_s: float = 60.0,
    beat_match_tolerance_s: float = 0.05,
    n_beats: int = 4,
):
    notes_with_beats = build_notes_with_beats(
        notes_data,
        beat_played_history,
        beat_match_tolerance_s
    )

    if notes_with_beats is None or len(notes_with_beats) == 0:
        return np.nan

    note_times = notes_with_beats[:, 0]
    note_pcs = notes_with_beats[:, 1].astype(int) % 12
    note_beats = notes_with_beats[:, 2].astype(int)

    diversity_scores = []

    for t in time_bins:
        t_min = max(min_time, t - diversity_window_s)
        t_max = t

        window_mask = (note_times >= t_min) & (note_times <= t_max)

        if not np.any(window_mask):
            diversity_scores.append(np.nan)
            continue

        beat_diversities = []

        for beat in range(n_beats):
            beat_mask = window_mask & (note_beats == beat)
            pcs_for_beat = note_pcs[beat_mask]

            if len(pcs_for_beat) == 0:
                continue

            diversity = normalized_entropy(pcs_for_beat, n_classes=12)
            beat_diversities.append(diversity)

        diversity_scores.append(safe_mean(beat_diversities))

    return safe_mean(diversity_scores)


def build_notes_with_beats(
    notes_data,
    beat_played_history,
    beat_match_tolerance_s: float = 0.05,
):
    """
    Returns array:
    (time_s, midi_note, beat)

    If notes_data contains a 4th column, it is assumed to be the beat.
    Otherwise, the function tries to match each note with the closest
    beat event from beat_played_history.
    """

    has_beat_in_notes = notes_data.shape[1] >= 4
    notes_with_beats = []

    if has_beat_in_notes:
        for row in notes_data:
            time_s = row[0]
            midi_note = int(row[1])
            beat = int(row[3])
            notes_with_beats.append((time_s, midi_note, beat))

        return np.array(notes_with_beats, dtype=float)

    if not beat_played_history:
        return None

    beat_data = np.array(beat_played_history, dtype=float)

    if beat_data.ndim != 2 or beat_data.shape[1] < 2:
        return None

    beat_data = beat_data[np.argsort(beat_data[:, 0])]
    beat_times = beat_data[:, 0]
    beats = beat_data[:, 1].astype(int)

    for row in notes_data:
        time_s = row[0]
        midi_note = int(row[1])

        idx = np.argmin(np.abs(beat_times - time_s))
        delta = abs(beat_times[idx] - time_s)

        if delta <= beat_match_tolerance_s:
            notes_with_beats.append((time_s, midi_note, int(beats[idx])))

    return np.array(notes_with_beats, dtype=float)

# ============================================================
# Plot
# ============================================================

def plot_musical_quality_scores(
    raw_scores: dict,
    final_score: float,
    base_name: str,
    folder: str,
):
    labels = list(raw_scores.keys())
    values = [
        raw_scores[k] * 100.0 if not np.isnan(raw_scores[k]) else 0.0
        for k in labels
    ]

    plt.figure(figsize=(10, 5))
    plt.bar(labels, values)

    if not np.isnan(final_score):
        plt.axhline(
            final_score * 100.0,
            linestyle="--",
            linewidth=2,
            label=f"Final score = {final_score * 100.0:.1f}%"
        )

    plt.ylabel("Score (%)")
    plt.title("Global Musical Quality Evaluation")
    plt.ylim(0, 100)
    plt.xticks(rotation=25, ha="right")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()

    png_path = build_filename(f"{base_name}_musical_quality", folder, file_extension="png")
    plt.savefig(png_path, dpi=180, bbox_inches="tight")
    plt.savefig("metrics/last/last_musical_quality.png", dpi=180, bbox_inches="tight")
    plt.close()

    logger.log("INFO", f"Saved musical quality graph: {png_path}")


def compute_weighted_final_score(raw_scores: dict, weights: dict):
    final_score = 0.0
    used_weight_sum = 0.0

    for key, score in raw_scores.items():
        if np.isnan(score):
            continue

        final_score += weights[key] * score
        used_weight_sum += weights[key]

    if used_weight_sum == 0:
        return np.nan

    return final_score / used_weight_sum