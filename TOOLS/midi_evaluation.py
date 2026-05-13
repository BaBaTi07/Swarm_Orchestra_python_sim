import mido
import numpy as np
from tools.qualityScoresHistory import QualityScoresHistory
from tools.evaluation import evaluate_musical_quality, compute_weighted_final_score, safe_mean
from tools.logger import logger

# Provides utilities to evaluate musical quality from MIDI files
def evaluate_musical_quality_from_midi(
    midi_path: list | str,
    base_name: str = "midi_evaluation",
    folder: str = "metrics/quality",
    cycle_time_s: float = 2.0,
    n_beats: int = 4,
    min_time: float = 200.0,
    time_interval: float = 2.0,
    diversity_window_s: float = 60.0,
    default_note_duration_s: float = 0.5,
    min_overlap_ratio: float = 0.8,
    repeat_threshold_s: float = 0.30,
    sync_window_s: float = 4.0,
    sync_step_s: float = 2.0,
    weights: dict | None = None,
    plot: bool = True,
):
    if isinstance(midi_path, list):
        all_results = []
        qualityScoresHistory = QualityScoresHistory()
        for path in midi_path:
            result = evaluate_musical_quality_from_midi(
                path,
                base_name=path.split("/")[-1].replace(".mid", ""),
                folder=folder,
                cycle_time_s=cycle_time_s,
                n_beats=n_beats,
                min_time=min_time,
                time_interval=time_interval,
                diversity_window_s=diversity_window_s,
                default_note_duration_s=default_note_duration_s,
                min_overlap_ratio=min_overlap_ratio,
                repeat_threshold_s=repeat_threshold_s,
                sync_window_s=sync_window_s,
                sync_step_s=sync_step_s,
                weights=weights,
                plot=plot,
            )
            if result is not None:
                all_results.append(result)
                qualityScoresHistory.add_scores(result["display_scores"] | {"final_score": result["final_score"]})
        qualityScoresHistory.plot_all_score_history(base_name, "metrics/quality/MIDI/multiple_trials")
        return all_results
    
    notes_history, beat_played_history, note_events_with_source = midi_to_histories(
        midi_path=midi_path,
        cycle_time_s=cycle_time_s,
        n_beats=n_beats,
    )

    if not notes_history:
        logger.log("WARNING", f"No valid notes found in MIDI file: {midi_path}")
        return None

    notes_after_convergence = [
        note for note in notes_history
        if note[0] >= min_time
    ]

    note_events_after_convergence = [
        event for event in note_events_with_source
        if event[0] >= min_time
    ]

    if not notes_after_convergence:
        logger.log("WARNING", f"No notes after convergence time min_time={min_time}s")
        return None

    observed_sync_score = compute_adaptive_midi_sync_score(
        note_events_after_convergence,
        cycle_time_s=cycle_time_s,
        repeat_threshold_s=repeat_threshold_s,
        window_s=sync_window_s,
        step_s=sync_step_s,
        n_beats=n_beats,
    )

    phase_sync_history = [
        (min_time, observed_sync_score),
        (notes_after_convergence[-1][0], observed_sync_score),
    ]

    if weights is None:
        weights = {
            "phase_sync": 0.20,
            "scale_harmony": 0.25,
            "chord_harmony": 0.20,
            "beat_evenness": 0.15,
            "note_diversity_by_beat": 0.20,
        }

    results = evaluate_musical_quality(
        phase_sync_history=phase_sync_history,
        notes_history=notes_history,
        beat_played_history=beat_played_history,
        base_name=base_name,
        folder=folder,
        min_time=min_time,
        time_interval=time_interval,
        diversity_window_s=diversity_window_s,
        default_note_duration_s=default_note_duration_s,
        min_overlap_ratio=min_overlap_ratio,
        beat_match_tolerance_s=0.05,
        weights=weights,
        plot=plot,
    )

    if results is not None:
        results["observed_midi_sync"] = observed_sync_score

    return results

def midi_to_histories(
    midi_path: str,
    cycle_time_s: float = 2.0,
    n_beats: int = 4,
    min_note_duration_s: float = 0.05,
):
    """
    Convert a MIDI file into:
    - notes_history: (time_s, midi_note, duration_s, beat)
    - beat_played_history: (time_s, beat)
    - note_events_with_source: (time_s, midi_note, duration_s, beat, source_id)
    """

    mid = mido.MidiFile(midi_path)

    notes_history = []
    beat_played_history = []
    note_events_with_source = []

    ticks_per_beat = mid.ticks_per_beat
    tempo = 500000  # 120 BPM default

    for track_id, track in enumerate(mid.tracks):
        current_time_s = 0.0
        active_notes = {}

        for msg in track:
            current_time_s += mido.tick2second(
                msg.time,
                ticks_per_beat,
                tempo
            )

            if msg.type == "set_tempo":
                tempo = msg.tempo
                continue

            if msg.type == "note_on" and msg.velocity > 0:
                channel = msg.channel if hasattr(msg, "channel") else 0
                key = (channel, msg.note)
                active_notes[key] = current_time_s

            elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                channel = msg.channel if hasattr(msg, "channel") else 0
                key = (channel, msg.note)

                if key not in active_notes:
                    continue

                start_s = active_notes.pop(key)
                duration_s = current_time_s - start_s

                if duration_s < min_note_duration_s:
                    continue

                beat_duration_s = cycle_time_s / n_beats
                phase_in_cycle = start_s % cycle_time_s
                beat = int(phase_in_cycle // beat_duration_s)
                beat = int(np.clip(beat, 0, n_beats - 1))

                source_id = track_id

                notes_history.append((
                    start_s,
                    int(msg.note),
                    duration_s,
                    beat
                ))

                beat_played_history.append((
                    start_s,
                    beat
                ))

                note_events_with_source.append((
                    start_s,
                    int(msg.note),
                    duration_s,
                    beat,
                    source_id
                ))

    notes_history = sorted(notes_history, key=lambda x: x[0])
    beat_played_history = sorted(beat_played_history, key=lambda x: x[0])
    note_events_with_source = sorted(note_events_with_source, key=lambda x: x[0])

    return notes_history, beat_played_history, note_events_with_source

def filter_repeated_note_attacks(
    note_events,
    repeat_threshold_s: float = 0.30
):
    """
    Remove fast repeated attacks from the same source and same pitch.

    note_events rows:
    (time_s, midi_note, duration_s, beat, source_id)
    """

    if not note_events:
        return []

    note_events = sorted(note_events, key=lambda x: x[0])

    filtered = []
    last_seen = {}

    for event in note_events:
        time_s = event[0]
        midi_note = int(event[1])
        source_id = int(event[4])

        key = (source_id, midi_note)

        if key in last_seen:
            if time_s - last_seen[key] < repeat_threshold_s:
                continue

        filtered.append(event)
        last_seen[key] = time_s

    return filtered

def compute_adaptive_midi_sync_score(
    note_events,
    cycle_time_s: float = 2.0,
    repeat_threshold_s: float = 0.30,
    window_s: float = 10.0,
    step_s: float = 2.0,
    n_beats: int = 4,
):
    """
    Estimate MIDI synchronization with an adaptive local phase score.

    This score does not assume a fixed global phase.
    It measures whether note attacks are concentrated around one common
    phase inside each temporal window.

    Good for swarms whose shared phase can drift over time.
    """

    if not note_events:
        return np.nan

    filtered_events = filter_repeated_note_attacks(
        note_events,
        repeat_threshold_s=repeat_threshold_s
    )

    if not filtered_events:
        return np.nan

    attack_times = np.array([event[0] for event in filtered_events], dtype=float)

    t_min = attack_times[0]
    t_max = attack_times[-1]

    window_scores = []

    for t0 in np.arange(t_min, t_max + 1e-9, step_s):
        t1 = t0 + window_s

        window_times = attack_times[
            (attack_times >= t0) &
            (attack_times < t1)
        ]

        if len(window_times) == 0:
            continue

        beat_duration_s = cycle_time_s / n_beats
        phases = 2.0 * np.pi * ((window_times % beat_duration_s) / beat_duration_s)
        
        R = np.abs(np.mean(np.exp(1j * phases)))

        window_scores.append(float(R))

    return safe_mean(window_scores)
