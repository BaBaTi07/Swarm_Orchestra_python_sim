from TOOLS.midi_evaluation import evaluate_musical_quality_from_midi

quality = evaluate_musical_quality_from_midi(
    "MIDI/midi_records/trial_10_2026-05-07_03-10-48.mid",
    folder="metrics/quality/MIDI",
)

print(quality)