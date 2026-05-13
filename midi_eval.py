from app.tools.midi_evaluation import evaluate_musical_quality_from_midi

quality = evaluate_musical_quality_from_midi(
    [ 
     "MIDI/midi_records/trial_10_2026-05-12_02-04-11.mid",
     "MIDI/midi_records/trial_9_2026-05-12_01-57-27.mid",
     "MIDI/midi_records/trial_8_2026-05-12_01-50-25.mid",
     "MIDI/midi_records/trial_7_2026-05-12_01-43-20.mid",
     "MIDI/midi_records/trial_6_2026-05-12_01-35-55.mid",
     "MIDI/midi_records/trial_5_2026-05-12_01-28-28.mid",
     "MIDI/midi_records/trial_4_2026-05-12_01-22-44.mid",
     "MIDI/midi_records/trial_3_2026-05-12_01-17-02.mid",
     "MIDI/midi_records/trial_2_2026-05-12_01-11-15.mid",
     "MIDI/midi_records/trial_1_2026-05-12_01-05-20.mid",
    ],
    folder="metrics/quality/MIDI",
)

print(quality)