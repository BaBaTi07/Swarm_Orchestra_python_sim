from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import threading
import pretty_midi
from pathlib import Path

@dataclass(frozen=True)
class MidiNoteEvent:
    track_id: int          # ex: robot.id ou channel_id
    pitch: int             # midi note number
    start_s: float
    duration_s: float
    velocity: int = 80     # 1..127 (intensitée de la note)


class MidiRecorder:
    """
    Enregistre des notes (note-on + duration) et exporte en .mid
    """
    def __init__(self, tempo_bpm: float = 120.0):
        self.tempo_bpm = float(tempo_bpm)
        self._enabled = False
        self._lock = threading.Lock()
        self._events: Dict[int, List[MidiNoteEvent]] = {}

    def start(self) -> None:
        with self._lock:
            self._enabled = True
            self._events.clear()

    def stop(self) -> None:
        with self._lock:
            self._enabled = False

    def is_enabled(self) -> bool:
        return self._enabled

    def record_note(
        self,
        track_id: int,
        pitch: int,
        start_s: float,
        duration_s: float,
        volume_0_1: float = 0.6,
        velocity: Optional[int] = None,
    ) -> None:
        if not self._enabled:
            return
        
        # Convert volume (0..1) to MIDI velocity (1..127) if velocity is not explicitly provided
        if velocity is None:
            v = int(round(max(0.0, min(1.0, float(volume_0_1))) * 127.0))
            velocity = max(1, min(127, v))

        ev = MidiNoteEvent(
            track_id=int(track_id),
            pitch=int(pitch),
            start_s=float(start_s),
            duration_s=max(0.0, float(duration_s)),
            velocity=int(velocity),
        )

        with self._lock:
            self._events.setdefault(ev.track_id, []).append(ev)

    def write_midi(self, output_path: str | Path, program: int = 0) -> None:
        output_path = str(output_path)

        with self._lock:
            snapshot = {tid: list(evts) for tid, evts in self._events.items()}

        pm = pretty_midi.PrettyMIDI(initial_tempo=self.tempo_bpm)

        for track_id in sorted(snapshot.keys()):
            inst = pretty_midi.Instrument(program=int(program), name=f"track_{track_id}")
            for ev in snapshot[track_id]:
                start = ev.start_s
                end = ev.start_s + ev.duration_s
                inst.notes.append(
                    pretty_midi.Note(
                        velocity=ev.velocity,
                        pitch=ev.pitch,
                        start=float(start),
                        end=float(end),
                    )
                )
            pm.instruments.append(inst)

        pm.write(output_path)
