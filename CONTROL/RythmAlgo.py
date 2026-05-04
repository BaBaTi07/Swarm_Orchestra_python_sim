import numpy as np


class RhythmAlgo:
    def __init__(
        self,
        nbr_beats: int,
        beat_duration_s: float,
        min_duration_factor: float = 0.25,
        max_duration_factor: float = 2.0,
    ):
        self.nbr_beats = nbr_beats
        self.beat_duration_s = beat_duration_s
        self.min_duration_factor = min_duration_factor
        self.max_duration_factor = max_duration_factor

        self.role_duration_factors = {
            "isolated": {
                "beats" :[[0, 0.5, 1], [0, 0.5, 1.5],[0, 0.5, 2]],
                "duration_factor": 0.5,
                "volume": 0.7
            },
            "no_scale": {
                "beats": [0],
                "duration_factor": 0.5,
                "volume": 0
            },
            "unstable_scale": {
                "beats": [0],
                "duration_factor": 0.75,
                "volume": 0.5
            },
            "crowded_unstable_scale": {
                "beats": [0],
                "duration_factor": 0.7,
                "volume": 0.4
            },
            "stable_scale": {
                "beats": [0],
                "duration_factor": 1.0,
                "volume": 0.8
            },
            "crowded_scale": {
                "beats": [0],
                "duration_factor": 0.9,
                "volume": 0.65
            },
            "stable_chord": {
                "beats": [0],
                "duration_factor": 2.0,
                "volume": 1.0
            },
            "crowded_chord": {
                "beats": [0],
                "duration_factor": 1.8,
                "volume": 0.8
            },
            
        }

    def infer_role(self, harmony_debug: dict) -> str:
        recent_neighbors = harmony_debug.get("recent_neighbors", 0)
        scale_confidence = harmony_debug.get("scale_confidence", 0.0)
        chord_notes = harmony_debug.get("chord_notes", None)
        reason = harmony_debug.get("reason", "")
        crowded = False

        if recent_neighbors == 0:
            return "isolated"
        
        if recent_neighbors >= 10:
            crowded = True

        if scale_confidence < 0.5:
            return "no_scale"

        if scale_confidence < 0.7:
            if crowded:
                return "crowded_unstable_scale"
            return "unstable_scale"

        if chord_notes is not None and "chord" in reason:
            if crowded:
                return "crowded_chord"
            return "stable_chord"
        
        if crowded:
            return "crowded_scale"
        return "stable_scale"

    def compute_note_duration(self, harmony_debug: dict):
        role = self.infer_role(harmony_debug)
        param = self.role_duration_factors[role]

        duration_factor = param["duration_factor"]

        volume = param["volume"]

        beats = param["beats"]
        if len(beats) > 1:
            beats = beats[np.random.randint(0, len(beats))]

        duration_factor = float(np.clip(
            duration_factor,
            self.min_duration_factor,
            self.max_duration_factor
        ))

        duration = duration_factor * self.beat_duration_s

        debug = {
            "rhythm_role": role,
            "duration_factor": duration_factor,
            "duration_s": duration,
            "beats": beats,
            "volume": volume
        }

        return duration, beats, volume, debug