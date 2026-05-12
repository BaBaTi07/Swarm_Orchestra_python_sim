import numpy as np
from TOOLS.logger import logger
from TOOLS.scales import Scales, CHORD_PATTERNS


class HarmonyAlgo:
    """
    Local harmony algorithm for decentralized robots.

    Goals:
    - keep local/global convergence toward a common scale
    - allow punctual emergence of local chords
    - keep current chord while neighbors remain compatible
    - avoid global collapse toward one single beat

    Input:
    - note messages only (payload >= 128)
    - current note event of the robot
    - current beat of the robot
    - current time

    Output:
    - chosen note event
    - chosen beat
    - debug info dict
    """

    def __init__(
        self,
        nbr_beats: int,
        beat_duration_s: float,
        note_memory_ttl_s: float = 80.0, #gamme
        chord_memory_ttl_s: float = 6.0,  #accord
        beat_memory_ttl_s: float = 40.0,   #beat
        same_captor_merge_ttl_s: float = 1.0,
        fallback_volume: float = 0.6,
        beat_change_eval_delay_s: float = 2.0,
        bad_beat_penalty_decay: float = 0.98,
        dominant_beat_window_s: float = 20.0,
        forbidden_pair_ttl_s: float = 2.0,

        chord_commitment_ttl_s: float = 12.0,
        chord_create_probability: float = 0.25,
        chord_creation_score: float = 1.5,
        chord_beat_join_boost: float = 2.5,
        max_chord_beat_occupancy: int = 5,

        candidate_scale_threshold: float = 0.95,
        strict_scale_threshold_for_chords: float = 1.0,
        disambiguation_probability: float = 0.80,

        min_stable_scale_updates: int = 3
    ):
        self.nbr_beats = nbr_beats
        self.beat_duration_s = beat_duration_s

        self.note_memory_ttl_s = note_memory_ttl_s
        self.chord_memory_ttl_s = chord_memory_ttl_s
        self.beat_memory_ttl_s = beat_memory_ttl_s
        self.same_captor_merge_ttl_s = same_captor_merge_ttl_s
        self.fallback_volume = fallback_volume

        # new memory / adaptation params
        self.beat_change_eval_delay_s = beat_change_eval_delay_s
        self.bad_beat_penalty_decay = bad_beat_penalty_decay
        self.dominant_beat_window_s = dominant_beat_window_s

        # memory of recent note messages
        # item = {"time_s", "captor_id", "note", "beat"}
        self.note_history = []

        self.last_distinct_notes = []
        self.last_distinct_notes_max_len = 5

        #from the last distinct notes, if a scale covers >95% of them,
        #consider it as stable and start playing chords from that scale.
        self.scale_stability_count = 0
        self.last_scale_name = None
        self.min_stable_scale_updates = min_stable_scale_updates

        # state of the current local harmonic commitment
        self.current_scale = None
        self.current_chord = None
        self.current_chord_root = None
        self.current_chord_beat = None

        # used to increase exploration if local beat consensus stalls
        self.failed_beat_consensus_count = 0

        # pending evaluation of last beat change
        # {
        #   "time_s": ...,
        #   "old_beat": ...,
        #   "new_beat": ...,
        #   "balance_cost_before": ...
        # }
        self.pending_beat_evaluation = None

        # learned local penalties on target beats
        # beat -> float penalty
        self.bad_beat_targets = {b: 0.0 for b in range(self.nbr_beats)}

        # track prolonged local beat dominance
        self.last_uniform_neighbor_beat = None
        self.last_uniform_neighbor_beat_start_s = None

        self.forbidden_pair_ttl_s = forbidden_pair_ttl_s
        # key: (note_mod, beat) -> until_time_s
        self.forbidden_note_beat_pairs = {}

        self.current_chord_name = None

        # chord formation params
        self.chord_commitment_ttl_s = chord_commitment_ttl_s
        self.chord_create_probability = chord_create_probability
        self.chord_creation_score = chord_creation_score
        self.chord_beat_join_boost = chord_beat_join_boost
        self.max_chord_beat_occupancy = max_chord_beat_occupancy

        self.current_chord_until_s = 0.0

        # scale disambiguation params
        self.candidate_scale_threshold = candidate_scale_threshold
        self.strict_scale_threshold_for_chords = strict_scale_threshold_for_chords
        self.disambiguation_probability = disambiguation_probability

        self.current_scale_candidates = []
        self.current_scale_confidence = 0.0
        self.current_scale_is_ambiguous = False

    # ------------------------------------------------------------------
    # Parsing / memory
    # ------------------------------------------------------------------

    def parse_note_messages(self, note_msgs: list, time_s: float):
        """
        Decode note messages into local events.
        Payload encoding:
            payload = 128 + (beat * 24 + note)
        """
        parsed = []
        for msg in note_msgs:
            raw = int(msg.payload) - 128
            if raw < 0:
                continue

            beat = raw // 24
            note = raw % 24
            pitch = note % 12
            octave = note // 12

            if beat < 0 or beat >= self.nbr_beats:
                continue

            parsed.append({
                "time_s": time_s,
                "captor_id": getattr(msg, "captor_id", None),
                "note": pitch,
                "beat": beat,
                "octave": octave
            })
        return parsed

    def update_memory(self, note_msgs: list, time_s: float):
        """
        Add parsed note messages to memory with a lightweight dedup rule:
        if the same captor sends the same (note, beat) very recently, ignore duplicate.
        """
        parsed = self.parse_note_messages(note_msgs, time_s)

        for event in parsed:
            duplicate = False
            for old in reversed(self.note_history):
                if old["captor_id"] != event["captor_id"]:
                    continue

                if (event["time_s"] - old["time_s"]) > self.same_captor_merge_ttl_s:
                    break

                if old["note"] == event["note"] and old["beat"] == event["beat"] and old["octave"] == event["octave"]:
                    duplicate = True
                    break

            if not duplicate:
                self.note_history.append(event)

        self.cleanup_memory(time_s)

    def cleanup_memory(self, time_s: float):
        """
        Remove events older than max(note TTL, beat TTL).
        Since beat info is embedded in note history, one structure is enough.
        """
        max_ttl = max(self.note_memory_ttl_s, self.beat_memory_ttl_s)
        self.note_history = [
            e for e in self.note_history
            if (time_s - e["time_s"]) <= max_ttl
        ]

    def get_recent_note_events(self, time_s: float):
        return [
            e for e in self.note_history
            if (time_s - e["time_s"]) <= self.note_memory_ttl_s
        ]

    def get_recent_beat_events(self, time_s: float):
        return [
            e for e in self.note_history
            if (time_s - e["time_s"]) <= self.beat_memory_ttl_s
        ]

    # ------------------------------------------------------------------
    # Scale inference
    # ------------------------------------------------------------------

    def infer_local_scale(self, recent_events: list, current_note_event):
        """
        Choose a compatible local scale.
        Selection criteria:
        1. cover all recent distinct notes + current note if present
        2. maximize number of compatible chords
        3. prefer keeping current note if possible
        4. random tie-break
        """
        notes = {e["note"] % 12 for e in recent_events}
        current_note_mod = None

        if current_note_event is not None:
            current_note_mod = int(current_note_event[0]) % 12
            notes.add(current_note_mod)

        if not notes:
            # no info: choose random scale
            chosen = np.random.choice(Scales)
            return chosen

        compatible_scales = [
            scale for scale in Scales
            if notes.issubset(set(scale.notes))
        ]

        if not compatible_scales:
            # fallback: keep previous if possible
            if self.current_scale is not None:
                return self.current_scale
            return np.random.choice(Scales)

        scored = []
        for scale in compatible_scales:
            chords = self.get_valid_chords_for_scale(scale)
            chord_count = len(chords)

            keep_note_bonus = 0
            if current_note_mod is not None and current_note_mod in scale.notes:
                keep_note_bonus = 1

            scored.append((chord_count, keep_note_bonus, np.random.rand(), scale))

        scored.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
        return scored[0][3]
    
    def infer_local_scale_with_confidence(self, current_note_event):
        """
        Infer dominant scale and keep all plausible candidate scales.

        Candidate scales are scales that cover at least
        self.candidate_scale_threshold of the recently heard distinct notes.

        Returns:
            best_scale, best_coverage_ratio
        """
        notes = list(self.last_distinct_notes)

        current_note_mod = None
        if current_note_event is not None:
            current_note_mod = int(current_note_event[0]) % 12
            if current_note_mod not in notes:
                notes.append(current_note_mod)

        if not notes:
            chosen = np.random.choice(Scales)

            self.current_scale_candidates = [{
                "scale": chosen,
                "coverage": 0.0,
                "score": 0.0,
                "unique_notes": set(chosen.notes),
            }]
            self.current_scale_confidence = 0.0
            self.current_scale_is_ambiguous = False

            return chosen, 0.0

        scored = []

        for scale in Scales:
            scale_notes = set(scale.notes)

            covered = sum(1 for n in notes if n in scale_notes)
            coverage_ratio = covered / max(1, len(notes))

            chord_count = len(self.get_valid_chords_for_scale(scale))

            keep_note_bonus = 0.0
            if current_note_mod is not None and current_note_mod in scale_notes:
                keep_note_bonus = 0.03

            stability_bonus = 0.0
            if (
                self.current_scale is not None
                and getattr(self.current_scale, "name", None) == getattr(scale, "name", None)
            ):
                stability_bonus = 0.05

            # Main criterion remains coverage.
            # Chord count and stability are only tie-breakers.
            score = coverage_ratio + keep_note_bonus + stability_bonus + 0.005 * chord_count

            scored.append({
                "scale": scale,
                "coverage": coverage_ratio,
                "score": score,
            })

        scored.sort(key=lambda x: (x["coverage"], x["score"], np.random.rand()), reverse=True)

        best = scored[0]
        best_scale = best["scale"]
        best_coverage = best["coverage"]

        candidates = [
            item for item in scored
            if item["coverage"] >= self.candidate_scale_threshold
        ]

        # Compute notes that are unique to each candidate compared with the other candidates.
        for item in candidates:
            scale_notes = set(item["scale"].notes)
            other_notes = set()

            for other in candidates:
                if other is item:
                    continue
                other_notes.update(set(other["scale"].notes))

            item["unique_notes"] = scale_notes - other_notes

        self.current_scale_candidates = candidates
        self.current_scale_confidence = best_coverage
        self.current_scale_is_ambiguous = len(candidates) > 1

        return best_scale, best_coverage
    
    def update_scale_stability(self, chosen_scale):
        scale_name = getattr(chosen_scale, "name", None)

        if self.last_scale_name == scale_name:
            self.scale_stability_count += 1
        else:
            self.last_scale_name = scale_name
            self.scale_stability_count = 1

    def is_scale_stable_enough(self, confidence: float) -> bool:
        """
        Scale is stable enough for chord formation only if all heard notes
        fit the dominant scale.
        """
        return (
            confidence >= self.strict_scale_threshold_for_chords
            and self.scale_stability_count >= self.min_stable_scale_updates
        )
    
    def choose_note_from_scale_only(self, scale, current_note_event):
        """
        Pre-harmonic mode:
        choose a note in the dominant inferred scale,
        prefer keeping current note if already valid.
        """
        if scale is None:
            return None

        if current_note_event is not None:
            current_note = int(current_note_event[0]) % 12
            if current_note in scale.notes:
                return (current_note, self.beat_duration_s, self.fallback_volume)

        note = int(np.random.choice(scale.notes)) % 12
        return (note, self.beat_duration_s, self.fallback_volume)
    
    def choose_disambiguation_note(self, best_scale, current_note_event):
        """
        If several scales are plausible, play a note that belongs only
        to the best scale among the candidate scales.

        This helps neighbors reject competing scales.
        """
        if best_scale is None:
            return None

        if len(self.current_scale_candidates) <= 1:
            return None

        best_item = None
        best_name = getattr(best_scale, "name", None)

        for item in self.current_scale_candidates:
            if getattr(item["scale"], "name", None) == best_name:
                best_item = item
                break

        if best_item is None:
            return None

        unique_notes = list(best_item.get("unique_notes", []))

        if not unique_notes:
            return None

        current_note_mod = None
        if current_note_event is not None:
            current_note_mod = int(current_note_event[0]) % 12

        # If current note is already discriminating, keep it.
        if current_note_mod in unique_notes:
            return (current_note_mod, self.beat_duration_s, self.fallback_volume)

        chosen_note = int(np.random.choice(unique_notes)) % 12
        return (chosen_note, self.beat_duration_s, self.fallback_volume)
    # ------------------------------------------------------------------
    # Chord generation / detection
    # ------------------------------------------------------------------

    def get_valid_chords_for_scale(self, scale):
        """
        Return all allowed chords fully contained in the given scale.
        """
        scale_notes = set(scale.notes)
        chords = []

        for root in range(12):
            for chord_name, intervals in CHORD_PATTERNS.items():
                chord_notes = {(root + interval) % 12 for interval in intervals}

                if chord_notes.issubset(scale_notes):
                    chords.append({
                        "root": root % 12,
                        "name": chord_name,
                        "notes": chord_notes,
                        "intervals": intervals
                    })

        return chords

    def group_events_by_beat(self, recent_events: list):
        grouped = {beat: [] for beat in range(self.nbr_beats)}
        for e in recent_events:
            grouped[e["beat"]].append(e)
        return grouped

    def is_current_chord_still_valid(self, recent_events: list, current_note_event, scale, time_s: float):
        if self.current_chord is None or self.current_chord_beat is None or scale is None:
            return False

        if time_s > self.current_chord_until_s:
            return False

        if current_note_event is None:
            return False

        current_note = int(current_note_event[0]) % 12

        if current_note not in self.current_chord:
            return False

        if not self.current_chord.issubset(set(scale.notes)):
            return False

        beat_events = [
            e for e in recent_events
            if e["beat"] == self.current_chord_beat
        ]

        if not beat_events:
            return False

        heard_notes = {e["note"] for e in beat_events}

        # On garde l'accord si les voisins entendus sur ce beat restent compatibles.
        if not heard_notes.issubset(self.current_chord):
            return False

        return True

    def find_chord_candidates(self, recent_events: list, scale, current_beat: int, scale_is_stable: bool):
        """
        Build candidate chords from recent events.

        Three candidate types:
        1. strong same-beat chord candidates
        2. weaker cross-beat harmonic candidates
        3. proposed chords when the scale is stable, even if no chord exists yet
        """
        chords = self.get_valid_chords_for_scale(scale)
        grouped = self.group_events_by_beat(recent_events)
        beat_usage = self.compute_local_beat_usage(recent_events)

        candidates = []

        # 1) same-beat candidates
        for beat, events in grouped.items():
            notes_on_beat = {e["note"] for e in events}
            count_on_beat = len(events)

            for chord in chords:
                present = notes_on_beat.intersection(chord["notes"])
                missing = chord["notes"] - notes_on_beat

                if len(present) == 0:
                    continue

                score = 0.0

                if len(present) >= 2:
                    score += 10.0
                else:
                    score += 3.0

                if beat == current_beat:
                    score += 5.0

                score += 0.3 * count_on_beat

                candidates.append({
                    "chord_root": chord["root"],
                    "chord_notes": chord["notes"],
                    "chord_name": chord["name"],
                    "beat": beat,
                    "present": present,
                    "missing": missing,
                    "score": score,
                    "same_beat_support": True,
                    "is_proposed": False,
                })

        # 2) cross-beat candidates
        all_notes = {e["note"] for e in recent_events}
        beat_set = {e["beat"] for e in recent_events}

        for chord in chords:
            present = all_notes.intersection(chord["notes"])
            missing = chord["notes"] - all_notes

            if len(present) >= 2 and len(beat_set) > 1:
                candidates.append({
                    "chord_root": chord["root"],
                    "chord_notes": chord["notes"],
                    "chord_name": chord["name"],
                    "beat": current_beat,
                    "present": present,
                    "missing": missing,
                    "score": 3.0 + (2.0 if current_beat in beat_set else 0.0),
                    "same_beat_support": False,
                    "is_proposed": False,
                })

        # 3) proposed chord candidates
        if scale_is_stable and np.random.rand() < self.chord_create_probability:
            target_beat = self.choose_best_unoccupied_beat(beat_usage)

            if target_beat is None:
                target_beat = current_beat

            # prefer a chord containing the current note if possible
            current_note_mod = None
            # current_note_event is not available here, so this preference is handled later

            for chord in chords:
                candidates.append({
                    "chord_root": chord["root"],
                    "chord_notes": chord["notes"],
                    "chord_name": chord["name"],
                    "beat": target_beat,
                    "present": set(),
                    "missing": set(chord["notes"]),
                    "score": self.chord_creation_score + np.random.rand() * 0.1,
                    "same_beat_support": False,
                    "is_proposed": True,
                })

        if not candidates:
            return []

        unique = {}
        for c in candidates:
            key = (tuple(sorted(c["chord_notes"])), c["beat"])
            if key not in unique or c["score"] > unique[key]["score"]:
                unique[key] = c

        result = list(unique.values())
        result.sort(key=lambda c: c["score"], reverse=True)
        return result

    def get_recent_chord_events(self, time_s: float):
        return [
            e for e in self.note_history
            if (time_s - e["time_s"]) <= self.chord_memory_ttl_s
        ]
    # ------------------------------------------------------------------
    # Note choice
    # ------------------------------------------------------------------

    def choose_note_for_candidate(self, candidate, current_note_event, scale, time_s: float):
        """
        Priority:
        1. keep current note if it is already in chord and scale and pair is not forbidden
        2. if exactly one note is missing, choose it if pair is not forbidden
        3. if chord already complete, do NOT double -> keep current if valid,
        else choose a note in scale (outside the chord if possible), otherwise None
        """
        chord = candidate["chord_notes"]
        missing = candidate["missing"]
        target_beat = candidate["beat"]

        current_note_mod = None
        if current_note_event is not None:
            current_note_mod = int(current_note_event[0]) % 12

        # keep same note if possible
        if (
            current_note_mod is not None
            and current_note_mod in chord
            and current_note_mod in scale.notes
            and not self.is_note_beat_pair_forbidden(current_note_mod, target_beat, time_s)
        ):
            return current_note_mod

        # best case: choose one of the missing chord notes
        # even if the chord is far from complete
        if len(missing) >= 1:
            missing_allowed = [
                int(n) % 12
                for n in missing
                if not self.is_note_beat_pair_forbidden(n, target_beat, time_s)
            ]

            if missing_allowed:
                return int(np.random.choice(missing_allowed))

        # if chord is already complete, do not double
        if len(missing) == 0:
            if (
                current_note_mod is not None
                and current_note_mod in scale.notes
                and not self.is_note_beat_pair_forbidden(current_note_mod, target_beat, time_s)
            ):
                return current_note_mod

            # choose another note from the same scale, preferably outside the chord
            outside = [n for n in scale.notes if n not in chord]
            for note in outside:
                note = int(note) % 12
                if not self.is_note_beat_pair_forbidden(note, target_beat, time_s):
                    return note

            for note in scale.notes:
                note = int(note) % 12
                if not self.is_note_beat_pair_forbidden(note, target_beat, time_s):
                    return note

            return None

        # weak candidate: keep current note only if allowed
        if (
            current_note_mod is not None
            and current_note_mod in scale.notes
            and not self.is_note_beat_pair_forbidden(current_note_mod, target_beat, time_s)
        ):
            return current_note_mod

        # otherwise search another allowed scale note
        for note in scale.notes:
            note = int(note) % 12
            if not self.is_note_beat_pair_forbidden(note, target_beat, time_s):
                return note

        return None
    
    def update_distinct_note_history(self, recent_events: list):
        """
        Keep a short ordered memory of last distinct heard notes (mod 12),
        similar to the previous scale-convergence logic.
        """
        for e in recent_events:
            note = int(e["note"]) % 12
            if not self.last_distinct_notes or self.last_distinct_notes[-1] != note:
                if note in self.last_distinct_notes:
                    self.last_distinct_notes.remove(note)
                self.last_distinct_notes.append(note)

                if len(self.last_distinct_notes) > self.last_distinct_notes_max_len:
                    self.last_distinct_notes.pop(0)

    # ------------------------------------------------------------------
    # forbiden pairs management
    # ------------------------------------------------------------------

    def cleanup_forbidden_pairs(self, time_s: float):
        expired = [
            pair for pair, until in self.forbidden_note_beat_pairs.items()
            if until <= time_s
        ]
        for pair in expired:
            del self.forbidden_note_beat_pairs[pair]

    def ban_note_beat_pair(self, note: int, beat: int, time_s: float):
        self.forbidden_note_beat_pairs[(note % 12, beat)] = time_s + self.forbidden_pair_ttl_s

    def is_note_beat_pair_forbidden(self, note: int, beat: int, time_s: float) -> bool:
        until = self.forbidden_note_beat_pairs.get((note % 12, beat), 0.0)
        return until > time_s
    
    def detect_same_note_same_beat_collision(self, recent_events: list, current_note_event, current_beat: int):
        if current_note_event is None:
            return False

        current_note = int(current_note_event[0]) % 12

        for e in recent_events:
            if e["note"] == current_note and e["beat"] == current_beat:
                return True

        return False
    
    def choose_forbidden_pair_alternative(self, scale, current_note_event, current_beat: int, beat_events: list, time_s: float):
        """
        When (current_note, current_beat) becomes forbidden because a neighbor plays
        the exact same note on the exact same beat, choose an alternative:
        - prefer changing beat first
        - then prefer keeping note if possible
        - otherwise choose another note in same scale
        - avoid forbidden (note, beat) pairs
        """
        if scale is None:
            return None, current_beat, "no_scale_for_forbidden_pair"

        beat_usage = self.compute_local_beat_usage(beat_events)

        current_note = None
        if current_note_event is not None:
            current_note = int(current_note_event[0]) % 12

        # 1) prefer another beat with same note
        if current_note is not None:
            candidate_beats = [b for b in range(self.nbr_beats) if b != current_beat]
            candidate_beats.sort(key=lambda b: beat_usage.get(b, 0))

            for beat in candidate_beats:
                if not self.is_note_beat_pair_forbidden(current_note, beat, time_s):
                    return (current_note, self.beat_duration_s, self.fallback_volume), beat, "change_beat_keep_note"

        # 2) otherwise keep beat and change note within scale
        if current_note is not None:
            for note in scale.notes:
                note = int(note) % 12
                if note == current_note:
                    continue
                if not self.is_note_beat_pair_forbidden(note, current_beat, time_s):
                    return (note, self.beat_duration_s, self.fallback_volume), current_beat, "keep_beat_change_note"

        # 3) otherwise change both note and beat
        candidate_beats = list(range(self.nbr_beats))
        candidate_beats.sort(key=lambda b: beat_usage.get(b, 0))

        for beat in candidate_beats:
            for note in scale.notes:
                note = int(note) % 12
                if not self.is_note_beat_pair_forbidden(note, beat, time_s):
                    return (note, self.beat_duration_s, self.fallback_volume), beat, "change_note_and_beat"

        return None, current_beat, "no_alternative_forbidden_pair"
    # ------------------------------------------------------------------
    # Beat choice
    # ------------------------------------------------------------------

    def beat_balance_cost(self, beat_usage: dict) -> float:
        """
        Cost of local beat imbalance.
        0 = perfectly balanced
        higher = more unbalanced
        """
        counts = np.array([beat_usage[b] for b in range(self.nbr_beats)], dtype=float)
        total = np.sum(counts)
        if total <= 0:
            return 0.0

        p = counts / total
        ideal = np.ones(self.nbr_beats, dtype=float) / self.nbr_beats
        return float(np.sum((p - ideal) ** 2))

    def decay_bad_beat_penalties(self):
        for b in range(self.nbr_beats):
            self.bad_beat_targets[b] *= self.bad_beat_penalty_decay
            if self.bad_beat_targets[b] < 1e-3:
                self.bad_beat_targets[b] = 0.0

    def compute_local_beat_usage(self, beat_events: list):
        """
        Count recent beat usage locally.
        """
        usage = {b: 0 for b in range(self.nbr_beats)}
        for e in beat_events:
            usage[e["beat"]] += 1
        return usage
    
    def start_beat_change_evaluation(self, old_beat: int, new_beat: int, beat_events: list, time_s: float):
        if old_beat == new_beat:
            return

        beat_usage = self.compute_local_beat_usage(beat_events)
        balance_cost_before = self.beat_balance_cost(beat_usage)

        self.pending_beat_evaluation = {
            "time_s": time_s,
            "old_beat": old_beat,
            "new_beat": new_beat,
            "balance_cost_before": balance_cost_before
        }

    def update_pending_beat_evaluation(self, beat_events: list, time_s: float):
        if self.pending_beat_evaluation is None:
            return

        age = time_s - self.pending_beat_evaluation["time_s"]
        if age < self.beat_change_eval_delay_s:
            return

        beat_usage_after = self.compute_local_beat_usage(beat_events)
        balance_cost_after = self.beat_balance_cost(beat_usage_after)
        balance_cost_before = self.pending_beat_evaluation["balance_cost_before"]
        new_beat = self.pending_beat_evaluation["new_beat"]

        # if local balance got worse, penalize that target beat
        if balance_cost_after > balance_cost_before + 1e-9:
            delta = balance_cost_after - balance_cost_before
            self.bad_beat_targets[new_beat] += 1.0 + 5.0 * delta
            logger.log(
                "DEBUG",
                f"HarmonyAlgo learned bad beat target: beat={new_beat}, "
                f"before={balance_cost_before:.4f}, after={balance_cost_after:.4f}, "
                f"penalty={self.bad_beat_targets[new_beat]:.3f}"
            )
        else:
            # if change improved or preserved balance, slightly forgive that beat
            self.bad_beat_targets[new_beat] *= 0.8

        self.pending_beat_evaluation = None

    def update_dominant_beat_tracking(self, beat_events: list, time_s: float):
        """
        Track if all recent neighbor beat events are concentrated on one single beat.
        """
        if not beat_events:
            self.last_uniform_neighbor_beat = None
            self.last_uniform_neighbor_beat_start_s = None
            return

        used_beats = {e["beat"] for e in beat_events}

        if len(used_beats) == 1:
            only_beat = next(iter(used_beats))
            if self.last_uniform_neighbor_beat == only_beat:
                if self.last_uniform_neighbor_beat_start_s is None:
                    self.last_uniform_neighbor_beat_start_s = time_s
            else:
                self.last_uniform_neighbor_beat = only_beat
                self.last_uniform_neighbor_beat_start_s = time_s
        else:
            self.last_uniform_neighbor_beat = None
            self.last_uniform_neighbor_beat_start_s = None
    
    def get_dominant_beat_duration(self, time_s: float) -> float:
        if self.last_uniform_neighbor_beat is None or self.last_uniform_neighbor_beat_start_s is None:
            return 0.0
        return max(0.0, time_s - self.last_uniform_neighbor_beat_start_s)

    def choose_best_unoccupied_beat(self, beat_usage: dict, forbidden_beats: set | None = None):
        if forbidden_beats is None:
            forbidden_beats = set()

        candidates = [b for b in range(self.nbr_beats) if b not in forbidden_beats]
        if not candidates:
            return None

        min_use = min(beat_usage[b] for b in candidates)
        best = [b for b in candidates if beat_usage[b] == min_use]
        return int(np.random.choice(best))

    def choose_beat_for_candidate(self, candidate, current_beat: int, beat_events: list, time_s: float):
        """
        Beat decision with:
        - local harmonic evidence
        - local saturation penalty
        - learned penalty from past bad beat changes
        - strong escape when one beat dominates too long
        """
        target_beat = candidate["beat"]
        same_beat_support = candidate["same_beat_support"]
        present_count = len(candidate["present"])

        beat_usage = self.compute_local_beat_usage(beat_events)
        current_usage = beat_usage.get(current_beat, 0)
        target_usage = beat_usage.get(target_beat, 0)

        total_local = sum(beat_usage.values())
        ideal_usage = max(1, int(np.ceil(total_local / self.nbr_beats))) if total_local > 0 else 1

        dominant_duration = self.get_dominant_beat_duration(time_s)
        dominant_beat = self.last_uniform_neighbor_beat

        # 1) if candidate already matches current beat, keep it
        if target_beat == current_beat:
            self.failed_beat_consensus_count = 0
            return current_beat, "keep_current_beat"

        # 2) base probability from harmonic evidence
        if same_beat_support and present_count >= 2:
            p_change = 0.75
        elif present_count >= 2:
            p_change = 0.30
        elif present_count == 1:
            p_change = 0.30
        else:
            p_change = 0.15

        # Proposed chord: allow some active beat joining
        if candidate.get("is_proposed", False):
            p_change = max(p_change, 0.25)

        # 3) strong penalty if target beat is already more crowded
        if target_usage > current_usage:
            diff = target_usage - current_usage
            if candidate.get("is_proposed", False) or present_count >= 1:
                p_change *= (0.70 ** diff)  # less harsh penalty for proposed chords
            p_change *= (0.35 ** diff)

        # 4) strong penalty if target beat exceeds ideal local occupancy
        if target_usage > ideal_usage:
            overflow = target_usage - ideal_usage
            if candidate.get("is_proposed", False) or present_count >= 1:
                p_change *= (0.7 ** overflow)  # less harsh penalty for proposed chords
            p_change *= (0.25 ** overflow)

        # 5)if target beat is dominant locally
        max_usage = max(beat_usage.values()) if beat_usage else 0
        if target_usage == max_usage and target_usage >= ideal_usage + 1:
            p_change *= 0.50

        # In chord formation mode, slightly boost joining a harmonic beat
        if present_count >= 1 and target_usage < self.max_chord_beat_occupancy:
            p_change *= self.chord_beat_join_boost

        # If this is a proposed chord, allow joining the proposed beat
        # but still avoid overcrowding
        if candidate.get("is_proposed", False) and target_usage < self.max_chord_beat_occupancy:
            p_change *= self.chord_beat_join_boost

        # 6) if chord already sufficiently represented on target beat, don't reinforce it much more
        if same_beat_support and present_count >= 2 and target_usage >= 2:
            p_change *= 0.80

        # 7) learned historical penalty on bad target beats
        learned_penalty = self.bad_beat_targets.get(target_beat, 0.0)
        p_change *= 1.0 / (1.0 + learned_penalty)

        # 8) if current beat is relatively underused, favor staying
        if current_usage < ideal_usage:
            p_change *= 0.70

        # 9) prolonged uniform dominance: strong pressure to escape
        # If all neighbors have used the same beat for too long,
        # do not keep feeding that beat; strongly prefer an alternative underused beat.
        if dominant_beat is not None and dominant_duration >= self.dominant_beat_window_s:
            if target_beat == dominant_beat:
                p_change *= 0.05  # almost refuse joining the dominant beat

            alternative_beats = [b for b in range(self.nbr_beats) if b != dominant_beat]
            if alternative_beats:
                min_use = min(beat_usage[b] for b in alternative_beats)
                best_alts = [b for b in alternative_beats if beat_usage[b] == min_use]
                forced_escape_beat = int(np.random.choice(best_alts))

                p_escape = min(0.90, 0.35 + 0.05 * (dominant_duration - self.dominant_beat_window_s))
                if np.random.rand() < p_escape:
                    self.failed_beat_consensus_count = 0
                    return forced_escape_beat, "escape_dominant_beat"

        # 10) exploration toward unused beat if local consensus stalls
        exploration_bonus = min(0.50, 0.10 * self.failed_beat_consensus_count)
        unused_beats = {b for b, u in beat_usage.items() if u == 0}

        if self.failed_beat_consensus_count >= 3 and unused_beats:
            exploratory_beat = self.choose_best_unoccupied_beat(beat_usage)
            if exploratory_beat is not None:
                p_explore = min(0.70, 0.25 + exploration_bonus)
                if np.random.rand() < p_explore:
                    self.failed_beat_consensus_count = 0
                    return exploratory_beat, "explore_unused_beat"

        p_change = min(0.95, max(0.0, p_change))

        # 11) final probabilistic decision
        if np.random.rand() < p_change:
            self.failed_beat_consensus_count = 0
            return target_beat, "change_to_candidate_beat"

        self.failed_beat_consensus_count += 1
        return current_beat, "stay_current_beat_probabilistic"

    # ------------------------------------------------------------------
    # Main update
    # ------------------------------------------------------------------

    def update(self, note_msgs: list, current_note_event, current_beat: int, time_s: float):
        """
        Returns:
            note_event_or_none,
            chosen_beat,
            debug_info
        """
        self.update_memory(note_msgs, time_s)

        recent_events = self.get_recent_note_events(time_s)
        chord_events = self.get_recent_chord_events(time_s)
        beat_events = self.get_recent_beat_events(time_s)
        self.cleanup_forbidden_pairs(time_s)
        self.update_distinct_note_history(recent_events)

        # new adaptive memory updates
        self.decay_bad_beat_penalties()
        self.update_pending_beat_evaluation(beat_events, time_s)
        self.update_dominant_beat_tracking(beat_events, time_s)

        debug = {
            "used_fallback": False,
            "reason": "",
            "scale": None,
            "chord_root": None,
            "chord_notes": None,
            "chord_name": None,
            "beat": current_beat,
            "recent_neighbors": len(recent_events),
            "dominant_beat": self.last_uniform_neighbor_beat,
            "dominant_duration": self.get_dominant_beat_duration(time_s),
            "bad_beat_targets": dict(self.bad_beat_targets),
        }

        # no recent neighbors -> keep current state, no aggressive recalculation
        if not recent_events:
            debug["reason"] = "no_recent_neighbors_keep_state"
            return current_note_event, current_beat, debug

        # infer local scale
        scale, scale_confidence = self.infer_local_scale_with_confidence(current_note_event)
        self.current_scale = scale
        self.update_scale_stability(scale)

        debug["scale"] = getattr(scale, "name", None)
        debug["scale_confidence"] = scale_confidence
        debug["scale_stability_count"] = self.scale_stability_count
        debug["scale_candidates"] = [{
                "name": getattr(item["scale"], "name", None),
                "coverage": item["coverage"],
                "unique_notes": sorted(list(item.get("unique_notes", []))),}
            for item in self.current_scale_candidates]
        
        debug["scale_is_ambiguous"] = self.current_scale_is_ambiguous

        scale_ready_for_chords = self.is_scale_stable_enough(scale_confidence)

        if not scale_ready_for_chords:
            # If several candidate scales are plausible, actively play
            # a note that only belongs to the dominant scale.
            if (
                self.current_scale_is_ambiguous
                and np.random.rand() < self.disambiguation_probability
            ):
                disambiguation_note_event = self.choose_disambiguation_note(
                    best_scale=scale,
                    current_note_event=current_note_event
                )

                if disambiguation_note_event is not None:
                    debug["reason"] = "scale_disambiguation_note"
                    debug["used_fallback"] = False
                    return disambiguation_note_event, current_beat, debug

            # Otherwise continue normal scale alignment.
            tonal_note_event = self.choose_note_from_scale_only(scale, current_note_event)

            debug["reason"] = "scale_alignment_only"
            debug["used_fallback"] = False

            if tonal_note_event is not None:
                return tonal_note_event, current_beat, debug

            debug["used_fallback"] = True
            debug["reason"] = "scale_alignment_failed"
            return None, current_beat, debug
        
        apply_collision_rule = True

        if self.current_chord is not None:
            current_chord_events = [
                e for e in chord_events
                if e["beat"] == current_beat and e["note"] in self.current_chord
            ]
            distinct_chord_notes = {e["note"] for e in current_chord_events}

            if len(distinct_chord_notes) < 2:
                apply_collision_rule = False

            # During chord bootstrap, tolerate some duplicates.
            # Once at least 2 chord notes exist locally, activate collision avoidance.
            if len(distinct_chord_notes) < 2:
                apply_collision_rule = False

        if apply_collision_rule and self.detect_same_note_same_beat_collision(chord_events, current_note_event, current_beat):
            current_note_mod = int(current_note_event[0]) % 12 if current_note_event is not None else None

            if current_note_mod is not None:
                self.ban_note_beat_pair(current_note_mod, current_beat, time_s)
                logger.log(
                    "DEBUG",
                    f"HarmonyAlgo banned pair due to collision: note={current_note_mod}, beat={current_beat}"
                )

                alt_note_event, alt_beat, alt_reason = self.choose_forbidden_pair_alternative(
                    scale=scale,
                    current_note_event=current_note_event,
                    current_beat=current_beat,
                    beat_events=beat_events,
                    time_s=time_s
                )

                if alt_note_event is not None:
                    # also evaluate beat change if any
                    if alt_beat != current_beat:
                        self.start_beat_change_evaluation(
                            old_beat=current_beat,
                            new_beat=alt_beat,
                            beat_events=beat_events,
                            time_s=time_s
                        )

                    # reset current chord because exact duplication broke local usefulness
                    self.current_chord = None
                    self.current_chord_root = None
                    self.current_chord_beat = None
                    self.current_chord_name = None
                    self.current_chord_until_s = 0.0

                    debug["reason"] = f"forbidden_pair_escape::{alt_reason}"
                    debug["beat"] = alt_beat
                    return alt_note_event, alt_beat, debug

        # maintain current chord if still valid
        if self.is_current_chord_still_valid(chord_events, current_note_event, scale, time_s):
            debug["reason"] = "maintain_current_chord"
            debug["chord_root"] = self.current_chord_root
            debug["chord_notes"] = sorted(list(self.current_chord))
            debug["beat"] = self.current_chord_beat
            return current_note_event, self.current_chord_beat, debug

        # find candidate chords
        candidates = self.find_chord_candidates(chord_events, scale, current_beat, scale_ready_for_chords)
        if not candidates:
            self.current_chord = None
            self.current_chord_root = None
            self.current_chord_beat = None
            self.current_chord_name = None
            self.current_chord_until_s = 0.0
            debug["used_fallback"] = True
            debug["reason"] = "no_chord_candidate"
            return None, current_beat, debug

        best_candidate = candidates[0]
        chosen_note = self.choose_note_for_candidate(best_candidate, current_note_event, scale, time_s)

        if chosen_note is None:
            self.current_chord = None
            self.current_chord_root = None
            self.current_chord_beat = None
            self.current_chord_name = None
            self.current_chord_until_s = 0.0
            debug["used_fallback"] = True
            debug["reason"] = "candidate_note_selection_failed"
            return None, current_beat, debug

        chosen_beat, beat_reason = self.choose_beat_for_candidate(
            best_candidate,
            current_beat,
            beat_events,
            time_s
        )

        # start before/after evaluation only if beat really changed
        if chosen_beat != current_beat:
            self.start_beat_change_evaluation(
                old_beat=current_beat,
                new_beat=chosen_beat,
                beat_events=beat_events,
                time_s=time_s
            )

        # commit chord state
        self.current_chord = set(best_candidate["chord_notes"])
        self.current_chord_root = best_candidate["chord_root"]
        self.current_chord_name = best_candidate["chord_name"]
        self.current_chord_beat = chosen_beat
        self.current_chord_until_s = time_s + self.chord_commitment_ttl_s

        note_event = (int(chosen_note), self.beat_duration_s, self.fallback_volume)

        debug["reason"] = f"new_or_updated_chord::{beat_reason}"
        debug["chord_root"] = self.current_chord_root
        debug["chord_notes"] = sorted(list(self.current_chord))
        debug["chord_name"] = self.current_chord_name
        debug["beat"] = chosen_beat
        debug["current_chord_until_s"] = self.current_chord_until_s
        debug["chord_commitment_remaining_s"] = max(0.0, self.current_chord_until_s - time_s)

        return note_event, chosen_beat, debug