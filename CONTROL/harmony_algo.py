import numpy as np
from TOOLS.logger import logger
from TOOLS.scales import Scales


class HarmonyAlgo:
    """
    Local harmony algorithm for decentralized robots.

    Goals:
    - keep local/global convergence toward a common scale
    - allow punctual emergence of local major triads
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

    MAJOR_TRIAD = (0, 4, 7)

    def __init__(
        self,
        nbr_beats: int,
        beat_duration_s: float,
        note_memory_ttl_s: float = 5.0,
        beat_memory_ttl_s: float = 5.0,
        same_captor_merge_ttl_s: float = 1.0,
        fallback_volume: float = 0.6,
        beat_change_eval_delay_s: float = 2.0,
        bad_beat_penalty_decay: float = 0.98,
        dominant_beat_window_s: float = 10.0,
        forbidden_pair_ttl_s: float = 10.0,
    ):
        self.nbr_beats = nbr_beats
        self.beat_duration_s = beat_duration_s

        self.note_memory_ttl_s = note_memory_ttl_s
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
        self.scale_confidence_threshold = 0.95 
        self.scale_stability_count = 0
        self.last_scale_name = None
        self.min_stable_scale_updates = 2

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

    # ------------------------------------------------------------------
    # Parsing / memory
    # ------------------------------------------------------------------

    def parse_note_messages(self, note_msgs: list, time_s: float):
        """
        Decode note messages into local events.
        Payload encoding:
            payload = 128 + (beat * 12 + note)
        """
        parsed = []
        for msg in note_msgs:
            raw = int(msg.payload) - 128
            if raw < 0:
                continue

            beat = raw // 12
            note = raw % 12

            if beat < 0 or beat >= self.nbr_beats:
                continue

            parsed.append({
                "time_s": time_s,
                "captor_id": getattr(msg, "captor_id", None),
                "note": note,
                "beat": beat
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

                if old["note"] == event["note"] and old["beat"] == event["beat"]:
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
        2. maximize number of compatible major triads
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
            triads = self.get_valid_major_triads_for_scale(scale)
            triad_count = len(triads)

            keep_note_bonus = 0
            if current_note_mod is not None and current_note_mod in scale.notes:
                keep_note_bonus = 1

            scored.append((triad_count, keep_note_bonus, np.random.rand(), scale))

        scored.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
        return scored[0][3]
    
    def infer_local_scale_with_confidence(self, current_note_event):
        """
        Infer a dominant local scale from the last distinct heard notes.
        Returns:
            chosen_scale, confidence in [0,1]
        """
        notes = set(self.last_distinct_notes)

        current_note_mod = None
        if current_note_event is not None:
            current_note_mod = int(current_note_event[0]) % 12
            notes.add(current_note_mod)

        if not notes:
            chosen = np.random.choice(Scales)
            return chosen, 0.0

        scored = []
        for scale in Scales:
            scale_notes = set(scale.notes)

            covered = sum(1 for n in self.last_distinct_notes if n in scale_notes)
            coverage_ratio = covered / max(1, len(self.last_distinct_notes))

            triad_count = len(self.get_valid_major_triads_for_scale(scale))

            keep_note_bonus = 0.0
            if current_note_mod is not None and current_note_mod in scale_notes:
                keep_note_bonus = 0.05

            stability_bonus = 0.0
            if self.current_scale is not None and getattr(self.current_scale, "name", None) == getattr(scale, "name", None):
                stability_bonus = 0.05

            score = coverage_ratio + 0.02 * triad_count + keep_note_bonus + stability_bonus
            scored.append((score, coverage_ratio, np.random.rand(), scale))

        scored.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
        best_score, best_coverage_ratio, _, best_scale = scored[0]

        confidence = best_coverage_ratio
        return best_scale, confidence
    
    def update_scale_stability(self, chosen_scale):
        scale_name = getattr(chosen_scale, "name", None)

        if self.last_scale_name == scale_name:
            self.scale_stability_count += 1
        else:
            self.last_scale_name = scale_name
            self.scale_stability_count = 1

    def is_scale_stable_enough(self, confidence: float) -> bool:
        return (
            confidence >= self.scale_confidence_threshold
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
    # ------------------------------------------------------------------
    # Chord generation / detection
    # ------------------------------------------------------------------

    def get_valid_major_triads_for_scale(self, scale):
        """
        Return all major triads fully contained in the given scale.
        """
        scale_notes = set(scale.notes)
        triads = []

        for root in range(12):
            chord = {
                root % 12,
                (root + self.MAJOR_TRIAD[1]) % 12,
                (root + self.MAJOR_TRIAD[2]) % 12
            }
            if chord.issubset(scale_notes):
                triads.append({
                    "root": root % 12,
                    "notes": chord
                })

        return triads

    def group_events_by_beat(self, recent_events: list):
        grouped = {beat: [] for beat in range(self.nbr_beats)}
        for e in recent_events:
            grouped[e["beat"]].append(e)
        return grouped

    def is_current_chord_still_valid(self, recent_events: list, current_note_event, scale):
        """
        Keep chord while:
        - neighbors still exist
        - current note still belongs to chord
        - chord remains in local scale
        - recent neighbor notes remain compatible with that chord
        """
        if self.current_chord is None or self.current_chord_beat is None or scale is None:
            return False

        if not recent_events:
            return False

        if current_note_event is None:
            return False

        current_note = int(current_note_event[0]) % 12
        if current_note not in self.current_chord:
            return False

        if not self.current_chord.issubset(set(scale.notes)):
            return False

        beat_events = [e for e in recent_events if e["beat"] == self.current_chord_beat]
        if not beat_events:
            return False

        heard_notes = {e["note"] for e in beat_events}
        if not heard_notes.issubset(self.current_chord):
            return False

        return True

    def find_chord_candidates(self, recent_events: list, scale, current_beat: int):
        """
        Build candidate triads from recent events.

        Candidate score is stronger when:
        - two distinct notes of a triad are heard on the same beat
        - current beat already matches
        - more neighbor notes support the chord
        """
        triads = self.get_valid_major_triads_for_scale(scale)
        grouped = self.group_events_by_beat(recent_events)

        candidates = []

        # same-beat candidates
        for beat, events in grouped.items():
            notes_on_beat = {e["note"] for e in events}
            count_on_beat = len(events)

            for triad in triads:
                present = notes_on_beat.intersection(triad["notes"])
                missing = triad["notes"] - notes_on_beat

                if len(present) == 0:
                    continue

                score = 0.0

                # strong signal if 2 different notes already on same beat
                if len(present) >= 2:
                    score += 10.0
                else:
                    score += 2.0

                # prefer staying on current beat
                if beat == current_beat:
                    score += 5.0

                # slightly reward chords supported by more recent activity
                score += 0.3 * count_on_beat

                candidates.append({
                    "triad_root": triad["root"],
                    "triad_notes": triad["notes"],
                    "beat": beat,
                    "present": present,
                    "missing": missing,
                    "score": score,
                    "same_beat_support": True
                })

        # cross-beat weaker candidates
        all_notes = {e["note"] for e in recent_events}
        beat_set = {e["beat"] for e in recent_events}

        for triad in triads:
            present = all_notes.intersection(triad["notes"])
            missing = triad["notes"] - all_notes

            if len(present) == 0:
                continue

            # if already represented on same-beat candidates, keep cross-beat only as weaker alternative
            if len(present) >= 2 and len(beat_set) > 1:
                candidates.append({
                    "triad_root": triad["root"],
                    "triad_notes": triad["notes"],
                    "beat": current_beat,
                    "present": present,
                    "missing": missing,
                    "score": 3.0 + (2.0 if current_beat in beat_set else 0.0),
                    "same_beat_support": False
                })

        if not candidates:
            return []

        # keep best unique (triad, beat)
        unique = {}
        for c in candidates:
            key = (tuple(sorted(c["triad_notes"])), c["beat"])
            if key not in unique or c["score"] > unique[key]["score"]:
                unique[key] = c

        result = list(unique.values())
        result.sort(key=lambda c: c["score"], reverse=True)
        return result

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
        triad = candidate["triad_notes"]
        missing = candidate["missing"]
        target_beat = candidate["beat"]

        current_note_mod = None
        if current_note_event is not None:
            current_note_mod = int(current_note_event[0]) % 12

        # keep same note if possible
        if (
            current_note_mod is not None
            and current_note_mod in triad
            and current_note_mod in scale.notes
            and not self.is_note_beat_pair_forbidden(current_note_mod, target_beat, time_s)
        ):
            return current_note_mod

        # best case: complete the triad
        if len(missing) == 1:
            note = list(missing)[0]
            if not self.is_note_beat_pair_forbidden(note, target_beat, time_s):
                return note

        # if chord is already complete, do not double
        if len(missing) == 0:
            if (
                current_note_mod is not None
                and current_note_mod in scale.notes
                and not self.is_note_beat_pair_forbidden(current_note_mod, target_beat, time_s)
            ):
                return current_note_mod

            # choose another note from the same scale, preferably outside the triad
            outside = [n for n in scale.notes if n not in triad]
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
            p_change = 0.10
        else:
            p_change = 0.02

        # 3) strong penalty if target beat is already more crowded
        if target_usage > current_usage:
            diff = target_usage - current_usage
            p_change *= (0.35 ** diff)

        # 4) strong penalty if target beat exceeds ideal local occupancy
        if target_usage > ideal_usage:
            overflow = target_usage - ideal_usage
            p_change *= (0.25 ** overflow)

        # 5) near-block if target beat is dominant locally
        max_usage = max(beat_usage.values()) if beat_usage else 0
        if target_usage == max_usage and target_usage >= ideal_usage + 1:
            p_change *= 0.10

        # 6) if chord already sufficiently represented on target beat, don't reinforce it much more
        if same_beat_support and present_count >= 2 and target_usage >= 2:
            p_change *= 0.20

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

        if not self.is_scale_stable_enough(scale_confidence):
            tonal_note_event = self.choose_note_from_scale_only(scale, current_note_event)

            debug["reason"] = "scale_alignment_only"
            debug["used_fallback"] = False

            if tonal_note_event is not None:
                return tonal_note_event, current_beat, debug

            debug["used_fallback"] = True
            debug["reason"] = "scale_alignment_failed"
            return None, current_beat, debug
        
        # hard collision rule: same note + same beat as a neighbor -> ban this pair temporarily
        if self.detect_same_note_same_beat_collision(recent_events, current_note_event, current_beat):
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

                    debug["reason"] = f"forbidden_pair_escape::{alt_reason}"
                    debug["beat"] = alt_beat
                    return alt_note_event, alt_beat, debug

        # maintain current chord if still valid
        if self.is_current_chord_still_valid(recent_events, current_note_event, scale):
            debug["reason"] = "maintain_current_chord"
            debug["chord_root"] = self.current_chord_root
            debug["chord_notes"] = sorted(list(self.current_chord))
            debug["beat"] = self.current_chord_beat
            return current_note_event, self.current_chord_beat, debug

        # find candidate chords
        candidates = self.find_chord_candidates(recent_events, scale, current_beat)
        if not candidates:
            self.current_chord = None
            self.current_chord_root = None
            self.current_chord_beat = None
            debug["used_fallback"] = True
            debug["reason"] = "no_chord_candidate"
            return None, current_beat, debug

        best_candidate = candidates[0]
        chosen_note = self.choose_note_for_candidate(best_candidate, current_note_event, scale, time_s)

        if chosen_note is None:
            self.current_chord = None
            self.current_chord_root = None
            self.current_chord_beat = None
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
        self.current_chord = set(best_candidate["triad_notes"])
        self.current_chord_root = best_candidate["triad_root"]
        self.current_chord_beat = chosen_beat

        note_event = (int(chosen_note), self.beat_duration_s, self.fallback_volume)

        debug["reason"] = f"new_or_updated_chord::{beat_reason}"
        debug["chord_root"] = self.current_chord_root
        debug["chord_notes"] = sorted(list(self.current_chord))
        debug["beat"] = chosen_beat

        return note_event, chosen_beat, debug