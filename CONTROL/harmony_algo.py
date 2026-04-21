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
    ):
        self.nbr_beats = nbr_beats
        self.beat_duration_s = beat_duration_s

        self.note_memory_ttl_s = note_memory_ttl_s
        self.beat_memory_ttl_s = beat_memory_ttl_s
        self.same_captor_merge_ttl_s = same_captor_merge_ttl_s
        self.fallback_volume = fallback_volume

        # memory of recent note messages
        # item = {"time_s", "captor_id", "note", "beat"}
        self.note_history = []

        # state of the current local harmonic commitment
        self.current_scale = None
        self.current_chord = None       # set[int] or None
        self.current_chord_root = None  # int or None
        self.current_chord_beat = None  # int or None

        # used to increase exploration if local beat consensus stalls
        self.failed_beat_consensus_count = 0

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

    def choose_note_for_candidate(self, candidate, current_note_event, scale):
        """
        Priority:
        1. keep current note if it is already in chord and scale
        2. if exactly one note is missing, choose it
        3. if chord already complete, do not double -> keep current if valid,
           else choose a note in scale (outside the chord if possible), otherwise None
        """
        triad = candidate["triad_notes"]
        missing = candidate["missing"]

        current_note_mod = None
        if current_note_event is not None:
            current_note_mod = int(current_note_event[0]) % 12

        # keep same note if possible
        if current_note_mod is not None and current_note_mod in triad and current_note_mod in scale.notes:
            return current_note_mod

        # best case: complete the triad
        if len(missing) == 1:
            return list(missing)[0]

        # if chord is already complete, do not double
        if len(missing) == 0:
            if current_note_mod is not None and current_note_mod in scale.notes:
                return current_note_mod

            # choose another note from the same scale, preferably outside the triad
            outside = [n for n in scale.notes if n not in triad]
            if outside:
                return int(np.random.choice(outside))

            # otherwise impossible to avoid duplication cleanly
            return None

        # if two notes are missing, candidate is weak; let caller decide with probabilities
        if current_note_mod is not None and current_note_mod in scale.notes:
            return current_note_mod

        return None

    # ------------------------------------------------------------------
    # Beat choice
    # ------------------------------------------------------------------

    def compute_local_beat_usage(self, beat_events: list):
        """
        Count recent beat usage locally.
        """
        usage = {b: 0 for b in range(self.nbr_beats)}
        for e in beat_events:
            usage[e["beat"]] += 1
        return usage

    def choose_best_unoccupied_beat(self, beat_usage: dict, forbidden_beats: set | None = None):
        if forbidden_beats is None:
            forbidden_beats = set()

        candidates = [b for b in range(self.nbr_beats) if b not in forbidden_beats]
        if not candidates:
            return None

        min_use = min(beat_usage[b] for b in candidates)
        best = [b for b in candidates if beat_usage[b] == min_use]
        return int(np.random.choice(best))

    def choose_beat_for_candidate(self, candidate, current_beat: int, beat_events: list):
        """
        Beat decision according to user's rules:

        - avoid changing beat first
        - if same-beat support is strong, strong probability to join that beat
        - if support is weaker/cross-beat, lower probability
        - if local beat usage suggests target beat is crowded, probability drops
        - if repeated failed local convergence, exploration toward a less used beat increases
        """
        target_beat = candidate["beat"]
        same_beat_support = candidate["same_beat_support"]
        present_count = len(candidate["present"])

        beat_usage = self.compute_local_beat_usage(beat_events)
        current_usage = beat_usage.get(current_beat, 0)
        target_usage = beat_usage.get(target_beat, 0)

        # 1) avoid changing if possible
        if target_beat == current_beat:
            self.failed_beat_consensus_count = 0
            return current_beat, "keep_current_beat"

        # 2) base probability depending on harmonic evidence
        if same_beat_support and present_count >= 2:
            p_change = 0.85
        elif present_count >= 2:
            p_change = 0.45
        elif present_count == 1:
            p_change = 0.20
        else:
            p_change = 0.05

        # 3) crowded target beat => less attractive
        if target_usage > current_usage:
            p_change *= 0.45
        elif target_usage < current_usage:
            p_change *= 1.15

        # 4) after repeated failed consensus, increase exploration toward least used beat
        exploration_bonus = min(0.35, 0.08 * self.failed_beat_consensus_count)
        unused_beats = {b for b, u in beat_usage.items() if u == 0}

        # if both robots/notes don't converge after several interactions,
        # bias toward a beat not used locally
        exploratory_beat = None
        if self.failed_beat_consensus_count >= 3 and unused_beats:
            exploratory_beat = self.choose_best_unoccupied_beat(beat_usage)
            if exploratory_beat is not None:
                p_explore = min(0.60, 0.20 + exploration_bonus)
                if np.random.rand() < p_explore:
                    self.failed_beat_consensus_count = 0
                    return exploratory_beat, "explore_unused_beat"

        # 5) normal target selection
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

        debug = {
            "used_fallback": False,
            "reason": "",
            "scale": None,
            "chord_root": None,
            "chord_notes": None,
            "beat": current_beat,
            "recent_neighbors": len(recent_events),
        }

        # no recent neighbors -> keep current state, no aggressive recalculation
        if not recent_events:
            debug["reason"] = "no_recent_neighbors_keep_state"
            return current_note_event, current_beat, debug

        # infer local scale
        scale = self.infer_local_scale(recent_events, current_note_event)
        self.current_scale = scale
        debug["scale"] = getattr(scale, "name", None)

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
        chosen_note = self.choose_note_for_candidate(best_candidate, current_note_event, scale)

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
            beat_events
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