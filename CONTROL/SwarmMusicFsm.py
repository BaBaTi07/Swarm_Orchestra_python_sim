import numpy as np
import copy
from numpy.typing import NDArray
from CONTROL.fsm  import Fsm
from TOOLS.logger import logger
from TOOLS.scales import Scales

class SwarmMusicFsm(Fsm):

    def __init__(self, rho: float, fm_length: int):

        super().__init__(rho, fm_length)
        self.start = True
        self.waiting_time = 0
        self.internal_clock = 0
        self.cicle_time_s = 2.0  # time of a full cycle of the internal clock
        self.nbr_beats = 4  # number of beats in a cycle
        self.beat_to_play = 1 #int(np.random.randint(0, self.nbr_beats))
        self.beat_duration_s = self.cicle_time_s / self.nbr_beats

        self.K = 0.5 # Kuramoto coupling strength
        self.theta = float(2.0 * np.pi * np.random.rand())  # initial phase of the internal clock
        self.kuramoto_conf = 0.5 #confidence in sync [0-1] increase when same theta, decrease when opposite theta

        self.communication_time = 3.0 # time for witch the robot will stop and comunicate when recieving a message
        self.communicating = False
        self.communication_timer = 0.0

        self.communication_cooldown_time = 5.0 # the robot will not communicate so it can move around 
        self.communication_cooldown_timer = 0.0
        self.communication_cooldown = False

        self.last_received_notes_max_length =3
        self.last_received_notes = []
        self.last_played_note = None

        self.neighbor_beats_played = []

        self.note_or_clock = False # if true, send note, if false send clock

        self.phase_levels = 128 

        # Memory-based circular averaging synchronization parameters

        self.self_phase_weight = 0.25 # relative weight of robot's own phase in final mean [0..1]
        self.max_memory_cycles = 60 
        self.phase_memory = [] 
        self.prev_phase_level = 0 # moment of a new cycle, used to know when to update sync memory

        self.phase_cont = ((self.theta % (2.0 * np.pi)) / (2.0 * np.pi)) * self.phase_levels
        self.prev_phase_cont = self.phase_cont
        self.prev_phase_level = int(round(self.phase_cont)) % self.phase_levels
        self.internal_clock = (self.phase_cont / self.phase_levels) * self.cicle_time_s   
        

    def update_communication(self, dt_s, msgs):
        """
        Handle if a robot should stop moving and communicate
        or not based on received messages and cooldowns.
        """
        wheels = (0.8, 0.8)

        if msgs and not self.communicating and not self.communication_cooldown:
            self.communicating = True
            self.communication_timer = 0.0

        # hndle current communication state
        if self.communicating:
            self.communication_timer += dt_s
            wheels = self.stop_wheels()
            if self.communication_timer >= self.communication_time:
                # communication done, reset timers and states
                self.communicating = False
                self.communication_cooldown = True
                self.communication_cooldown_timer = 0.0

        # Handle cooldown timer        
        if self.communication_cooldown:
            self.communication_cooldown_timer += dt_s
            if self.communication_cooldown_timer >= self.communication_cooldown_time:
                self.communication_cooldown = False
                self.communication_cooldown_timer = 0.0
        return wheels
    
    def check_for_collisions( self, ir_readings: NDArray[np.float64] ):
        index = [3,4]
        ir = np.delete(ir_readings, index)
        for i in ir:
            if i > 0.98:
                return False
        return True
    
    def check_for_things_around( self, ir_readings: NDArray[np.float64] ):
        for i in ir_readings:
            if i > 0.5: #closer -> 1, farther -> 0, middle -> 0.5
                return True
        return False
    
    def generate_random_note_event(self):
        """(midi, duration_s, volume)"""
        return (int(12*np.random.rand()), self.beat_duration_s, self.kuramoto_conf)

    def choose_beat_to_play(self, neighbor_beats):
        """chose the beat to play based on the beat recived from neighbors,
           if neighbor play the same beat, probabilisticly (0.7) change to a non played beat,
           otherwise keep the same beat"""
        
        if self.beat_to_play in neighbor_beats:
            if np.random.rand() < 0.7:
                potential_beats = [beat for beat in range(self.nbr_beats) if beat not in neighbor_beats]
                if potential_beats:
                    new_beat = np.random.choice(potential_beats)
                    logger.log("DEBUG", f"Changing beat from {self.beat_to_play} to {new_beat} to avoid neighbor beats {neighbor_beats}")
                    return new_beat
                
        return self.beat_to_play
    
    def generate_note_to_play(self, msgs):
        """chose the note to play based on the three last diferent recived notes"""
        if not msgs or self.last_played_note is None:
            if self.last_played_note is None:
                self.last_played_note = self.generate_random_note_event() 
            return self.last_played_note
        
        # extract notes from messages and keep only the last three different ones
        for msg in msgs:
            note = msg.payload - 128  # convert back to note value
            if note not in self.last_received_notes: #new note
                self.last_received_notes.append(note)

                if len(self.last_received_notes) > self.last_received_notes_max_length:
                    self.last_received_notes.pop(0)

                chosen_note = self.choose_note_from_scale(self.last_received_notes, self.last_played_note)

                self.last_played_note = (chosen_note, self.beat_duration_s, 0.6)
        
        return self.last_played_note
    
    def choose_note_from_scale(self, notes, last_note):
        """choose a note from the scales based on the last received notes"""
        note_mod = [note % 12 for note in notes]

        potential_scales = [
            scale for scale in Scales
            if all(note in scale.notes for note in note_mod)
        ]

        logger.log("INFO",f"Potential scales based on received notes {notes}: {[scale.name for scale in potential_scales]}")

        last_note_mod = last_note[0] % 12
        if not potential_scales:
            return last_note_mod
        if any(last_note_mod in scale.notes for scale in potential_scales):
            return last_note_mod

        chosen_scale = np.random.choice(potential_scales)
        return np.random.choice(chosen_scale.notes)
                        
    def generate_ir_message(self, note=None, beat=None):
        """return internal clock (7 bits)msb = 0 or note (7 bits) msb = 1"""
        if note is not None:
            value = (beat * 12) + note 
            return value  + 128
        else:
            # Send internal clock message
            return int(self.internal_clock / self.cicle_time_s * 128) % 128

    def theta_to_payload(self):
        """Convert the internal clock phase (theta) to an 8-bit payload for IR communication."""
        return int((self.theta % (2.0 * np.pi)) / (2.0 * np.pi) * 128) % 128
    
    def payload_to_theta(self, payload):
        """Convert an 8-bit payload from IR communication back to an internal clock phase (theta)."""
        return (payload / 128.0) * (2.0 * np.pi)
    
    def Kuramoto_update_confidence_based(self, msgs, cicle_time_s, dt_s, theta):
        """ Kuramoto-like synchronization: adjust internal clock phase based on received messages"""
        T = float(cicle_time_s)
        omega = (2.0 * np.pi) / T
        theta_dot = omega 

        # Add coupling from received messages
        for msg in msgs:
            theta_j = self.payload_to_theta(msg.payload)
            self.update_kuramoto_confidence(theta_j, theta)
            theta_dot += ( self.K * (1-self.kuramoto_conf)) * np.sin(theta_j - theta)
            #print(f"theta_j={theta_j:.2f}, current theta={theta:.2f}, confidence={self.kuramoto_conf:.2f}, theta_dot={theta_dot:.2f}")
        
        # Update the internal clock phase
        theta = (theta + theta_dot * dt_s) % (2.0 * np.pi)

        # return internal clock time and phase
        return (theta / (2.0 * np.pi)) * T , theta
    
    def update_kuramoto_confidence(self, theta_j, theta):
        """Update the confidence in synchronization based on communication success."""
        diff = abs((theta_j - theta+np.pi) % (2.0 * np.pi) - np.pi) 

        if diff < 0.1:  
            self.kuramoto_conf = min(1.0, self.kuramoto_conf + 0.015)  
        elif diff < 0.5: 
            self.kuramoto_conf = min(1.0, self.kuramoto_conf + 0.0045)
        elif diff < 1.0: 
            self.kuramoto_conf = max(0.0, self.kuramoto_conf + 0.0015)
        else:
            self.kuramoto_conf = max(0.0, self.kuramoto_conf - 0.001)
        # if diff is moderate, we do not change confidence
        self.kuramoto_conf = max(0.0, min(1.0, self.kuramoto_conf))  #[0, 1]

    def angle_diff(self, a, b):
        d = (a - b + np.pi) % (2.0 * np.pi) - np.pi
        return d

    def Kuramoto_update_local_error_based(self, msgs, cycle_time_s, dt_s, theta):
        T = float(cycle_time_s)
        omega = (2.0 * np.pi) / T

        theta_dot = omega

        if msgs:
            s = 0.0
            err = 0.0
            n = 0

            for msg in msgs:
                theta_j = self.payload_to_theta(msg.payload)
                d = self.angle_diff(theta_j, theta)

                s += np.sin(d)
                err += abs(d)
                n += 1

            s /= n
            err /= n 

            # erreur max pertinente ~ pi
            err_norm = min(1.0, err / np.pi)

            K_min = 0.05
            K_max = self.K
            K_eff = K_min + (K_max - K_min) * err_norm

            theta_dot += K_eff * s

        theta = (theta + theta_dot * dt_s) % (2.0 * np.pi)
        return (theta / (2.0 * np.pi)) * T, theta

    def Kuramoto_update_basic(self, msgs, cycle_time_s, dt_s, theta):
        T = float(cycle_time_s)
        omega = (2.0 * np.pi) / T
        theta_dot = omega
        s = 0.0

        for msg in msgs:
            theta_j = self.payload_to_theta(msg.payload)
            s += np.sin(theta_j - theta)
        s/= len(msgs) if msgs else 1
        theta_dot += self.K * s

        theta = (theta + theta_dot * dt_s) % (2.0 * np.pi)
        return (theta / (2.0 * np.pi)) * T, theta

    def payload_to_phase_level(self, payload: int) -> int:
        """Convert a sync payload [0..127] to a discrete phase level."""
        return int(payload) % self.phase_levels

    def phase_level_to_theta(self, phase_level: int) -> float:
        """Convert discrete phase level [0..127] to theta in [0, 2pi)."""
        return (float(phase_level % self.phase_levels) / self.phase_levels) * (2.0 * np.pi)

    def phase_level_to_internal_clock(self, phase_level: int) -> float:
        """Convert discrete phase level [0..127] to internal clock in [0, cycle_time)."""
        return (float(phase_level % self.phase_levels) / self.phase_levels) * self.cicle_time_s

    def wrap_phase_level(self, phase_level: float) -> int:
        """Wrap any phase value to an integer discrete phase level in [0..127]."""
        return int(np.floor(phase_level)) % self.phase_levels

    def advance_phase_level_by_elapsed_time(self, phase_level: int, elapsed_s: float) -> int:
        """
        Advance a received discrete phase level according to elapsed time,
        assuming all robots run at the same nominal clock speed.
        """
        levels_advanced = int(round((elapsed_s / self.cicle_time_s) * self.phase_levels))
        return (phase_level + levels_advanced) % self.phase_levels

    def linear_memory_weight(self, age_s: float) -> float:
        """
        Linear decay by cycle:
            age in [0,1 cycle)   -> 1.0
            age in [1,2 cycles)  -> 0.95
            ...
            age in [9,10 cycles) -> 0.55
            age >= 10 cycles     -> 0.50 (invalid, should be removed)
        """
        cycle_age = int(age_s / self.cicle_time_s)
        if cycle_age >= self.max_memory_cycles:
            return 0.0
        if cycle_age <= self.max_memory_cycles /2:
            weight = 1.0
        else:
            weight = 1.0 - ((cycle_age - self.max_memory_cycles /2) / (self.max_memory_cycles /2))
        return max(0.0, weight)

    def cleanup_phase_memory(self, current_time_s: float):
        """Remove messages older than max_memory_cycles cycles."""
        max_age_s = self.max_memory_cycles * self.cicle_time_s
        self.phase_memory = [
            entry for entry in self.phase_memory
            if (current_time_s - entry["time_s"]) < max_age_s
        ]

    def store_sync_messages(self, synch_msgs: list, current_time_s: float):
        """Store received sync messages with their timestamp."""
        for msg in synch_msgs:
            self.phase_memory.append({
                "phase": self.payload_to_phase_level(msg.payload),
                "time_s": current_time_s,
                "captor_id": msg.captor_id
            })

    def phase_levels_are_similar(self, p1: int, p2: int, tolerance: int = 1) -> bool:
        """
        Return True if two discrete phase levels are similar on the circle,
        i.e. circular distance <= tolerance.
        """
        diff = abs(((p1 - p2 + self.phase_levels // 2) % self.phase_levels) - self.phase_levels // 2)
        return diff <= tolerance

    def weighted_circular_mean_phase_level(self, own_phase_level: int, current_time_s: float) -> int:
        """
        Compute weighted circular mean of:
          - robot's own current phase (relative weight = self.self_phase_weight)
          - all valid memory messages, reprojected to current time

        Deduplication rule:
          if two projected messages come from the same captor_id and have
          similar projected phase (±1 phase level), they are considered redundant
          and only counted once.

        If no valid external messages exist, returns own_phase_level.
        """
        self.cleanup_phase_memory(current_time_s)

        deduped_msgs = []   # list of dicts: {"captor_id": ..., "projected_phase": ..., "weight": ...}

        for entry in reversed(self.phase_memory): # reverse to keep newer messages
            age_s = current_time_s - entry["time_s"]
            w = self.linear_memory_weight(age_s)
            if w <= 0.0:
                continue

            projected_phase = self.advance_phase_level_by_elapsed_time(entry["phase"], age_s)
            captor_id = entry["captor_id"]

            # Check if this projected message is redundant with one already kept
            redundant = False
            for kept in deduped_msgs:
                if kept["captor_id"] == captor_id and self.phase_levels_are_similar(
                    projected_phase, kept["projected_phase"], tolerance=1
                ):
                    redundant = True
                    break

            if not redundant:
                deduped_msgs.append({
                    "captor_id": captor_id,
                    "projected_phase": projected_phase,
                    "weight": w
                })

        # No received sync info -> keep own phase
        if not deduped_msgs:
            return own_phase_level

        # Normalize external weights so they sum to (1 - self.self_phase_weight)
        ext_weights = np.array([msg["weight"] for msg in deduped_msgs], dtype=float)
        ext_weights_sum = np.sum(ext_weights)

        if ext_weights_sum <= 0.0:
            return own_phase_level

        ext_weights = ((1.0 - self.self_phase_weight) / ext_weights_sum) * ext_weights

        # Own phase contribution
        own_angle = (own_phase_level / self.phase_levels) * (2.0 * np.pi)
        z = self.self_phase_weight * np.exp(1j * own_angle)

        # External contributions
        for msg, w in zip(deduped_msgs, ext_weights):
            angle = (msg["projected_phase"] / self.phase_levels) * (2.0 * np.pi)
            z += w * np.exp(1j * angle)

        # Fallback if vector is numerically almost zero
        if np.abs(z) < 1e-12:
            return own_phase_level

        mean_angle = np.angle(z)
        if mean_angle < 0.0:
            mean_angle += 2.0 * np.pi

        new_phase_level = int(round((mean_angle / (2.0 * np.pi)) * self.phase_levels)) % self.phase_levels

        logger.log(
            "DEBUG",
            f"Weighted mean: own={own_phase_level}, kept_msgs={len(deduped_msgs)}, "
            f"raw_memory={len(self.phase_memory)}, new={new_phase_level}"
        )

        return new_phase_level

    def update_memory_based_sync(self, synch_msgs: list, time_s: float, dt_s: float):
        """
        Main synchronization update:
        1. store incoming sync messages
        2. advance own phase continuously in float
        3. detect wrap
        4. on wrap, compute new phase by weighted circular mean and jump to it

        Returns:
        internal_clock, theta
        """
        # 1) Store received messages
        self.store_sync_messages(synch_msgs, time_s)

        # 2) Advance own phase continuously (FLOAT, not integer)
        self.prev_phase_cont = self.phase_cont
        phase_increment = (dt_s / self.cicle_time_s) * self.phase_levels
        self.phase_cont = (self.phase_cont + phase_increment) % self.phase_levels

        # 3) Detect wrap on continuous phase
        wrapped = self.phase_cont < self.prev_phase_cont

        # 4) At end of cycle, recalculate phase from memory
        if wrapped:
            current_phase_level = int(round(self.phase_cont)) % self.phase_levels
            new_phase_level = self.weighted_circular_mean_phase_level(current_phase_level, time_s)

            logger.log(
                "DEBUG",
                f"Memory sync wrap: old_phase={current_phase_level}, new_phase={new_phase_level}, "
                f"memory_size={len(self.phase_memory)}"
            )
            self.phase_cont = float(new_phase_level)

        # Update discrete + continuous derived values
        current_phase_level = int(round(self.phase_cont)) % self.phase_levels
        self.prev_phase_level = current_phase_level
        self.theta = self.phase_level_to_theta(current_phase_level)
        self.internal_clock = (self.phase_cont / self.phase_levels) * self.cicle_time_s

        return self.internal_clock, self.theta

    def update(self, ir_readings: NDArray[np.float64], msgs: list, time_s: float, dt_s: float):

        note_event = None
        msg_snd = None
        wheels = (0.5, 0.5)
        synch_msg = []
        note_msg = []
        neighbor_beats_played = []

        # Wait 0-5 seconds before moving to simulate manual initialization and desynchronization
        if self.start:
            self.start = False
            self.waiting_time = float(5 * np.random.rand())
        if time_s < self.waiting_time:
            return wheels, note_event, msg_snd
        
        #sort synch and note messages
        for msg in msgs:
            if msg.payload >= 128:  # note message
                note_msg.append(msg)
                beat = (msg.payload - 128)//12
                neighbor_beats_played.append(beat)

            else:  # synchronization message
                synch_msg.append(msg)

        # Memory-based synchronization
        self.internal_clock, self.theta = self.update_memory_based_sync(synch_msg, time_s, dt_s)

        note_to_play = self.generate_note_to_play(note_msg)
        self.beat_to_play = self.choose_beat_to_play(neighbor_beats_played)
        if self.internal_clock < (dt_s + self.beat_duration_s * self.beat_to_play) and self.internal_clock + dt_s >= (self.beat_duration_s * self.beat_to_play): 
            note_event = note_to_play
            
        if self.check_for_things_around( ir_readings ):
            if self.note_or_clock:
                msg_snd = self.generate_ir_message(note=note_to_play[0], beat=self.beat_to_play)  # send note message
                self.note_or_clock = False
            else:
                msg_snd = self.generate_ir_message()
                self.note_or_clock = True

        self.no_obstacle = self.check_for_collisions( ir_readings )

        # if receiving a message, stop and communicate
        wheels = self.update_communication(dt_s, msgs)
        if wheels != (0.8, 0.8):
             return wheels, note_event, msg_snd

        if( self.no_obstacle ):
            self.just_hit_obstacle = False
            if( self.count_step_forward < (self.forward_movement_length + (np.random.randint(20) - 10)) ):
                wheels = self.move_forward( )
            else:
                wheels = self.turn_on_spot( )
        else:
            wheels = self.turn_to_avoid_obstacle( )

        return wheels, note_event, msg_snd