import numpy as np
from numpy.typing import NDArray
from CONTROL.fsm import Fsm
from TOOLS.logger import logger
from TOOLS.scales import Scales
from CONTROL.sync_algo import SyncAlgo
from CONTROL.harmony_algo import HarmonyAlgo

class SwarmMusicFsm(Fsm):

    def __init__(self, rho: float, fm_length: int):

        super().__init__(rho, fm_length)
        self.start = True
        self.waiting_time = 0
        self.cicle_time_s = 2.0  # time of a full cycle of the internal clock
        self.nbr_beats = 4  # number of beats in a cycle
        self.beat_to_play = int(np.random.randint(0, self.nbr_beats))
        self.beat_duration_s = self.cicle_time_s / self.nbr_beats

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

        self.sync_algo = SyncAlgo(
            algo_type="memory",  #"kuramoto_basic", "kuramoto_confidence", "kuramoto_local_error", "memory"
            cycle_time_s=self.cicle_time_s,
            phase_levels=128,
            K=0.5,
            self_phase_weight=0.25,
            max_memory_cycles=60
        )

        self.harmony_algo = HarmonyAlgo(
            nbr_beats=self.nbr_beats,
            beat_duration_s=self.beat_duration_s,
            note_memory_ttl_s=10.0,
            beat_memory_ttl_s=10.0,
            same_captor_merge_ttl_s=1.0, # anti-duplication
            fallback_volume=0.6,
            beat_change_eval_delay_s=3.0,   # délai avant comparaison avant/après
            bad_beat_penalty_decay=0.995,   # oubli lent
            dominant_beat_window_s=5.0,     # si un beat domine > 5s, fuite forte
            forbidden_pair_ttl_s=10.0
        )

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
        return (int(12*np.random.rand()), self.beat_duration_s, 0.6)

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
            note = (msg.payload - 128)%12  # convert back to note value
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
            return self.sync_algo.theta_to_payload()
  
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

        # time synchronization
        self.sync_algo.update(synch_msg, time_s, dt_s)

        note_to_play, chosen_beat, harmony_debug = self.harmony_algo.update(
            note_msgs=note_msg,
            current_note_event=self.last_played_note,
            current_beat=self.beat_to_play,
            time_s=time_s
        )

        if note_to_play is None and self.last_played_note is None:
            note_to_play = self.generate_random_note_event()
            chosen_beat = self.beat_to_play

        if harmony_debug["used_fallback"]:
            logger.log("WARN", f"HarmonyAlgo fallback: {harmony_debug['reason']}")
            note_to_play = self.generate_note_to_play(note_msg)
            chosen_beat = self.choose_beat_to_play(neighbor_beats_played)
        else:
            logger.log(
                "DEBUG",
                f"HarmonyAlgo ok: scale={harmony_debug['scale']}, "
                f"chord={harmony_debug['chord_notes']}, beat={harmony_debug['beat']}, "
                f"reason={harmony_debug['reason']}"
            )

        self.beat_to_play = chosen_beat
        self.last_played_note = note_to_play

        #déclanchement de la note au bon moment
        if self.sync_algo.internal_clock < (dt_s + self.beat_duration_s * self.beat_to_play) and self.sync_algo.internal_clock + dt_s >= (self.beat_duration_s * self.beat_to_play): 
            note_event = note_to_play
            
        #send a message if someting around    
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