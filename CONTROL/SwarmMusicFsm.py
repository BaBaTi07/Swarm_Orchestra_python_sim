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

        self.K = 1.5 # Kuramoto coupling strength
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

        self.note_or_clock = False # if true, send note, if false send clock
        

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
                #print("something around", ir_readings)
                return True
        return False
    
    def generate_random_note_event(self):
        """(midi, duration_s, volume)"""
        return (int(12*np.random.rand())+60, self.beat_duration_s, self.kuramoto_conf)
    
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

                chosen_note = 60 + self.choose_note_from_scale(self.last_received_notes, self.last_played_note)

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
                        
    def generate_ir_message(self, note=None):
        """return internal clock (7 bits)msb = 0 or note (7 bits) msb = 1"""
        if note is not None:
            # Send a note message
            return 128 + int(note) % 128
        else:
            # Send internal clock message
            return int(self.internal_clock / self.cicle_time_s * 128) % 128

    def theta_to_payload(self):
        """Convert the internal clock phase (theta) to an 8-bit payload for IR communication."""
        return int((self.theta % (2.0 * np.pi)) / (2.0 * np.pi) * 128) % 128
    
    def payload_to_theta(self, payload):
        """Convert an 8-bit payload from IR communication back to an internal clock phase (theta)."""
        return (payload / 128.0) * (2.0 * np.pi)
    
    def Kuramoto_update(self, msgs, cicle_time_s, dt_s, theta):
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

    def update(self, ir_readings: NDArray[np.float64], msgs: list, time_s: float, dt_s: float):

        note_event = None
        msg_snd = None
        wheels = (0.5, 0.5)
        synch_msg = []
        note_msg = []

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
            else:  # synchronization message
                synch_msg.append(msg)

        # Kuramoto --> internal clock time and phase"""
        self.internal_clock, self.theta = self.Kuramoto_update(synch_msg, self.cicle_time_s, dt_s, self.theta)

        note_to_play = self.generate_note_to_play(note_msg)
            
        if self.internal_clock < (dt_s + self.beat_duration_s * self.beat_to_play) and self.internal_clock + dt_s >= (self.beat_duration_s * self.beat_to_play): 
            note_event = note_to_play
            
        if self.check_for_things_around( ir_readings ):
            if self.note_or_clock:
                msg_snd = self.generate_ir_message(note=note_to_play[0])  # send note message
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