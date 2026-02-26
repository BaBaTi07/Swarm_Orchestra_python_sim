import numpy as np
from numpy.typing import NDArray
from CONTROL.fsm  import Fsm
from TOOLS.logger import logger


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

        self.K = 0.1 # Kuramoto coupling strength
        self.theta = float(2.0 * np.pi * np.random.rand())  # initial phase of the internal clock


    def generate_random_note_event(self):
        """(midi, duration_s, volume)"""
        return (int(13*np.random.rand())+60, 0.5, 0.6)
    
    def generate_ir_message(self):
        """return internal clock (8 bits)"""
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
            theta_dot += self.K * np.sin(theta_j - theta)
        
        # Update the internal clock phase
        theta = (theta + theta_dot * dt_s) % (2.0 * np.pi)

        # return internal clock time and phase
        return (theta / (2.0 * np.pi)) * T , theta
    
    def update(self, ir_readings: NDArray[np.float64], msgs: list, time_s: float, dt_s: float):

        note_event = None
        msg_snd = None
        wheels = (0.5, 0.5)

        # Wait 0-5 seconds before moving to simulate manual initialization and desynchronization
        if self.start:
            self.start = False
            self.waiting_time = float(5 * np.random.rand())
        if time_s < self.waiting_time:
            return wheels, note_event, msg_snd

        # Kuramoto --> internal clock time and phase"""
        self.internal_clock, self.theta = self.Kuramoto_update(msgs, self.cicle_time_s, dt_s, self.theta)

        if self.internal_clock < (dt_s + self.beat_duration_s * self.beat_to_play) and self.internal_clock + dt_s >= (self.beat_duration_s * self.beat_to_play): 
            note_event = self.generate_random_note_event()
            
        if self.check_for_things_around( ir_readings ):
            msg_snd = self.generate_ir_message()

        self.no_obstacle = self.check_for_collisions( ir_readings )

        if( self.no_obstacle ):
            self.just_hit_obstacle = False
            if( self.count_step_forward < (self.forward_movement_length + (np.random.randint(20) - 10)) ):
                wheels = self.move_forward( )
            else:
                wheels = self.turn_on_spot( )
        else:
            wheels = self.turn_to_avoid_obstacle( )

        return wheels, note_event, msg_snd