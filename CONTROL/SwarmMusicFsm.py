import numpy as np
from numpy.typing import NDArray
from CONTROL.fsm  import Fsm


class SwarmMusicFsm(Fsm):

    def __init__(
        self,
        rho: float,
        fm_length: int,
    ):
        super().__init__(rho, fm_length)

    def generate_random_note_event(self):
        """(midi, duration_s, volume)"""
        return (int(13*np.random.rand())+60, 0.1+1.0*np.random.rand(), 0.8*np.random.rand())
    
    def generate_ir_message(self):
        """number between 60 and 72 (8 bits)"""
        return int(np.random.rand()*13 + 60)
    
    def update(self, ir_readings: NDArray[np.float64], msgs: list, time_s: float, dt_s: float):
        
        note_event = None
        msg_snd = None
        wheels = (0.5, 0.5)

        self.no_obstacle = self.check_for_collisions( ir_readings )

        if self.check_for_things_around( ir_readings ):
            msg_snd = self.generate_ir_message()

        for msg in msgs:
            note = msg.sender_id % 13+ 60
            strenght = msg.strenght
            note_event = (note, 1.1-np.random.rand(), strenght)

        if( self.no_obstacle ):
            self.just_hit_obstacle = False
            if( self.count_step_forward < (self.forward_movement_length + (np.random.randint(20) - 10)) ):
                wheels = self.move_forward( )
            else:
                wheels = self.turn_on_spot( )
        else:
            wheels = self.turn_to_avoid_obstacle( )

        return wheels, note_event, msg_snd