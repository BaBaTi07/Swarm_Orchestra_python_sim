import numpy as np
from CONTROL.fsm import *
from WORLD.arena import *
from TOOLS.logger import logger

class Exp( ):
    num_trials     = 0
    num_iterations = 0
    my_controller  = np.array([]) #This is the array where it is stored the robot controllers
    trial         = 0
    iter          = 0
 
    def reset():
        Exp.my_controller = np.array([])
        Exp.trial = 0
        Exp.iter  = 0
        Exp.num_trials = 0
        Exp.num_iterations = 0

    def init_all_trials():
        Exp.trial = 0
        for e in range (len(Arena.robot)):
            Exp.my_controller = np.append(Exp.my_controller, Fsm(0.6, 50))
    
    def init_single_trial():
        for e in range (len(Arena.robot)):
            id = Arena.robot[e].id
            np.copyto(Arena.robot[id].pos, Arena.robot[id].init_pos )
            np.copyto(Arena.robot[id].rot, Arena.robot[id].init_rot )
        Exp.iter = 0
        
    
    def finalise_single_trial():
        if( Exp.iter >= Exp.num_iterations):
            Exp.trial += 1
            return False
        else:
            return True
    
    def finalise_all_trials( ):
        if( Exp.trial >= Exp.num_trials):
            return False
        else:
            return True
    
    def exp_engine():
        Exp.init_all_trials()
        while ( Exp.finalise_all_trials() ):
            Exp.init_single_trial()
            while ( Exp.finalise_single_trial() ):
                Exp.make_iteration()

    def make_iteration():
        for rb in Arena.robot:
            rb.update_sensors( )
            wheels, music_event = Exp.my_controller[rb.id].update( rb.Dst_rd.reading  )
            rb.make_movement( np.array(wheels) )
            if music_event is not None and hasattr(rb, 'play_note'):
                logger.log("DEBUG",f"Robot {rb.id} plays note: {music_event[0]} for {music_event[1]} seconds at volume {music_event[2]}")
                rb.play_note(music_event[0], music_event[1], volume=music_event[2])
        Exp.iter += 1