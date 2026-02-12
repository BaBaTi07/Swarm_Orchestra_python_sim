import numpy as np
from CONTROL.fsm import *
from WORLD.world import *

class Exp( ):
    num_trials     = 0
    num_iterations = 0
    my_controller  = np.array([]) #This is the array where it is stored the epuck controllers
    trial         = 0
    iter          = 0
 
    def init_all_trials():
        Exp.trial = 0
        for e in range (len(Arena.epuck)):
            Exp.my_controller = np.append(Exp.my_controller, Fsm(0.6, 50))
    
    def init_single_trial():
        for e in range (len(Arena.epuck)):
            id = Arena.epuck[e].id
            np.copyto(Arena.epuck[id].pos, Arena.epuck[id].init_pos )
            np.copyto(Arena.epuck[id].rot, Arena.epuck[id].init_rot )
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
        for ep in Arena.epuck:
            ep.update_ir_sensors( )
            ep.make_movement( np.array(Exp.my_controller[ep.id].update( ep.IR.reading  )) )
            #ep.make_movement( np.array([0.4, 0.6]) )
        Exp.iter += 1