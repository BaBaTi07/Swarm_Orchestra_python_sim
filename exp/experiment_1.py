import numpy as np
from world.world import Arena
from robot.epuck import Epuck_robot

class Exp( ):
    #This class assumes Arena parameters have been loaded from JSON before instantiation.
    
    num_trials     = 0
    num_iterations = 0
    
    def __init__( self ):
        self.trial = 0
        self.iter = 0
        self.epuck = np.empty(0)
        for r in range (len(Arena.epucks_xyz_rx_ry_rz_colour)):
                self.epuck = np.append(self.epuck, Epuck_robot(r, 
                                                               Arena.epucks_xyz_rx_ry_rz_colour[r][0:3],
                                                               Arena.epucks_xyz_rx_ry_rz_colour[r][3:6],
                                                               Arena.epucks_xyz_rx_ry_rz_colour[r][6:9],
                                                               np.zeros(2),
                                                               Arena.DeltaT ))
    def init_all_trials(self):
        self.trial = 0
    
    #For now, each trial resets all robots to a fixed start position
    def init_single_trial(self):
        for e in range (len(self.epuck)):
            self.epuck[e].set_pos(np.array((0.0, 0.952, 0.0)))
            # self.epuck[e].set_rot(np.array((0.0, 0.0, 0.0)))
        self.iter = 0
    
    def finalise_single_trial(self):
        if( self.iter == self.num_iterations):
            self.trial += 1
            return False
        else:
            return True
    
    def finalise_all_trials(self):
        if( self.trial == self.num_trials):
            return False
        else:
            return True
    
    def exp_engine(self):
        self.init_all_trials()
        while ( self.finalise_all_trials() ):
            self.init_single_trial()
            while ( self.finalise_single_trial() ):
                self.make_iteration()
                
    #This is a placeholder controller: constant wheel speeds, only for testing sensing + motion integration
    def make_iteration(self):
        for e in range (len(self.epuck)):
            outputs = np.array((0.45, 0.55))
            self.epuck[e].update_ir_sensors()
            self.epuck[e].make_movement( outputs )
        self.iter += 1
            # print(f"iter = {e} - epuck x_pos = {self.epuck[e].pos[0]}, y_pos = {self.epuck[e].pos[1]}, z_pos = {self.epuck[e].pos[2]}, x_rot = {self.epuck[e].rot[0]}, y_rot = {self.epuck[e].rot[1]}, z_rot = {self.epuck[e].rot[2]}")
    