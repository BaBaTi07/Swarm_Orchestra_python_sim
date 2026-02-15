import numpy as np
from numpy.typing import NDArray

class Fsm( ):
    
    def __init__(self, rho:np.float64, fm_length:np.int64):
        self.rho                             = rho
        self.forward_movement_length         = fm_length
        self.count_step_forward              = 0
        self.turning_length                  = 0
        self.count_step_turn                 = 0
        self.turning_length_already_computed = False
        self.turning_side_already_computed   = False
        self.no_obstacle                     = True
        self.output                          = np.zeros(2)

     #  --------------------------------------------------------
    
    def random_turn_length( self ):
        c = ( (2.0 * self.rho) / (1.0 + (self.rho * self.rho)) )
        V = np.cos(np.random.rand()* (2.0*np.pi))
        sigma = np.arccos((V+c)/(1+(c*V)) )  # [0, PI];
        return int((np.rint(25.0 * sigma)/np.pi))
    

    #  --------------------------------------------------------
 
    def turn_to_avoid_obstacle( self ):
        if not self.turning_side_already_computed:
            self.turning_side_already_computed = True
            if( np.random.rand()  > 0.5 ):
                self.output = np.array([0.3, 0.7])
            else:
                self.output = np.array([0.7, 0.3])
        self.count_step_forward = 0
        return self.output
        
    #  --------------------------------------------------------
 
    def turn_on_spot( self ):
        if not self.turning_side_already_computed:
            self.turning_side_already_computed = True
            if( np.random.rand()  > 0.5 ):
                self.output = np.array([0.3, 0.7])
            else:
                self.output = np.array([0.7, 0.3])
        
        if( not self.turning_length_already_computed ):
            self.turning_length_already_computed = True
            self.turning_length  = self.random_turn_length( )
            self.count_step_turn = 0

        self.count_step_turn += 1
        if self.count_step_turn >= self.turning_length:
            self.count_step_forward              = 0
            self.turning_side_already_computed   = False
            self.turning_length_already_computed = False
            
        return self.output
        
    #  --------------------------------------------------------
       
    def check_for_collisions( self, ir_readings: NDArray[np.float64] ):
        index = [3,4]
        ir = np.delete(ir_readings, index)
        for i in ir:
            if not i < 0.2:
                return False
        return True
    
     #  --------------------------------------------------------
     
    def move_forward(self ):
        self.count_step_forward += 1
        return (0.8, 0.8)
            
    #  --------------------------------------------------------
       
    def update ( self, ir_readings: NDArray[np.float64]):
        self.no_obstacles = self.check_for_collisions( ir_readings )
        if( self.no_obstacles ):
            if( self.count_step_forward < (self.forward_movement_length + (np.random.randint(20) - 10)) ):
                return self.move_forward( )
            else:
                return self.turn_on_spot( )
        else:
            return self.turn_to_avoid_obstacle( )