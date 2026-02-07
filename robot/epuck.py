import numpy as np
from sensors.ir_sensors import Ir_sensors

class Epuck_robot( ):
    
    # dimensions in meters
    wheel_distance          = 0.052 
    half_wheel_distance     = 0.026
    wheel_radius            = 0.0205
    robot_radius            = 0.037 
    robot_height           =  0.055

    pos_noise               = 0.0 # 0.01
    rot_noise               = 0.0 # 0.0087
    vel_noise               = 0.0
    prob_long_term_change   = 0.02
    prob_medium_term_change = 0.1
    
    def __init__ ( self, id, pos, rot, colour, linear_vel, deltat):
        if not hasattr(Epuck_robot, 'DeltaT' ):
            Epuck_robot.DeltaT  = deltat
        self.IR             = Ir_sensors( Epuck_robot.robot_radius )
        self.id             = id
        self.pos            = pos
        self.old_pos        = pos
        self.rot            = (rot * np.pi)/180.0
        self.old_rot        = rot
        self.colour         = colour
        self.linear_vel     = linear_vel # [0] left wheel - [1] right wheel 

    def set_pos(self, _pos):
        self.pos[0] = _pos[0]
        self.pos[1] = _pos[1]
        
    def set_rot(self, _rot):
        self.rot[2] = _rot[2]
    
    def set_colour(self, _col):
        self.colour[0] = _col[0]
        self.colour[1] = _col[1]
        self.colour[2] = _col[2]
    
    def set_linear_vel(self, _lin_vel):
        self.linear_vel[0] = _lin_vel[0]
        self.linear_vel[1] = _lin_vel[1]
        
    def add_noise( self, element, noise_level, high_threshold, low_threshold):
        for i in range(len(element)):
            element[i] += 2.0 * np.random.rand() * noise_level - noise_level
            if( element[i] > high_threshold ):
                element[i] = high_threshold 
            elif ( element[i] < low_threshold ):
                element[i] = low_threshold

    def update_ir_sensors( self ):
        self.IR.update_sensors( self.pos, self.rot, Epuck_robot.robot_radius )

    def make_movement( self, outputs ):
        self.old_pos[0] = self.pos[0]
        self.old_pos[1] = self.pos[1]
        self.old_rot[2] = self.rot[2]
        
        self.add_noise( outputs, 0.0, 1.0, 0.0)
        for i in range(len(outputs)):
            # wheel rotation speed between [-2PI, 2PI]
            outputs[i] = outputs[i] * (4.0 * np.pi) - (2.0 * np.pi) 
            
        # Set linear velocity
        for i in range (len(self.linear_vel)):
            self.linear_vel[i] = outputs[i] * Epuck_robot.wheel_radius
        
        if ( self.linear_vel[0] != self.linear_vel[1] ):
            R = (0.5 * Epuck_robot.wheel_distance) * ((self.linear_vel[1]+self.linear_vel[0])/(self.linear_vel[1]-self.linear_vel[0]))
            omega = (self.linear_vel[1]-self.linear_vel[0])/Epuck_robot.wheel_distance
            ICC_x = self.pos[0] - R * np.sin(self.rot[2])
            ICC_y = self.pos[1] + R * np.cos(self.rot[2])
            omegaDeltaT = omega * Epuck_robot.DeltaT
            a1 = [[np.cos(omegaDeltaT), -np.sin(omegaDeltaT), 0.0], [np.sin(omegaDeltaT),  np.cos(omegaDeltaT), 0.0], [0.0, 0.0, 1.0]]
            a1 = np.array(a1)
            a2 = np.array((self.pos[0] - ICC_x, self.pos[1] - ICC_y, self.rot[2]))
            a3 = np.array((ICC_x, ICC_y, (omegaDeltaT) ))
            a4 = np.add(np.dot(a1, a2), a3)
        else:
            v = self.linear_vel[0]
            a1 = [v * np.cos(self.rot[2]), v * np.sin(self.rot[2]), 0]
            a1 = np.array(a1)
            a2 = np.array((self.pos[0], self.pos[1], self.rot[2]))
            a4 = np.add(a1 * Epuck_robot.DeltaT, a2)
            
        self.pos[0] = a4[0]
        self.pos[1] = a4[1]
        self.rot[2] = a4[2]
        if ( self.rot[2] >= (2.0 * np.pi) ) :
            self.rot[2] -= (2.0 * np.pi)
        if ( self.rot[2] < 0.0 ) :
            self.rot[2] += (2.0 * np.pi)
            
if __name__ == "__main__" :
    rob = Epuck_robot(0, np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(2), 0.1)