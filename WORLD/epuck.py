import numpy as np
from numpy.typing import NDArray
from SENSORS.ir_sensors import Ir_sensors
from WORLD.shapes import *

class Epuck_robot( Diff_drive_robot ):
    # Epuck dimensions in meters
    wheel_distance  = 0.052 
    wheel_radius    = 0.0205
    robot_radius    = 0.037 
    robot_height    =  0.055

    def __init__ ( self, id: np.int64, pos: NDArray[np.float64], rot: NDArray[np.float64], colour: NDArray[np.float64], linear_vel: NDArray[np.float64] ):
        super().__init__(id, pos, rot, linear_vel, Epuck_robot.wheel_distance, Epuck_robot.wheel_radius, Epuck_robot.robot_radius, Epuck_robot.robot_height, colour)
        self.IR  = Ir_sensors( )
        
    def update_ir_sensors( self ):
        self.IR.update_sensors( self.id )
    
    def draw(self):
        super().draw ()
        
        glPushMatrix()
        glTranslatef(self.pos[0], self.pos[2], self.pos[1])    
        # ---- Draw epuck headings on the cylinder upper side ----
        #Rotation (headings) of the Diff drive robot body    
        x = self.radius * np.cos(self.rot[2])
        z = self.radius * np.sin(self.rot[2])
        glDisable(GL_LIGHTING)
        glLineWidth(3.0)
        glColor3f(1.0, 1.0, 0.0)
        glBegin(GL_LINES)
        glVertex3f(0.0, self.height + 0.001, 0.0)
        glVertex3f(x, self.height + 0.001, z)
        glEnd()
        glLineWidth(1.0)    
        glEnable(GL_LIGHTING)
        
        # ---- Draw robot ir sensors rays ----
        new_ir_angle = np.zeros(self.IR.nb_sensors)
        full_lenght  = np.zeros(self.IR.nb_sensors)
        for ir in range(self.IR.nb_sensors):
            new_ir_angle[ir] = self.rot[2] + self.IR.ir_angle[ir]
            if new_ir_angle[ir] < 0.0:
                new_ir_angle[ir] += 2.0*np.pi
            elif new_ir_angle[ir] > (2.0 * np.pi):
                new_ir_angle[ir] -= (2.0 * np.pi)
            full_lenght[ir] = self.radius + self.IR.distance[ir]
            
            x = full_lenght[ir] * np.cos(new_ir_angle[ir])
            z = full_lenght[ir] * np.sin(new_ir_angle[ir])
            glDisable(GL_LIGHTING)
            glLineWidth(3.0)
            glColor3f(1.0, 0.0, 0.0)
            glBegin(GL_LINES)
            glVertex3f(0.0, self.height - 0.002, 0.0)
            glVertex3f(x, self.height - 0.002, z)
            glEnd()
            glLineWidth(1.0)    
            glEnable(GL_LIGHTING)
        glPopMatrix()

    
if __name__ == "__main__" :
    rob = Epuck_robot(0, np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(2), 0.1)