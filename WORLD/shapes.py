import numpy as np
from numpy.typing import NDArray

from OpenGL.GL import *
from OpenGL.GLU import *

class Shapes ():
    def __init__ ( self, pos: NDArray[np.float64], rot: NDArray[np.float64], colour: NDArray[np.float64] ):
        self.pos = pos # pos[0] = x, pos[1] = y pos[2] = z
        self.rot = rot # in RADIANT : rot[0] = roll-rx, rot[1] = pitch-ry rot[2] = yaw-rz
        self.colour = colour # colour[0] = RED in [0,1],  
                             # colour[1] = GREEN in [0,1], 
                             # colour[2] = BLUE in [0,1], 
    
class Ring(Shapes):
    def __init__ (self, pos: NDArray[np.float64], radius: np.float64, height: np.float64, rot: NDArray[np.float64], colour: NDArray[np.float64]):
        super().__init__(pos, rot, colour)
        self.radius = radius
        self.height = height
        
    def draw (self, slices=48):
        glPushMatrix()
        glTranslatef(self.pos[0], self.pos[2], self.pos[1])
        #Rotation of the ring body
        DEG_rot = np.rad2deg(self.rot) # from RADIANTS to DEGREES
        glRotatef(DEG_rot[0], 1.0, 0.0, 0.0)
        glRotatef(DEG_rot[1], 0.0, 0.0, 1.0)
        glRotatef(DEG_rot[2], 0.0, 1.0, 0.0)
        glColor3f(self.colour[0], self.colour[1], self.colour[2]) 
        # glMaterialfv(GL_FRONT, GL_DIFFUSE, (0.9, 0.6, 0.2, 1))
        # half_h = height / 2.0
        step = 2 * np.pi / slices
        glBegin(GL_QUAD_STRIP)
        for i in range(slices + 1):
            angle = i * step
            x = np.cos(angle)
            z = np.sin(angle)
            # Normal
            glNormal3f(x, 0, z)
            # Low Vertex
            glVertex3f(self.radius * x, 0.0, self.radius * z)
            # High Vertex
            glVertex3f(self.radius * x, self.height, self.radius * z)
        glEnd()
        glPopMatrix()

class Cuboid (Shapes):
    def __init__ ( self, pos: NDArray[np.float64], dim: NDArray[np.float64], rot: NDArray[np.float64], colour: NDArray[np.float64] ):
        super().__init__(pos, rot, colour)
        self.dim = dim # dims[0] = length, dim[1] = width dim[2] = height
        
    def draw(self):
        glPushMatrix()
        # Translation of cuboud body
        glTranslatef(self.pos[0], self.pos[2]+self.dim[2]/2, self.pos[1])
        #Rotation of the cuboid body
        DEG_rot = np.rad2deg(self.rot) # from RADIANTS to DEGREES
        glRotatef(DEG_rot[0], 1.0, 0.0, 0.0)
        glRotatef(DEG_rot[1], 0.0, 0.0, 1.0)
        glRotatef(DEG_rot[2], 0.0, 1.0, 0.0)
        glScalef(self.dim[0], self.dim[2], self.dim[1])
        
        glColor3f(self.colour[0], self.colour[1], self.colour[2])
        s = 0.5 
        glBegin(GL_QUADS)

        # Front (Z = +0.5)
        glVertex3f(-s, -s,  s)
        glVertex3f( s, -s,  s)
        glVertex3f( s,  s,  s)
        glVertex3f(-s,  s,  s)

        # Back (Z = -0.5)
        glVertex3f(-s, -s, -s)
        glVertex3f(-s,  s, -s)
        glVertex3f( s,  s, -s)
        glVertex3f( s, -s, -s)

        # Top (Y = +0.5)
        glVertex3f(-s,  s, -s)
        glVertex3f(-s,  s,  s)
        glVertex3f( s,  s,  s)
        glVertex3f( s,  s, -s)
    
        # Bottom (Y = -0.5)
        glVertex3f(-s, -s, -s)
        glVertex3f( s, -s, -s)
        glVertex3f( s, -s,  s)
        glVertex3f(-s, -s,  s)

        # Right (X = +0.5)
        glVertex3f( s, -s, -s)
        glVertex3f( s,  s, -s)
        glVertex3f( s,  s,  s)
        glVertex3f( s, -s,  s)

        # Left (X = -0.5)
        glVertex3f(-s, -s, -s)
        glVertex3f(-s, -s,  s)
        glVertex3f(-s,  s,  s)
        glVertex3f(-s,  s, -s)

        glEnd()
        
        # ----- 2) Arêtes noires (wireframe) -----
        glDisable(GL_LIGHTING)
        glColor3f(0.0, 0.0, 0.0)
        glLineWidth(2.0)

        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        glBegin(GL_QUADS)

        # EXACTEMENT les mêmes faces
        glVertex3f(-s, -s,  s); glVertex3f( s, -s,  s); glVertex3f( s,  s,  s); glVertex3f(-s,  s,  s)
        glVertex3f(-s, -s, -s); glVertex3f(-s,  s, -s); glVertex3f( s,  s, -s); glVertex3f( s, -s, -s)
        glVertex3f(-s,  s, -s); glVertex3f(-s,  s,  s); glVertex3f( s,  s,  s); glVertex3f( s,  s, -s)
        glVertex3f(-s, -s, -s); glVertex3f( s, -s, -s); glVertex3f( s, -s,  s); glVertex3f(-s, -s,  s)
        glVertex3f( s, -s, -s); glVertex3f( s,  s, -s); glVertex3f( s,  s,  s); glVertex3f( s, -s,  s)
        glVertex3f(-s, -s, -s); glVertex3f(-s, -s,  s); glVertex3f(-s,  s,  s); glVertex3f(-s,  s, -s)

        glEnd()
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

        glLineWidth(1.0)
        glEnable(GL_LIGHTING)

        glPopMatrix()
        
        
class Cylinder(Shapes):
    def __init__ (self, pos: NDArray[np.float64], radius: np.float64, height: np.float64, rot: NDArray[np.float64], colour: NDArray[np.float64]):
        super().__init__(pos, rot, colour)
        self.radius = radius
        self.height = height

    def draw (self, slices=64):
        glPushMatrix()
        # ---- Trslate the cylinder body ----
        glTranslatef(self.pos[0], self.pos[2], self.pos[1]) 
        #Rotation of the cylinder body
        DEG_rot = np.rad2deg(self.rot) # from RADIANTS to DEGREES
        glRotatef(DEG_rot[0], 1.0, 0.0, 0.0)
        glRotatef(DEG_rot[1], 0.0, 0.0, 1.0)
        glRotatef(DEG_rot[2], 0.0, 1.0, 0.0)

        glColor3f(self.colour[0], self.colour[1], self.colour[2])  
        step = 2 * np.pi / slices
        # ---- Cylinder body ----
        glBegin(GL_QUAD_STRIP)
        for i in range(slices + 1):
            a = i * step
            x = np.cos(a)
            z = np.sin(a)
            glNormal3f(x, 0, z)
            glVertex3f(self.radius * x, 0.0, self.radius * z)
            glVertex3f(self.radius * x, self.height, self.radius * z)
        glEnd()

        # ---- Lower side ----
        glNormal3f(0, -1, 0)
        glBegin(GL_TRIANGLE_FAN)
        glVertex3f(0, 0, 0)
        for i in range(slices + 1):
            a = -i * step
            glVertex3f(self.radius * np.cos(a), 0.0, self.radius * np.sin(a))
        glEnd()
        # ---- Upper side ----
        glNormal3f(0, 1, 0)
        glBegin(GL_TRIANGLE_FAN)
        glVertex3f(0, self.height, 0)
        for i in range(slices + 1):
            a = i * step
            glVertex3f(self.radius * np.cos(a), self.height, self.radius * np.sin(a))
        glEnd()
        
        # ---- Contours noirs des disques (bas + haut) ----
        glDisable(GL_LIGHTING)
        glColor3f(0.0, 0.0, 0.0)
        glLineWidth(2.0)

        glBegin(GL_LINE_LOOP)
        for i in range(slices):
            a = i * step
            glVertex3f(self.radius * np.cos(a), 0.0, self.radius * np.sin(a))
        glEnd()

        glBegin(GL_LINE_LOOP)
        for i in range(slices):
            a = i * step
            glVertex3f(self.radius * np.cos(a), self.height, self.radius * np.sin(a))
        glEnd()

        glLineWidth(1.0)
        glEnable(GL_LIGHTING)
        glPopMatrix()

class Diff_drive_robot(Cylinder):
    
    delta_t  = 0.0
    
    def __init__(self, id, pos: NDArray[np.float64], rot: NDArray[np.float64], linear_vel: NDArray[np.float64], wheel_distance: np.float64, wheel_radius: np.float64, radius: np.float64, height: np.float64, colour: NDArray[np.float64]):
        super().__init__ (pos, radius, height, rot, colour)
        self.id                  = id
        self.init_pos            = np.zeros(3)
        self.init_rot            = np.zeros(3)
        self.old_pos             = np.zeros(3)
        self.old_rot             = np.zeros(3)
        self.linear_vel          = linear_vel # [0] left wheel - [1] right wheel 
        
        np.copyto(self.init_pos, pos )
        np.copyto(self.init_rot, rot )
        np.copyto(self.old_pos,  pos )
        np.copyto(self.old_rot,  rot )
        
        self.wheel_distance      = wheel_distance 
        self.wheel_radius        = wheel_radius

        self.pos_noise            = 0.0 #0.01
        self.rot_noise            = 0.0 #np.pi/360.0 #  about 0.0087
        self.vel_noise            = 0.0
    
    def add_noise( self, element: NDArray[np.float64], noise_level: np.float64, high_threshold: np.float64, low_threshold: np.float64):
        for i in range(len(element)):
            element[i] += 2.0 * np.random.rand() * noise_level - noise_level
            if( element[i] > high_threshold ):
                element[i] = high_threshold 
            elif ( element[i] < low_threshold ):
                element[i] = low_threshold
    
    def update_old_pos_rot( self ):
        np.copyto(self.old_pos, self.pos )
        np.copyto(self.old_rot, self.rot )
                
    def make_movement(self, to_motors: NDArray[np.float64]):
        self.update_old_pos_rot()
        self.add_noise( to_motors, self.vel_noise, 1.0, 0.0)
        self.move( to_motors )
        # self.add_noise( self.pos, self.pos_noise, 1.0, 0.0)
        self.add_noise( self.rot, self.rot_noise, 2.0*np.pi, 0.0)
                
    def move( self, to_motors: NDArray[np.float64] ):
        rot_speed = np.zeros(2)
        for i in range(len(to_motors)):
            # wheel rotation speed set between [-2PI, 2PI]
            rot_speed[i] = to_motors[i] * (4.0 * np.pi) - (2.0 * np.pi) 
            
        # Set linear velocity from wheels' rotational speed
        for i in range (len(self.linear_vel)):
            self.linear_vel[i] = rot_speed[i] * self.wheel_radius
        
        if ( self.linear_vel[0] != self.linear_vel[1] ):
            R = (0.5 * self.wheel_distance) * ((self.linear_vel[1]+self.linear_vel[0])/(self.linear_vel[1]-self.linear_vel[0]))
            omega = (self.linear_vel[1]-self.linear_vel[0])/self.wheel_distance
            ICC_x = self.pos[0] - R * np.sin(self.rot[2])
            ICC_y = self.pos[1] + R * np.cos(self.rot[2])
            omegaDeltaT = omega * Diff_drive_robot.delta_t
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
            a4 = np.add(a1 * Diff_drive_robot.delta_t, a2)
            
        self.pos[0] = a4[0]
        self.pos[1] = a4[1]
        self.rot[2] = a4[2]
        if ( self.rot[2] >= (2.0 * np.pi) ) :
            self.rot[2] -= (2.0 * np.pi)
        if ( self.rot[2] < 0.0 ) :
            self.rot[2] += (2.0 * np.pi)
    
    def draw(self ):
        super().draw ()
