from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from OpenGL.GL import *
from OpenGL.GLU import *
import math
import numpy as np

from world.world import Arena
from exp.experiment_1 import Exp

# from OpenGL.GLU import *
# from OpenGL.GLUT import glutInit, glutWireSphere


class Viewer(QOpenGLWidget):
    def __init__(self, my_exp: Exp ):
        super().__init__()
        self.my_exp = my_exp
        self.cam_x  = 0.0
        self.cam_y  = 0.0
        self.cam_dist = 10.0
        self.cam_rot_x = 25.0
        self.cam_rot_y = -30.0
        self.last_pos = None
        self.setMouseTracking(True)


    # ---------- OpenGL setup ----------
    def initializeGL(self):
        glClearColor(0.15, 0.15, 0.18, 1.0)
        glEnable(GL_DEPTH_TEST)
        glDisable(GL_CULL_FACE)
        # ---- Uniform Lighting ----
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        # Luce ambientale globale al 100% (niente ombre)
        glLightModelfv(
            GL_LIGHT_MODEL_AMBIENT,
            (1.0, 1.0, 1.0, 1.0)
        )
        # Niente luce direzionale
        glLightfv(GL_LIGHT0, GL_DIFFUSE, (0.0, 0.0, 0.0, 1.0))
        glLightfv(GL_LIGHT0, GL_SPECULAR, (0.0, 0.0, 0.0, 1.0))
        # Permette a glColor di funzionare con le luci accese
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)

    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, w / max(1, h), 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)
        
    # ---------- Rendering ----------
    def paintGL(self):
        # print(" IN PAINT GL ")
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()  
        
        glTranslatef(-self.cam_x, -self.cam_y, -self.cam_dist)
        glRotatef(self.cam_rot_x, 1, 0, 0)
        glRotatef(self.cam_rot_y, 0, 1, 0)

        self.draw_floor()
        
        # ---- draw arena here ----
        glPushMatrix()
        glTranslatef(Arena.perimetral_wall_xyz_radius_height_colour[0], Arena.perimetral_wall_xyz_radius_height_colour[2], Arena.perimetral_wall_xyz_radius_height_colour[1]) 
        self.draw_arena_wall(Arena.perimetral_wall_xyz_radius_height_colour[3] , Arena.perimetral_wall_xyz_radius_height_colour[4], Arena.perimetral_wall_xyz_radius_height_colour[5:8])
        glPopMatrix()
        
        # ---- draw cylinders here ----
        for id in range(len(Arena.round_obst_xyz_radius_height_colour)):
            glPushMatrix()
            glTranslatef(Arena.round_obst_xyz_radius_height_colour[id][0], Arena.round_obst_xyz_radius_height_colour[id][2], Arena.round_obst_xyz_radius_height_colour[id][1]) 
            self.draw_cylinder(Arena.round_obst_xyz_radius_height_colour[id][3], Arena.round_obst_xyz_radius_height_colour[id][4],Arena.round_obst_xyz_radius_height_colour[id][5:8])
            glPopMatrix()
        
        # ---- draw epuck robots here ----
        for id in range(len(self.my_exp.epuck)):
            glPushMatrix()
            glTranslatef(self.my_exp.epuck[id].pos[0], self.my_exp.epuck[id].pos[2], self.my_exp.epuck[id].pos[1])  
            self.draw_robot(id)
            glPopMatrix()
            
        # ---- draw cubes here ----
        for id in range(len(Arena.cuboid_obst_xyz_lwh_rxryrz_colour)):
            # Translation
            glTranslatef(Arena.cuboid_obst_xyz_lwh_rxryrz_colour[id][0], Arena.cuboid_obst_xyz_lwh_rxryrz_colour[id][1], Arena.cuboid_obst_xyz_lwh_rxryrz_colour[id][2])
            # Rotation
            glRotatef(Arena.cuboid_obst_xyz_lwh_rxryrz_colour[id][6], 1.0, 0.0, 0.0)
            glRotatef(Arena.cuboid_obst_xyz_lwh_rxryrz_colour[id][7], 0.0, 0.0, 1.0)
            glRotatef(Arena.cuboid_obst_xyz_lwh_rxryrz_colour[id][8], 0.0, 1.0, 0.0)
            glScalef(Arena.cuboid_obst_xyz_lwh_rxryrz_colour[id][3], Arena.cuboid_obst_xyz_lwh_rxryrz_colour[id][5], Arena.cuboid_obst_xyz_lwh_rxryrz_colour[id][4])
            self.draw_cube(Arena.cuboid_obst_xyz_lwh_rxryrz_colour[id][9:12])
        
    def draw_arena_wall(self, radius, height, colour, slices=48):
        glColor3f(colour[0], colour[1], colour[2])  # green
        # glMaterialfv(GL_FRONT, GL_DIFFUSE, (0.9, 0.6, 0.2, 1))
        # half_h = height / 2.0
        step = 2 * math.pi / slices
        glBegin(GL_QUAD_STRIP)
        for i in range(slices + 1):
            angle = i * step
            x = math.cos(angle)
            z = math.sin(angle)
            # Normale (radiale)
            glNormal3f(x, 0, z)
            # Vertice basso
            glVertex3f(radius * x, 0.0, radius * z)
            # Vertice alto
            glVertex3f(radius * x, height, radius * z)
        glEnd()
        
    def draw_robot(self, id, slices=64):
        glColor3f(self.my_exp.epuck[id].colour[0], self.my_exp.epuck[id].colour[1], self.my_exp.epuck[id].colour[2])  
        step = 2 * math.pi / slices
        
        # ---- Cylinder body ----
        glBegin(GL_QUAD_STRIP)
        for i in range(slices + 1):
            a = i * step
            x = math.cos(a)
            z = math.sin(a)
            glNormal3f(x, 0, z)
            glVertex3f(self.my_exp.epuck[id].robot_radius * x, 0.0, self.my_exp.epuck[id].robot_radius * z)
            glVertex3f(self.my_exp.epuck[id].robot_radius * x, self.my_exp.epuck[id].robot_height, self.my_exp.epuck[id].robot_radius * z)
        glEnd()

        # ---- Lower side ----
        glNormal3f(0, -1, 0)
        glBegin(GL_TRIANGLE_FAN)
        glVertex3f(0, 0, 0)
        for i in range(slices + 1):
            a = -i * step
            glVertex3f(self.my_exp.epuck[id].robot_radius * math.cos(a), 0.0, self.my_exp.epuck[id].robot_radius * math.sin(a))
        glEnd()
        # ---- Upper side ----
        glNormal3f(0, 1, 0)
        glBegin(GL_TRIANGLE_FAN)
        glVertex3f(0, self.my_exp.epuck[id].robot_height, 0)
        for i in range(slices + 1):
            a = i * step
            glVertex3f(self.my_exp.epuck[id].robot_radius * math.cos(a), self.my_exp.epuck[id].robot_height, self.my_exp.epuck[id].robot_radius * math.sin(a))
        glEnd()
        
        # ---- Draw epuck headings on the cylinder upper side ----
        x = self.my_exp.epuck[id].robot_radius * math.cos(self.my_exp.epuck[id].rot[2])
        z = self.my_exp.epuck[id].robot_radius * math.sin(self.my_exp.epuck[id].rot[2])
        glDisable(GL_LIGHTING)
        glLineWidth(3.0)
        glColor3f(1.0, 1.0, 0.0)
        glBegin(GL_LINES)
        glVertex3f(0.0, self.my_exp.epuck[id].robot_height + 0.001, 0.0)
        glVertex3f(x, self.my_exp.epuck[id].robot_height + 0.001, z)
        glEnd()
        glLineWidth(1.0)    
        glEnable(GL_LIGHTING)
        
        # ---- Draw epuck ir sensors rays ----
        for ir in range(self.my_exp.epuck[id].IR.nb_sensors):
            ir_angle = self.my_exp.epuck[id].rot[2] + self.my_exp.epuck[id].IR.ir_angle[ir]
            if ir_angle < 0.0:
                ir_angle += 2.0*np.pi
            elif ir_angle > (2.0 * np.pi):
                ir_angle -= (2.0 * np.pi)
            full_lenght = self.my_exp.epuck[id].robot_radius + self.my_exp.epuck[id].IR.distance[ir]
            
            x = full_lenght * math.cos(ir_angle)
            z = full_lenght * math.sin(ir_angle)
            glDisable(GL_LIGHTING)
            glLineWidth(3.0)
            glColor3f(1.0, 0.0, 0.0)
            glBegin(GL_LINES)
            glVertex3f(0.0, self.my_exp.epuck[id].robot_height - 0.002, 0.0)
            glVertex3f(x, self.my_exp.epuck[id].robot_height - 0.002, z)
            glEnd()
            glLineWidth(1.0)    
            glEnable(GL_LIGHTING)
            
    def draw_cylinder(self, cyl_radius,cyl_height, cyl_colour, slices=64):
        glColor3f(cyl_colour[0], cyl_colour[1], cyl_colour[2])  
        step = 2 * math.pi / slices
        
        # ---- Cylinder body ----
        glBegin(GL_QUAD_STRIP)
        for i in range(slices + 1):
            a = i * step
            x = math.cos(a)
            z = math.sin(a)
            glNormal3f(x, 0, z)
            glVertex3f(cyl_radius * x, 0.0, cyl_radius * z)
            glVertex3f(cyl_radius * x, cyl_height, cyl_radius * z)
        glEnd()

        # ---- Lower side ----
        glNormal3f(0, -1, 0)
        glBegin(GL_TRIANGLE_FAN)
        glVertex3f(0, 0, 0)
        for i in range(slices + 1):
            a = -i * step
            glVertex3f(cyl_radius * math.cos(a), 0.0, cyl_radius * math.sin(a))
        glEnd()
        # ---- Upper side ----
        glNormal3f(0, 1, 0)
        glBegin(GL_TRIANGLE_FAN)
        glVertex3f(0, cyl_height, 0)
        for i in range(slices + 1):
            a = i * step
            glVertex3f(cyl_radius * math.cos(a), cyl_height, cyl_radius * math.sin(a))
        glEnd()
        
        
    def draw_cube(self, cube_colour):
        glColor3f(cube_colour[0], cube_colour[1], cube_colour[2])
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


    # def draw_cube(self, cube_dims, cube_colour):
    #     glBegin(GL_QUADS)

    #     # Faccia frontale
    #     glColor3f(cube_colour[0], cube_colour[1], cube_colour[2])
    #     glVertex3f(-1, -1,  1)
    #     glVertex3f( 1, -1,  1)
    #     glVertex3f( 1,  1,  1)
    #     glVertex3f(-1,  1,  1)

    #     # Faccia posteriore
    #     glVertex3f(-1, -1, -1)
    #     glVertex3f(-1,  1, -1)
    #     glVertex3f( 1,  1, -1)
    #     glVertex3f( 1, -1, -1)

    #     # Faccia sinistra
    #     glVertex3f(-1, -1, -1)
    #     glVertex3f(-1, -1,  1)
    #     glVertex3f(-1,  1,  1)
    #     glVertex3f(-1,  1, -1)

    #     # Faccia destra
    #     glVertex3f(1, -1, -1)
    #     glVertex3f(1,  1, -1)
    #     glVertex3f(1,  1,  1)
    #     glVertex3f(1, -1,  1)

    #     # Faccia superiore
    #     glVertex3f(-1, 1, -1)
    #     glVertex3f(-1, 1,  1)
    #     glVertex3f( 1, 1,  1)
    #     glVertex3f( 1, 1, -1)

    #     # Faccia inferiore
    #     glVertex3f(-1, -1, -1)
    #     glVertex3f( 1, -1, -1)
    #     glVertex3f( 1, -1,  1)
    #     glVertex3f(-1, -1,  1)

    #     glEnd()
    
    
    # ---------- Mouse ----------
    def mousePressEvent(self, event):
        self.last_pos = event.position()

    def mouseMoveEvent(self, event):
        if self.last_pos is None:
            return
        dx = event.position().x() - self.last_pos.x()
        dy = event.position().y() - self.last_pos.y()
        sensitivity = 0.3
        self.cam_rot_y += dx * sensitivity
        self.cam_rot_x += dy * sensitivity
        self.cam_rot_x = max(-89, min(89, self.cam_rot_x))
        self.last_pos = event.position()
        self.update()
    
    def wheelEvent(self, event):
        self.cam_dist -= event.angleDelta().y() * 0.01
        self.cam_dist = max(2.0, min(50.0, self.cam_dist))
        self.update()
    



    def mouseMoveEvent(self, event):
        if self.last_pos is None:
            return
        dx = event.position().x() - self.last_pos.x()
        dy = event.position().y() - self.last_pos.y()
        sensitivity = 0.01
        if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            # PAN
            self.cam_x -= dx * sensitivity
            self.cam_y += dy * sensitivity
        else:
            # ROTAZIONE
            self.cam_rot_y += dx * 0.3
            self.cam_rot_x += dy * 0.3
        self.last_pos = event.position()
        self.update()



    def draw_floor(self):
        glDisable(GL_LIGHTING)
        glColor3f(0.5, 0.5, 0.5)
        # pavimento uniforme
        size = 20.0
        glBegin(GL_QUADS)
        glVertex3f(-size, 0.0, -size)
        glVertex3f(size, 0.0, -size)
        glVertex3f(size, 0.0, size)
        glVertex3f(-size, 0.0, size)
        glEnd()
    
        # pavimento con griglia
        # size = 10
        # step = 1
        # glBegin(GL_LINES)
        # for i in range(-size, size + 1, step):
        #     glVertex3f(i, 0, -size)
        #     glVertex3f(i, 0, size)
        #     glVertex3f(-size, 0, i)
        #     glVertex3f(size, 0, i)
        # glEnd()

        glEnable(GL_LIGHTING)


if __name__ == "__main__":
    app = QApplication()
    viewer = Viewer()
    viewer.resize(600, 600)
    viewer.show()
    app.exec()