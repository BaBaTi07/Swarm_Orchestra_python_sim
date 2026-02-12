from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from OpenGL.GL import *
from OpenGL.GLU import *

from WORLD.world import Arena

class Viewer(QOpenGLWidget):
    def __init__(self ):
        super().__init__()
        self.cam_x  = 0.0
        self.cam_y  = 0.0
        self.cam_dist = 10.0
        self.cam_rot_x = 25.0
        self.cam_rot_y = -30.0
        self.last_pos = None
        # self.setMouseTracking(True)

    # ---------- OpenGL setup ----------
    def initializeGL(self):
        glClearColor(0.1, 0.1, 0.1, 1.0)
        glEnable(GL_DEPTH_TEST)
        glDisable(GL_CULL_FACE)
        # Light 100%
        glLightModelfv(
            GL_LIGHT_MODEL_AMBIENT,
            (1.0, 1.0, 1.0, 1.0)
        )
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
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()  
        
        glTranslatef(-self.cam_x, -self.cam_y, -self.cam_dist)
        glRotatef(self.cam_rot_x, 1, 0, 0)
        glRotatef(self.cam_rot_y, 0, 1, 0)

        # ---- draw arena floor here ----
        self.draw_floor()
        
        # ---- draw arena perimetral wall here ----
        for r in Arena.ring:
            r.draw()
            
        # ---- draw cylinders here ----
        for cyl in Arena.cylinder:
            cyl.draw()
        
        # ---- draw cubes here ----
        for cuboid in Arena.cuboid:
            cuboid.draw()
            
        # ---- draw epuck robots here ----
        for ep in Arena.epuck:
            ep.draw()
            
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
            # ROT
            self.cam_rot_y += dx * 0.3
            self.cam_rot_x += dy * 0.3
        self.last_pos = event.position()
        self.update()

    def draw_floor(self):
        glDisable(GL_LIGHTING)
        glColor3f(0.7, 0.5, 0.5)
        # Uniform floor
        # size = 20.0
        # glBegin(GL_QUADS)
        # glVertex3f(-size, 0.0, -size)
        # glVertex3f(size, 0.0, -size)
        # glVertex3f(size, 0.0, size)
        # glVertex3f(-size, 0.0, size)
        # glEnd()
    
        # Floor with grid
        size = 10
        step = 1
        glBegin(GL_LINES)
        for i in range(-size, size + 1, step):
            glVertex3f(i, 0, -size)
            glVertex3f(i, 0, size)
            glVertex3f(-size, 0, i)
            glVertex3f(size, 0, i)
        glEnd()

        glEnable(GL_LIGHTING)


if __name__ == "__main__":
    app = QApplication()
    viewer = Viewer()
    viewer.resize(600, 600)
    viewer.show()
    app.exec()