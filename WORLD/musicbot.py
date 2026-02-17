import numpy as np
from numpy.typing import NDArray

from SENSORS.ultrasonic_sensors import Ultrasonic_sensors
from SENSORS.ir_comm import IRComm
from SENSORS.music_module import MusicModule

from WORLD.shapes import Diff_drive_robot
from OpenGL.GL import (
    glPushMatrix, glPopMatrix, glTranslatef,
    glDisable, glEnable, glLineWidth, glColor3f,
    glBegin, glEnd, glVertex3f, GL_LINES, GL_LIGHTING
)

class MusicBot(Diff_drive_robot):
    # --- Dimensions (mètres) ---
    robot_radius = 0.150   # 300 mm diameter
    robot_height = 0.050   # 50 mm high

    wheel_distance = 0.250   # placeholder (m) TO Change
    wheel_radius   = 0.030   # placeholder (m)

    def __init__(self, id: np.int64, pos: NDArray[np.float64], rot: NDArray[np.float64], colour: NDArray[np.float64], linear_vel: NDArray[np.float64]):
        super().__init__(id=id, pos=pos, rot=rot, linear_vel=linear_vel, wheel_distance=MusicBot.wheel_distance, wheel_radius=MusicBot.wheel_radius, radius=MusicBot.robot_radius, height=MusicBot.robot_height, colour=colour)

        self.Dst_rd = Ultrasonic_sensors()

        # --- IR communication (placeholder) ---
        self.ir_comm = IRComm()

        # --- Music module (placeholder) ---
        self.music = MusicModule()

    #update all sensor of th robot    
    def update_sensors(self) -> None:
        self.update_ultrasonic_sensors()
        # Placeholder for IR communication and music module updates

    def update_ultrasonic_sensors(self) -> None:
        self.Dst_rd.update_sensors(self.id)

    def draw(self):
        # Draw body (Diff_drive_robot / Cylinder etc.)
        super().draw()

        glPushMatrix()
        glTranslatef(self.pos[0], self.pos[2], self.pos[1])

        # Heading line
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

        # Draw ultrasonic rays
        new_angle = np.zeros(self.Dst_rd.nb_sensors)
        full_len  = np.zeros(self.Dst_rd.nb_sensors)

        for i in range(self.Dst_rd.nb_sensors):
            new_angle[i] = self.rot[2] + self.Dst_rd.us_angle[i]
            new_angle[i] = new_angle[i] % (2.0 * np.pi)

            full_len[i] = self.radius + self.Dst_rd.distance[i]
            x = full_len[i] * np.cos(new_angle[i])
            z = full_len[i] * np.sin(new_angle[i])

            glDisable(GL_LIGHTING)
            glLineWidth(2.0)
            glColor3f(0.0, 0.6, 1.0)  # bleu clair pour sonar
            glBegin(GL_LINES)
            glVertex3f(0.0, self.height - 0.002, 0.0)
            glVertex3f(x, self.height - 0.002, z)
            glEnd()
            glLineWidth(1.0)
            glEnable(GL_LIGHTING)

        glPopMatrix()
