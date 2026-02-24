import numpy as np
from numpy.typing import NDArray

from SENSORS.ultrasonic_sensors import Ultrasonic_sensors
from SENSORS.ir_comm import IRComm
from SENSORS.music_module import MusicModule
from MIDI.midi_recorder import MidiRecorder
from SENSORS.ir_comm import IRComm, IRCommConfig
from TOOLS.angle_to_sector import angle_to_sector
from TOOLS.note_to_color import note_to_color


from WORLD.shapes import Diff_drive_robot
from OpenGL.GLU import gluNewQuadric, gluCylinder, gluDisk
from OpenGL.GL import (
    glPushMatrix, glPopMatrix, glTranslatef,
    glDisable, glEnable, glLineWidth, glColor3f,
    glBegin, glEnd, glVertex3f,glRotatef, GL_LINES, GL_LIGHTING, GL_QUADS
)

class MusicBot(Diff_drive_robot):
    # --- Dimensions (mètres) ---
    robot_radius = 0.150   # 300 mm diameter
    robot_height = 0.050   # 50 mm high

    wheel_distance = 0.250   # placeholder (m) TO Change
    wheel_radius   = 0.030   # placeholder (m)

    def __init__(self, id: np.int64, pos: NDArray[np.float64], rot: NDArray[np.float64],
                  colour: NDArray[np.float64], linear_vel: NDArray[np.float64],midi_recorder: MidiRecorder | None = None):
        
        super().__init__(id=id, pos=pos, rot=rot, linear_vel=linear_vel, wheel_distance=MusicBot.wheel_distance,
                          wheel_radius=MusicBot.wheel_radius, radius=MusicBot.robot_radius, height=MusicBot.robot_height, colour=colour)

        # --- Ultrasonic sensors ---
        self.Dst_rd = Ultrasonic_sensors()

        # --- IR communication  ---
        self.ir_comm = IRComm(robot_id=int(self.id), config=IRCommConfig( 
            range_m=0.5,
            robot_rad_m=MusicBot.robot_radius,
            fov_deg=180.0,
            max_process_rate_s=6.0,
            max_inbox=3,
            msg_ttl_s = 0.5,
            drop_prob=0.0,       
            enabled=True         
        ))

        # --- Music module ---
        self.music = MusicModule(channel_id = int(self.id))  # Each robot has its own music channel

        # --- MIDI recorder ---
        self.midi_recorder = midi_recorder  

        # led state
        self.time_s : float = 0.0
        self.led_until_s : float = 0.0
        self.led_color : tuple[float, float, float] = (0.0, 0.0, 0.0)


    #update all sensor of th robot    
    def update_sensors(self) -> None:
        self.update_ultrasonic_sensors()
        # Placeholder for IR communication and music module updates

    def update_ultrasonic_sensors(self) -> None:
        self.Dst_rd.update_sensors(self.id)
    
    def play_note(self, note: int, duration_s: float, volume: float = 1.0, now_s: float|None = None):
        # audio
        self.music.play_note(note, duration_s, volume) 

        self.led_color = note_to_color(note)
        if now_s is not None:
            self.led_until_s = now_s + duration_s

        # midi recording
        if self.midi_recorder is not None and now_s is not None:
            self.midi_recorder.record_note(
                track_id=self.id,          # 1 track par robot
                pitch=note,
                start_s=now_s,
                duration_s=duration_s,
                volume_0_1=volume
            )
    def draw_led(self):

        # éteint si expiré
        if self.time_s >= self.led_until_s:
            self.led_color = (0.1, 0.1, 0.1)

        r, g, b = self.led_color

        rear_offset = self.radius * 0.65
        x = -rear_offset * np.cos(self.rot[2])
        z = -rear_offset * np.sin(self.rot[2])

        led_radius = 0.018
        led_height = 0.005
        y = self.height + 0.002

        glDisable(GL_LIGHTING)
        glColor3f(r, g, b)
        glPushMatrix()
        glTranslatef(x, y, z)
        glRotatef(-90.0, 1.0, 0.0, 0.0)
        quad = gluNewQuadric()
        gluDisk(quad, 0.0, led_radius, 16, 1)
        gluCylinder(quad, led_radius, led_radius, led_height, 16, 1)
        glTranslatef(0.0, 0.0, led_height)
        gluDisk(quad, 0.0, led_radius, 16, 1)
        glPopMatrix()
        glEnable(GL_LIGHTING)

    def draw(self):
        
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

        self.draw_led()

        # draw IR communication rays 
        # TODO: montrer quel secteur est actif (ex: rouge si message reçu dans ce secteur)
        ir_fov_rad = np.deg2rad(self.ir_comm.cfg.fov_deg)
        n_sectors = self.ir_comm.cfg.num_captors
        sector_angle = ir_fov_rad / float(n_sectors)
        len = self.ir_comm.cfg.range_m
        for i in range(n_sectors+1):
            angle = self.rot[2] - 0.5 * ir_fov_rad + float(i) * sector_angle
            x = self.radius * np.cos(angle)
            z = self.radius * np.sin(angle)
            len
            glDisable(GL_LIGHTING)
            glLineWidth(1.0)
            glColor3f(1.0, 0.0, 0.0)  #red
            glBegin(GL_LINES)
            glVertex3f(x, self.height + 0.002, z)
            glVertex3f(x + len * np.cos(angle), self.height + 0.002, z + len * np.sin(angle))
            glEnd()
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
