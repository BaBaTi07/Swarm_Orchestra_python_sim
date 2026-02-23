import numpy as np
from numpy.typing import NDArray

from SENSORS.ultrasonic_sensors import Ultrasonic_sensors
from SENSORS.ir_comm import IRComm
from SENSORS.music_module import MusicModule
from MIDI.midi_recorder import MidiRecorder
from SENSORS.ir_comm import IRComm, IRCommConfig


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

    def __init__(self, id: np.int64, pos: NDArray[np.float64], rot: NDArray[np.float64],
                  colour: NDArray[np.float64], linear_vel: NDArray[np.float64],midi_recorder: MidiRecorder | None = None):
        
        super().__init__(id=id, pos=pos, rot=rot, linear_vel=linear_vel, wheel_distance=MusicBot.wheel_distance,
                          wheel_radius=MusicBot.wheel_radius, radius=MusicBot.robot_radius, height=MusicBot.robot_height, colour=colour)

        # --- Ultrasonic sensors ---
        self.Dst_rd = Ultrasonic_sensors()

        # --- IR communication (placeholder) ---
        self.ir_comm = IRComm(robot_id=int(self.id), config=IRCommConfig( 
            range_m=0.5,
            fov_deg=180.0,
            max_process_rate_s=6.0,
            max_inbox=64,
            drop_prob=0.0,       
            enabled=True         
        ))

        # --- Music module ---
        self.music = MusicModule(channel_id = int(self.id))  # Each robot has its own music channel

        # --- MIDI recorder ---
        self.midi_recorder = midi_recorder  


    #update all sensor of th robot    
    def update_sensors(self) -> None:
        self.update_ultrasonic_sensors()
        # Placeholder for IR communication and music module updates

    def update_ultrasonic_sensors(self) -> None:
        self.Dst_rd.update_sensors(self.id)
    
    def play_note(self, note: str, duration_s: float, volume: float = 1.0, now_s: float|None = None):
        # audio
        self.music.play_note(note, duration_s, volume) 
        # midi recording
        if self.midi_recorder is not None and now_s is not None:
            self.midi_recorder.record_note(
                track_id=self.id,          # 1 track par robot
                pitch=note,
                start_s=now_s,
                duration_s=duration_s,
                volume_0_1=volume
            )


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
