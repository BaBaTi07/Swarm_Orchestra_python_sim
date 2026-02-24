import numpy as np
from CONTROL.fsm import *
from WORLD.arena import *
from TOOLS.logger import logger
from MIDI.midi_recorder import MidiRecorder
from WORLD.musicbot import MusicBot
from datetime import datetime
from pathlib import Path
from SENSORS.ir_comm import IRMedium, IRCommConfig

class Exp( ):
    num_trials     = 0
    num_iterations = 0
    my_controller  = np.array([]) #This is the array where it is stored the robot controllers
    trial         = 0
    iter          = 0
    dt_s          = 0.2
    sim_time_s     = 0.0
    midi          = MidiRecorder(tempo_bpm=120.0)  # Global MIDI recorder for the experiment
    ir_medium      = IRMedium(config=IRCommConfig( 
        range_m=0.5,
        fov_deg=180.0,
        max_process_rate_s=6.0,
        max_inbox=64,
        drop_prob=0.0,       
        enabled=True         
    ))
 
    def reset():
        Exp.my_controller = np.array([])
        Exp.trial = 0
        Exp.iter  = 0
        Exp.num_trials = 0
        Exp.num_iterations = 0
        Exp.sim_time_s = 0.0

    def init_all_trials():
        Exp.trial = 0
        for e in range (len(Arena.robot)):
            Exp.my_controller = np.append(Exp.my_controller, Fsm(0.6, 50))
    
    def init_single_trial():
        for e in range (len(Arena.robot)):
            id = Arena.robot[e].id
            np.copyto(Arena.robot[id].pos, Arena.robot[id].init_pos )
            np.copyto(Arena.robot[id].rot, Arena.robot[id].init_rot )
        Exp.iter = 0
        Exp.sim_time_s = 0.0

        # Start the Midi recording if the robots are music bots
        if any(isinstance(rb, MusicBot) for rb in Arena.robot):
            
            if not Exp.midi.is_enabled():
                Exp.midi.start()  # Start MIDI recording for the trial
                logger.log("INFO", f"Trial {Exp.trial+1} started. MIDI recording enabled.")
            else:
                logger.log("WARN", f"MIDI recording was already enabled at the start of trial {Exp.trial}. This may lead to overwriting previous recordings.")
        else:
            logger.log("INFO", "MIDI recording is disabled. No MIDI file will be generated.")
        
    
    def finalise_single_trial():
        if( Exp.iter >= Exp.num_iterations):
            Exp.trial += 1
            if Exp.midi.is_enabled():
                Exp.midi.write_midi(Exp.build_midi_filename(f"trial_{Exp.trial}", "MIDI/midi_records") )
                logger.log("INFO", f"Trial {Exp.trial} ended. MIDI file saved as 'trial_{Exp.trial}.mid'.")
                Exp.midi.stop() 
            return False
        else:
            return True
    
    def finalise_all_trials( ):
        if( Exp.trial >= Exp.num_trials):
            return False
        else:
            return True
    
    def exp_engine():
        Exp.init_all_trials()
        while ( Exp.finalise_all_trials() ):
            Exp.init_single_trial()
            while ( Exp.finalise_single_trial() ):
                Exp.make_iteration()
    
    def build_midi_filename( base_name: str, folder: str ) -> Path:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)

        return folder / f"{base_name}_{timestamp}.mid"

    def get_ir_messages(rb, time_s: float, dt_s: float) -> list:
        msgs = []
        # Consume IR messages if the robot has a communication module
        if hasattr(rb, 'ir_comm') and rb.ir_comm is not None:
            msgs = rb.ir_comm.consume(time_s=time_s, dt_s=dt_s)
            if msgs:
                logger.log("DEBUG", f"Robot {rb.id} received IR messages: {msgs}")
        return msgs

    def make_iteration():
        now_s = Exp.sim_time_s
        dt_s = Exp.dt_s

        Exp.ir_medium.step(Arena.robot, time_s=now_s, dt_s=dt_s)

        for rb in Arena.robot:

            # Get IR messages for this robot
            msgs = Exp.get_ir_messages(rb, now_s, dt_s)
            
            rb.update_sensors()

            # get wheels, music event and IR message to send from the controller
            wheels, music_event, msg_snd = Exp.my_controller[rb.id].update( rb.Dst_rd.reading, msgs)

            rb.make_movement(np.array(wheels))
            
            # Send IR message if any
            if msg_snd is not None and hasattr(rb, 'ir_comm') and rb.ir_comm is not None:
                rb.ir_comm.send(payload=msg_snd, time_s=now_s)

            # Play music event if any
            if music_event is not None and hasattr(rb, 'play_note'):
                logger.log("DEBUG",f"Robot {rb.id} plays note: {music_event[0]} for {music_event[1]} seconds at volume {music_event[2]}")
                rb.play_note(music_event[0], music_event[1], volume=music_event[2], now_s=now_s)
        Exp.iter += 1
        Exp.sim_time_s += dt_s