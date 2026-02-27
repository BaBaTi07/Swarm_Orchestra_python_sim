import numpy as np
from CONTROL.fsm import *
from CONTROL.SwarmMusicFsm import *
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
    my_controller  = []
    trial         = 0
    iter          = 0
    dt_s          = 0.2
    sim_time_s     = 0.0
    has_music = [False]* len(Arena.robot)
    has_ir_comm = [False]* len(Arena.robot)
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
        Exp.my_controller = []
        Exp.trial = 0
        Exp.iter  = 0
        Exp.num_trials = 0
        Exp.num_iterations = 0
        Exp.sim_time_s = 0.0
        Exp.has_music = [False]* len(Arena.robot)
        Exp.has_ir_comm = [False]* len(Arena.robot)

    def init_all_trials():
        Exp.trial = 0
        Exp.my_controller = [None] * len(Arena.robot)
        Exp.has_music = [False]* len(Arena.robot)
        Exp.has_ir_comm = [False]* len(Arena.robot)
        
        for rb in Arena.robot:
            if hasattr(rb, 'play_note'):
                Exp.has_music[rb.id] = True
                Exp.my_controller[rb.id] = SwarmMusicFsm(0.6, 50)
            if hasattr(rb, 'ir_comm'):
                Exp.has_ir_comm[rb.id] = True
            else:
                Exp.my_controller[rb.id] = Fsm(0.6, 50)
    
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
    
    def exp_engine(mute=False):
        Exp.init_all_trials()
        while ( Exp.finalise_all_trials() ):
            Exp.init_single_trial()
            while ( Exp.finalise_single_trial() ):
                Exp.make_iteration(mute)
    
    def build_midi_filename( base_name: str, folder: str ) -> Path:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)

        return folder / f"{base_name}_{timestamp}.mid"

    def get_ir_messages(rb, time_s: float, dt_s: float) -> list:
        msgs = []
        # Consume IR messages if the robot has a communication module
        if Exp.has_ir_comm[rb.id]:
            msgs = rb.ir_comm.consume(time_s=time_s, dt_s=dt_s)
            if msgs:
                logger.log("DEBUG", f"Robot {rb.id} received IR messages: {msgs}")
        return msgs

    def make_iteration(mute=False):
        now_s = Exp.sim_time_s
        dt_s = Exp.dt_s

        # log when mute to folow the simulation when not viewing
        if mute and now_s%10 <= dt_s:
            logger.log("TIME", f"Iteration {Exp.iter}, Simulation time: {now_s:.2f} seconds")

        Exp.ir_medium.step(Arena.robot, time_s=now_s, dt_s=dt_s)

        for rb in Arena.robot:

            # Update robot's internal time (used for LED timing)
            rb.time_s = now_s  
            rb.update_sensors()

            # get wheels, music event and IR message to send from the controller
            if Exp.has_ir_comm[rb.id] or Exp.has_music[rb.id]:
                msgs = Exp.get_ir_messages(rb, now_s, dt_s)
                wheels, music_event, msg_snd = Exp.my_controller[rb.id].update( rb.Dst_rd.reading, msgs, time_s=now_s, dt_s=dt_s)
            else:
                wheels = Exp.my_controller[rb.id].update( rb.Dst_rd.reading)
                music_event = None
                msg_snd = None

            rb.make_movement(np.array(wheels))
            
            # Send IR message if any
            if msg_snd is not None and Exp.has_ir_comm[rb.id]:
                rb.ir_comm.send(payload=msg_snd, time_s=now_s)

            # Play music event if any
            if music_event is not None and Exp.has_music[rb.id]:
                logger.log("DEBUG",f"Robot {rb.id} plays note: {music_event[0]} for {music_event[1]} seconds at volume {music_event[2]}")
                rb.play_note(music_event[0], music_event[1], volume=music_event[2], now_s=now_s, mute=mute)
        Exp.iter += 1
        Exp.sim_time_s += dt_s