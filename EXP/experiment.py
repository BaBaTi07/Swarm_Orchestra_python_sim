import numpy as np
from CONTROL.fsm import *
from CONTROL.SwarmMusicFsm import *
from WORLD.arena import *
from TOOLS.logger import logger
from MIDI.midi_recorder import MidiRecorder
from WORLD.musicbot import MusicBot
from SENSORS.ir_comm import IRMedium, IRCommConfig
from TOOLS.plot_gen import *

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
    phase_sync_history = [[]]   # list of list for multiples trials
    notes_history = [[]]        # list of list
    beat_played_history = [[]]   # list of list

    current_phase_sync_history = []   # list of (time_s, sync) for the current trial
    current_notes_history = []        # list of (time_s, note) for the current trial
    current_beat_played_history = []   # list of (time_s, beat) for the current trial

    ir_medium      = IRMedium(config=IRCommConfig( 
        range_m=0.5,
        fov_deg=180.0,
        max_process_rate_s=6.0,
        max_inbox=64,
        drop_prob=0.0,       
        enabled=True         
    ))

    def set_name(name):
        Exp.name = name
 
    def reset():
        """
        reset to initial state for all trials, should be only called before starting the experiment
        not to be called between trials, use reset_single_trial instead
        """
        Exp.my_controller = []
        Exp.trial = 0
        Exp.iter  = 0
        Exp.num_trials = 0
        Exp.num_iterations = 0
        Exp.sim_time_s = 0.0
        Exp.has_music = [False]* len(Arena.robot)
        Exp.has_ir_comm = [False]* len(Arena.robot)

    def reset_single_trial():
        # reset robot position and rotation
        for e in range (len(Arena.robot)):
            id = Arena.robot[e].id
            np.copyto(Arena.robot[id].pos, Arena.robot[id].init_pos )
            np.copyto(Arena.robot[id].rot, Arena.robot[id].init_rot )

        Exp.iter = 0
        Exp.sim_time_s = 0.0

        #reset contrlers
        Exp.my_controller = [None] * len(Arena.robot)
        Exp.has_music = [False]* len(Arena.robot)
        Exp.has_ir_comm = [False]* len(Arena.robot)

        Exp.ir_medium = IRMedium(config=IRCommConfig(
            range_m=0.5,
            fov_deg=180.0,
            max_process_rate_s=6.0,
            max_inbox=64,
            drop_prob=0.0,
            enabled=True
        ))
        
        for rb in Arena.robot:
            Exp.has_music[rb.id] = hasattr(rb, 'play_note')
            Exp.has_ir_comm[rb.id] = hasattr(rb, 'ir_comm')

            if Exp.has_ir_comm[rb.id]:
                rb.ir_comm.reset()

            if Exp.has_music[rb.id]:
                Exp.my_controller[rb.id] = SwarmMusicFsm(0.6, 50)
            else:
                Exp.my_controller[rb.id] = Fsm(0.6, 50)

        #reste current history
        Exp.current_phase_sync_history = []
        Exp.current_notes_history = []
        Exp.current_beat_played_history = []

    def init_all_trials():
        Exp.trial = 0
        Exp.phase_sync_history = []
        Exp.notes_history = [] 
        Exp.beat_played_history = []
    
    def init_single_trial():
        Exp.reset_single_trial()

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
                Exp.midi.write_midi(build_filename(f"trial_{Exp.trial}", "MIDI/midi_records") )
                logger.log("INFO", f"Trial {Exp.trial} ended. MIDI file saved as 'trial_{Exp.trial}.mid'.")
                Exp.midi.stop() 
                save_beat_played_plot(Exp.current_beat_played_history, Exp.name if Exp.name else f"trial_{Exp.trial}", "metrics/beat_played")
                save_harmonic_scale_plot(Exp.current_notes_history, Exp.name if Exp.name else f"trial_{Exp.trial}", "metrics/harmonic_scales")
                # add the history to the list of history for all trials
                Exp.phase_sync_history.append(Exp.current_phase_sync_history)
                Exp.notes_history.append(Exp.current_notes_history)
                Exp.beat_played_history.append(Exp.current_beat_played_history)
            return False
        else:
            return True
    
    def finalise_all_trials( ):
        if( Exp.trial >= Exp.num_trials):
            save_sync_plot(Exp.phase_sync_history, Exp.name if Exp.name else f"trial_{Exp.trial}", "metrics/phase_sync")
            generate_multiple_execution_beat_evenness_graph(Exp.beat_played_history, Exp.name if Exp.name else f"trial_{Exp.trial}", "metrics/beat_played/multiple_trials")
            generate_multiple_execution_harmonic_graph(Exp.notes_history, Exp.name if Exp.name else f"trial_{Exp.trial}", "metrics/harmonic_scales/multiple_trials")
            return False
        else:
            return True
    
    def exp_engine(mute=False):
        Exp.init_all_trials()
        while ( Exp.finalise_all_trials() ):
            Exp.init_single_trial()
            while ( Exp.finalise_single_trial() ):
                Exp.make_iteration(mute)

    def get_ir_messages(rb, time_s: float, dt_s: float) -> list:
        msgs = []
        # Consume IR messages if the robot has a communication module
        if Exp.has_ir_comm[rb.id]:
            msgs = rb.ir_comm.consume(time_s=time_s, dt_s=dt_s)
            if msgs:
                logger.log("DEBUG", f"Robot {rb.id} received IR messages: {msgs}")
        return msgs
    
    def compute_phase_sync():
        thetas = []

        for rb in Arena.robot:
            if hasattr(Exp.my_controller[rb.id], "theta"):
                thetas.append(Exp.my_controller[rb.id].theta)

        if not thetas:
            return None

        thetas = np.array(thetas)

        r = np.abs(np.mean(np.exp(1j * thetas))) #1j = sqrt(-1)

        return r
    
    def make_iteration(mute=False):
        now_s = Exp.sim_time_s
        dt_s = Exp.dt_s

        # log when mute to folow the simulation when not viewing
        if mute and now_s%10 <= dt_s:
            logger.log("TIME", f"Iteration {Exp.iter}, Simulation time: {now_s:.2f} seconds")
        
        # Compute and log synchronization metric
        if now_s%2 <= dt_s:  
            sync = Exp.compute_phase_sync()
            kuramoto_conf_min = np.min([Exp.my_controller[rb.id].kuramoto_conf for rb in Arena.robot if hasattr(Exp.my_controller[rb.id], "kuramoto_conf")])
            kuramoto_conf_max = np.max([Exp.my_controller[rb.id].kuramoto_conf for rb in Arena.robot if hasattr(Exp.my_controller[rb.id], "kuramoto_conf")])
            kuramoto_conf_mean = np.mean([Exp.my_controller[rb.id].kuramoto_conf for rb in Arena.robot if hasattr(Exp.my_controller[rb.id], "kuramoto_conf")])
            if sync is not None:
                Exp.current_phase_sync_history.append((now_s, sync, kuramoto_conf_min, kuramoto_conf_mean, kuramoto_conf_max))
                logger.log("TIME", f"sync={sync:.3f}, Kuramoto confidence (min/mean/max)={kuramoto_conf_min:.3f}/{kuramoto_conf_mean:.3f}/{kuramoto_conf_max:.3f}")

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
                rb.play_note((music_event[0]%12)+60, music_event[1], volume=music_event[2], now_s=now_s, mute=mute)
                Exp.current_notes_history.append((now_s, music_event[0]))
                if Exp.my_controller[rb.id].beat_to_play is not None:
                    Exp.current_beat_played_history.append((now_s, Exp.my_controller[rb.id].beat_to_play))

        Exp.iter += 1
        Exp.sim_time_s += dt_s