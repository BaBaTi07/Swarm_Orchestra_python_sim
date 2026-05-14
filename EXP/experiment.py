import numpy as np
from CONTROL.fsm import *
from CONTROL.SwarmMusicFsm import *
from WORLD.arena import *
from TOOLS.logger import logger
from MIDI.midi_recorder import MidiRecorder
from WORLD.musicbot import MusicBot
from SENSORS.ir_comm import IRMedium, IRCommConfig
from TOOLS.plot_gen import *
from CONTROL.sync_algo import SyncAlgo
from TOOLS.evaluation import evaluate_musical_quality
from TOOLS.qualityScoresHistory import QualityScoresHistory
from TOOLS.bounded_normal import bounded_normal
from datetime import datetime

class Exp( ):
    num_trials     = 0
    num_iterations = 0
    my_controller  = []
    trial         = 0
    iter          = 0
    dt_s          = 0.2
    sim_time_s     = 0.0
    training_mode  = False
    training_args  = {
        "note_memory_ttl_s": 88.9,
        "chord_memory_ttl_s": 0.2,
        "beat_memory_ttl_s": 37.6,
        "dominant_beat_window_s": 16.0,
        "chord_commitment_ttl_s": 15.4,
        "chord_create_probability": 0.18,
        "chord_creation_score": 0.015,
        "chord_beat_join_boost": 2.5,
        "candidate_scale_threshold": 0.87,
        "disambiguation_probability": 0.67,
        "min_stable_scale_updates": 1,
    }

    best_training_args = training_args.copy()
    good_training_args_history = []
    name           = None
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

    qualityScoresHistory = QualityScoresHistory()

    def set_training_mode(training_mode: bool):
        Exp.training_mode = training_mode

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
        Exp.qualityScoresHistory = QualityScoresHistory()

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
                if not Exp.training_mode:
                    Exp.my_controller[rb.id] = SwarmMusicFsm(0.6, 50)
                else: 
                    Exp.my_controller[rb.id] = SwarmMusicFsm(0.6, 50, training_args=Exp.training_args)
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
                result=evaluate_musical_quality(Exp.current_phase_sync_history, Exp.current_notes_history, Exp.current_beat_played_history, base_name=Exp.name if Exp.name else f"trial_{Exp.trial}", folder="metrics/quality/EXP", plot=True)
                Exp.qualityScoresHistory.add_scores(result["display_scores"] | {"final_score": result["final_score"]})
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
            Exp.qualityScoresHistory.plot_all_score_history(Exp.name if Exp.name else f"trial_{Exp.trial}", "metrics/quality/EXP/multiple_trials")
            
            return False
        else:
            return True
    
    def exp_engine(mute=False):
        if not Exp.training_mode:
            Exp.init_all_trials()
            while ( Exp.finalise_all_trials() ):
                Exp.init_single_trial()
                while ( Exp.finalise_single_trial() ):
                    Exp.make_iteration(mute)
        else:
            best_score = 0
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            logger.log("WRITE", f"--------Nouvelle entrainement - {timestamp} ----------")
            while best_score < 0.9: # arbitrary threshold to stop training, can be changed or made into an argument
                Exp.init_all_trials()
                n = 0
                total_score = 0
                while ( Exp.finalise_all_trials() ):
                    Exp.init_single_trial()

                    while ( Exp.finalise_single_trial() ):
                        Exp.make_iteration(mute)

                    last_score = Exp.qualityScoresHistory.get_final_score_history()[-1] if Exp.qualityScoresHistory.get_final_score_history() else 0
                    total_score += last_score
                    n += 1  
                    if last_score < 0.6:
                        
                        logger.log("WRITE", f"Trial {Exp.trial}, Iteration {Exp.iter}, Score: {last_score:.3f} - Not good enough, moving to next trial.")
                        break # if the score is not good enough, we can stop the current trial and start a new one to save time during training
                    
                    if n > 3 and total_score/n < 0.8*best_score:
                        logger.log("WRITE", f"Trial {Exp.trial}, Iteration {Exp.iter}, Average Score: {total_score/n:.3f} - Average score is low after 3 trials, moving to next training.")
                        break # if after 3 trials the average score can not compete with the best_score, moving on
                    
                    if n > 5 and total_score / n < 0.9*best_score: # if after 5 trials the average score can not compete with the best_score, moving on
                        logger.log("WRITE", f"Trial {Exp.trial}, Iteration {Exp.iter}, Average Score: {total_score/n:.3f} - Average score is low after 5 trials, moving to next training.")
                        break
                    
                    print(last_score)
                    logger.log("WRITE", f"Trial {Exp.trial} ended with final score: {last_score:.3f}")
                
                mean_score = total_score / n if n > 0 else 0  
                if mean_score > best_score:
                    best_score = mean_score
                    Exp.best_training_args = Exp.training_args.copy()
                    logger.log("WRITE", f"$$$$ New best score achieved: {best_score:.3f} with training args: {Exp.training_args}")
                    #remove old training (max 3)
                    if len(Exp.good_training_args_history) >= 3:
                        Exp.good_training_args_history = Exp.good_training_args_history[-3:]
                        
                elif mean_score > best_score * 0.96: # if the score is close to the best score, we can still consider it as an improvement and keep the training args
                    Exp.good_training_args_history.append(Exp.training_args.copy())
                    logger.log("WRITE", f"$$$$ New good score achieved: {mean_score:.3f} with training args: {Exp.training_args}, but not better than the best score: {best_score:.3f}")
                else:
                    logger.log("WRITE", f"Current score {mean_score:.3f} does not improve the best score: {best_score:.3f}")
                
                # update training args for the next trial     
                Exp.training_args = {
                    "note_memory_ttl_s": bounded_normal(Exp.best_training_args["note_memory_ttl_s"], 2.0, 0.0, 100.0),
                    "chord_memory_ttl_s": bounded_normal(Exp.best_training_args["chord_memory_ttl_s"], 2.0, 0.0, 100.0),
                    "beat_memory_ttl_s": bounded_normal(Exp.best_training_args["beat_memory_ttl_s"], 2.0, 0.0, 100.0),
                    "dominant_beat_window_s": bounded_normal(Exp.best_training_args["dominant_beat_window_s"], 2.0, 0.0, 100.0),
                    "chord_commitment_ttl_s": bounded_normal(Exp.best_training_args["chord_commitment_ttl_s"], 2.0, 0.0, 100.0),
                    "chord_create_probability": bounded_normal(Exp.best_training_args["chord_create_probability"], 0.02, 0.0, 1.0),
                    "chord_creation_score": bounded_normal(Exp.best_training_args["chord_creation_score"], 0.2, 0.0, 10.0),
                    "chord_beat_join_boost": bounded_normal(Exp.best_training_args["chord_beat_join_boost"], 0.2, 0.0, 10.0),
                    "candidate_scale_threshold": bounded_normal(Exp.best_training_args["candidate_scale_threshold"], 0.02, 0.0, 1.0),
                    "disambiguation_probability": bounded_normal(Exp.best_training_args["disambiguation_probability"], 0.02, 0.0, 1.0),
                    "min_stable_scale_updates": int(bounded_normal(Exp.best_training_args["min_stable_scale_updates"], 0.2, 0.0, 10.0)),
                }

                #pick random good training args from the history to replace some current training args
                if len(Exp.good_training_args_history) > 0:
                    random_good_args = Exp.good_training_args_history[np.random.randint(len(Exp.good_training_args_history))]
                    for key in Exp.training_args.keys():
                        if np.random.rand() < 0.3: # 30% chance to replace the current training arg with a good one from the history
                            Exp.training_args[key] = random_good_args[key]
                            logger.log("WRITE", f"Randomly replaced training arg '{key}' with value from good history: {random_good_args[key]}")

                logger.log("WRITE", f"Updated training args for next trial: {Exp.training_args}")

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
            if hasattr(Exp.my_controller[rb.id], "sync_algo") and hasattr(Exp.my_controller[rb.id].sync_algo, "theta"):
                thetas.append(Exp.my_controller[rb.id].sync_algo.theta)

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
            kuramoto_conf_min = np.min([Exp.my_controller[rb.id].sync_algo.kuramoto_conf for rb in Arena.robot if hasattr(Exp.my_controller[rb.id], "sync_algo") and hasattr(Exp.my_controller[rb.id].sync_algo, "kuramoto_conf")])
            kuramoto_conf_max = np.max([Exp.my_controller[rb.id].sync_algo.kuramoto_conf for rb in Arena.robot if hasattr(Exp.my_controller[rb.id], "sync_algo") and hasattr(Exp.my_controller[rb.id].sync_algo, "kuramoto_conf")])
            kuramoto_conf_mean = np.mean([Exp.my_controller[rb.id].sync_algo.kuramoto_conf for rb in Arena.robot if hasattr(Exp.my_controller[rb.id], "sync_algo") and hasattr(Exp.my_controller[rb.id].sync_algo, "kuramoto_conf")])
            if sync is not None:
                Exp.current_phase_sync_history.append((now_s, sync, kuramoto_conf_min, kuramoto_conf_mean, kuramoto_conf_max))
                logger.log("TIME", f"sync={sync:.3f}")

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
                rb.play_note((music_event[0]%24)+56, music_event[1], volume=music_event[2], now_s=now_s, mute=mute)
                Exp.current_notes_history.append((now_s, music_event[0]))
                if Exp.my_controller[rb.id].beat_to_play is not None:
                    Exp.current_beat_played_history.append((now_s, Exp.my_controller[rb.id].beat_to_play))

        Exp.iter += 1
        Exp.sim_time_s += dt_s