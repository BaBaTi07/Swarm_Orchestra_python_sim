import sys
import numpy as np
import argparse
from PyQt6.QtWidgets import QApplication
from GRAPHICS.interface import MainWindow

from TOOLS.read_json import * 
from EXP.experiment import *
from WORLD.arena import *

def preamble( ):
    parser = argparse.ArgumentParser()
    parser.add_argument("-seed", type=int, help="add a custom seed for numpy random seeding")
    parser.add_argument("-file", type=str, help="add a json_files/file_name with details of the experiment")
    parser.add_argument("-viewing", type=bool, help="if true it triggers the graphical mode")
    parser.add_argument("-log", type=str, help="set the log level (DEBUG, INFO, WARN, ERROR, NONE)")
    parser.add_argument("-instrument", type=str, help="set the instrument directory (if none -> Basic sound waveforms will be used)")
    parser.set_defaults(seed = 0, viewing=False)
    args = parser.parse_args()
    
    if args.seed:
        n_seed = args.seed
    else:
        n_seed = None
        
    if not args.file:
        f_name = "json_files/experiment_0.json"
    else:
        f_name = args.file       
    
    if not args.viewing:
        flag_viewing = False
    else:
        flag_viewing = True
    
    if args.log:
        if args.log in logger.levels:
            logger.curent_level = args.log
        else:
            print(f"Invalid log level '{args.log}'. Valid levels are: {sorted(logger.levels.keys())} running with default level NONE.")
            logger.curent_level = "NONE"
    else:
        logger.curent_level = "NONE"
    
    if args.instrument:
        instrument_path = args.instrument
    else:
        instrument_path = None
        
    return n_seed, f_name, flag_viewing, instrument_path

    
if __name__ == "__main__" :
    
    n_seed = None
    flag_viewing = False
    f_name = ""
    
    n_seed, f_name, flag_viewing, instrument_path = preamble( )
    
    MusicModule.init_global(num_mixer_channels=128)
    if instrument_path is not None:
        MusicModule.set_instrument_samples(instrument_path, preload=True)
        
    seed_from_file, delta_t_ms = read_json_file(f_name)
    if n_seed == None:
        n_seed = seed_from_file    
    np.random.seed(n_seed)
    
    
    if flag_viewing:
        app = QApplication([]) # create the QApplication
        window = MainWindow(delta_t_ms) # create the main window
        window.show()
        app.exec() # start the event loop
    else:
        Exp.exp_engine()
        sys.exit()
    