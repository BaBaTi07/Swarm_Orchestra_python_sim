import sys
import numpy as np
import argparse
from PyQt6.QtWidgets import QApplication
from GRAPHICS.interface import MainWindow

from json_files.read_json import * 
from EXP.experiment import *
from WORLD.arena import *

def preamble( ):
    parser = argparse.ArgumentParser()
    parser.add_argument("-seed", type=int, help="add a custom seed for numpy random seeding")
    parser.add_argument("-file", type=str, help="add a required json_files/file_name with details of the experiment")
    parser.add_argument("-viewing", type=bool, help="if true it triggers the graphical mode")
    parser.set_defaults(seed = 0, viewing=False)
    args = parser.parse_args()
    
    if args.seed:
        n_seed = args.seed
    else:
        n_seed = None
        
    if not args.file:
        print("Add -f or -file followed by a json_files/file_name with details of the experiment")
        exit(0)
    else:
        f_name = args.file       
    
    if not args.viewing:
        flag_viewing = False
    else:
        flag_viewing = True
        
    return n_seed, f_name, flag_viewing

    
if __name__ == "__main__" :
    
    n_seed = None
    flag_viewing = False
    f_name = ""
    n_seed, f_name, flag_viewing = preamble( )
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
    