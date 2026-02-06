import numpy as np
import argparse
from PyQt6.QtWidgets import QApplication

from graphics.interface import MainWindow
from json_files.read_json import * 
from exp.experiment_1 import Exp
from world.world import Arena

def preamble( ):
    parser = argparse.ArgumentParser()
    parser.add_argument("-seed", type=int, help="add a required custom seed for numpy random seeding")
    parser.add_argument("-file", type=str, help="add a required json_files/file_name with details of the experiment")
    parser.add_argument("-viewing", type=bool, help="if added it triggers the graphical mode")
    parser.set_defaults(seed = 0, viewing=False)
    args = parser.parse_args()
    
    if not args.seed:
        print("Add -s or -seed followed by a custom seed for seeding numpy random")
        exit(0)
    else:
        n_seed = args.seed   
        
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
    
    n_seed = 0
    flag_viewing = False
    f_name = ""
    n_seed, f_name, flag_viewing = preamble( )
    
    np.random.seed(n_seed)
    read_json_file(f_name)
    my_exp = Exp( )
    
    if flag_viewing:
        app = QApplication([]) # create the QApplication
        window = MainWindow( my_exp, int(1.0/Arena.DeltaT) ) # create the main window
        window.show()
        app.exec() # start the event loop
    else:
        my_exp.exp_engine()