import json
import numpy as np
from world.world import Arena
from exp.experiment_1 import Exp

def read_json_file (file_name):
    try:
        with open(file_name, 'r') as file:
            data = json.load(file)
            for k, v in data["experiment"][0]["duration"].items():
                if k == "num_trials":
                    Exp.num_trials = v
                    # print(f"Trials = {Exp.num_trials}")
                elif k == "num_iterations":
                    Exp.num_iterations = v
                    # print(f"Iterations = {Exp.num_iterations}")
 
            for k, v in data["experiment"][1]["arena"].items():
                # print(k)
                if k == "delta_t":
                    Arena.DeltaT = v 
                    # print(Arena.DeltaT)
                elif k == "perimetral round wall [x,y,z,radius,height,colour]":
                    Arena.perimetral_wall_xyz_radius_height_colour = np.array(v) 
                    # print(Arena.perimetral_wall_xyz_radius_height_colour)

            for k, v in data["experiment"][2]["round_obstacle"].items():
                # print(k)
                if k == "[x,y,z,radius,h,colour]":
                    Arena.round_obst_xyz_radius_height_colour = np.array(v) 
                    # print(Arena.round_obst_xyz_radius_h_colour)
                    
            for k, v in data["experiment"][3]["cuboid_obstacle"].items():
                # print(k)
                if k == "[x,y,z,l,w,h,rx,ry,rz,colour]":
                    Arena.cuboid_obst_xyz_lwh_rxryrz_colour = np.array(v)
                    # print(Arena.cuboid_obst_xyz_lwh)
                
            for k, v in data["experiment"][4]["e_pucks"].items():
                # print(k)
                if k == "[x,y,z,rot_x,rot_y,rot_z,colour]":
                    Arena.epucks_xyz_rx_ry_rz_colour = np.array(v) 
                    # print(Arena.epucks_xyz_rx_ry_rz_colour)
                    
    except FileNotFoundError:
        print("Error: The file 'world.json' was not found.")
