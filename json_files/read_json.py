import json
import numpy as np

from GRAPHICS.interface import *
from WORLD.world import *
from EXP.experiment import *
from WORLD.epuck import *
from WORLD.shapes import *
from WORLD.world import *

def read_json_file (file_name: str):
    try:
        with open(file_name, 'r') as file:
            data = json.load(file)
            for k, v in data["experiment"][0]["duration"].items():
                if k == "num_trials":
                    Exp.num_trials = v
                elif k == "num_iterations":
                    Exp.num_iterations = v
                elif k == "seed":
                    exp_seed = int(v)
    
            for k, v in data["experiment"][1]["arena"].items():
                if k == "delta_t(ms)":
                    MainWindow.delta_t_ms    = v
                    Diff_drive_robot.delta_t = (1.0/v)
                    
                elif k == "perimetral round wall [x,y,z,radius,height,rx,ry,rz,colour]":
                    if np.any(v):
                        for n in range(len(v)):
                            Arena.ring = np.append(Arena.ring, Ring(v[n][0:3], v[n][3], v[n][4], v[n][5:8], v[n][8:11]))
    
                        
            for k, v in data["experiment"][2]["round_obstacle"].items():
                if k == "[x,y,z,radius,height,rx,ry,rz,colour]":
                    if np.any(v):
                        for n in range(len(v)):
                            Arena.cylinder = np.append(Arena.cylinder, Cylinder(v[n][0:3], v[n][3], v[n][4], v[n][5:8], v[n][8:11]))

            for k, v in data["experiment"][3]["cuboid_obstacle"].items():
                # print(k)
                if k == "[x,y,z,l,w,h,rx,ry,rz,colour]":
                    if np.any(v):
                        for n in range(len(v)):
                            Arena.cuboid = np.append(Arena.cuboid, Cuboid(v[n][0:3], v[n][3:6], v[n][6:9], v[n][9:12]))
                
            for k, v in data["experiment"][4]["e_pucks"].items():
                if k == "[x,y,z,rx,ry,rz,colour]":
                    if np.any(v):
                        v = np.array(v)
                        for id in range(len(v)):
                            Arena.epuck = np.append(Arena.epuck, Epuck_robot(id, v[id][0:3], v[id][3:6], v[id][6:9], np.zeros(2) ))
                    
    except FileNotFoundError:
        print("Error: The file '.json' was not found.")
    return exp_seed
