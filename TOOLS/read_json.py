import json
import numpy as np

from GRAPHICS.interface import *
from WORLD.arena import *
from EXP.experiment import *
from WORLD.epuck import *
from WORLD.musicbot import *
from WORLD.shapes import *
from TOOLS.logger import logger



def read_json_file(file_name: str):
    try:
        with open(file_name, "r") as file:
            data = json.load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file '{file_name}' was not found.")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in '{file_name}': {e}")

    exp_seed = 0
    delta_t_ms = 0

    #validate global structure of JSON
    exp_list = data.get("experiment", None)
    if not isinstance(exp_list, list):
        raise ValueError("JSON format error: root key 'experiment' must be a list.")

    #required section in json
    required_sections = {"duration", "arena"}
    all_sections = {"duration", "arena", "round_obstacle", "cuboid_obstacle", "e_pucks"}
    seen_sections = set()

    #Handlers for each section 
    def handle_comment(section: dict):
        for k, v in section.items():
            logger.log("INFO", f"{k}: {v}")

    def handle_duration(section: dict):
        nonlocal exp_seed
        nonlocal delta_t_ms
        
        known_keys = {"num_trials", "num_iterations", "seed", "delta_t(ms)"}
        for k, v in section.items():
            if k == "num_trials":
                Exp.num_trials = v
            elif k == "num_iterations":
                Exp.num_iterations = v
            elif k == "seed":
                exp_seed = int(v)
            elif k == "delta_t(ms)":
                delta_t_ms = float(v)
                if delta_t_ms <= 0:
                    raise ValueError("duration.delta_t(ms) must be > 0")
                Diff_drive_robot.delta_t = (1.0 /delta_t_ms)
            else:
                logger.log("WARN", f"Unknown key in 'duration': '{k}' (ignored). Expected keys: {sorted(known_keys)}")

        for must in ("num_trials", "num_iterations", "seed", "delta_t(ms)"):
            if must not in section:
                logger.log("WARN", f"Missing key in 'duration': '{must}'")
    def handle_arena(section: dict):
        known_keys = {
            "perimetral round wall [x,y,z,radius,height,rx,ry,rz,colour]",
        }

        for k, v in section.items():

            if k == "perimetral round wall [x,y,z,radius,height,rx,ry,rz,colour]":

                for n in range(len(v)):
                    row = v[n]
                    if len(row) < 11:
                        logger.log("WARN", f"Ring entry #{n} has invalid length {len(row)} (expected 11). Ignored.")
                        continue
                    Arena.ring = np.append(
                        Arena.ring,
                        Ring(row[0:3], row[3], row[4], row[5:8], row[8:11])
                    )
            else:
                logger.log("WARN", f"Unknown key in 'arena': '{k}' (ignored). Expected keys: {sorted(known_keys)}")


    def handle_round_obstacle(section: dict):
        known_keys = {"[x,y,z,radius,height,rx,ry,rz,colour]"}
        for k, v in section.items():
            if k == "[x,y,z,radius,height,rx,ry,rz,colour]":
                if v is None or len(v) == 0:
                    return
                for n in range(len(v)):
                    row = v[n]
                    if len(row) < 11:
                        logger.log("WARN", f"Cylinder entry #{n} has invalid length {len(row)} (expected 11). Ignored.")
                        continue
                    Arena.cylinder = np.append(
                        Arena.cylinder,
                        Cylinder(row[0:3], row[3], row[4], row[5:8], row[8:11])
                    )
            else:
                logger.log("WARN", f"Unknown key in 'round_obstacle': '{k}' (ignored). Expected keys: {sorted(known_keys)}")

    def handle_cuboid_obstacle(section: dict):
        known_keys = {"[x,y,z,l,w,h,rx,ry,rz,colour]"}
        for k, v in section.items():
            if k == "[x,y,z,l,w,h,rx,ry,rz,colour]":
                if v is None or len(v) == 0:
                    return
                for n in range(len(v)):
                    row = v[n]
                    if len(row) < 12:
                        logger.log("WARN", f"Cuboid entry #{n} has invalid length {len(row)} (expected 12). Ignored.")
                        continue
                    Arena.cuboid = np.append(
                        Arena.cuboid,
                        Cuboid(row[0:3], row[3:6], row[6:9], row[9:12])
                    )
            else:
                logger.log("WARN", f"Unknown key in 'cuboid_obstacle': '{k}' (ignored). Expected keys: {sorted(known_keys)}")

    def handle_robot(section_label: str, robot_class):
        """
        Returns a handler function for robot
        sections (e_pucks, music_bots, etc.)
        """
        known_keys = {"[x,y,z,rx,ry,rz,colour]"}
        
        def handler(section: dict):
            for k, v in section.items():
                if k == "[x,y,z,rx,ry,rz,colour]":

                    if v is None or len(v) == 0:
                        logger.log("INFO", f"{section_label} list is empty.")
                        return
                    
                    arr = np.array(v, dtype=float)
                    for rid in range(len(arr)):
                        row = arr[rid]
                        if len(row) < 9:
                            logger.log("WARN", f"{section_label} entry #{rid} has invalid length {len(row)} (expected 9). Ignored.")
                            continue

                        global_robot_id = len(Arena.robot)

                        Arena.robot = np.append(
                            Arena.robot,
                            robot_class(global_robot_id, row[0:3], row[3:6], row[6:9], np.zeros(2))
                        )
                else:
                    logger.log("WARN", f"Unknown key in '{section_label}': '{k}' (ignored). Expected keys: {sorted(known_keys)}")
        
        #handler function is returned by handle_robot and added to handlers dict
        return handler 
       
    handlers = {
        "comment": handle_comment,
        "duration": handle_duration,
        "arena": handle_arena,
        "round_obstacle": handle_round_obstacle,
        "cuboid_obstacle": handle_cuboid_obstacle,
        "e_pucks": handle_robot("e_pucks", Epuck_robot),
        "music_bots": handle_robot("music_bots", MusicBot),
    }

    # Process each item in the experiment list
    for item_idx, item in enumerate(exp_list):
        if not isinstance(item, dict) or len(item) == 0:
            logger.log("WARN", f"Ignoring invalid experiment entry at index {item_idx}: {item}")
            continue

        
        for section_name, section_body in item.items():
            if section_name in handlers:
                seen_sections.add(section_name)
                if not isinstance(section_body, dict):
                    logger.log("WARN", f"Section '{section_name}' should be a dict, got {type(section_body).__name__}. Ignored.")
                    continue
                handlers[section_name](section_body)
            else:
                logger.log("WARN", f"Unused/unknown section '{section_name}' (ignored). Try to reformat or update parser.")

    # missing section
    # mandatory
    for req in sorted(required_sections):
        if req not in seen_sections:
            logger.log("CRITIC", f"Missing mandatory section '{req}' in JSON.")
    # non mandatory
    for sec in sorted(all_sections):
        if sec not in seen_sections:
            logger.log("INFO", f"Non mandatory section '{sec}' not found in JSON.")

    return exp_seed, delta_t_ms