import numpy as np
import math
import random

from TOOLS.logger import logger
from WORLD.arena import Arena    

def get_robot_radius(robot_class):
    for attr in ("radius", "robot_radius", "body_radius", "r"):
        if hasattr(robot_class, attr):
            return float(getattr(robot_class, attr))
    logger.log("WARN", f"Could not infer radius for {robot_class.__name__}, fallback to 0.035")
    return 0.035

def is_inside_ring(x, y, margin=0.0):
    if len(Arena.ring) == 0:
        return True  # pas de mur défini => on laisse passer

    ring = Arena.ring[0]
    cx, cy = ring.pos[0], ring.pos[1]
    dist = math.hypot(x - cx, y - cy)
    return dist <= (ring.radius - margin)

def collides_with_other_robots(x, y, robot_radius):
    for rb in Arena.robot:
        rx, ry = rb.pos[0], rb.pos[1]
        other_radius = getattr(rb, "radius", robot_radius)
        if math.hypot(x - rx, y - ry) < (robot_radius + other_radius):
            return True
    return False

def is_valid_robot_position(x, y, robot_radius):
    if not is_inside_ring(x, y, margin=robot_radius):
        return False
    if collides_with_other_robots(x, y, robot_radius):
        return False
    return True

def jitter_robot_pose(base_pos, base_rot, robot_radius,
                        pos_jitter=0.03, rz_jitter=0.35, max_tries=100):
    """
    base_pos : [x,y,z]
    base_rot : [rx,ry,rz]

    pos_jitter en mètres (0.03 = 3 cm)
    rz_jitter en radians (~0.35 = ~20°)
    """
    x0, y0, z0 = float(base_pos[0]), float(base_pos[1]), float(base_pos[2])
    rx0, ry0, rz0 = float(base_rot[0]), float(base_rot[1]), float(base_rot[2])

    for _ in range(max_tries):
        x = x0 + random.uniform(-pos_jitter, pos_jitter)
        y = y0 + random.uniform(-pos_jitter, pos_jitter)
        rz = rz0 + random.uniform(-rz_jitter, rz_jitter)

        if is_valid_robot_position(x, y, robot_radius):
            return np.array([x, y, z0], dtype=float), np.array([rx0, ry0, rz], dtype=float)

    logger.log("WARN", f"Could not jitter robot around ({x0:.3f}, {y0:.3f}) without collision. Keeping original pose.")
    return np.array([x0, y0, z0], dtype=float), np.array([rx0, ry0, rz0], dtype=float)