import numpy as np
from numpy.typing import NDArray
import math
from WORLD.shapes import *

class Arena():
    DeltaT                       = np.array([])
    ring                         = np.array([])
    cylinder                     = np.array([])
    cuboid                       = np.array([])
    robot                        = np.array([]) #This is the array where it is stored the robot objects
    def reset():
        Arena.DeltaT   = np.array([])
        Arena.ring     = np.array([])
        Arena.cylinder = np.array([])
        Arena.cuboid   = np.array([])
        Arena.robot    = np.array([])
    
    def compute_dist_to_perimetral_wall(ring: Ring, id: np.int64, ir_range:np.float64,ir_cos: NDArray[np.float64], ir_sin: NDArray[np.float64],cosyaw:float ,sinyaw:float, distance: NDArray[np.float64] ):
        
        px, py = Arena.robot[id].pos[0], Arena.robot[id].pos[1]    # centre du robot en 2D
        cx, cy= ring.pos[0], ring.pos[1]               # centre du cercle
        R = float(ring.radius)                          # rayon du cercle
        r_robot = float(Arena.robot[id].radius)         # rayon du robot

        mx = px - cx                                   # Vector from circle centre to ray origin
        my = py - cy

        m = np.array([mx, my], dtype=float)

        for ir in range(len(ir_cos)):

            ux = cosyaw * ir_cos[ir]- sinyaw * ir_sin[ir]
            uy = sinyaw * ir_cos[ir] + cosyaw * ir_sin[ir]
            # Solve intersection: || m + t*u ||^2 = R^2
            # -> t^2 + 2*(u·m)*t + (m·m - R^2) = 0

            b = 2.0 * (ux *mx + uy * my)
            c = float(np.dot(m, m) - R*R)

            # Discriminant
            disc = b*b - 4.0*c  

            # If disc < 0 -> no intersection with infinite ray
            if disc < 0.0:
                continue

            sqrt_disc = math.sqrt(disc)

            # Two solutions along the ray
            t1 = (-b - sqrt_disc) * 0.5
            t2 = (-b + sqrt_disc) * 0.5

            # the closest intersection IN FRONT of the ray origin: t >= 0
            t_candidates = []
            if t1 >= 0.0:
                t_candidates.append(t1)
            if t2 >= 0.0:
                t_candidates.append(t2)

            if not t_candidates:
                continue

            t = min(t_candidates)  # closest hit in front

            # distance from robot surface
            d_surface = t - r_robot

            # If robot is overlapping/outside, d_surface can be negative.
            if d_surface < 0.0:
                d_surface = 0.0

            # If within sensor range, update minimum distance for that sensor
            if d_surface <= ir_range and d_surface < distance[ir]:
                distance[ir] = d_surface

    def compute_dist_to_perimetral_wall_old(ring: Ring, id: np.int64, rob_arena_centre_dist:np.float64, ir_range:np.float64, ir_angle: NDArray[np.float64], distance: NDArray[np.float64] ):
        """deprecated """
        dist = distance
    
        for ir in range( len(ir_angle) ):
            ray_end_pos = np.array([])
            ir_angle_in2pi = Arena.robot[id].rot[2] + ir_angle[ir]
            if ( ir_angle_in2pi  >= (2.0 * np.pi) ) :
                ir_angle_in2pi  -= (2.0 * np.pi)
            elif ( ir_angle_in2pi  < 0.0 ) :
                ir_angle_in2pi  += (2.0 * np.pi)
            ray_end_pos = np.append( ray_end_pos, Arena.robot[id].pos[0] + ((Arena.robot[id].radius+ir_range) * np.cos(ir_angle_in2pi)) )
            ray_end_pos = np.append( ray_end_pos, Arena.robot[id].pos[1] + ((Arena.robot[id].radius+ir_range) * np.sin(ir_angle_in2pi)) )
            
        
            if np.linalg.norm(ray_end_pos - ring.pos[0:2]) > ring.radius:
                theta = np.arctan2(Arena.robot[id].pos[1]-ring.pos[1], Arena.robot[id].pos[0]-ring.pos[0])
                if (theta < 0):
                    theta = (np.pi*2.0) + theta

                if np.around(ir_angle_in2pi, 4) == np.around(theta, 4) :
                    rob_arena_centre_dist = np.linalg.norm(Arena.robot[id].pos - ring.pos)
                    dist[ir] = np.around(ring.radius - rob_arena_centre_dist, 4) - Arena.robot[id].radius 
                else:
                    if ir_angle_in2pi < theta: 
                        compl_alpha = theta - ir_angle_in2pi
                    else: 
                        compl_alpha = ir_angle_in2pi - theta
                    alpha = np.pi - compl_alpha
                    
                    beta = np.arcsin((np.sin(alpha) * rob_arena_centre_dist)/ring.radius)
                    if beta < 0:
                        beta = (2.0 * np.pi) + beta
                    gamma = np.pi - beta - alpha
                    dist[ir] = ((np.sin(gamma) * ring.radius)/ np.sin(alpha)) - Arena.robot[id].radius
                    if( dist[ir] > ir_range) :
                        print(f"ERROR dist[{ir}] = {dist[ir]}")
                    elif ( dist[ir] < distance[ir] ):
                        distance[ir] = dist[ir]
                    
    def compute_dist_to_robot(id: int, other_id: int,ir_range: float, ir_cos: NDArray[np.float64], ir_sin: NDArray[np.float64], cosyaw: float, sinyaw: float, distance: NDArray[np.float64]):

            px, py = float(Arena.robot[id].pos[0]), float(Arena.robot[id].pos[1])           # centre robot courant
            qx, qy = float(Arena.robot[other_id].pos[0]), float(Arena.robot[other_id].pos[1])     # centre robot cible

            r_self = float(Arena.robot[id].radius)
            r_other = float(Arena.robot[other_id].radius)

            dx = px - qx
            dy = py - qy

            # too far -> no computation 
            centre_dist = math.sqrt(dx*dx + dy*dy)
            if centre_dist - (r_self + r_other) > ir_range:
                return

            for ir in range(len(ir_cos)):
                
                ux = cosyaw * ir_cos[ir]- sinyaw * ir_sin[ir]
                uy = cosyaw * ir_sin[ir] + sinyaw * ir_cos[ir]

                # Solve intersection of ray with circle of radius r_other around Q:
                m = np.array([dx, dy])
                b = 2.0 * (ux * m[0] + uy * m[1])
                c = (m[0] * m[0] + m[1] * m[1]) - (r_other * r_other)

                disc = b * b - 4.0 * c
                if disc < 0.0:
                    continue

                sqrt_disc = float(np.sqrt(disc))
                t1 = (-b - sqrt_disc) * 0.5
                t2 = (-b + sqrt_disc) * 0.5

                # Intersection devant le capteur (t >= 0)
                t = None
                if t1 >= 0.0 and t2 >= 0.0:
                    t = min(t1, t2)
                elif t1 >= 0.0:
                    t = t1
                elif t2 >= 0.0:
                    t = t2
                else:
                    continue

                # distance from robot surface
                d_surface = t - r_self
                if d_surface < 0.0:
                    d_surface = 0.0

                if d_surface <= ir_range and d_surface < distance[ir]:
                    distance[ir] = d_surface

    def compute_dist_to_cylinder(id: int, cyl: Cylinder, ir_range: float, ir_cos: NDArray[np.float64], ir_sin: NDArray[np.float64], cosyaw: float, sinyaw: float, distance: NDArray[np.float64]):
        #similar torobot distance but with cylinder
        
        px, py = float(Arena.robot[id].pos[0]), float(Arena.robot[id].pos[1])           # centre robot
        qx, qy = float(cyl.pos[0]), float(cyl.pos[1])           # centre cylindre

        r_self = float(Arena.robot[id].radius)
        r_cyl  = float(cyl.radius)

        dx = px - qx
        dy = py - qy

        centre_dist = math.sqrt(dx*dx + dy*dy)
        if centre_dist - (r_self + r_cyl) > ir_range:
            return

        m = np.array([dx, dy])

        for ir in range(len(ir_cos)):
            ux = cosyaw * ir_cos[ir]- sinyaw * ir_sin[ir]
            uy = cosyaw * ir_sin[ir] + sinyaw * ir_cos[ir]

            b = 2.0 * (ux * m[0] + uy * m[1])
            c = (m[0]*m[0] + m[1]*m[1]) - (r_cyl*r_cyl)

            disc = b*b - 4.0*c
            if disc < 0.0:
                continue

            sqrt_disc = math.sqrt(disc)
            t1 = (-b - sqrt_disc) * 0.5
            t2 = (-b + sqrt_disc) * 0.5

            t_candidates = []
            if t1 >= 0.0: t_candidates.append(t1)
            if t2 >= 0.0: t_candidates.append(t2)
            if not t_candidates:
                continue

            t = min(t_candidates)

            d_surface = t - r_self
            if d_surface < 0.0:
                d_surface = 0.0

            if d_surface <= ir_range and d_surface < distance[ir]:
                distance[ir] = d_surface

    def compute_dist_to_cuboid(id: int,cub: Cuboid,ir_range: float,ir_cos: NDArray[np.float64],ir_sin: NDArray[np.float64], cosyaw: float, sinyaw: float, distance: NDArray[np.float64]):
    
        rx, ry = float(Arena.robot[id].pos[0]), float(Arena.robot[id].pos[1]) # centre du robot en 2D
        cx, cy = float(cub.pos[0]), float(cub.pos[1])             # centre du cuboid en 2D

        r_self = float(Arena.robot[id].radius)

        hx = 0.5 * float(cub.dim[0]) 
        hy = 0.5 * float(cub.dim[1])  

        # Rotation yaw du cuboid
        yaw_c = float(cub.rot[2]) if hasattr(cub, "rot") else 0.0

        dx = rx - cx
        dy = ry - cy

        # Bounding circle radius for quick rejection test
        bounding = math.sqrt(hx*hx + hy*hy) + r_self
        if math.sqrt(dx*dx + dy*dy) - bounding > ir_range:
            return

        # Matrice rotation inverse 
        # on fait tourner le repere autour du centre du cuboid pour que le cuboid soit aligné
        cos_y = math.cos(-yaw_c)
        sin_y = math.sin(-yaw_c)

        px = cos_y*dx - sin_y*dy
        py = sin_y*dx + cos_y*dy

        eps = 1e-12
        for ir in range(len(ir_cos)):
            
            ux_world = cosyaw * ir_cos[ir]- sinyaw * ir_sin[ir]
            uy_world = sinyaw * ir_cos[ir] + cosyaw * ir_sin[ir]
            
            ux_local = cos_y*ux_world - sin_y* uy_world
            uy_local = sin_y*ux_world + cos_y* uy_world

            # Intersection rayon (p_local + t*u_local) avec AABB [-hx,hx] x [-hy,hy]
            tmin = -1e30
            tmax =  1e30

            # Axe X
            if abs(ux_local) < eps:
                # Rayon parallèle aux faces verticales
                if px < -hx or px > hx:
                    continue
            else:
                tx1 = (-hx - px) / ux_local
                tx2 = ( hx - px) / ux_local
                tmin = max(tmin, min(tx1, tx2))
                tmax = min(tmax, max(tx1, tx2))

            # Axe Y
            if abs(uy_local) < eps:
                if py < -hy or py > hy:
                    continue
            else:
                ty1 = (-hy - py) / uy_local
                ty2 = ( hy - py) / uy_local
                tmin = max(tmin, min(ty1, ty2))
                tmax = min(tmax, max(ty1, ty2))

            # Pas d'intersection
            if tmax < 0.0 or tmin > tmax:
                continue

            # Première intersection devant le rayon
            t = tmin if tmin >= 0.0 else tmax 

            # Distance depuis surface du robot
            d_surface = t - r_self
            if d_surface < 0.0:
                d_surface = 0.0

            if d_surface <= ir_range and d_surface < distance[ir]:
                distance[ir] = d_surface

    def compute_min_dist_to_objects(id: np.int64, ir_range:np.float64, ir_angle: NDArray[np.float64], ir_cos: NDArray[np.float64], ir_sin: NDArray[np.float64], distance: NDArray[np.float64] ):
        # we compute yaw and cos/sin here to avoid recomputing them for each object type
        yaw = float(Arena.robot[id].rot[2])
        cosyaw = math.cos(yaw)
        sinyaw = math.sin(yaw)

        for r in Arena.ring:             
            Arena.compute_dist_to_perimetral_wall(r, id, ir_range, ir_cos, ir_sin, cosyaw, sinyaw, distance)

        
        for cyl in Arena.cylinder:
            rob_cylinder_dist = np.linalg.norm(Arena.robot[id].pos - cyl.pos)
            if (rob_cylinder_dist - cyl.radius - Arena.robot[id].radius) <= ir_range:
                Arena.compute_dist_to_cylinder(id, cyl, ir_range, ir_cos, ir_sin, cosyaw, sinyaw, distance)
                
                
        for cub in Arena.cuboid:
            Arena.compute_dist_to_cuboid(id, cub, ir_range, ir_cos, ir_sin, cosyaw, sinyaw, distance)
                
        for e in range(len(Arena.robot)):
            if id != Arena.robot[e].id:
                other_id = Arena.robot[e].id
                rob_rob_dist = np.linalg.norm(Arena.robot[id].pos - Arena.robot[other_id].pos)
                if (rob_rob_dist - Arena.robot[id].radius - Arena.robot[other_id].radius) <= ir_range:
                    Arena.compute_dist_to_robot(id, other_id, ir_range, ir_cos, ir_sin, cosyaw, sinyaw, distance)