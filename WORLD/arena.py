import numpy as np
from numpy.typing import NDArray
from WORLD.shapes import *

class Arena():
    DeltaT                       = np.array([])
    ring                         = np.array([])
    cylinder                     = np.array([])
    cuboid                       = np.array([])
    robot                        = np.array([]) #This is the array where it is stored the robot objects
    
    def compute_dist_to_perimetral_wall(ring: Ring, id: np.int64, rob_arena_centre_dist:np.float64, ir_range:np.float64, ir_angle: NDArray[np.float64], distance: NDArray[np.float64] ):
        
        P = np.array(Arena.robot[id].pos[0:2], dtype=float)      # centre du robot en 2D
        C = np.array(ring.pos[0:2], dtype=float)                # centre du cercle
        R = float(ring.radius)                          # rayon du cercle
        r_robot = float(Arena.robot[id].radius)         # rayon du robot

        m = P - C                                   # Vector from circle centre to ray origin

        for ir in range(len(ir_angle)):

            # Direction of the IR ray in world frame
            phi = float(Arena.robot[id].rot[2]) + float(ir_angle[ir])

            u = np.array([np.cos(phi), np.sin(phi)], dtype=float)  # unité

            # Solve intersection: || m + t*u ||^2 = R^2
            # -> t^2 + 2*(u·m)*t + (m·m - R^2) = 0

            b = 2.0 * float(np.dot(u, m))
            c = float(np.dot(m, m) - R*R)

            # Discriminant
            disc = b*b - 4.0*c  

            # If disc < 0 -> no intersection with infinite ray
            if disc < 0.0:
                continue

            sqrt_disc = float(np.sqrt(disc))

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
                    
    def compute_dist_to_robot(id: int, other_id: int,ir_range: float, ir_angle: NDArray[np.float64], distance: NDArray[np.float64]):

            P = np.array(Arena.robot[id].pos[0:2], dtype=float)           # centre robot courant
            Q = np.array(Arena.robot[other_id].pos[0:2], dtype=float)     # centre robot cible

            r_self = float(Arena.robot[id].radius)
            r_other = float(Arena.robot[other_id].radius)

            # too far -> no computation 
            centre_dist = np.linalg.norm(Q - P)
            if centre_dist - (r_self + r_other) > ir_range:
                return
            
            # Vector from other robot centre to current robot centre
            m = P - Q 

            # Angle of the current robot in world frame
            yaw = float(Arena.robot[id].rot[2])

            for ir in range(len(ir_angle)):
                phi = yaw + float(ir_angle[ir])
                ux = float(np.cos(phi))
                uy = float(np.sin(phi))

                # Solve intersection of ray with circle of radius r_other around Q:
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

    def compute_dist_to_cylinder(id: int, cyl: Cylinder, ir_range: float, ir_angle: NDArray[np.float64], distance: NDArray[np.float64]):
        #similar torobot distance but with cylinder
        
        P = np.array(Arena.robot[id].pos[0:2], dtype=float)
        Q = np.array(cyl.pos[0:2], dtype=float)

        r_self = float(Arena.robot[id].radius)
        r_cyl  = float(cyl.radius)

        centre_dist = np.linalg.norm(Q - P)
        if centre_dist - (r_self + r_cyl) > ir_range:
            return

        m = P - Q
        yaw = float(Arena.robot[id].rot[2])

        for ir in range(len(ir_angle)):
            phi = yaw + float(ir_angle[ir])
            ux = float(np.cos(phi))
            uy = float(np.sin(phi))

            b = 2.0 * (ux * m[0] + uy * m[1])
            c = (m[0]*m[0] + m[1]*m[1]) - (r_cyl*r_cyl)

            disc = b*b - 4.0*c
            if disc < 0.0:
                continue

            sqrt_disc = float(np.sqrt(disc))
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

    def compute_dist_to_cuboid(id: int,cub: Cuboid,ir_range: float,ir_angle: NDArray[np.float64],distance: NDArray[np.float64]):
    
        P = np.array(Arena.robot[id].pos[0:2], dtype=float)# centre du robot en 2D
        C = np.array(cub.pos[0:2], dtype=float)             # centre du cuboid en 2D

        r_self = float(Arena.robot[id].radius)

        hx = 0.5 * float(cub.dim[0]) 
        hy = 0.5 * float(cub.dim[1])  

        # Rotation yaw du cuboid
        yaw_c = float(cub.rot[2]) if hasattr(cub, "rot") else 0.0

        # Bounding circle radius for quick rejection test
        bounding = np.sqrt(hx*hx + hy*hy) + r_self
        if np.linalg.norm(P - C) - bounding > ir_range:
            return

        # Matrice rotation inverse 
        # on fait tourner le repere autour du centre du cuboid pour que le cuboid soit aligné
        cos_y = np.cos(-yaw_c)
        sin_y = np.sin(-yaw_c)

        def world_to_local(v: np.ndarray) -> np.ndarray:
            return np.array([
                cos_y * v[0] - sin_y * v[1],
                sin_y * v[0] + cos_y * v[1],
            ], dtype=float)

        # Origine of rayon in cuboid local frame
        p_local = world_to_local(P - C)

        yaw_r = float(Arena.robot[id].rot[2])

        eps = 1e-12
        for ir in range(len(ir_angle)):
            phi = yaw_r + float(ir_angle[ir])
            u_world = np.array([np.cos(phi), np.sin(phi)], dtype=float)
            u_local = world_to_local(u_world)

            # Intersection rayon (p_local + t*u_local) avec AABB [-hx,hx] x [-hy,hy]
            tmin = -np.inf
            tmax =  np.inf

            # Axe X
            if abs(u_local[0]) < eps:
                # Rayon parallèle aux faces verticales
                if p_local[0] < -hx or p_local[0] > hx:
                    continue
            else:
                tx1 = (-hx - p_local[0]) / u_local[0]
                tx2 = ( hx - p_local[0]) / u_local[0]
                tmin = max(tmin, min(tx1, tx2))
                tmax = min(tmax, max(tx1, tx2))

            # Axe Y
            if abs(u_local[1]) < eps:
                if p_local[1] < -hy or p_local[1] > hy:
                    continue
            else:
                ty1 = (-hy - p_local[1]) / u_local[1]
                ty2 = ( hy - p_local[1]) / u_local[1]
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

    def compute_min_dist_to_objects(id: np.int64, ir_range:np.float64, ir_angle: NDArray[np.float64], distance: NDArray[np.float64] ):
        for r in Arena.ring:
            rob_arena_centre_dist = np.linalg.norm(Arena.robot[id].pos - r.pos)
            if np.around( r.radius - (rob_arena_centre_dist + Arena.robot[id].radius), 2)  > 0 :                
                Arena.compute_dist_to_perimetral_wall(r, id, rob_arena_centre_dist, ir_range, ir_angle, distance)

        
        for cyl in Arena.cylinder:
            rob_cylinder_dist = np.linalg.norm(Arena.robot[id].pos - cyl.pos)
            if (rob_cylinder_dist - cyl.radius - Arena.robot[id].radius) <= ir_range:
                Arena.compute_dist_to_cylinder(id, cyl, ir_range, ir_angle, distance)
                
                
        for cub in Arena.cuboid:
            Arena.compute_dist_to_cuboid(id, cub, ir_range, ir_angle, distance)
                
        for e in range(len(Arena.robot)):
            if id != Arena.robot[e].id:
                other_id = Arena.robot[e].id
                rob_rob_dist = np.linalg.norm(Arena.robot[id].pos - Arena.robot[other_id].pos)
                if (rob_rob_dist - Arena.robot[id].radius - Arena.robot[other_id].radius) <= ir_range:
                    Arena.compute_dist_to_robot(id, other_id, ir_range, ir_angle, distance)