import numpy as np
from numpy.typing import NDArray
from WORLD.shapes import *

class Arena():
    DeltaT                       = np.array([])
    ring                         = np.array([])
    cylinder                     = np.array([])
    cuboid                       = np.array([])
    epuck                        = np.array([]) #This is the array where it is stored the epuck objects
    
    def compute_dist_to_perimetral_wall(ring: Ring, id: np.int64, rob_arena_centre_dist:np.float64, ir_range:np.float64, ir_angle: NDArray[np.float64], distance: NDArray[np.float64] ):
        dist = distance
    
        for ir in range( len(ir_angle) ):
            ray_end_pos = np.array([])
            ir_angle_in2pi = Arena.epuck[id].rot[2] + ir_angle[ir]
            if ( ir_angle_in2pi  >= (2.0 * np.pi) ) :
                ir_angle_in2pi  -= (2.0 * np.pi)
            elif ( ir_angle_in2pi  < 0.0 ) :
                ir_angle_in2pi  += (2.0 * np.pi)
            ray_end_pos = np.append( ray_end_pos, Arena.epuck[id].pos[0] + ((Arena.epuck[id].radius+ir_range) * np.cos(ir_angle_in2pi)) )
            ray_end_pos = np.append( ray_end_pos, Arena.epuck[id].pos[1] + ((Arena.epuck[id].radius+ir_range) * np.sin(ir_angle_in2pi)) )
            
        
            if np.linalg.norm(ray_end_pos - ring.pos[0:2]) > ring.radius:
                theta = np.arctan2(Arena.epuck[id].pos[1]-ring.pos[1], Arena.epuck[id].pos[0]-ring.pos[0])
                if (theta < 0):
                    theta = (np.pi*2.0) + theta

                if np.around(ir_angle_in2pi, 4) == np.around(theta, 4) :
                    rob_arena_centre_dist = np.linalg.norm(Arena.epuck[id].pos - ring.pos)
                    dist[ir] = np.around(ring.radius - rob_arena_centre_dist, 4) - Arena.epuck[id].radius 
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
                    dist[ir] = ((np.sin(gamma) * ring.radius)/ np.sin(alpha)) - Arena.epuck[id].radius
                    if( dist[ir] > ir_range) :
                        print(f"ERROR dist[{ir}] = {dist[ir]}")
                    elif ( dist[ir] < distance[ir] ):
                        distance[ir] = dist[ir]
                    
        
    def compute_min_dist_to_objects(id: np.int64, ir_range:np.float64, ir_angle: NDArray[np.float64], distance: NDArray[np.float64] ):
        for r in Arena.ring:
            rob_arena_centre_dist = np.linalg.norm(Arena.epuck[id].pos - r.pos)
            if np.around( r.radius - (rob_arena_centre_dist + Arena.epuck[id].radius), 2)  > 0 :                
                Arena.compute_dist_to_perimetral_wall(r, id, rob_arena_centre_dist, ir_range, ir_angle, distance)

        
        for cyl in Arena.cylinder:
            rob_cylinder_dist = np.linalg.norm(Arena.epuck[id].pos - cyl.pos)
            # if ( rob_cylinder_dist - cyl.radius - Arena.epuck[id].radius) < ir_range :
                # TASK A - This piece of code has to be completed with a function 
                # that updates the np.array distance (only smaller distances have to be updates)
                # taking into account robot-cylinder interactions
                
                
        for cub in Arena.cuboid:
            rob_cuboid_dist = np.linalg.norm(Arena.epuck[id].pos - cub.pos)
            # if ( rob_cuboid_dist - cub.dim[1] - Arena.epuck[id].radius) < ir_range :
                 # TASK B - This piece of code has to be completed with a function 
                # that updates the np.array distance (only smaller distances have to be updates)
                # taking into account robot-cuboid interactions
                
        for e in range(len(Arena.epuck)):
            if id != Arena.epuck[e].id:
                rob_rob_dist = np.linalg.norm(Arena.epuck[id].pos - Arena.epuck[Arena.epuck[e].id].pos)
                # if ( rob_rob_dist - Arena.epuck[id].radius - Arena.epuck[Arena.epuck[e].id].radius) < ir_range :
                # TASK C - This piece of code has to be completed with a function 
                # that updates the np.array distance (only smaller distances have to be updates)
                # taking into account robot-robot interactions