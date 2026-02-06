import json
import numpy as np


class Arena():
    DeltaT                                    = np.empty(0)
    perimetral_wall_xyz_radius_height_colour  = np.empty(0)
    round_obst_xyz_radius_height_colour       = np.empty(0)
    cuboid_obst_xyz_lwh_rxryrz_colour         = np.empty(0)
    epucks_xyz_rx_ry_rz_colour                = np.empty(0)
    
    def compute_dist_to_perimetral_wall(pos, rot, robot_radius, rob_arena_centre_dist, ir_range, ir_angle, distance ):
        dist = np.linspace(ir_range, ir_range, len(ir_angle)) # np.zeros( len(ir_angle) ) 
        # print(f" dist = {dist}")
        for ir in range( len(ir_angle) ):
            ir_angle_in2pi = rot[2] + ir_angle[ir]
            if ( ir_angle_in2pi  >= (2.0 * np.pi) ) :
                ir_angle_in2pi  -= (2.0 * np.pi)
            elif ( ir_angle_in2pi  < 0.0 ) :
                ir_angle_in2pi  += (2.0 * np.pi)
            ray_end_pos = np.array( pos[0] + ((robot_radius+ir_range) * np.cos(ir_angle_in2pi)) )
            ray_end_pos = np.append( ray_end_pos, pos[1] + ((robot_radius+ir_range) * np.sin(ir_angle_in2pi)) )
            # ray_end_pos = np.array( pos[0] + ((robot_radius+ir_range) * np.cos(rot[2] - ir_angle[ir])) )
            # ray_end_pos = np.append( ray_end_pos, pos[1] + ((robot_radius+ir_range) * np.sin(rot[2] - ir_angle[ir])) )
            # print( f"BEFORE IR = {ir}, ray_pos = {ray_end_pos}, dist= {np.linalg.norm(ray_end_pos - Arena.perimetral_wall_xyz_radius_height_colour[0:2])}" )
            #  print(f" rob rot={(rot*180.0)/np.pi} ir_pos[{ir}] = {(ir_angle_in2pi*180.0)/np.pi }")
            if np.linalg.norm(ray_end_pos - Arena.perimetral_wall_xyz_radius_height_colour[0:2]) > Arena.perimetral_wall_xyz_radius_height_colour[3]:
                theta = np.arctan2(pos[1]-Arena.perimetral_wall_xyz_radius_height_colour[1], pos[0]-Arena.perimetral_wall_xyz_radius_height_colour[0])
                if (theta < 0):
                    theta = (np.pi*2.0) + theta
                # theta_180 = theta + np.pi
                # if theta_180 > (2.0 * np.pi):
                #     theta_180 -= (2.0 * np.pi) theta_180 = {theta_180}, 
                # print(f"IR = {ir}, theta={theta}, ir_angle_in2pi = {ir_angle_in2pi}")
                    
                if np.around(ir_angle_in2pi, 4) == np.around(theta, 4) :
                # or np.around(ir_angle_in2pi,4) == np.around(theta_180, 4):
                    # print(f"A ")
                    rob_arena_centre_dist = np.linalg.norm(pos - Arena.perimetral_wall_xyz_radius_height_colour[0:3])
                    dist[ir] = np.around(Arena.perimetral_wall_xyz_radius_height_colour[3] - rob_arena_centre_dist, 4) - robot_radius 
                else:
                    if ir_angle_in2pi < theta: 
                        compl_alpha = theta - ir_angle_in2pi
                        # print(f"B compl_alpha = {compl_alpha}")
                    else: #if ir_angle_in2pi > theta:
                        compl_alpha = ir_angle_in2pi - theta
                        # print(f"C compl_alpha = {compl_alpha}")
                    # elif ir_angle_in2pi > theta_180:
                    #     compl_alpha = ir_angle_in2pi - theta_180
                    #     print(f"D compl_alpha = {compl_alpha}")
                    # elif ir_angle_in2pi < theta_180:
                    #     compl_alpha = theta_180 - ir_angle_in2pi
                    #     print(f"E compl_alpha = {compl_alpha}")
                    alpha = np.pi - compl_alpha
                
                    beta = np.arcsin(np.sin(alpha) * rob_arena_centre_dist)/Arena.perimetral_wall_xyz_radius_height_colour[3]
                    if beta < 0:
                        beta = (2.0 * np.pi) + beta
                    gamma = np.pi - beta - alpha
                    # print(f"alpha = {(alpha*180.0)/np.pi}, beta = {(beta*180.0)/np.pi}, gamma = {(gamma*180.0)/np.pi}")
                    dist[ir] = ((np.sin(gamma) * Arena.perimetral_wall_xyz_radius_height_colour[3])/ np.sin(alpha)) - robot_radius
                    if( dist[ir] > ir_range) :
                        print(f"ERROR dist[{ir}] = {dist[ir]}")
                    elif ( dist[ir] < distance[ir] ):
                        distance[ir] = dist[ir]
                    # print( f"AFTER IR = {ir}, dist[{ir}]= {dist[ir]} distance[{ir}]= {distance[ir]}" )
                    
                    
    def compute_dist_to_round_obst(pos, rot, robot_radius, rob_cylinder_dist, ir_range, ir_angle, distance ):
        print("compute_dist_to_round_obst() - function has to be completed")
        return
        

    def compute_min_dist_to_objects(pos, rot, robot_radius, ir_range, ir_angle, distance ):
        rob_arena_centre_dist = np.linalg.norm(pos - Arena.perimetral_wall_xyz_radius_height_colour[0:3])
        if (np.around(Arena.perimetral_wall_xyz_radius_height_colour[3] - rob_arena_centre_dist, 4) - robot_radius) < ir_range :
            Arena.compute_dist_to_perimetral_wall(pos, rot, robot_radius, rob_arena_centre_dist, ir_range, ir_angle, distance)
        
        for c in range(len(Arena.round_obst_xyz_radius_height_colour)):
            rob_cylinder_dist = np.linalg.norm(pos - Arena.round_obst_xyz_radius_height_colour[c][0:3])
            if ( rob_cylinder_dist - Arena.round_obst_xyz_radius_height_colour[c][3] - robot_radius) < ir_range :
                Arena.compute_dist_to_round_obst(pos, rot, robot_radius, rob_cylinder_dist, ir_range, ir_angle, distance)
        
        for c in range(len(Arena.cuboid_obst_xyz_lwh_rxryrz_colour )):
            rob_cuboid_dist = np.linalg.norm(pos - Arena.cuboid_obst_xyz_lwh_rxryrz_colour[c][0:3])
            if ( rob_cuboid_dist - Arena.cuboid_obst_xyz_lwh_rxryrz_colour[c][4] - robot_radius) < ir_range :
                print("This piece of code has to be completed")
        
        