import numpy as np
from numpy.typing import NDArray
from WORLD.arena import Arena

class Ir_sensors():
    nb_sensors              = 8
    ir_range                = 0.04
    max_ir_reading          = 4095.0
    noise_level             = 1.0
    twopi                   = (2.0 * np.pi)
    halfpi                  = 0.5 * np.pi
    ir_angle                = np.array([0.3, 0.8, halfpi, 2.64, -2.64, -halfpi, -0.8, -0.3])
    ir_cos                  = np.cos(ir_angle)
    ir_sin                  = np.sin(ir_angle)
    
    def __init__ ( self ):
        self.reading  = np.zeros(Ir_sensors.nb_sensors)
        self.distance = np.array([])
        for i in range( Ir_sensors.nb_sensors ):
            self.distance = np.append(self.distance, Ir_sensors.ir_range )
        
    def update_sensors( self, id:np.int64 ):
        for i in range( Ir_sensors.nb_sensors ):
            self.distance[i] = Ir_sensors.ir_range
        Arena.compute_min_dist_to_objects( id, Ir_sensors.ir_range,Ir_sensors.ir_angle, Ir_sensors.ir_cos, Ir_sensors.ir_sin, self.distance )
        self.compute_reading ( self.distance )
        self.add_noise( )
    
    def compute_reading( self, dist: NDArray[np.float64]):
        for ir in range( Ir_sensors.nb_sensors ):
            self.reading[ir] = -1
            if (dist[ir] > 0.03 and dist[ir] <= 0.04):
                self.reading[ir] = -20600 * dist[ir] + 924
            elif ( dist[ir] > 0.02 and dist[ir] <= 0.03):
                self.reading[ir] = -37000 * dist[ir] + 1416
            elif ( dist[ir] > 0.01 and dist[ir] <= 0.02):
                self.reading[ir] = -153500 * dist[ir] + 3746
            elif ( dist[ir] > 0.005 and dist[ir] <= 0.01):
                self.reading[ir] = -252600 * dist[ir] + 4737
            elif ( dist[ir] >= 0.0 and dist[ir] <= 0.005 ):
                self.reading[ir] = -124200 * dist[ir] + 4095
    
    def add_noise( self ):
        for ir in range( Ir_sensors.nb_sensors ):
            if( self.reading[ir] == -1.0 ):
                # just background noise
                self.reading[ir] = np.random.randint(0, 150) * Ir_sensors.noise_level
            else:
                noise = int(np.random.normal(0.0, 50.0)) * Ir_sensors.noise_level
                self.reading[ir] += noise
                if( self.reading[ir] > Ir_sensors.max_ir_reading ):
                    self.reading[ir] = Ir_sensors.max_ir_reading
                elif ( self.reading[ir] < 0.0 ):
                    self.reading[ir] = 0.0
            self.reading[ir] /= float(Ir_sensors.max_ir_reading)
