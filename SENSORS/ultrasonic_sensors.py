import numpy as np
from numpy.typing import NDArray
from WORLD.arena import Arena

class Ultrasonic_sensors():
    nb_sensors              = 8
    us_range                = 1.0
    max_us_reading          = 4095.0
    noise_level             = 1.0

    twopi                   = (2.0 * np.pi)
    halfpi                  = 0.5 * np.pi

    us_angle                = np.array([0.3, 0.8, halfpi, 2.64, -2.64, -halfpi, -0.8, -0.3])
    
    def __init__ ( self ):
        self.reading  = np.zeros(Ultrasonic_sensors.nb_sensors)
        self.distance = np.array([])
        for i in range( Ultrasonic_sensors.nb_sensors ):
            self.distance = np.append(self.distance, Ultrasonic_sensors.us_range )
        
    def update_sensors( self, id:np.int64 ):
        for i in range( Ultrasonic_sensors.nb_sensors ):
            self.distance[i] = Ultrasonic_sensors.us_range
        Arena.compute_min_dist_to_objects( id, Ultrasonic_sensors.us_range, Ultrasonic_sensors.us_angle, self.distance )
        self.compute_reading ( self.distance )
        self.add_noise( )
    
    def compute_reading( self, dist: NDArray[np.float64]):
        for us in range( Ultrasonic_sensors.nb_sensors ):
            self.reading[us] = -1
            if (dist[us] > 0.03 and dist[us] <= 0.04):
                self.reading[us] = -20600 * dist[us] + 924
            elif ( dist[us] > 0.02 and dist[us] <= 0.03):
                self.reading[us] = -37000 * dist[us] + 1416
            elif ( dist[us] > 0.01 and dist[us] <= 0.02):
                self.reading[us] = -153500 * dist[us] + 3746
            elif ( dist[us] > 0.005 and dist[us] <= 0.01):
                self.reading[us] = -252600 * dist[us] + 4737
            elif ( dist[us] >= 0.0 and dist[us] <= 0.005 ):
                self.reading[us] = -124200 * dist[us] + 4095
    
    def add_noise( self ):
        for us in range( Ultrasonic_sensors.nb_sensors ):
            if( self.reading[us] == -1.0 ):
                # just background noise
                self.reading[us] = np.random.randint(0, 150) * Ultrasonic_sensors.noise_level
            else:
                noise = int(np.random.normal(0.0, 50.0)) * Ultrasonic_sensors.noise_level
                self.reading[us] += noise
                if( self.reading[us] > Ultrasonic_sensors.max_us_reading ):
                    self.reading[us] = Ultrasonic_sensors.max_us_reading
                elif ( self.reading[us] < 0.0 ):
                    self.reading[us] = 0.0
            self.reading[us] /= float(Ultrasonic_sensors.max_us_reading)
