import numpy as np

def velocity_update(distance, noise, neighbours, weight, force, time_update, velocity_neighbour):

    def neighbour_weight(neighbours, weight, velocity_neighbour):
        for i in range(neighbours):
            j = weight[i] * velocity_neighbour[i]
        return j
    
    def attraction_force(force, distance, noise, neighbours):
        for i in range(neighbours):
            k = (force[i] * (distance[i]/np.mod(distance[i])) + noise[i])
        return k
    
    return (1/neighbours) * (neighbour_weight(neighbours, weight, velocity_neighbour) + attraction_force(force, distance, noise, neighbours))        
