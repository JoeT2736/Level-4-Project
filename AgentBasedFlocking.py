import numpy as np

time_update = 0.5
distance = np.ones(10)
neighbours = 4


def velocity_update(distance, noise, neighbours, weight, force, time_update, velocity_neighbour):

    def neighbour_weight(neighbours, weight, velocity_neighbour):
        j = np.zeros(np.len(neighbours))
        for i in range(neighbours):
            j[i] = weight[i] * velocity_neighbour[i]
        return j
    
    def attraction_force(force, distance, neighbours):
        k = np.zeros(np.len(neighbours))
        for i in range(neighbours):
            k[i] = force[i] * (distance[i]/np.mod(distance[i]))
        return k
    
    return (1/neighbours) * (neighbour_weight(neighbours, weight, velocity_neighbour) 
                            + attraction_force(force, distance, noise, neighbours)) + noise  

