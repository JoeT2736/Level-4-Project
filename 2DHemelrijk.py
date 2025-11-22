import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import scipy.spatial

#For infinite potential wall version -> if agent within some range of a wall, add some direction to its direction at each time step.


plt.rcParams["font.family"] = "Times New Roman"

N=5  #Number of agents
D=5  #Size of domain
T=6000   #Total number of time steps (frames) in simulation
timestep=0.2  #change in time between calculation of position and direction
eta=0.15   #Random noise added to directions
v0=0.03   #Starting velocity
v_SD = 0.003   #Standard deviation for gaussian of speed
R=1    #Interaction radius
SD = np.pi/72   #Standard deviation for gaussian distribution of new directions
w_def = np.pi/2     #Default turning rate (per second)
blind_back = 2*np.pi/3     #Area behind fish where it cant see
blind_front = 2*np.pi/3     #Area in front of fish, where it does not align
attraction_scale = 1 
repulsion_scale = 1
aligning_scale = 1
attraction_range = 4
repulsion_range = 0.6
aligning_range = 2   
eccentricity = 2
scale = 2
force_scale = 0.002
fake_wall = 1
w_repel = np.deg2rad(60)    #angle in front of fish for repulsion
w_attract = np.deg2rad(300)
epsilon = 1e-6      #prevents divide by zero in wall force calculations




pos = np.random.uniform(0, D, size=(N, 2))
#direction = np.random.uniform(0, 2*np.pi, size=N)
direction = np.random.uniform(0, 2*np.pi, size=N)

#print(pos)
#print(pos[:, 0])
#print(pos[0, :])

def speed():
    global v0
    vel0 = np.full(shape=N, fill_value=v0)
    return np.random.normal(vel0, scale=v_SD)


def wall_force_vector(position, angle):
                         # x and y direction of the boid
    direction = np.array([np.cos(angle), np.sin(angle)])  
    Force = np.zeros(2)     # setting array for force due to walls

    # distance to walls
    dist_left = position[0]         # x = 0
    dist_right = D - position[0]    # x = D
    dist_bot = position[1]          # y = 0
    dist_top = D - position[1]      # y = D

    # normal vectors pointing away from each wall, alongside the boids respective distance to that wall
    normals = [
        (np.array([1.0, 0.0]), dist_left),       
        (np.array([-1.0, 0.0]), dist_right),    
        (np.array([0.0, 1.0]), dist_bot),        
        (np.array([0.0, -1.0]), dist_top)        
    ]

    # calculate force due to wall, and update 'Force', with the force and its direction using the normal vectors
    for norm_vec, dist in normals:
        if dist < fake_wall:
            mag = 1.0 / ((dist + epsilon)**scale)   # epsilon for if distance = 0
            Force += mag * norm_vec   

    # changes the force from the wall, to a force that changes the direction of the boid, not the position
    torque = direction[0]*Force[1] - direction[1]*Force[0]

    return torque


#print(speed())

def Hemelrijk():
    #print(i)
    global pos
    global direction

    #Meandirection = np.zeros(N, )
    noise = np.random.uniform(-eta/2, eta/2, size=(N))
    #repel_direction=np.full(shape=(N,), fill_value=np.pi/12)
    #r=np.zeros(N,)
    align_angle = np.zeros(N,)
    repel_angle = np.zeros(N,)
    attract_angle = np.zeros(N,)
    w_a = np.zeros(N,)
    w_r = np.zeros(N,)
    w_p = np.zeros(N,)
    rotation = np.zeros(N,)
    weight_a = np.zeros((N,))
    weight_r = np.zeros((N,))
    weight_p = np.zeros((N,))
    force_wall_left = np.zeros(N)
    force_wall_right = np.zeros(N)
    force_wall_bot = np.zeros(N)
    force_wall_top = np.zeros(N)
    vector = np.zeros(shape=(N, N, 2))
    vector_norm = np.zeros(shape=(N))
    direction_i = np.zeros(shape=(N, 2))
    direction_vectors = np.zeros(shape=(N, N, 2))
    dots = np.zeros((N, N))
    angle_difference = np.zeros((N, N))
    total_wall_torque = np.zeros(N)



    DistanceMatrix = scipy.spatial.distance.pdist(pos)  #Scipy function to calculate distance between two agents
    DistanceMatrix = scipy.spatial.distance.squareform(DistanceMatrix)   #matrix of form [i, j] => distance between agents i and j
    #returns the distance of each agent to all other agents, the array is of size [N, N]

    for i in range(N):

        #Gives array of True/False, if distance less than R, this returns True
        #Still circles here, hve to change to be ellipse
        dist_attract_mask = np.logical_and(DistanceMatrix <= attraction_range, DistanceMatrix > repulsion_range, DistanceMatrix > aligning_range) 
        dist_repel_mask = np.logical_and(DistanceMatrix <= repulsion_range, DistanceMatrix > 0)
        dist_align_mask = np.logical_and(DistanceMatrix <= aligning_range, DistanceMatrix > repulsion_range)

        #Vector of boid i to all others (change in x and y)
        vector[i] = pos - pos[i]

        #Direction of i
        direction_i[i] = np.array([np.cos(direction[i]), np.sin(direction[i])])
        #direction_difference = np.array([np.cos(direction), np.sin(direction)]) - np.array([np.cos(direction[i]), np.sin(direction[i])])

        #size of vector from boid i to each other boid
        vector_norm = np.linalg.norm(vector, axis=1)
        vector_norm[vector_norm == 0] = 1e-9   #No divide by zero in next step 
        direction_vectors[i] = vector[i] / vector_norm[i, None]     #scales down the vector between each of the boids to be a unit vector

        #
        dots[i] = np.sum(direction_vectors[i] * direction_i[i], axis=1)     #Relative 'turn' of each boid from all the other boids
        dots = np.clip(dots, -1, 1)     #if values above/below 1 or -1, give those values 1 or -1
        angle_difference[i] = np.arccos(dots[i])    #Difference in direction of each boid relative to all other boids (in radians)
        #gives value of pi/2 when looking at the difference with itself

        LOS_attract_mask = angle_difference <= w_attract/2
        LOS_repel_mask = angle_difference <= w_repel
        LOS_align_mask = np.logical_and(angle_difference > w_repel, angle_difference < np.deg2rad(180 + w_repel))
        
        #each row gives the True/False values for wach boid
        attract_Neighbours = dist_attract_mask & LOS_attract_mask
        repel_Neighbours = dist_repel_mask & LOS_repel_mask
        align_Neighbours = dist_align_mask & LOS_align_mask
        
        ### repulsion rotation ###
        #angle between heading direction of i and the direction of i to j
        #repel_angle[i] = np.angle((np.sqrt(np.sum(x[repel[:, i+1]] - x[repel[:, i]])**2 - np.sum(y[repel[:, i+1]] - y[repel[:, i]])**2), direction[repel[:, i]]))

        repel_angle[i] = np.sum(angle_difference[repel_Neighbours[:, i]])

        if np.sum(repel_Neighbours[:, i]) > 0:
            if repel_angle[i] > np.pi/2:
                w_r[i] = -w_def
            if repel_angle[i] < np.pi/2:
                w_r[i] = w_def
        else: 
            w_r[i] = 0
        
        ##########################

        
        ### attraction rotation ###
        #attract_angle[i] = np.angle((np.sqrt((x[attract[:, i + 1]] - x[attract[:, i]])**2 - (y[attract[:, i + 1]] - y[attract[:, i]])**2) , direction[attract[:, i]]))
        
        attract_angle[i] = np.sum(angle_difference[attract_Neighbours[:, i]])
        w_a[i] = w_def * attract_angle[i]

        ###########################


        ### alignment rotation ###
        #difference between angle of diections of i and j
        #align_angle[i] = direction[align[:, i]] - direction[align[:, i + 1]]

        if np.sum(align_Neighbours) > 0:
            direction_difference = direction - direction[i]
            align_angle[i] = np.sum(direction_difference[align_Neighbours[:, i]])
            w_p[i] = w_def * align_angle[i]
        else:
            w_p[i] = 0

        ##########################

        
        #weight of repulsion
        if np.sum(repel_Neighbours[:, i]) > 0:
            DistanceMatrix[DistanceMatrix == 0] = 1e-9
            weight_r[i] = (0.005 * repulsion_scale / np.sum(DistanceMatrix[repel_Neighbours[:, i]])**3)
            weight_r[weight_r == np.inf] = 0

        #weight of attraction
        weight_a[i] = 0.2 * attraction_scale * np.exp(-((np.sum(DistanceMatrix[attract_Neighbours[:, i]]) - 0.5*(attraction_range + aligning_range))/(attraction_range - aligning_range))**2)

        #weight of alignment
        weight_p[i] = aligning_scale * np.exp(-((np.sum(DistanceMatrix[align_Neighbours[:, i]]) - 0.5*(aligning_range + repulsion_range))/(aligning_range - repulsion_range))**2)


        #behavioural reaction = weighted sum
        rotation = weight_r*w_r + weight_a*w_a + weight_p*w_p

        #wall_torque = wall_force_vector(pos[i], Meandirection[i])
        #total_wall_torque[i] = wall_torque
    
    #Meandirection = Meandirection + total_wall_torque * force_scale

        #new direction of ith agent
        Meandirection = np.random.normal(direction + rotation*timestep, scale=SD)
        
    #force_wall = force_wall_left + force_wall_right + force_wall_bot + force_wall_top

    #Meandirection = Meandirection + force_wall*force_scale
    
    #x and y directions accoring to new direction
    cos = (np.cos(Meandirection))
    sin = (np.sin(Meandirection))

        #Updating the position of the agents
    pos[:, 0] += cos * speed() * timestep
    pos[:, 1] += sin * speed() * timestep
    
    pos = np.mod(pos, D)

    #direction = Meandirection

    #pos = np.mod(pos, D)    #Agent appears on other side of domain when goes off one end

    return pos, cos, sin, weight_r*w_r , weight_p*w_p, weight_a*w_a, np.sum(DistanceMatrix[align_Neighbours[:, 0]]), - 0.5*(aligning_range + repulsion_range)/(aligning_range - repulsion_range), weight_a#, angle_difference[repel_Neighbours[:, 4]], angle_difference, align_Neighbours, w_p


#print((Hemelrijk()[3]))
#print((Hemelrijk()[4]))
#print((Hemelrijk()[5]))
print((Hemelrijk()[3]))
print((Hemelrijk()[4]))
print((Hemelrijk()[5]))
print((Hemelrijk()[6]))



#fig, ax = plt.subplots()
#ax.set_xlim([0, D])
#ax.set_ylim([0, D])
#colours=['g', 'r', 'b', 'y', 'black']
#ax.quiver(pos[:, 0], pos[:, 1], np.cos(direction), np.sin(direction), clim=[-np.pi, np.pi], color=colours)
#plt.show()


### Code to get animation ###

fig, ax = plt.subplots()
ax.set_xlim([0, D])
ax.set_ylim([0, D])
animated_plot_quiver = ax.quiver(pos[:, 0], pos[:, 1], np.cos(direction), np.sin(direction), clim=[-np.pi, np.pi])#, direction)


plt.rcParams["font.family"] = "Times New Roman"
plt.title(f'Noise level = {eta}', fontsize=16)
plt.xlabel('X-axis', fontsize=14)
plt.ylabel('Y-axis', fontsize=14)
plt.setp(ax.get_xticklabels(), visible='False')
plt.setp(ax.get_yticklabels(), visible='False')
plt.locator_params(axis='y', nbins=1)
plt.locator_params(axis='x', nbins=1)
ax.set_aspect('equal', adjustable='box')


def Animate_quiver(frame):
    pos, cos, sin, k, l, j, m, i = Hemelrijk()
    animated_plot_quiver.set_offsets(pos)
    animated_plot_quiver.set_UVC(cos, sin)#, Vicsek()[3])
    #print(j)
    return (animated_plot_quiver, )
#Animate_quiver

anim = FuncAnimation(fig = fig, func = Animate_quiver, interval = 1, frames = T, blit = False, repeat=False)

#anim.save(f"Hemelrijk, Pop={N}.gif", dpi=400)
#plt.savefig("2DVicsekAnimation.png", dpi=400)
plt.show()