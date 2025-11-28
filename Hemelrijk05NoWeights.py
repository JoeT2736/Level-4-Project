'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import scipy.spatial
import scipy.constants


N=40  #Number of agents
D=20  #Size of domain
T=1000   #Total number of time steps (frames) in simulation
stepsize=0.2 #seconds      #change in time between calculation of position and angle
eta=0.15   #Random noise added to angles
v0=0.3   # 0.3m/s     #Starting velocity
R=1    #Interaction radius
scale = 3
force_scale = 0.01
fake_wall = 1
epsilon = 1e-6
v_SD = 0.03     # 0.03m/s
direction_SD = np.pi/72
turning_rate = (np.pi/2)*stepsize #pi/2 rad/s
repulsion_range_s = 0.3   #meters  for small fish
repulsion_range_l = 0.6   #m  for large fish
attraction_range = 5    #m  
aligning_range_s = 1      #m  for small fish
aligning_range_l = 2      #m  for large fish

#np.random.seed(10)

pos = np.random.uniform(0, D, size=(N, 2))
angle = np.random.uniform(0, 2*np.pi, size=N)


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
            mag = 1.0 / ((dist + epsilon)**scale)
            Force += mag * norm_vec   

    # changes the force from the wall, to a force that changes the direction of the boid, not the position
    torque = direction[0]*Force[1] - direction[1]*Force[0]

    return torque


def speed():
    global v0
    vel0 = np.full(shape=N, fill_value=v0)
    return np.random.normal(vel0, scale=v_SD)


def update():

    global pos
    global angle

    Meandirection = angle
    noise = np.random.uniform(-eta/2, eta/2, size=(N))
    total_wall_torque = np.zeros(N)
    repel_angle = np.zeros(N)
    repel_torque = np.zeros(N)
    align_angle = np.zeros(N)
    align_torque = np.zeros(N)
    attract_angle = np.zeros(N)
    attract_torque = np.zeros(N)
    direction_difference = np.zeros(N)

    DistanceMatrix = scipy.spatial.distance.pdist(pos)  #Scipy function to calculate distance between two agents
    DistanceMatrix = scipy.spatial.distance.squareform(DistanceMatrix)   #matrix of form [i, j] => distance between agents i and j
        #returns the distance of each agent to all other agents, the array is of size [N, N]

    for i in range(N):

        #Distance_Mask = DistanceMatrix <= R #Gives array of True/False, if distance less than R, this returns True
        
        #Vector of boid i to all others (change in x and y)
        vector = pos - pos[i]

        #Direction of i
        direction_i = np.array([np.cos(angle[i]), np.sin(angle[i])])

        #size of vector from boid i to each other boid
        vector_norm = np.linalg.norm(vector, axis=1)
        vector_norm[vector_norm == 0] = 1e-9   #No divide by zero in next step 
        direction_vectors = vector / vector_norm[:, None]

        #
        dots = np.sum(direction_vectors * direction_i, axis=1)
        dots = np.clip(dots, -1, 1)
        angle_difference = np.arccos(dots)
        angle_difference2 = np.arctan(dots)

        repel_mask_distance_s = DistanceMatrix <= repulsion_range_s
        align_mask_distance_s = np.logical_and(DistanceMatrix <= aligning_range_s, DistanceMatrix > repulsion_range_s) 
        attract_mask_distance_s = np.logical_and(DistanceMatrix <= attraction_range, DistanceMatrix > aligning_range_s)


        #Change mask values to True for when DistanceMatrix value is the distance of an agent to itself
        for j in range (N):
            align_mask_distance_s[:, j][j] = True
            #attract_mask_distance_s[:, j][j] = True   


        #repel_angle[i] = (angle_difference[repel_mask_distance_s[:, i]])

        ##### repulsion #####
        repel_angle[i] = np.sum(angle_difference[repel_mask_distance_s[:, i]])

        if np.sum(repel_mask_distance_s[:, i]) > 0:
            if repel_angle[i] > np.pi/2:
                repel_torque[i] = -turning_rate
            if repel_angle[i] < np.pi/2:
                repel_torque[i] = turning_rate
        else: 
            repel_torque[i] = 0


        ##### alignment #####
        if np.sum(align_mask_distance_s[:, i]) > 0:
            direction_difference = angle - angle[i]
            align_angle[i] = np.sum(direction_difference[align_mask_distance_s[:, i]])
            align_torque[i] = turning_rate * align_angle[i]     #times by 0.7 or less if they still spin about
        else:
            align_torque[i] = 0

        
        ##### attraction #####
        if np.sum(attract_mask_distance_s[:, i]) > 0:
            attract_angle[i] = np.sum(angle_difference2[attract_mask_distance_s[:, i]])
            attract_torque[i] = turning_rate * attract_angle[i]
        else:
            attract_torque[i] = 0

            

        Meandirection[i] += repel_torque[i] + align_torque[i] + attract_torque[i]
        


        #total_wall_torque[i] = wall_force_vector(pos[i], Meandirection[i])
    
    #Meandirection = Meandirection + total_wall_torque * force_scale

    update_angle = np.random.normal(Meandirection, direction_SD)

    cos = np.cos(update_angle)
    sin = np.sin(update_angle)

    vx = cos * v0 * stepsize
    vy = sin * v0 * stepsize

    #Updating the position of the agents
    pos[:, 0] += vx
    pos[:, 1] += vy

    #pos[:, 0] = np.clip(pos[:, 0], 0.0, D)
    #pos[:, 1] = np.clip(pos[:, 1], 0.0, D)

    angle = update_angle

    pos = np.mod(pos, D)

    return pos, cos, sin, attract_mask_distance_s, np.sum(angle_difference[attract_mask_distance_s[:, 4]])

#print((update()[3]))
#print((update()[4]))



fig, ax = plt.subplots()
ax.set_xlim([0, D])
ax.set_ylim([0, D])
animated_plot_quiver = ax.quiver(pos[:, 0], pos[:, 1], np.cos(angle), np.sin(angle), clim=[-np.pi, np.pi])#, angle)


plt.rcParams["font.family"] = "Times New Roman"
plt.title(f'Potential Walls', fontsize=16)
plt.xlabel('X-axis', fontsize=14)
plt.ylabel('Y-axis', fontsize=14)
plt.setp(ax.get_xticklabels(), visible='False')
plt.setp(ax.get_yticklabels(), visible='False')
plt.locator_params(axis='y', nbins=1)
plt.locator_params(axis='x', nbins=1)
ax.set_aspect('equal', adjustable='box')


def Animate_quiver(frame):
    pos, cos, sin, k, m = update()
    animated_plot_quiver.set_offsets(pos)
    animated_plot_quiver.set_UVC(cos, sin)
    return (animated_plot_quiver,)
#Animate_quiver

anim = FuncAnimation(fig = fig, func = Animate_quiver, interval = 1, frames = T, blit = False, repeat=False)

anim.save(f"Hemelrijk, no LOS mask.gif", dpi=400)
#plt.savefig("2DVicsekAnimation.png", dpi=400)
plt.show()
'''



####### With LOS ######
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import scipy.spatial
import scipy.constants


N=40  #Number of agents
D=8  #Size of domain
T=1000   #Total number of time steps (frames) in simulation
stepsize=0.2 #seconds      #change in time between calculation of position and angle
eta=0.15   #Random noise added to angles
v0=0.3   # 0.3m/s     #Starting velocity
R=1    #Interaction radius
scale = 3
force_scale = 0.01
fake_wall = 1
epsilon = 1e-6
v_SD = 0.03     # 0.03m/s
direction_SD = np.pi/72
turning_rate = (np.pi/2)/10 #pi/2 rad/s
repulsion_range_s = 0.3   #meters  for small fish
repulsion_range_l = 0.6   #m  for large fish
attraction_range = 5    #m  
aligning_range_s = 1      #m  for small fish
aligning_range_l = 2      #m  for large fish
eccentricity_s = 2
eccentricity_l = 4
repel_LOS = np.deg2rad(60)
attract_LOS = np.deg2rad(300)

np.random.seed(12)

pos = np.random.uniform(0, D, size=(N, 2))
angle = np.random.uniform(0, 2*np.pi, size=N)


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
            mag = 1.0 / ((dist + epsilon)**scale)
            Force += mag * norm_vec   

    # changes the force from the wall, to a force that changes the direction of the boid, not the position
    torque = direction[0]*Force[1] - direction[1]*Force[0]

    return torque


def speed():
    global v0
    vel0 = np.full(shape=N, fill_value=v0)
    return np.random.normal(vel0, scale=v_SD)


def update():

    global pos
    global angle

    Meandirection = angle
    noise = np.random.uniform(-eta/2, eta/2, size=(N))
    total_wall_torque = np.zeros(N)
    repel_angle = np.zeros(N)
    repel_torque = np.zeros(N)
    align_angle = np.zeros(N)
    align_torque = np.zeros(N)
    attract_angle = np.zeros(N)
    attract_torque = np.zeros(N)
    direction_difference = np.zeros(N)
    repel_mask_LOS = np.zeros((N, N))
    align_mask_LOS = np.zeros((N, N))
    attract_mask_LOS = np.zeros((N, N))
    direction_i = np.zeros((N, 2))
    dots = np.zeros((N, N))
    angle_difference = np.zeros((N, N))
    angle_difference2 = np.zeros((N, N))
    vector = np.zeros((N, N, 2))
    vector_norm = np.zeros(N)
    direction_vectors = np.zeros((N, N, 2))
    

    DistanceMatrix = scipy.spatial.distance.pdist(pos)  #Scipy function to calculate distance between two agents
    DistanceMatrix = scipy.spatial.distance.squareform(DistanceMatrix)   #matrix of form [i, j] => distance between agents i and j
        #returns the distance of each agent to all other agents, the array is of size [N, N]

    for i in range(N):

        #Distance_Mask = DistanceMatrix <= R #Gives array of True/False, if distance less than R, this returns True
        
        #Vector of boid i to all others (change in x and y)
        vector[i] = pos - pos[i]
        
        #Direction of i
        direction_i[i] = np.array([np.cos(angle[i]), np.sin(angle[i])])
        
        #size of vector from boid i to each other boid
        vector_norm[i] = np.linalg.norm(vector[i])
        direction_vectors[i] = vector[i] / vector_norm[i]

        
        #values from 0 to 1 (rate of full turn needed for boid to align with another) = dots
        dots[i] = np.sum(direction_vectors[i] * direction_i[i], axis=1)
        dots[i] = np.clip(dots[i], -1, 1)   #incase gives value above/below 1 or -1 (if above/below set equal to 1 or -1)
        angle_difference[i] = np.arccos(dots[i])      #used for LOS angles
        angle_difference2[i] = np.arctan(dots[i])     #used for attract angle
        
        repel_mask_distance_s = DistanceMatrix <= repulsion_range_s
        align_mask_distance_s = np.logical_and(DistanceMatrix <= aligning_range_s, DistanceMatrix > repulsion_range_s) 
        attract_mask_distance_s = np.logical_and(DistanceMatrix <= attraction_range, DistanceMatrix > aligning_range_s)

        repel_mask_LOS = angle_difference <= repel_LOS/2
        align_mask_LOS = np.logical_and(angle_difference > repel_LOS/2, angle_difference <= np.deg2rad(90) + repel_LOS/2)
        attract_mask_LOS = angle_difference <= attract_LOS/2


        repel_mask_s = np.logical_and(repel_mask_distance_s, repel_mask_LOS)
        align_mask_s = np.logical_and(align_mask_distance_s, align_mask_LOS)
        attract_mask_s = np.logical_and(attract_mask_distance_s, attract_mask_LOS)


        ##### repulsion #####
        repel_angle[i] = np.sum(angle_difference[repel_mask_s[:, i]])

        if np.sum(repel_mask_s[:, i]) > 0:
            if repel_angle[i] > np.pi/2:
                repel_torque[i] = -turning_rate
            if repel_angle[i] < np.pi/2:
                repel_torque[i] = turning_rate
        else: 
            repel_torque[i] = 0


        ##### alignment #####
        if np.sum(align_mask_s[:, i]) > 0:
            direction_difference = angle - angle[i]
            align_angle[i] = np.sum(direction_difference[align_mask_s[:, i]])
            align_torque[i] = turning_rate * align_angle[i]     #times by 0.7 or less if they still spin about
        else:
            align_torque[i] = 0

        
        ##### attraction #####
        if np.sum(attract_mask_distance_s[:, i]) > 0:
            attract_angle[i] = np.sum(angle_difference2[attract_mask_distance_s[i]])
            attract_torque[i] = turning_rate * attract_angle[i]
        else:
            attract_torque[i] = 0

            

        #Meandirection[i] += repel_torque[i] + align_torque[i] 
        Meandirection[i] += attract_torque[i]
    

        #total_wall_torque[i] = wall_force_vector(pos[i], Meandirection[i])
    
    #Meandirection = Meandirection + total_wall_torque * force_scale

    update_angle = np.random.normal(Meandirection, direction_SD)

    cos = np.cos(update_angle)
    sin = np.sin(update_angle)

    vx = cos * v0 * stepsize
    vy = sin * v0 * stepsize

    #Updating the position of the agents
    pos[:, 0] += vx
    pos[:, 1] += vy

    angle = update_angle

    pos = np.mod(pos, D)

    return pos, cos, sin, attract_mask_distance_s, attract_mask_LOS, attract_mask_s

#print((update()[3]))
#print((update()[4]))
#print((update()[5]))




fig, ax = plt.subplots()
ax.set_xlim([0, D])
ax.set_ylim([0, D])
animated_plot_quiver = ax.quiver(pos[:, 0], pos[:, 1], np.cos(angle), np.sin(angle), clim=[-np.pi, np.pi])


plt.rcParams["font.family"] = "Times New Roman"
plt.title(f'Potential Walls', fontsize=16)
plt.xlabel('X-axis', fontsize=14)
plt.ylabel('Y-axis', fontsize=14)
plt.setp(ax.get_xticklabels(), visible='False')
plt.setp(ax.get_yticklabels(), visible='False')
plt.locator_params(axis='y', nbins=1)
plt.locator_params(axis='x', nbins=1)
ax.set_aspect('equal', adjustable='box')


def Animate_quiver(frame):
    pos, cos, sin, k, k, l = update()
    animated_plot_quiver.set_offsets(pos)
    animated_plot_quiver.set_UVC(cos, sin)
    return (animated_plot_quiver,)


anim = FuncAnimation(fig = fig, func = Animate_quiver, interval = 1, frames = T, blit = False, repeat=False)

#anim.save(f"Hemelrijk, with LOS mask.gif", dpi=400)
#plt.savefig("2DVicsekAnimation.png", dpi=400)
plt.show()


