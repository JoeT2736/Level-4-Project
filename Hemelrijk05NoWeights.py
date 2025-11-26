import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import scipy.spatial
import scipy.constants


N=10  #Number of agents
D=5  #Size of domain
T=600   #Total number of time steps (frames) in simulation
stepsize=0.2 #s      #change in time between calculation of position and angle
eta=0.15   #Random noise added to angles
v0=0.03   #Starting velocity
R=1    #Interaction radius
scale = 3
force_scale = 0.01
fake_wall = 1
epsilon = 1e-6
v_SD = 0.003     #m/s
direction_SD = np.pi/72
turning_rate = (np.pi/2)/10 #rad/s
repulsion_range_s = 0.3   #m  for small fish
repulsion_range_l = 0.6   #m  for large fish
attraction_range = 5    #m  
aligning_range_s = 1      #m  for small fish
aligning_range_l = 2      #m  for large fish

np.random.seed(10)

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

        repel_mask_distance_s = DistanceMatrix <= repulsion_range_s
        align_mask_distance_s = np.logical_and(DistanceMatrix <= aligning_range_s, DistanceMatrix > repulsion_range_s) 
        attract_mask_distance_s = np.logical_and(DistanceMatrix <= attraction_range, DistanceMatrix > aligning_range_s)


        #Change mask values to True for when DistanceMatrix value is the distance of an agent to itself
        align_mask_distance_s[:, i][i] = True
        attract_mask_distance_s[:, i][i] = True


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
            attract_angle[i] = np.sum(angle_difference[attract_mask_distance_s[:, i]])
            attract_torque[i] = turning_rate * attract_angle[i]
        else:
            attract_torque[i] = 0

            

    Meandirection += repel_torque + align_torque + attract_torque
        


        #total_wall_torque[i] = wall_force_vector(pos[i], Meandirection[i])
    
    #Meandirection = Meandirection + total_wall_torque * force_scale

    cos = np.cos(Meandirection)
    sin = np.sin(Meandirection)

    vx = cos * v0 * stepsize
    vy = sin * v0 * stepsize

    #Updating the position of the agents
    pos[:, 0] += vx
    pos[:, 1] += vy

    #pos[:, 0] = np.clip(pos[:, 0], 0.0, D)
    #pos[:, 1] = np.clip(pos[:, 1], 0.0, D)

    angle = Meandirection

    pos = np.mod(pos, D)

    return pos, cos, sin, align_mask_distance_s

print((update()[3]))




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
    pos, cos, sin, k = update()
    animated_plot_quiver.set_offsets(pos)
    animated_plot_quiver.set_UVC(cos, sin)
    return (animated_plot_quiver,)
#Animate_quiver

anim = FuncAnimation(fig = fig, func = Animate_quiver, interval = 1, frames = T, blit = False, repeat=False)

#anim.save(f"Potential Walls, Noise Level = {eta}, N={N}, D={D}, potential scale = {force_scale}.gif", dpi=400)
#plt.savefig("2DVicsekAnimation.png", dpi=400)
plt.show()

