import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import scipy.spatial
import scipy.constants


N=30  #Number of agents
D=5  #Size of domain
T=6000   #Total number of time steps (frames) in simulation
stepsize=1  #change in time between calculation of position and angle
eta=0.15   #Random noise added to angles
v0=0.03   #Starting velocity
R=1    #Interaction radius
scale = 3
force_scale = 0.01
w = np.pi/2   #FOV
fake_wall = 1
epsilon = 1e-6

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
            mag = 1.0 / ((dist + epsilon)**scale)   # epsilon for if distance = 0
            Force += mag * norm_vec   

    # changes the force from the wall, to a force that changes the direction of the boid, not the position
    torque = direction[0]*Force[1] - direction[1]*Force[0]

    return torque


def Vicsek():
    global pos
    global angle

    Meandirection = np.zeros(N, )
    noise = np.random.uniform(-eta/2, eta/2, size=(N))
    total_wall_torque = np.zeros(N)

    DistanceMatrix = scipy.spatial.distance.pdist(pos)  #Scipy function to calculate distance between two agents
    DistanceMatrix = scipy.spatial.distance.squareform(DistanceMatrix)   #matrix of form [i, j] => distance between agents i and j
    #returns the distance of each agent to all other agents, the array is of size [N, N]

    for i in range(N):

        Distance_mask = DistanceMatrix <= R #Gives array of True/False, if distance less than R, this returns True

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

        LOS_mask = angle_difference <= w

        Neighbours = Distance_mask & LOS_mask

        Meandirection[i] = (np.arctan2(np.sum(np.sin(angle[Neighbours[:, i]])), np.sum(np.cos(angle[Neighbours[:, i]])))) + noise[i]
                                                        #^^^^^Angles of the agents within R, the True values in 'Neighbours'
        #Equation as in Vicsek 1995 to get the average angle of all the neighbours

        #### Instead of closest distance to wall to find force, force should be based off of distance to wall in the LOS of the boid ####

        wall_torque = wall_force_vector(pos[i], Meandirection[i])
        total_wall_torque[i] = wall_torque
    
    Meandirection = Meandirection + total_wall_torque * force_scale

    cos = np.cos(Meandirection)
    sin = np.sin(Meandirection)

    vx = cos * v0 * stepsize
    vy = sin * v0 * stepsize

    #Updating the position of the agents
    pos[:, 0] += vx
    pos[:, 1] += vy

    angle = Meandirection

    return pos, cos, sin


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
    pos, cos, sin = Vicsek()
    animated_plot_quiver.set_offsets(pos)
    animated_plot_quiver.set_UVC(cos, sin)
    return (animated_plot_quiver,)

anim = FuncAnimation(fig = fig, func = Animate_quiver, interval = 1, frames = T, blit = False, repeat=False)

#anim.save(f"Hemelrijk, Noise Level = {eta}, N={N}, D={D}, potential scale = {force_scale}.gif", dpi=400)
#plt.savefig("2DVicsekAnimation.png", dpi=400)
plt.show()




