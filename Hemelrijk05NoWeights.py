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
turning_rate = (np.pi/2) #pi/2 rad/s
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


def circ_mean(angles):
    sin = np.mean(np.sin(angles))
    cos = np.mean(np.cos(angles))
    return np.arctan2(sin, cos)


def angle_wrap(a):
    """Wrap to [-pi, pi]."""
    return (a + np.pi) % (2 * np.pi) - np.pi


def angular_difference_signed(target, source):
    """Signed smallest angle target - source in [-pi, pi]."""
    return angle_wrap(target - source)


def update():

    global pos
    global angle

    DistanceMatrix = scipy.spatial.distance.pdist(pos)  #Scipy function to calculate distance between two agents
    DistanceMatrix = scipy.spatial.distance.squareform(DistanceMatrix)   #matrix of form [i, j] => distance between agents i and j
        #returns the distance of each agent to all other agents, the array is of size [N, N]

    vec_ij = pos[None, :, :] - pos[:, None, :]  

    dist = np.linalg.norm(vec_ij, axis=2)  

    np.fill_diagonal(dist, np.inf)

    bearing_ij = np.arctan2(vec_ij[:, :, 1], vec_ij[:, :, 0])
    rel_bearing_ij = angle_wrap(bearing_ij - angle[:, None])
    
    repel_mask_distance_s = dist <= repulsion_range_s
    align_mask_distance_s = (dist > repulsion_range_s) & (dist <= aligning_range_s)
    attract_mask_distance_s = (dist > aligning_range_s) & (dist <= attraction_range)

    rotation = np.zeros(N)


    for i in range(N):


        ##### repulsion #####
        repel_idx = np.where(repel_mask_distance_s[i])[0]

        if repel_idx.size > 0:
            
            signs = np.sign(rel_bearing_ij[i, repel_idx])

            signs[signs == 0] = np.random.choice([-1, 1], size=(signs == 0).sum())

            repel_rotate = -turning_rate * signs

            rotation[i] = np.mean(repel_rotate)

        else:

            ##### alignment #####
            align_idx = np.where(align_mask_distance_s[i])[0]
            algin_contribution = 0

            if align_idx.size > 0:

                angle_difference = angular_difference_signed(angle[align_idx], angle[i])

                align_rotate = turning_rate * angle_difference

                algin_contribution = np.mean(align_rotate)
            

            ##### attraction #####
            attract_idx = np.where(attract_mask_distance_s[i])[0]
            attract_contribution = 0

            if attract_idx.size > 0:

                heading_vals = np.cos(rel_bearing_ij[i, attract_idx])

                sign_to_j = np.sign(rel_bearing_ij[i, attract_idx])
                sign_to_j[sign_to_j == 0] = 1

                attract_rotate = turning_rate * heading_vals * sign_to_j
                #w_attr = 1.0 / (dist[i, attract_idx] + epsilon)**2

                attract_contribution = np.mean(attract_rotate)# * w_attr)

            rotation[i] = algin_contribution + attract_contribution


        #rotation[i] = np.clip(rotation[i], -10*turning_rate, 10*turning_rate)


        #total_wall_torque[i] = wall_force_vector(pos[i], Meandirection[i])
    
    #Meandirection = Meandirection + total_wall_torque * force_scale

    mean_heading = angle + rotation * stepsize

    update_angle = np.random.normal(mean_heading, direction_SD)

    cos = np.cos(update_angle)
    sin = np.sin(update_angle)

    vx = cos * v0 * stepsize
    vy = sin * v0 * stepsize

    #Updating the position of the agents
    pos[:, 0] += vx
    pos[:, 1] += vy

    angle[:] = update_angle

    pos[:] = np.mod(pos, D)

    return pos, cos, sin, rotation



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
    pos, cos, sin, r = update()
    animated_plot_quiver.set_offsets(pos)
    animated_plot_quiver.set_UVC(cos, sin)
    #print(r/(np.pi*2))
    return (animated_plot_quiver,)
#Animate_quiver


anim = FuncAnimation(fig = fig, func = Animate_quiver, interval = 1, frames = T, blit = False, repeat=False)

#anim.save(f"Hemelrijk, no LOS mask.gif", dpi=400)
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
D=15  #Size of domain
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
turning_rate_attraction = (np.pi/2)
turning_rate = (np.pi/2) #pi/2 rad/s
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


def circ_mean(angles):
    sin = np.mean(np.sin(angles))
    cos = np.mean(np.cos(angles))
    return np.arctan2(sin, cos)


def angle_wrap(a):
    """Wrap to [-pi, pi]."""
    return (a + np.pi) % (2 * np.pi) - np.pi


def angular_difference_signed(target, source):
    """Signed smallest angle target - source in [-pi, pi]."""
    return angle_wrap(target - source)



def update():

    global pos
    global angle

    #angle_difference = np.zeros((N, N))
    

    DistanceMatrix = scipy.spatial.distance.pdist(pos)  #Scipy function to calculate distance between two agents
    DistanceMatrix = scipy.spatial.distance.squareform(DistanceMatrix)   #matrix of form [i, j] => distance between agents i and j
        #returns the distance of each agent to all other agents, the array is of size [N, N]

    vec_ij = pos[None, :, :] - pos[:, None, :]  

    dist = np.linalg.norm(vec_ij, axis=2)  

    np.fill_diagonal(dist, np.inf)

    bearing_ij = np.arctan2(vec_ij[:, :, 1], vec_ij[:, :, 0])
    rel_bearing_ij = angle_wrap(bearing_ij - angle[:, None])
    
    repel_mask_distance_s = dist <= repulsion_range_s
    align_mask_distance_s = (dist > repulsion_range_s) & (dist <= aligning_range_s)
    attract_mask_distance_s = (dist > aligning_range_s) & (dist <= attraction_range)

    repel_mask_LOS = np.abs(rel_bearing_ij) <= (repel_LOS/2)
    align_mask_LOS = (np.abs(rel_bearing_ij) > (repel_LOS/2)) & (np.abs(rel_bearing_ij)  <= (np.deg2rad(90) + repel_LOS/2))
    attract_mask_LOS = np.abs(rel_bearing_ij) <= (attract_LOS/2)

    repel_mask_s = repel_mask_distance_s & repel_mask_LOS
    align_mask_s = align_mask_distance_s & align_mask_LOS
    attract_mask_s = attract_mask_distance_s & attract_mask_LOS

    rotation = np.zeros(N)

    for i in range(N):

        ##### repulsion #####
        repel_idx = np.where(repel_mask_s[i])[0]

        if repel_idx.size > 0:
            
            signs = np.sign(rel_bearing_ij[i, repel_idx])

            signs[signs == 0] = np.random.choice([-1, 1], size=(signs == 0).sum())

            repel_rotate = -turning_rate * signs

            rotation[i] = np.mean(repel_rotate)

        else:

            ##### alignment #####
            align_idx = np.where(align_mask_s[i])[0]
            algin_contribution = 0

            if align_idx.size > 0:

                angle_difference = angular_difference_signed(angle[align_idx], angle[i])

                align_rotate = turning_rate * angle_difference

                algin_contribution = np.mean(align_rotate)
            

            ##### attraction #####
            attract_idx = np.where(attract_mask_s[i])[0]
            attract_contribution = 0

            if attract_idx.size > 0:

                heading_vals = np.cos(rel_bearing_ij[i, attract_idx])

                sign_to_j = np.sign(rel_bearing_ij[i, attract_idx])
                sign_to_j[sign_to_j == 0] = 1

                attract_rotate = turning_rate * heading_vals * sign_to_j
                #w_attr = 1.0 / (dist[i, attract_idx] + epsilon)**2

                attract_contribution = np.mean(attract_rotate)# * w_attr)

            rotation[i] = algin_contribution + attract_contribution


        #rotation[i] = np.clip(rotation[i], -10*turning_rate, 10*turning_rate)

    

        #total_wall_torque[i] = wall_force_vector(pos[i], Meandirection[i])
    

    mean_heading = angle + rotation * stepsize

    update_angle = np.random.normal(mean_heading, direction_SD)


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
animated_plot_quiver = ax.quiver(pos[:, 0], pos[:, 1], np.cos(angle), np.sin(angle), clim=[-np.pi, np.pi], pivot='mid')
#animated_plot_repel_zones = ax.annotate("", xy=pos, xytext=pos)


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
    pos, cos, sin, k, d, l = update()
    animated_plot_quiver.set_offsets(pos)
    animated_plot_quiver.set_UVC(cos, sin)
    #print(k, d, l)
    return (animated_plot_quiver,)


anim = FuncAnimation(fig = fig, func = Animate_quiver, interval = 1, frames = T, blit = False, repeat=False)

anim.save(f"Hemelrijk, with LOS mask.gif", dpi=400)
#plt.savefig("2DVicsekAnimation.png", dpi=400)
plt.show()


