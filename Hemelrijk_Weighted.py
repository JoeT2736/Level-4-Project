'''
####### With LOS ######
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import scipy.spatial
import scipy.constants


N=100  #Number of agents
D=10  #Size of domain
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
#turning_rate_attraction = (np.pi/2)/80
#turning_rate = (np.pi/2)/2 #pi/2 rad/s
turning_rate = (np.pi/2)/5 #pi/2 rad/s
repulsion_range_s = 0.3   #meters  for small fish
repulsion_range_l = 0.6   #m  for large fish
attraction_range = 5    #m  
aligning_range_s = 1      #m  for small fish
aligning_range_l = 2      #m  for large fish
eccentricity_s = 2
eccentricity_l = 4
repel_LOS = np.deg2rad(60)
attract_LOS = np.deg2rad(300)
repel_scalefactor_s = 1
repel_scalefactor_l = 2
align_scalefactor = 1
attract_scalefactor = 1

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
    #repel_mask_LOS = np.zeros((N, N))
    #align_mask_LOS = np.zeros((N, N))
    #attract_mask_LOS = np.zeros((N, N))
    direction_i = np.zeros((N, 2))
    dots = np.zeros((N, N))
    angle_difference = np.zeros((N, N))
    angle_difference2 = np.zeros((N, N))
    vector = np.zeros((N, N, 2))
    vector_norm = np.zeros(N)
    direction_vectors = np.zeros((N, N, 2))
    repel_weight = np.zeros(N)
    align_weight = np.zeros(N)
    attract_weight = np.zeros(N)
    

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
        #vector_norm[i] = np.linalg.norm(vector[i])
        #direction_vectors[i] = vector[i] / vector_norm[i]

        #corrected version
        vector_norm[i] = np.linalg.norm(vector[i], axis=1)
        direction_vectors[i] = vector[i] / (vector_norm[i][:,None] + 1e-9)

        
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

        #Change mask values to True for when DistanceMatrix value is the distance of an agent to itself
        #for j in range (N):
        #    repel_mask_s[:, j][j] = True
        #    align_mask_s[:, j][j] = True
        #    attract_mask_s[:, j][j] = True


        ##### repulsion #####
        repel_angle[i] = np.sum((angle_difference[repel_mask_s[i]]))

        if np.sum(repel_mask_s[i]) > 0:
            if repel_angle[i] >= np.pi/2:
                repel_torque[i] = -turning_rate
            if repel_angle[i] < np.pi/2:
                repel_torque[i] = turning_rate
        else: 
            repel_torque[i] = 0

        
        repel_weight[i] = np.sum((0.05*repel_scalefactor_s)/DistanceMatrix[repel_mask_s[i]])

        repel_torque[i] *= repel_weight[i]


        ##### alignment #####
        if np.sum(align_mask_s[i]) > 0:
            direction_difference = (angle - angle[i])
            align_angle[i] = np.mean(direction_difference[align_mask_s[i]])
            align_torque[i] = turning_rate * align_angle[i]     #times by 0.7 or less if they still spin about
        else:
            align_torque[i] = 0

        
        align_weight[i] = np.sum(align_scalefactor * np.exp(-(DistanceMatrix[align_mask_s[i]] - 0.5 * (aligning_range_s + repulsion_range_s)/(aligning_range_s - repulsion_range_s))**2))

        align_torque[i] *= align_weight[i]

        
        ##### attraction #####
        if np.sum(attract_mask_distance_s[i]) > 0:
            angle_difference2 = (angle_difference2)
            attract_angle[i] = np.mean(angle_difference2[attract_mask_distance_s[i]])
            attract_torque[i] = turning_rate * attract_angle[i]
        else:
            attract_torque[i] = 0

        
        attract_weight[i] = np.sum(0.2 * attract_scalefactor * np.exp(-(DistanceMatrix[attract_mask_s[i]] - 0.5 * (attraction_range + aligning_range_s)/(attraction_range - aligning_range_s))**2))

        attract_torque[i] *= attract_weight[i]

        #attract_torque[i] = np.mean(attract_torque)

            
        Meandirection[i] += repel_torque[i] 
        Meandirection[i] += align_torque[i] 
        Meandirection[i] += attract_torque[i]

        #Meandirection[i] += (align_torque[i])# + attract_torque[i])
    

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

    return pos, cos, sin, DistanceMatrix, repel_mask_LOS[:, 1], DistanceMatrix[repel_mask_s[:, 0]], repel_weight, repel_torque, direction_difference

#print((update()[3]))
#print((update()[4]))
#print((update()[5]))
#print((update()[6]))
#print((update()[7]))
print((update()[8]))





fig, ax = plt.subplots()
ax.set_xlim([0, D])
ax.set_ylim([0, D])
animated_plot_quiver = ax.quiver(pos[:, 0], pos[:, 1], np.cos(angle), np.sin(angle), clim=[-np.pi, np.pi], pivot='mid')


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
    pos, cos, sin, k, d, l, j, o, p = update()
    animated_plot_quiver.set_offsets(pos)
    animated_plot_quiver.set_UVC(cos, sin)
    #print(d)
    return (animated_plot_quiver,)


anim = FuncAnimation(fig = fig, func = Animate_quiver, interval = 1, frames = T, blit = False, repeat=False)

#anim.save(f"Hemelrijk, with weights.gif", dpi=400)
#plt.savefig("2DVicsekAnimation.png", dpi=400)
plt.show()
'''





########## altered ##########

'''
####### With LOS ######
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import scipy.spatial
import scipy.constants


N=100  #Number of agents
D=10  #Size of domain
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
#turning_rate_attraction = (np.pi/2)/80
#turning_rate = (np.pi/2)/2 #pi/2 rad/s
turning_rate = (np.pi/2) #pi/2 rad/s
repulsion_range_s = 0.3   #meters  for small fish
repulsion_range_l = 0.6   #m  for large fish
attraction_range = 5      #m  
aligning_range_s = 1      #m  for small fish
aligning_range_l = 2      #m  for large fish
eccentricity_s = 2
eccentricity_l = 4
repel_LOS = np.deg2rad(60)
attract_LOS = np.deg2rad(300)
repel_scalefactor_s = 1
repel_scalefactor_l = 2
align_scalefactor = 1
attract_scalefactor = 1
active_sort_repel = 2
active_sort_align = 2
active_sort_attract = 2
risk_avoidance = 20#np.random.uniform(0, 40, size=N)

np.random.seed(3)

pos = np.random.uniform(1, 3.5, size=(N, 2))
angle = np.random.uniform(0, np.pi/2, size=N)

size_s = np.zeros(25)
size_l = np.ones(75)
size = np.concatenate((size_s, size_l))



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



def angle_wrap(a):
    #angles between -pi and pi
    return (a + np.pi) % (2 * np.pi) - np.pi

def angular_difference_signed(target, source):
    """Signed smallest angle target - source in [-pi, pi]."""
    return angle_wrap(target - source)


def repel(bearing, distance_array, scale_factor):

    bearing = np.atleast_1d(bearing)
    distance_array = np.atleast_1d(distance_array)

    signs = np.sign(bearing)

    signs[signs == 0] = np.random.choice([-1, 1], size=(signs == 0).sum())

    repel_weight = ((0.05*scale_factor)/distance_array)

    repel_rotate = (-turning_rate * signs * repel_weight)

    return np.mean(repel_rotate)


def align(distance_array, target, source, scale_factor, align_range, repel_range):

    distance_array = np.atleast_1d(distance_array)
    target = np.atleast_1d(target)
    source = np.atleast_1d(source)

    angle_difference = angular_difference_signed(target, source)

    align_weight = (scale_factor * np.exp(-(distance_array - 0.5 * (align_range + repel_range)/(align_range - repel_range))**2))

    align_rotate = turning_rate * angle_difference * align_weight

    return np.mean(align_rotate)


def attract(bearing, distance_array, scale_factor, attract_range, align_range):

    bearing = np.atleast_1d(bearing)
    distance_array = np.atleast_1d(distance_array)

    heading_vals = np.cos(bearing)

    sign_to_j = np.sign(bearing)
    sign_to_j[sign_to_j == 0] = 1

    attract_rotate = turning_rate * heading_vals * sign_to_j
                
    attract_weight = (0.2 * scale_factor * np.exp(-(distance_array - 0.5 * (attract_range + align_range)/(attract_range - align_range))**2))

    return np.mean(attract_rotate * attract_weight)



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

    repel_mask_distance_l = dist <= repulsion_range_l
    align_mask_distance_l = (dist > repulsion_range_l) & (dist <= aligning_range_l)
    attract_mask_distance_l = (dist > aligning_range_l) & (dist <= attraction_range)

    repel_mask_LOS = np.abs(rel_bearing_ij) <= (repel_LOS/2)
    align_mask_LOS = (np.abs(rel_bearing_ij) > (repel_LOS/2)) & (np.abs(rel_bearing_ij)  <= (np.deg2rad(90) + repel_LOS/2))
    attract_mask_LOS = np.abs(rel_bearing_ij) <= (attract_LOS/2)

    repel_mask_s = repel_mask_distance_s & repel_mask_LOS
    align_mask_s = align_mask_distance_s & align_mask_LOS
    attract_mask_s = attract_mask_distance_s & attract_mask_LOS

    repel_mask_l = repel_mask_distance_l & repel_mask_LOS
    align_mask_l = align_mask_distance_l & align_mask_LOS
    attract_mask_l = attract_mask_distance_l & attract_mask_LOS

    rotation_s = np.zeros(N)
    rotation_l = np.zeros(N)
        
    ############################## small ##############################

    for i in range(N):      #change to 'in range(len(size == 0))' to make faster (have to change other things too)

        if size[i] == 0:    #check to see if the boid is small 

            ##### repulsion #####
            repel_s = np.where(repel_mask_s[i])[0]      #gives the number value of all boids within the mask (1-100)
            repel_contribution = np.zeros(len(repel_s))      #set contribution as 0, for if no neighbour in range

            if repel_s.size > 0:    #only do calculations if there is a boid in the repel range, repel takes priority 

                for j in range(len(repel_s)):   #loop over each neighbour in that mask (might mess up averages in repel func???)
                    
                    if size[repel_s][j] == 0:    #find size of neighbour fish, to find how to interact

                        repel_contribution[j] = repel(rel_bearing_ij[i, repel_s[j]], DistanceMatrix[i, repel_s[j]], repel_scalefactor_s/active_sort_repel)
                   
                    if size[repel_s][j] == 1:

                        repel_contribution[j] = repel(rel_bearing_ij[i, repel_s[j]], DistanceMatrix[i, repel_s[j]], repel_scalefactor_s*active_sort_repel*risk_avoidance)
   
                #rotation_s[i] = repel(rel_bearing_ij[i, repel_s], DistanceMatrix[i, repel_s], repel_scalefactor_s)
                rotation_s[i] = 0 if repel_s.size == 0 else np.mean(repel_contribution)
        
            else:   #if no neighbours in repel, rotation is the average of the interactions of alignment and attraction

                ##### alignment #####
                align_s = np.where(align_mask_s[i])[0]
                align_contribution_s = np.zeros(len(align_s))

                if align_s.size > 0:

                    for j in range(len(align_s)):

                        if size[align_s][j] == 0:

                            align_contribution_s[j] = align(DistanceMatrix[i, align_s[j]], angle[align_s[j]], angle[i], align_scalefactor*active_sort_align, aligning_range_s, repulsion_range_s)

                        if size[align_s][j] == 1:

                            align_contribution_s[j] = align(DistanceMatrix[i, align_s[j]], angle[align_s[j]], angle[i], align_scalefactor/active_sort_align, aligning_range_s, repulsion_range_s)

                    #align_contribution_s = align(DistanceMatrix[i, align_s], angle[align_s], angle[i], align_scalefactor, aligning_range_s, repulsion_range_s)

            
                ##### attraction #####
                attract_s = np.where(attract_mask_s[i])[0]
                attract_contribution_s = np.zeros(len(attract_s))

                if attract_s.size > 0:

                    for j in range(len(attract_s)):

                        if size[attract_s][j] == 0:

                            attract_contribution_s[j] = attract(rel_bearing_ij[i, attract_s[j]], DistanceMatrix[i, attract_s[j]], attract_scalefactor*active_sort_attract, attraction_range, aligning_range_s)

                        if size[attract_s][j] == 1:

                            attract_contribution_s[j] = attract(rel_bearing_ij[i, attract_s[j]], DistanceMatrix[i, attract_s[j]], attract_scalefactor/active_sort_attract, attraction_range, aligning_range_s)

                    #attract_contribution_s = attract(rel_bearing_ij[i, attract_s], DistanceMatrix[i, attract_s], attract_scalefactor, attraction_range, aligning_range_s)

                rotation_s[i] = (0 if align_s.size == 0 else np.mean(align_contribution_s)) + (0 if attract_s.size == 0 else np.mean(attract_contribution_s))

        
    ############################## large ##############################

    for i in range(N):

        if size[i] == 1:

            repel_l = np.where(repel_mask_l[i])[0]
            repel_contribution = np.zeros(len(repel_l))

            if repel_l.size > 0:

                for j in range(len(repel_l)):

                    if size[repel_l][j] == 0:

                        repel_contribution[j] = repel(rel_bearing_ij[i, repel_l[j]], DistanceMatrix[i, repel_l[j]], repel_scalefactor_l*active_sort_repel)
                
                    if size[repel_l][j] == 1:

                        repel_contribution[j] = repel(rel_bearing_ij[i, repel_l[j]], DistanceMatrix[i, repel_l[j]], repel_scalefactor_l/active_sort_repel)

                #rotation_l[i] = repel(rel_bearing_ij[i, repel_l], DistanceMatrix[i, repel_l], repel_scalefactor_l)
                rotation_l[i] = 0 if repel_l.size == 0 else np.mean(repel_contribution)

            else:

                align_l = np.where(align_mask_l[i])[0]
                align_contribution_l = np.zeros(len(align_l))

                if align_l.size > 0:

                    for j in range(len(align_l)):

                        if size[align_l][j] == 0:

                            align_contribution_l[j] = align(DistanceMatrix[i, align_l[j]], angle[align_l[j]], angle[i], align_scalefactor/active_sort_align, aligning_range_l, repulsion_range_l)

                        if size[align_l][j] == 1:

                            align_contribution_l[j] = align(DistanceMatrix[i, align_l[j]], angle[align_l[j]], angle[i], align_scalefactor*active_sort_align, aligning_range_l, repulsion_range_l)

                    #align_contribution_l = align(DistanceMatrix[i, align_l], angle[align_l], angle[i], align_scalefactor, aligning_range_l, repulsion_range_l)

                attract_l = np.where(attract_mask_l[i])[0]
                attract_contribution_l = np.zeros(len(attract_l))

                if attract_l.size > 0:
                        
                    for j in range(len(attract_l)):

                        if size[attract_l][j] == 0:

                            attract_contribution_l[j] = attract(rel_bearing_ij[i, attract_l[j]], DistanceMatrix[i, attract_l[j]], attract_scalefactor/active_sort_attract, attraction_range, aligning_range_l)

                        if size[attract_l][j] == 1:

                            attract_contribution_l[j] = attract(rel_bearing_ij[i, attract_l[j]], DistanceMatrix[i, attract_l[j]], attract_scalefactor*active_sort_attract, attraction_range, aligning_range_l)

                    #attract_contribution_l = attract(rel_bearing_ij[i, attract_l], DistanceMatrix[i, attract_l], attract_scalefactor, attraction_range, aligning_range_l)

                rotation_l[i] = (0 if align_l.size == 0 else np.mean(align_contribution_l)) + (0 if attract_l.size == 0 else np.mean(attract_contribution_l))

    
    mean_heading = angle + (rotation_s + rotation_l) * stepsize

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

    return pos, cos, sin#, size[attract_l][5]


#print(update()[3])
#print(update()[4])
#print(update()[5])



fig, ax = plt.subplots()
ax.set_xlim([0, D])
ax.set_ylim([0, D])

color1 = np.full(len(size_s), fill_value='black')
color2 = np.full(len(size_l), fill_value='red')
color = np.concatenate((color1, color2))
    
animated_plot_quiver = ax.quiver(pos[:, 0], pos[:, 1], np.cos(angle), np.sin(angle), clim=[-np.pi, np.pi], pivot='mid', color = color)


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
    pos, cos, sin = update()
    animated_plot_quiver.set_offsets(pos)
    animated_plot_quiver.set_UVC(cos, sin)
    #print(k, n, m)
    return (animated_plot_quiver,)


anim = FuncAnimation(fig = fig, func = Animate_quiver, interval = 1, frames = T, blit = False, repeat=False)

#anim.save(f"Hemelrijk, with weights, np.random.seed(3), 75_25 big_small, active sorting.gif", dpi=400)
#plt.savefig("2DVicsekAnimation.png", dpi=400)
plt.show()
'''





######################### averages plots #########################


####### With LOS ######
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import scipy.spatial
import scipy.constants


N=100  #Number of agents
D=10  #Size of domain
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
#turning_rate_attraction = (np.pi/2)/80
#turning_rate = (np.pi/2)/2 #pi/2 rad/s
turning_rate = (np.pi/2) #pi/2 rad/s
repulsion_range_s = 0.3   #meters  for small fish
repulsion_range_l = 0.6   #m  for large fish
attraction_range = 5      #m  
aligning_range_s = 1      #m  for small fish
aligning_range_l = 2      #m  for large fish
eccentricity_s = 2
eccentricity_l = 4
repel_LOS = np.deg2rad(60)
attract_LOS = np.deg2rad(300)
repel_scalefactor_s = 1
repel_scalefactor_l = 2
align_scalefactor = 1
attract_scalefactor = 1
active_sort_repel = 2
active_sort_align = 2
active_sort_attract = 2
risk_avoidance = 20#np.random.uniform(0, 40, size=N)

#np.random.seed(3)

pos = np.random.uniform(1, 3.5, size=(N, 2))
angle = np.random.uniform(0, np.pi/2, size=N)

size_s = np.zeros(25)
size_l = np.ones(75)
size = np.concatenate((size_s, size_l))



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



def angle_wrap(a):
    #angles between -pi and pi
    return (a + np.pi) % (2 * np.pi) - np.pi

def angular_difference_signed(target, source):
    """Signed smallest angle target - source in [-pi, pi]."""
    return angle_wrap(target - source)


def repel(bearing, distance_array, scale_factor):

    bearing = np.atleast_1d(bearing)
    distance_array = np.atleast_1d(distance_array)

    signs = np.sign(bearing)

    signs[signs == 0] = np.random.choice([-1, 1], size=(signs == 0).sum())

    repel_weight = ((0.05*scale_factor)/distance_array)

    repel_rotate = (-turning_rate * signs * repel_weight)

    return np.mean(repel_rotate)


def align(distance_array, target, source, scale_factor, align_range, repel_range):

    distance_array = np.atleast_1d(distance_array)
    target = np.atleast_1d(target)
    source = np.atleast_1d(source)

    angle_difference = angular_difference_signed(target, source)

    align_weight = (scale_factor * np.exp(-(distance_array - 0.5 * (align_range + repel_range)/(align_range - repel_range))**2))

    align_rotate = turning_rate * angle_difference * align_weight

    return np.mean(align_rotate)


def attract(bearing, distance_array, scale_factor, attract_range, align_range):

    bearing = np.atleast_1d(bearing)
    distance_array = np.atleast_1d(distance_array)

    heading_vals = np.cos(bearing)

    sign_to_j = np.sign(bearing)
    sign_to_j[sign_to_j == 0] = 1

    attract_rotate = turning_rate * heading_vals * sign_to_j
                
    attract_weight = (0.2 * scale_factor * np.exp(-(distance_array - 0.5 * (attract_range + align_range)/(attract_range - align_range))**2))

    return np.mean(attract_rotate * attract_weight)



def update(ratio):

    global pos
    global angle

    for r in ratio:

        size_s = np.zeros(N * r)
        size_l = np.ones(N * (1 - r))
        size = np.concatenate((size_s, size_l))

        d = []

        for _ in range(T):

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

            repel_mask_distance_l = dist <= repulsion_range_l
            align_mask_distance_l = (dist > repulsion_range_l) & (dist <= aligning_range_l)
            attract_mask_distance_l = (dist > aligning_range_l) & (dist <= attraction_range)

            repel_mask_LOS = np.abs(rel_bearing_ij) <= (repel_LOS/2)
            align_mask_LOS = (np.abs(rel_bearing_ij) > (repel_LOS/2)) & (np.abs(rel_bearing_ij)  <= (np.deg2rad(90) + repel_LOS/2))
            attract_mask_LOS = np.abs(rel_bearing_ij) <= (attract_LOS/2)

            repel_mask_s = repel_mask_distance_s & repel_mask_LOS
            align_mask_s = align_mask_distance_s & align_mask_LOS
            attract_mask_s = attract_mask_distance_s & attract_mask_LOS

            repel_mask_l = repel_mask_distance_l & repel_mask_LOS
            align_mask_l = align_mask_distance_l & align_mask_LOS
            attract_mask_l = attract_mask_distance_l & attract_mask_LOS

            rotation_s = np.zeros(N)
            rotation_l = np.zeros(N)
                
            ############################## small ##############################

            for i in range(N):      #change to 'in range(len(size == 0))' to make faster (have to change other things too)

                if size[i] == 0:    #check to see if the boid is small 

                    ##### repulsion #####
                    repel_s = np.where(repel_mask_s[i])[0]      #gives the number value of all boids within the mask (1-100)
                    repel_contribution = np.zeros(len(repel_s))      #set contribution as 0, for if no neighbour in range

                    if repel_s.size > 0:    #only do calculations if there is a boid in the repel range, repel takes priority 

                        for j in range(len(repel_s)):   #loop over each neighbour in that mask (might mess up averages in repel func???)
                            
                            if size[repel_s][j] == 0:    #find size of neighbour fish, to find how to interact

                                repel_contribution[j] = repel(rel_bearing_ij[i, repel_s[j]], DistanceMatrix[i, repel_s[j]], repel_scalefactor_s/active_sort_repel)
                        
                            if size[repel_s][j] == 1:

                                repel_contribution[j] = repel(rel_bearing_ij[i, repel_s[j]], DistanceMatrix[i, repel_s[j]], repel_scalefactor_s*active_sort_repel*risk_avoidance)
        
                        #rotation_s[i] = repel(rel_bearing_ij[i, repel_s], DistanceMatrix[i, repel_s], repel_scalefactor_s)
                        rotation_s[i] = 0 if repel_s.size == 0 else np.mean(repel_contribution)
                
                    else:   #if no neighbours in repel, rotation is the average of the interactions of alignment and attraction

                        ##### alignment #####
                        align_s = np.where(align_mask_s[i])[0]
                        align_contribution_s = np.zeros(len(align_s))

                        if align_s.size > 0:

                            for j in range(len(align_s)):

                                if size[align_s][j] == 0:

                                    align_contribution_s[j] = align(DistanceMatrix[i, align_s[j]], angle[align_s[j]], angle[i], align_scalefactor*active_sort_align, aligning_range_s, repulsion_range_s)

                                if size[align_s][j] == 1:

                                    align_contribution_s[j] = align(DistanceMatrix[i, align_s[j]], angle[align_s[j]], angle[i], align_scalefactor/active_sort_align, aligning_range_s, repulsion_range_s)

                            #align_contribution_s = align(DistanceMatrix[i, align_s], angle[align_s], angle[i], align_scalefactor, aligning_range_s, repulsion_range_s)

                    
                        ##### attraction #####
                        attract_s = np.where(attract_mask_s[i])[0]
                        attract_contribution_s = np.zeros(len(attract_s))

                        if attract_s.size > 0:

                            for j in range(len(attract_s)):

                                if size[attract_s][j] == 0:

                                    attract_contribution_s[j] = attract(rel_bearing_ij[i, attract_s[j]], DistanceMatrix[i, attract_s[j]], attract_scalefactor*active_sort_attract, attraction_range, aligning_range_s)

                                if size[attract_s][j] == 1:

                                    attract_contribution_s[j] = attract(rel_bearing_ij[i, attract_s[j]], DistanceMatrix[i, attract_s[j]], attract_scalefactor/active_sort_attract, attraction_range, aligning_range_s)

                            #attract_contribution_s = attract(rel_bearing_ij[i, attract_s], DistanceMatrix[i, attract_s], attract_scalefactor, attraction_range, aligning_range_s)

                        rotation_s[i] = (0 if align_s.size == 0 else np.mean(align_contribution_s)) + (0 if attract_s.size == 0 else np.mean(attract_contribution_s))

                
            ############################## large ##############################

            for i in range(N):

                if size[i] == 1:

                    repel_l = np.where(repel_mask_l[i])[0]
                    repel_contribution = np.zeros(len(repel_l))

                    if repel_l.size > 0:

                        for j in range(len(repel_l)):

                            if size[repel_l][j] == 0:

                                repel_contribution[j] = repel(rel_bearing_ij[i, repel_l[j]], DistanceMatrix[i, repel_l[j]], repel_scalefactor_l*active_sort_repel)
                        
                            if size[repel_l][j] == 1:

                                repel_contribution[j] = repel(rel_bearing_ij[i, repel_l[j]], DistanceMatrix[i, repel_l[j]], repel_scalefactor_l/active_sort_repel)

                        #rotation_l[i] = repel(rel_bearing_ij[i, repel_l], DistanceMatrix[i, repel_l], repel_scalefactor_l)
                        rotation_l[i] = 0 if repel_l.size == 0 else np.mean(repel_contribution)

                    else:

                        align_l = np.where(align_mask_l[i])[0]
                        align_contribution_l = np.zeros(len(align_l))

                        if align_l.size > 0:

                            for j in range(len(align_l)):

                                if size[align_l][j] == 0:

                                    align_contribution_l[j] = align(DistanceMatrix[i, align_l[j]], angle[align_l[j]], angle[i], align_scalefactor/active_sort_align, aligning_range_l, repulsion_range_l)

                                if size[align_l][j] == 1:

                                    align_contribution_l[j] = align(DistanceMatrix[i, align_l[j]], angle[align_l[j]], angle[i], align_scalefactor*active_sort_align, aligning_range_l, repulsion_range_l)

                            #align_contribution_l = align(DistanceMatrix[i, align_l], angle[align_l], angle[i], align_scalefactor, aligning_range_l, repulsion_range_l)

                        attract_l = np.where(attract_mask_l[i])[0]
                        attract_contribution_l = np.zeros(len(attract_l))

                        if attract_l.size > 0:
                                
                            for j in range(len(attract_l)):

                                if size[attract_l][j] == 0:

                                    attract_contribution_l[j] = attract(rel_bearing_ij[i, attract_l[j]], DistanceMatrix[i, attract_l[j]], attract_scalefactor/active_sort_attract, attraction_range, aligning_range_l)

                                if size[attract_l][j] == 1:

                                    attract_contribution_l[j] = attract(rel_bearing_ij[i, attract_l[j]], DistanceMatrix[i, attract_l[j]], attract_scalefactor*active_sort_attract, attraction_range, aligning_range_l)

                            #attract_contribution_l = attract(rel_bearing_ij[i, attract_l], DistanceMatrix[i, attract_l], attract_scalefactor, attraction_range, aligning_range_l)

                        rotation_l[i] = (0 if align_l.size == 0 else np.mean(align_contribution_l)) + (0 if attract_l.size == 0 else np.mean(attract_contribution_l))

            
            mean_heading = angle + (rotation_s + rotation_l) * stepsize

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

            #average centre of all agents to the cnetre of the school
            d.append(np.mean(np.sqrt((pos[:, 0] - pos[:, 1])**2)))

    return pos, cos, sin, d



ratio_of_agents = [0, 0.25, 0.5, 0.75, 1]

fig, ax = plt.subplots(figsize=(7, 7))

for r in zip(ratio_of_agents):
    d = update(r)
    plt.plot(ratio_of_agents, d, linestyle='-', label=f'N={N}')

ax.set_xlabel('ratio of small agents', fontsize=14)
ax.set_ylabel('d (m)', fontsize=14)
ax.set_title('average centre distance')
ax.tick_params(direction='out', length=4, width=1, labelsize=12, top=False, right=False)
ax.legend(fontsize=14)
#ax.minorticks_on()
#plt.savefig(f"Polar Order Parameter, periodic boundary conditions Plot.png", dpi=400)
plt.show()
