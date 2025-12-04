


####### With LOS ######
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import scipy.spatial
import scipy.constants


N=100  #Number of agents
D=20 #Size of domain
T=3000   #Total number of time steps (frames) in simulation
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
risk_avoidance = 1#20#np.random.uniform(0, 40, size=N)

#np.random.seed(3)

#pos = np.random.uniform(1, 3.5, size=(N, 2))
#angle = np.random.uniform(0, np.pi/2, size=N)

pos = np.random.uniform(0, D, size=(N, 2))
angle = np.random.uniform(0, np.pi*2, size=N)

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


def interaction_area_repel(size):

    if size == 0:   #small fish
        area = eccentricity_s * np.pi * repulsion_range_s**2 * 60/360

    if size == 1:   #large fish
        area = eccentricity_l * np.pi * repel_scalefactor_l**2 * 60/360

    return area


def length_of_fish(size):

    if size == 0:
        length = 0.1

    if size == 1:
        length = 0.2

    return length



def predator_interaction(bearing, distance_array, scale_factor):

    bearing = np.atleast_1d(bearing)
    distance_array = np.atleast_1d(distance_array)

    signs = np.sign(bearing)
    signs[signs == 0] = np.random.choice([-1, 1], size=(signs == 0).sum())

    flee_weight = scale_factor/distance_array**3

    flee_rotate = -turning_rate * signs * flee_weight

    return np.mean(flee_rotate)


p_N = 1

p_pos = np.random.uniform(15, 17, size=(p_N, 2))
p_angle = np.random.uniform(-np.pi, np.pi, size=p_N)

p_v0 = 0.6
p_v_SD = 0.06
p_interaction_range = 8
p_LOS = 320


CoG_distances = []     # stores avg distance of boids to their CoG each step
simulation_step = 0    # step counter


def Centre_of_grav_dist():
    CoG = np.mean(pos, axis=0)                     # centre of gravity
    distances = np.linalg.norm(pos - CoG, axis=1)  # distance of each boid to CoG
    return np.mean(distances)


def update(ratios):

    global pos
    global angle
    global simulation_step
    global size

    results = []
    for r in ratios:
        # r is fraction of small boids (0..1)
        small_count = int(round(N * r))
        # assign small indices randomly so small and large are distributed
        indices = np.arange(N)
        np.random.shuffle(indices)
        small_idx = indices[:small_count]
        size = np.ones(N, dtype=int)   # 1 = large, 0 = small
        size[small_idx] = 0

        # initialize state
        pos = pos.copy()
        angle = angle.copy()



    DistanceMatrix = scipy.spatial.distance.pdist(pos)  #Scipy function to calculate distance between two agents
    DistanceMatrix = scipy.spatial.distance.squareform(DistanceMatrix)   #matrix of form [i, j] => distance between agents i and j
        #returns the distance of each agent to all other agents, the array is of size [N, N]

    vec_ij = pos[None, :, :] - pos[:, None, :]   #change in x and y positions of each boid with all other boids size=(N x N x 2)

    dist = np.linalg.norm(vec_ij, axis=2)   #length of each of the vectors of vec_ij size=(N x N)

    np.fill_diagonal(dist, np.inf)  #changes the 0 from distance too itself to infinite to counteract putting itself in the masks below

    bearing_ij = np.arctan2(vec_ij[:, :, 1], vec_ij[:, :, 0])   #direction of each boid to all other boids size=(N x N)
    rel_bearing_ij = angle_wrap(bearing_ij - angle[:, None])    #changes the angle to be between pi and -pi
    
    #masks for distance only (small)
    repel_mask_distance_s = dist <= repulsion_range_s
    align_mask_distance_s = (dist > repulsion_range_s) & (dist <= aligning_range_s)
    attract_mask_distance_s = (dist > aligning_range_s) & (dist <= attraction_range)

    #masks for distance only (large)
    repel_mask_distance_l = dist <= repulsion_range_l
    align_mask_distance_l = (dist > repulsion_range_l) & (dist <= aligning_range_l)
    attract_mask_distance_l = (dist > aligning_range_l) & (dist <= attraction_range)

    #masks for line of sight
    repel_mask_LOS = np.abs(rel_bearing_ij) <= (repel_LOS/2)
    align_mask_LOS = (np.abs(rel_bearing_ij) > (repel_LOS/2)) & (np.abs(rel_bearing_ij)  <= (np.deg2rad(90) + repel_LOS/2))
    attract_mask_LOS = np.abs(rel_bearing_ij) <= (attract_LOS/2)

    #total mask for small fish
    repel_mask_s = repel_mask_distance_s & repel_mask_LOS
    align_mask_s = align_mask_distance_s & align_mask_LOS
    attract_mask_s = attract_mask_distance_s & attract_mask_LOS

    #total mask for large fish
    repel_mask_l = repel_mask_distance_l & repel_mask_LOS
    align_mask_l = align_mask_distance_l & align_mask_LOS
    attract_mask_l = attract_mask_distance_l & attract_mask_LOS


    #for predators 
    #####################################################################
    #p_Distance_matrix = scipy.spatial.distance.cdist(pos, p_pos)
    #p_Distance_matrix = scipy.spatial.distance.squareform(p_Distance_matrix)

    #p_vec_ij = pos[None, :, :] - p_pos[:, None, :]
    #p_dist = np.linalg.norm(p_vec_ij, axis=2)

    #np.fill_diagonal(p_dist, np.inf)

    #p_bearing_ij = np.arctan2(p_vec_ij[:, :, 1], p_vec_ij[:, :, 0])
    #p_rel_bearing_ij = angle_wrap(p_bearing_ij - angle[:, None])

    #pred_distance_mask = p_dist <= p_interaction_range
    #fish_distance_mask = p_dist <= attraction_range

    #pred_LOS_mask = np.abs(p_rel_bearing_ij) <= p_LOS/2
    #fish_LOS_mask = np.abs(p_rel_bearing_ij) <= attract_LOS/2

    #pred_mask = pred_distance_mask & pred_LOS_mask
    #fish_mask = fish_distance_mask & fish_LOS_mask

    ######################################################################

    rotation_s = np.zeros(N)    #setting array for how much each boid will turn for each time step
    rotation_l = np.zeros(N)
    #rotation_from_pred = np.zeros(N)
        
    ############################## small ##############################

    for i in range(N):      #change to 'in range(len(size == 0))' to make faster (have to change other things too)

        #fish_see_pred = np.where(fish_mask[i])[0]
        #fish_contribution = np.zeros(len(fish_see_pred))

        #if fish_see_pred.size > 0:

            #for k in range(len(fish_see_pred)):

                #fish_contribution[k] = predator_interaction(p_rel_bearing_ij[i, fish_see_pred[k]], p_Distance_matrix[i, fish_see_pred[k]], 1)

            #rotation_from_pred[i] = 0 if fish_see_pred.size == 0 else np.mean(fish_contribution)


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

        elif size[i] == 1:

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

    
    mean_heading = angle + (rotation_s + rotation_l) * stepsize #+ rotation_from_pred) * stepsize

    update_angle = np.random.normal(mean_heading, direction_SD)

    cos = np.cos(update_angle)
    sin = np.sin(update_angle)

    p_cos = 1 #np.cos(p_updating_angle)
    p_sin = 1 #np.sin(p_updating_angle)

    vx = cos * v0 * stepsize
    vy = sin * v0 * stepsize

    #Updating the position of the agents
    pos[:, 0] += vx
    pos[:, 1] += vy

    angle = update_angle

    pos = np.mod(pos, D)

    #CoG_distances.append(Centre_of_grav_dist())
    #simulation_step += 1

    return pos, cos, sin



def Nearest_Neigbour_Dist(ratio):

    distance = []

    for _ in range(T):

        positions = update(ratio)[0]
        length = scipy.spatial.distance.pdist(positions)
        length = scipy.spatial.distance.squareform(length)

        min_length = np.min(length, axis=1)

        distance.append(np.mean(min_length))

    return distance


ratio_of_agents = np.array([0, 0.25, 0.5, 0.75, 1], dtype=int)


print(Nearest_Neigbour_Dist(ratio_of_agents))

fig, ax = plt.subplots(figsize=(7, 7))

plt.plot(ratio_of_agents, Nearest_Neigbour_Dist(ratio_of_agents), linestyle='-', label=f'N={N}', marker='s')

ax.set_xlabel('ratio of small agents', fontsize=14)
ax.set_ylabel('d (m)', fontsize=14)
ax.set_title('Average Nearest Neighbour distance')
ax.tick_params(direction='out', length=4, width=1, labelsize=12, top=False, right=False)
ax.legend(fontsize=14)
#ax.minorticks_on()
#plt.savefig(f"Polar Order Parameter, periodic boundary conditions Plot.png", dpi=400)
plt.show()



'''

# After simulation is complete, plot CoG distances for final 1000 steps
if len(CoG_distances) >= 1000:
    last_1000 = CoG_distances[-1000:]
    plt.figure(figsize=(7,4))
    plt.plot(last_1000)
    plt.title("Average Distance to Centre of Gravity (last 1000 steps)")
    plt.xlabel("Simulation Step (last 1000)")
    plt.ylabel("Mean Distance to CoG")
    plt.grid(True)
    plt.show()
else:
    print("Not enough simulation steps to compute last 1000 CoG distances.")

'''


fig, ax = plt.subplots()
ax.set_xlim([0, D])
ax.set_ylim([0, D])

def qlook(A, B):
    s = np.full(len(size_s), fill_value=A)
    l = np.full(len(size_l), fill_value=B)
    return np.concatenate((s, l))
    
animated_plot_quiver = ax.quiver(pos[:, 0], pos[:, 1], np.cos(angle), np.sin(angle), clim=[-np.pi, np.pi], 
                                 angles = 'xy',
                                 scale_units = 'xy',
                                 #scale = qlook(1.5, 1),
                                 pivot='mid', 
                                 color = qlook('black', 'green'), 
                                 )


#animated_plot_quiver_pred = ax.quiver(p_pos[:, 0], p_pos[:, 1], np.cos(p_angle), np.sin(p_angle),
#                                      angles = 'xy',
#                                      scale_units = 'xy',
#                                      scale = 0.75,
#                                      pivot='mid', 
#                                      color = 'red', 
#                                      ) 



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
    #animated_plot_quiver_pred.set_offsets(p_pos)
    #animated_plot_quiver_pred.set_UVC(p_cos, p_sin)
    #print(k, n, m)
    return (animated_plot_quiver,)# animated_plot_quiver_pred,)


anim = FuncAnimation(fig = fig, func = Animate_quiver, interval = 1, frames = T, blit = False, repeat=False)

#anim.save(f"Hemelrijk, with weights, np.random.seed(3), 75_25 big_small, active sorting.gif", dpi=400)
#plt.savefig("2DVicsekAnimation.png", dpi=400)
plt.show()






######################### averages plots #########################

'''
####### With LOS ######
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import scipy.spatial
import scipy.constants


N=100  #Number of agents
D=999999999999999999999999999  #Size of domain
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

    d = []

    for r in ratio:

        size_s = np.zeros((N * r))
        size_l = np.ones((N * (1 - r)))
        size = np.concatenate((size_s, size_l))

        d1 = []

        c_g = []

        position = []

        T = 3000

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

            mean_heading = np.zeros(N)
            update_angle = np.zeros(N)
            cos = np.zeros(N)
            sin = np.zeros(N)
            vx = np.zeros(N)
            vy = np.zeros(N)
                
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

            cos = np.cos(update_angle[i])
            sin = np.sin(update_angle[i])

            vx = cos * speed() * stepsize
            vy = sin * speed() * stepsize

            #Updating the position of the agents
            pos[:, 0] += vx
            pos[:, 1] += vy

            angle = update_angle

            pos = np.mod(pos, D)

            position.append(pos)

            #f = (np.sqrt((centre_of_grav - pos[-1000:])**2))

            #f = np.linalg.norm(f)

            #d1.append(np.mean(f))

        #only want the last 1000 time steps
        grav = np.mean(np.array(position[-1000:]))  # should be 1000 x 2
        p = np.array(position[-1000:])     #1000 x 100 x 2

        #distance to centre for each boid at each time step
        #f = (grav[:, None, :] - p)  #1000 x 100 x 2
        #g = np.linalg.norm(f)   #1000 x 100

        g = np.sqrt(((grav[:, 0] - p[:, 0]) - (grav[:, 1], p[:, 1]))**2)

        #average distance (from all boids) to centre at each time step
        d1 = np.mean(g)    #1000

        #average over all the timesteps
        d.append(np.mean(d1))   #one vale for each ratio of boids

    return d


ratio_of_agents = np.array([0, 0.25, 0.5, 0.75, 1], dtype=int)


print(update(ratio_of_agents))

fig, ax = plt.subplots(figsize=(7, 7))

plt.plot(ratio_of_agents, update(ratio_of_agents), linestyle='-', label=f'N={N}', marker='s')

ax.set_xlabel('ratio of small agents', fontsize=14)
ax.set_ylabel('d (m)', fontsize=14)
ax.set_title('average distance to centre')
ax.tick_params(direction='out', length=4, width=1, labelsize=12, top=False, right=False)
ax.legend(fontsize=14)
#ax.minorticks_on()
#plt.savefig(f"Polar Order Parameter, periodic boundary conditions Plot.png", dpi=400)
plt.show()
'''




'''

import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial

# --- Parameters (kept mostly from your original) ---
N = 100
D = 99999999999                # domain size (periodic). Set to a reasonable finite value.
total_steps = 3000      # total simulation steps
keep_last = 1000        # keep last 1000 timesteps for averaging
stepsize = 0.2
direction_SD = np.pi/72
v0 = 0.3
v_SD = 0.03

# interaction ranges
repulsion_range_s = 0.3
repulsion_range_l = 0.6
aligning_range_s = 1.0
aligning_range_l = 2.0
attraction_range = 5.0

# turning / force scales
turning_rate = np.pi/2
repel_scalefactor_s = 1.0
repel_scalefactor_l = 2.0
align_scalefactor = 1.0
attract_scalefactor = 1.0

# line-of-sight angles
repel_LOS = np.deg2rad(60)
attract_LOS = np.deg2rad(300)

active_sort_repel = 2.0
active_sort_align = 2.0
active_sort_attract = 2.0
risk_avoidance = 20.0

epsilon = 1e-9

# seed for reproducibility
#np.random.seed(3)

# initial positions and headings
pos0 = np.random.uniform(1, 3.5, size=(N, 2))
angle0 = np.random.uniform(0, 2*np.pi, size=N)

# helper functions (kept structure from your original but vectorized usage expected)
def angle_wrap(a):
    return (a + np.pi) % (2*np.pi) - np.pi

def angular_difference_signed(target, source):
    return angle_wrap(target - source)

def repel(bearing, distance_array, scale_factor):
    # bearing: scalar, distance: scalar
    sign = np.sign(bearing) if bearing != 0 else np.random.choice([-1, 1])
    repel_weight = ((0.05 * scale_factor) / (distance_array + epsilon))
    repel_rotate = (-turning_rate * sign * repel_weight)
    return repel_rotate

def align(distance, target_angle, source_angle, scale_factor, align_range, repel_range):
    # scalar inputs
    d = distance
    ang_diff = angular_difference_signed(target_angle, source_angle)
    # a Gaussian-like weight centered in the middle of ranges (follows your original intent)
    center = 0.5 * (align_range + repel_range) / max(align_range - repel_range, 1e-6)
    align_weight = (scale_factor * np.exp(-(d - center)**2))
    return turning_rate * ang_diff * align_weight

def attract(bearing, distance, scale_factor, attract_range, align_range):
    heading_vals = np.cos(bearing)
    sign_to_j = np.sign(bearing) if bearing != 0 else 1
    attract_rotate = turning_rate * heading_vals * sign_to_j
    center = 0.5 * (attract_range + align_range) / max(attract_range - align_range, 1e-6)
    attract_weight = (0.2 * scale_factor * np.exp(-(distance - center)**2))
    return attract_rotate * attract_weight

def speed():
    return np.random.normal(loc=v0, scale=v_SD, size=N)

# Main update: runs simulation for many ratios and returns average distances
def update(ratios):
    results = []
    for r in ratios:
        # r is fraction of small boids (0..1)
        small_count = int(round(N * r))
        # assign small indices randomly so small and large are distributed
        indices = np.arange(N)
        np.random.shuffle(indices)
        small_idx = indices[:small_count]
        size = np.ones(N, dtype=int)   # 1 = large, 0 = small
        size[small_idx] = 0

        # initialize state
        pos = pos0.copy()
        angle = angle0.copy()

        positions_history = []  # will hold copies of pos for the last keep_last steps (we'll append all and slice later)

        for step in range(total_steps):
            # compute pairwise vectors and distances
            vec_ij = pos[None, :, :] - pos[:, None, :]   # shape (N,N,2)
            dist = np.linalg.norm(vec_ij, axis=2)       # shape (N,N)
            np.fill_diagonal(dist, np.inf)

            bearing_ij = np.arctan2(vec_ij[:, :, 1], vec_ij[:, :, 0])    # absolute bearing from j to i
            rel_bearing_ij = angle_wrap(bearing_ij - angle[:, None])    # relative bearing in boid i's frame

            # masks per boid (vectorized booleans)
            repel_mask_LOS = np.abs(rel_bearing_ij) <= (repel_LOS / 2.0)
            align_mask_LOS = (np.abs(rel_bearing_ij) > (repel_LOS/2.0)) & (np.abs(rel_bearing_ij) <= (np.deg2rad(90) + repel_LOS/2.0))
            attract_mask_LOS = np.abs(rel_bearing_ij) <= (attract_LOS / 2.0)

            # distance masks (two types: small and large ranges)
            repel_mask_distance_s = dist <= repulsion_range_s
            align_mask_distance_s = (dist > repulsion_range_s) & (dist <= aligning_range_s)
            attract_mask_distance_s = (dist > aligning_range_s) & (dist <= attraction_range)

            repel_mask_distance_l = dist <= repulsion_range_l
            align_mask_distance_l = (dist > repulsion_range_l) & (dist <= aligning_range_l)
            attract_mask_distance_l = (dist > aligning_range_l) & (dist <= attraction_range)

            # combine LOS and distance masks
            repel_mask_s = repel_mask_distance_s & repel_mask_LOS
            align_mask_s = align_mask_distance_s & align_mask_LOS
            attract_mask_s = attract_mask_distance_s & attract_mask_LOS

            repel_mask_l = repel_mask_distance_l & repel_mask_LOS
            align_mask_l = align_mask_distance_l & align_mask_LOS
            attract_mask_l = attract_mask_distance_l & attract_mask_LOS

            rotation = np.zeros(N)   # net rotation per boid

            # compute interactions for each boid (vectorization is tricky because contributions differ per neighbour type)
            for i in range(N):
                if size[i] == 0:   # small boid
                    # repulsion (priority)
                    neighbors = np.where(repel_mask_s[i])[0]
                    if neighbors.size > 0:
                        contributions = []
                        for j in neighbors:
                            if size[j] == 0:    #calculate scale factor based on the size of the neighbour
                                sf = repel_scalefactor_s / active_sort_repel    
                            else:
                                sf = repel_scalefactor_s * active_sort_repel * risk_avoidance
                            contributions.append(repel(rel_bearing_ij[i, j], dist[i, j], sf))
                        rotation[i] = np.mean(contributions)
                        continue  # repulsion dominates for small
                    # alignment + attraction
                    # alignment
                    align_neigh = np.where(align_mask_s[i])[0]
                    align_contribs = []
                    for j in align_neigh:
                        sf = align_scalefactor * active_sort_align if size[j] == 0 else align_scalefactor / active_sort_align
                        align_contribs.append(align(dist[i, j], angle[j], angle[i], sf, aligning_range_s, repulsion_range_s))
                    # attraction
                    attract_neigh = np.where(attract_mask_s[i])[0]
                    attract_contribs = []
                    for j in attract_neigh:
                        sf = attract_scalefactor * active_sort_attract if size[j] == 0 else attract_scalefactor / active_sort_attract
                        attract_contribs.append(attract(rel_bearing_ij[i, j], dist[i, j], sf, attraction_range, aligning_range_s))
                    rotation[i] = (np.mean(align_contribs) if len(align_contribs)>0 else 0.0) + (np.mean(attract_contribs) if len(attract_contribs)>0 else 0.0)

                else:  # large boid
                    neighbors = np.where(repel_mask_l[i])[0]
                    if neighbors.size > 0:
                        contributions = []
                        for j in neighbors:
                            if size[j] == 0:
                                sf = repel_scalefactor_l * active_sort_repel
                            else:
                                sf = repel_scalefactor_l / active_sort_repel
                            contributions.append(repel(rel_bearing_ij[i, j], dist[i, j], sf))
                        rotation[i] = np.mean(contributions)
                        continue
                    # alignment + attraction for large
                    align_neigh = np.where(align_mask_l[i])[0]
                    align_contribs = []
                    for j in align_neigh:
                        sf = align_scalefactor / active_sort_align if size[j] == 0 else align_scalefactor * active_sort_align
                        align_contribs.append(align(dist[i, j], angle[j], angle[i], sf, aligning_range_l, repulsion_range_l))
                    attract_neigh = np.where(attract_mask_l[i])[0]
                    attract_contribs = []
                    for j in attract_neigh:
                        sf = attract_scalefactor / active_sort_attract if size[j] == 0 else attract_scalefactor * active_sort_attract
                        attract_contribs.append(attract(rel_bearing_ij[i, j], dist[i, j], sf, attraction_range, aligning_range_l))
                    rotation[i] = (np.mean(align_contribs) if len(align_contribs)>0 else 0.0) + (np.mean(attract_contribs) if len(attract_contribs)>0 else 0.0)

            # update angles and positions (vectorized)
            mean_heading = angle + rotation * stepsize
            update_angle = np.random.normal(loc=mean_heading, scale=direction_SD)

            spd = speed()  # vector length N
            vx = spd * np.cos(update_angle) * stepsize
            vy = spd * np.sin(update_angle) * stepsize

            pos[:, 0] = (pos[:, 0] + vx) % D
            pos[:, 1] = (pos[:, 1] + vy) % D

            angle = angle_wrap(update_angle)

            # store a copy for history
            positions_history.append(pos.copy())

        # get last keep_last steps and compute average distance to CoG
        p = np.array(positions_history[-keep_last:])   # shape (keep_last, N, 2)
        centres = p.mean(axis=1)                       # shape (keep_last, 2)
        # distances at each time step: (keep_last, N)
        distances = np.linalg.norm(p - centres[:, None, :], axis=2)
        # average over boids then over time
        avg_dist = distances.mean()
        results.append(avg_dist)

    return results

def avg_dist_C_mult(ratios, runs):

    avg = []
    for l in range(runs):
        print(f"run {l+1}/{runs}")
        vals = update(ratios)
        avg.append(vals)

    avg = np.array(avg)

    means = avg.mean(axis=0)
    std = avg.std(axis=0)

    return means, std


# run
ratio_of_agents = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
xs, errs = avg_dist_C_mult(ratio_of_agents, 10)
print("average distances:", xs)

# quick plot
plt.figure(figsize=(7,6))
plt.errorbar(ratio_of_agents, xs, yerr=errs, marker='s', linestyle='-')
plt.xlabel('ratio of small agents')
plt.ylabel('average distance (m)')
plt.title('Average distance to centre of gravity')
plt.grid(True, alpha=0.3)
plt.show()
'''
