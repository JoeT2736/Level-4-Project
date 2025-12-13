

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import scipy.spatial



#N = 10         #number of agents
D = 25          #domain size (square [0, D] x [0, D])
T = 3000        #number of frames
stepsize = 0.2  #seconds per update
#eta = 0.15      #angle noise 
v0 = 0.3        #speed
scale = 3
epsilon = 1e-6
v_SD = 0.03     #standard deviation of speed
direction_SD = np.pi/72     #standard deviation of angle
turning_rate = (np.pi)  #rad / s


repulsion_range_s = 0.3   #small fish
repulsion_range_l = 0.6   #large fish
aligning_range_s = 1.0
aligning_range_l = 2.0
attraction_range = 5.0    
eccentricity_s = 2
eccentricity_l = 4

repel_LOS = np.deg2rad(60)    
attract_LOS = np.deg2rad(300) 

repel_scalefactor_s = 1.0
repel_scalefactor_l = 2.0
align_scalefactor = 1.0
attract_scalefactor = 1.0

active_sort_repel = 2
active_sort_align = 2
active_sort_attract = 2

risk_avoidance = 20.0

#length of fish (m)
length_small = 0.1
length_large = 0.2

#"point", "line", or "ellipse"
MODE = "line"  


#np.random.seed(3)




def angle_wrap(a):
    #angles between -pi and pi instead of 0 and 2
    return (a + np.pi) % (2 * np.pi) - np.pi

def angular_difference_signed(target, source):
    #positive or negative difference in direction between agent i and j
    return angle_wrap(target - source)



def point_to_point_distances(positions):
    #pairwise distance for point agents
    vec = positions[None, :, :] - positions[:, None, :]  #(N,N,2)
    dist = np.linalg.norm(vec, axis=2)
    bearing = np.arctan2(vec[:, :, 1], vec[:, :, 0])  #angle from i to j
    return dist, bearing, vec


def segment_endpoints(positions, angles, lengths):
    #gives the x and y coords of the min and max positions of the fish with a length (position of tail and head)
    half = lengths / 2.0
    dx = half * np.cos(angles)  
    dy = half * np.sin(angles)
    max_point = np.stack([positions[:, 0] + dx, positions[:, 1] + dy], axis=1)  #(N, 2)
    min_point = np.stack([positions[:, 0] - dx, positions[:, 1] - dy], axis=1)
    return min_point, max_point


def vectorized_point_to_segment_distances(positions, min_point, max_point):
    #distance from centre of agent i to the closest position on the line segment of agent j (for models with a length)
    #Expand dims to (N, N, 2) where first index is point i and second is segment j
    P = positions[:, None, :]   #(N,1,2)
    A = min_point[None, :, :]   #(1,N,2)
    B = max_point[None, :, :]   #(1,N,2)

    AB = B - A                  #(1,N,2) vector from min point to max point of line segment
    AP = P - A                  #(N,N,2) vector from midpoint of i, to min point of segment of j

    AB_dot_AB = np.sum(AB * AB, axis=2)  #(1,N) square of length of segment
    #avoid divide-by-zero
    AB_dot_AB = np.where(AB_dot_AB == 0, epsilon, AB_dot_AB)

    #position of i onto infinite line segment through j
    t = np.sum(AP * AB, axis=2) / AB_dot_AB  #(N,N), projection scalar
    #keeps the position of where the closest point on the line segment is, actually on the line segment (not on the infinite line)
    t_clamped = np.clip(t, 0.0, 1.0)    #t<0 = projection is before the start of the segment (lies before min point)
                                        #t>1 = projection lies after end of segment (lies after max point)
                                        #0<t<1 = projection lies on the line segment
    

    closest = A + (t_clamped[..., None] * AB)  #(N,N,2) #finds the closest point (min point + how far along the vector of the line segment the projection of i is)
    diff = P - closest  #vector difference from position i to line j
    dist = np.linalg.norm(diff, axis=2)     #norm of the vector to find the distance
    bearing = np.arctan2(closest[:, :, 1] - P[:, :, 1], closest[:, :, 0] - P[:, :, 0])  #angle from point i to closest point on j
    vec = closest - P  #from i to closest on j
    return dist, bearing, vec

'''
def ellipse_distance_transform(positions, angles, a_vals, b_vals):
    """
    Compute pairwise distance-like metric from i to j using j's ellipse axes.
    Returns dist (N,N), bearing (N,N), vec (N,N,2).
    """

    N = positions.shape[0]

    # relative vector from i to j
    rel = positions[None, :, :] - positions[:, None, :]   # (N,N,2)

    # rotation of coordinates into j's body frame
    cos_j = np.cos(angles)[None, :]    # (1,N)
    sin_j = np.sin(angles)[None, :]    # (1,N)

    # expand to (N,N)
    cos_j = np.repeat(cos_j, N, axis=0)    # (N,N)
    sin_j = np.repeat(sin_j, N, axis=0)    # (N,N)

    # rotate rel by -theta_j
    x_rel = rel[:, :, 0]
    y_rel = rel[:, :, 1]

    x_rot = x_rel * cos_j + y_rel * sin_j
    y_rot = -x_rel * sin_j + y_rel * cos_j

    # axes arrays for each j
    a = a_vals[None, :]     # (1,N)
    b = b_vals[None, :]     # (1,N)

    a = np.repeat(a, N, axis=0)    # (N,N)
    b = np.repeat(b, N, axis=0)    # (N,N)

    # Mahalanobis-like distance
    scaled = (x_rot / a)**2 + (y_rot / b)**2
    dist = np.sqrt(scaled) * np.sqrt(a * b)

    # bearings and vector in world frame
    bearing = np.arctan2(rel[:, :, 1], rel[:, :, 0])
    vec = rel

    return dist, bearing, vec
'''

def ellipse_distance_hemlrijk(positions, angles, eccentricity):

    N = positions.shape[0]

    #relative vectors of every agent to every other agent
    rel = positions[None, :, :] - positions[:, None, :]   # (N,N,2)

    #x and y headings of each agent
    cos_i = np.cos(angles)[:, None]   # (N,1)
    sin_i = np.sin(angles)[:, None]   # (N,1)

    #expand to (N,N)
    cos_i = np.repeat(cos_i, N, axis=1)
    sin_i = np.repeat(sin_i, N, axis=1)

    #body-frame of agent i
    x_rel = rel[:, :, 0]
    y_rel = rel[:, :, 1]

    #rotate relative vectors into body frame of agent i (apply the rotation matrix)
    u = x_rel * cos_i + y_rel * sin_i       #forward axis  u>0 = j in front of i, u<0 = j behind i
    z = x_rel * -sin_i + y_rel * cos_i       #side axis     z>0 = j is to left of i, z<0 = j to right of i

    #eccentricity per agent i
    e = eccentricity[:, None]     # (N,1)
    e = np.repeat(e, N, axis=1)   # (N,N)

    #Equation 13 in Hemelrijk and Kunz 2003 (to find distance)
    dist = np.sqrt(((u**2) / (e)) + e*(z**2))

    #Bearings (not in agent i's frame)
    bearing = np.arctan2(rel[:, :, 1], rel[:, :, 0])

    return dist, bearing, rel



##### weights of different the types of interactions #####

def repel(bearing, distance_array, scale_factor):
    bearing = np.atleast_1d(bearing)
    distance_array = np.atleast_1d(distance_array)
    #signs of bearing: decide left/right
    signs = np.sign(bearing)
    signs[signs == 0] = np.random.choice([-1, 1], size=(signs == 0).sum())  #if signs is 0, pick a random choice of positive or negative for the interaction

    repel_weight = ((0.05 * scale_factor) / (distance_array + epsilon))
    repel_rotate = (-turning_rate * signs * repel_weight)
    return np.mean(repel_rotate)


def align(distance_array, target_angles, source_angle, scale_factor, align_range, repel_range):
    distance_array = np.atleast_1d(distance_array)
    target = np.atleast_1d(target_angles)
    angle_difference = angular_difference_signed(target, source_angle)
    align_weight = (scale_factor *  np.exp(-(distance_array - 0.5 * (align_range + repel_range)/(align_range - repel_range))**2))
    align_rotate = turning_rate * angle_difference * align_weight
    return np.mean(align_rotate)


def attract(bearing, distance_array, scale_factor, attract_range, align_range):
    bearing = np.atleast_1d(bearing)
    distance_array = np.atleast_1d(distance_array)

    heading_vals = np.cos(bearing)
    sign_to_j = np.sign(bearing)
    sign_to_j[sign_to_j == 0] = 1

    attract_rotate = turning_rate * heading_vals * sign_to_j
    attract_weight = (0.2 * scale_factor * np.exp(-(distance_array - 0.5 * (attract_range + align_range)/(attract_range - align_range, epsilon))**2))
    return np.mean(attract_rotate * attract_weight)

##########################################################




class HemelrijkSimulation:
    def __init__(self, mode="point", N=None, size=None):

        #assert mode in ("point", "line", "ellipse")
        self.mode = mode
        self.N = N
        self.D = D

        #random spawn across domain
        #self.pos = np.random.uniform(0, D, size=(N, 2))
        #self.angle = np.random.uniform(-np.pi, np.pi, size=N)

        #one school (2.5 x 2.5m box, with angles up to 90deg)
        self.pos = np.random.uniform(5, 7.5, size=(N, 2))
        self.angle = np.random.uniform(0, np.pi/2, size=N)

        self.v0 = v0
        self.v_SD = v_SD


        ############### use for animations, mix of large and small ###############
        
        #Size array (0 = small, 1 = large) 
        ratio = N   #number of small fish
        #size_s = np.zeros(ratio, dtype=int)
        #size_l = np.ones(N - ratio, dtype=int)
        #self.size = np.concatenate((size_s, size_l))
        #self.eccentricity = np.where(self.size == 0, eccentricity_s, eccentricity_l)

        ##########################################################################


        ############### use for static plots, for all large or all small ###############

        if size == 'small':
            self.size = np.zeros(ratio)
        else:
            self.size = np.ones(ratio)

        self.eccentricity = np.where(self.size == 0, eccentricity_s, eccentricity_l)

        ################################################################################


        self.repulsion_range = np.where(self.size == 0, repulsion_range_s, repulsion_range_l)
        self.aligning_range = np.where(self.size == 0, aligning_range_s, aligning_range_l)
        self.attraction_range = attraction_range

        #lengths for line agents
        self.lengths = np.where(self.size == 0, length_small, length_large)     #when self.size == 0, gives value of self.lengths 0.1m (length of small fish)
                                                                                #when not == 0, gives length 0.2m

        #lists to store statistics for plots
        self.centre_dist = []
        self.near1 = []
        self.near2 = []
        self.homogeneity_ = []

    #function for random speed element
    def speed(self):
        return np.random.normal(loc=self.v0, scale=self.v_SD, size=self.N)

    

    #centre of gravity of school
    def c_grav(self):
        return np.mean(self.pos, axis=0)

    #average centre distance (c^t in Hemelrijk and Kunz 2003/5)
    def avg_c_dist(self):
        diff = self.pos - self.c_grav()          
        distances = np.linalg.norm(diff, axis=1) 
        return np.mean(distances)  


    #nearest neighbour distance
    def near_n(self):
        dist_x = self.pos[:, None, 0] - self.pos[None, :, 0]
        dist_y = self.pos[:, None, 1] - self.pos[None, :, 1]
        dist = np.sqrt(dist_x**2 + dist_y**2)

        np.fill_diagonal(dist, np.inf)

        #sort distances, shortest to farthest
        sort_dist = np.sort(dist, axis=1)

        n1 = np.mean(sort_dist[:, 0])   #average distance to nearest neighbour (over all fish)
        n2 = np.mean(sort_dist[:, 1])   #average distance to second nearest neighbour

        return n1, n2
    

    def homogeneity(self):
        n1, n2 = self.near_n()
        return n1 / n2
    

    #all stats in one function
    def stats(self):
        c = self.avg_c_dist()
        n1, n2 = self.near_n()
        h = n1 / n2

        return {"center_distance": c,
            "nearest_neighbor": n1,
            "second_neighbor": n2,
            "homogeneity": h}


    #function for one step of simulation
    def step(self):
        pos = self.pos
        angle = self.angle
        N = self.N

        #different sims for the three different modes in described in Hemelrijk and Kunz 2003
        if self.mode == "point":
            dist, bearing, vec = point_to_point_distances(pos)

        elif self.mode == "line":
            min_pt, max_pt = segment_endpoints(pos, angle, self.lengths)
            dist, bearing, vec = vectorized_point_to_segment_distances(pos, min_pt, max_pt)

        elif self.mode == "ellipse":
            dist, bearing, vec = ellipse_distance_hemlrijk(self.pos, self.angle, self.eccentricity)

        else:
            raise ValueError("Unknown mode")

        #fill diagonal of distance matrix so the boids dont interact with themselves
        if dist.ndim == 2 and dist.shape[0] == dist.shape[1]:
            np.fill_diagonal(dist, np.inf)
        else:
            raise RuntimeError(f"Distance matrix not square! dist shape={dist.shape}")

        #relative angle of j from frame of i (how agent i views j)
        #bearing = angle from i to j (in world frame)
        rel_bearing = angle_wrap(bearing - angle[:, None])

        #line of sight masks
        repel_LOS_mask = np.abs(rel_bearing) <= (repel_LOS / 2.0)
        align_LOS_mask = (np.abs(rel_bearing) > (repel_LOS / 2.0)) & (np.abs(rel_bearing) <= (np.deg2rad(90) + repel_LOS / 2.0))
        attract_LOS_mask = np.abs(rel_bearing) <= (attract_LOS / 2.0)

        # Distance masks depend on each focal agent i's view of neighbors j:
        # For point mode, each agent's repulsion/align ranges depend on the focal agent's own size.
        # For line/ellipse we'll follow same logic: each focal agent i uses its own repulsion/align ranges.
        rep_range_i = self.repulsion_range[:, None]     # (N,1)
        align_range_i = self.aligning_range[:, None]    # (N,1)
        attract_range = self.attraction_range

        repel_mask_distance = dist <= rep_range_i
        align_mask_distance = (dist > rep_range_i) & (dist <= align_range_i)
        attract_mask_distance = (dist > align_range_i) & (dist <= attract_range)

        
        repel_mask = repel_mask_distance & repel_LOS_mask
        align_mask = align_mask_distance & align_LOS_mask
        attract_mask = attract_mask_distance & attract_LOS_mask

        
        rotation = np.zeros(N)

        #find rotation at each time step for all agents i
        for i in range(N):

            i_small = (self.size[i] == 0)   #check if focus agent (i) is small or large

            ##### repulsion #####

            repel_ = np.where(repel_mask[i])[0]

            if repel_.size > 0:  #compute if there are neighbours in repulsion zone
                
                contribution = np.zeros(len(repel_))     #set up array to store the contribution of repulsion due to each neighbour

                # k = counter, j = element ???
                for k, j in enumerate(repel_):  #for each value in repel_ (for each neighbour j)        enumerate adds a counter to the element depending on its position in the list

                    j_small = (self.size[j] == 0)   #check size of neighbour

                    if i_small:     #if focus agent small

                        if j_small:     #if neighbour small, change scale factor
                            scale_factor = repel_scalefactor_s / active_sort_repel
                        
                        else:   #if neighbour large
                            scale_factor = repel_scalefactor_s * active_sort_repel * risk_avoidance
                    
                    else:   #if focus agent large

                        if j_small:
                            scale_factor = repel_scalefactor_l * active_sort_repel

                        else:
                            scale_factor = repel_scalefactor_l / active_sort_repel
                    
                    contribution[k] = repel(rel_bearing[i, j], dist[i, j], scale_factor)
                
                rotation[i] = np.mean(contribution)     #take average from all neighbours
                continue    

            ##### alignment #####

            align_ = np.where(align_mask[i])[0]
            align_contribution = np.zeros(len(align_))

            for k, j in enumerate(align_):

                j_small = (self.size[j] == 0)

                if i_small:

                    if j_small:
                        scale_factor = align_scalefactor * active_sort_align

                    else:
                        scale_factor = align_scalefactor / active_sort_align

                    align_range_i = self.aligning_range[i]
                    repel_range_i = self.repulsion_range[i]
                
                else:

                    if j_small:
                        scale_factor = align_scalefactor / active_sort_align
                    
                    else:
                        scale_factor = align_scalefactor * active_sort_align
                    
                    align_range_i = self.aligning_range[i]
                    repel_range_i = self.repulsion_range[i]
                
                align_contribution[k] = align(dist[i, j], self.angle[j], self.angle[i], scale_factor, align_range_i, repel_range_i)

            
            ##### attraction #####

            attract_ = np.where(attract_mask[i])[0]
            attract_contribution = np.zeros(len(attract_))

            for k, j in enumerate(attract_):

                j_small = (self.size[j] == 0)

                if i_small:

                    if j_small:
                        scale_factor = attract_scalefactor * active_sort_attract

                    else:
                        scale_factor = attract_scalefactor / active_sort_attract

                else:

                    if j_small:
                        scale_factor = attract_scalefactor / active_sort_attract
                    
                    else:
                        scale_factor = attract_scalefactor * active_sort_attract
                    
                attract_contribution[k] = attract(rel_bearing[i, j], dist[i, j], scale_factor, self.attraction_range, self.aligning_range[i])
            

            rotation_align = 0 if align_.size == 0 else np.mean(align_contribution)
            rotation_attract = 0 if attract_.size == 0 else np.mean(attract_contribution)

            rotation[i] = rotation_align + rotation_attract
            

        #adding rotation to original angle
        mean_heading = angle + rotation * stepsize
                                                      #vvvvvvvvvvv = noise
        update_angle = np.random.normal(mean_heading, direction_SD)     

        self.v = self.speed()

        cos = np.cos(update_angle)
        sin = np.sin(update_angle)
        vx = cos * self.v * stepsize   #speed in x-direction
        vy = sin * self.v * stepsize   #speed in y-direction

        #update state
        self.pos[:, 0] = (self.pos[:, 0] + vx) % self.D
        self.pos[:, 1] = (self.pos[:, 1] + vy) % self.D
        self.angle = update_angle

        stats = self.stats()

        self.centre_dist.append(stats["center_distance"])
        self.near1.append(stats["nearest_neighbor"])
        self.near2.append(stats["second_neighbor"])
        self.homogeneity_.append(stats["homogeneity"])

        return self.pos.copy(), cos, sin, dist#, self.centre_dist.copy(), self.near1.copy(), self.near2.copy(), self.homogeneity_.copy()
    
def run_sim(N_vals, fish_size, repeats=3):
    modes = ['point', 'line', 'ellipse']

    avg_centre = {m: [] for m in modes}
    avg_near_n = {m: [] for m in modes}

    for mode in modes:
        print(f"\n=== {mode} ===")

        for N in N_vals:
            print(f"  N = {N}")

            centre_runs = []
            near_runs = []

            for r in range(repeats):
                sim = HemelrijkSimulation(mode=mode, N=N, size=fish_size)

                sim.pos = np.random.uniform(5, 7.5, size=(N, 2))
                sim.angle = np.random.uniform(0, np.pi/2, size=N)
                
                for t in range(T):
                    sim.step()

                #last 1000 time steps
                centre_runs.append(np.mean(sim.centre_dist[-1000:]))
                near_runs.append(np.mean(sim.near1[-1000:]))

            
            avg_centre[mode].append(np.mean(centre_runs))
            avg_near_n[mode].append(np.mean(near_runs))

    return avg_centre, avg_near_n




plt.rcParams["font.family"] = "Times New Roman"





N_vals = [3, 4, 6, 10, 25, 50, 75, 100]

print("\n=== Running SMALL ===")
centre_s, near_s = run_sim(N_vals, 'small')

print("\n=== Running LARGE ===")
centre_l, near_l = run_sim(N_vals, 'large')


modes = ['point', 'line', 'ellipse']
linestyles = {'point':':', 'line':'--', 'ellipse':'-'}

fig, axs = plt.subplots(2, 2, figsize=(6, 6))

ax = axs[0, 0]
for m in modes:
    ax.plot(N_vals, centre_s[m], '-o', linestyle=linestyles[m], label=m)
ax.set_title('Average centre distance (small fish)')
ax.set_xlabel('group size')
ax.set_ylabel('centre distance')
ax.legend()

ax = axs[0, 1]
for m in modes:
    ax.plot(N_vals, centre_l[m], '-o', linestyle=linestyles[m])
ax.set_title('Average centre distance (large fish)')
ax.set_xlabel('group size')
ax.set_ylabel('centre distance')
ax.legend()

ax = axs[1, 0]
for m in modes:
    ax.plot(N_vals, near_s[m], '-o', linestyle=linestyles[m])
ax.set_title('nearest neighbour distance (small fish)')
ax.set_xlabel('group size')
ax.set_ylabel('nearest neighbour')
ax.legend()

ax = axs[1, 1]
for m in modes:
    ax.plot(N_vals, near_l[m], '-o', linestyle=linestyles[m])
ax.set_title('nearest neighbour distance (large fish)')
ax.set_xlabel('group size')
ax.set_ylabel('nearest neighbour')
ax.legend()

plt.tight_layout()
plt.show()


    
#sim = HemelrijkSimulation(mode=MODE)
#print(sim.step()[3])





#for animation
'''
if __name__ == "__main__":
    sim = HemelrijkSimulation(mode=MODE)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(0, D)
    ax.set_ylim(0, D)
    ax.set_aspect('equal', adjustable='box')
    # colors by size
    colors = np.where(sim.size == 0, 'black', 'green')

    quiv = ax.quiver(sim.pos[:, 0], sim.pos[:, 1], np.cos(sim.angle), np.sin(sim.angle),
                     angles='xy', scale_units='xy', pivot='mid', color=colors)

    plt.title(f"Hemelrijk-style simulation — mode: {MODE}", fontsize=14)

    def animate(frame):
        pos, cos, sin, dist = sim.step()
        quiv.set_offsets(pos)
        quiv.set_UVC(cos, sin)
        return (quiv,)

    anim = FuncAnimation(fig, animate, frames=T, interval=20, blit=False, repeat=False)
    plt.show()
'''