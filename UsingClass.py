

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import scipy.spatial



#N = 10         #number of agents
D = 20          #domain size (square [0, D] x [0, D])
T = 3000        #number of frames
stepsize = 0.2  #seconds per update
#eta = 0.15      #angle noise 
v0 = 0.3        #speed
scale = 3
epsilon = 1e-6
v_SD = 0.03     #standard deviation of speed
direction_SD = np.pi/72     #standard deviation of angle
turning_rate = (np.pi) #rad / s


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
MODE = "ellipse"  


#np.random.seed(3)
np.random.seed(2)





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


def vectorized_point_to_segment_distances(positions, angles, lengths):
    #distance from centre of agent i to the closest position on the line segment of agent j (for models with a length)
    #Expand dims to (N, N, 2) where first index is point i and second is segment j

    min_point, max_point = segment_endpoints(positions, angles, lengths)

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


def ellipse_distance_hemlrijk(positions, angles, eccentricity, lengths):

    N = positions.shape[0]

    #min_point, max_point = segment_endpoints(positions, angles, lengths)

    dist_line, bearing, vec = vectorized_point_to_segment_distances(positions, angles, lengths)

    
    #relative vectors of every agent to every other agent
    rel = positions[None, :, :] - positions[:, None, :]   # (N,N,2)

    #dist_line = dist_line[None, :]  ###### NOT SURE ABOUT THIS ######

    #x and y headings of each agent
    cos_i = np.cos(angles)[:, None]   # (N,1)
    sin_i = np.sin(angles)[:, None]   # (N,1)

    #expand to (N,N)
    cos_i = np.repeat(cos_i, N, axis=1)
    sin_i = np.repeat(sin_i, N, axis=1)

    #body-frame of agent i
    #x_rel = rel[:, :, 0]
    #y_rel = rel[:, :, 1]
    
    x_rel = dist_line * np.cos(bearing)
    y_rel = dist_line * np.sin(bearing)

    #rotate relative vectors into body frame of agent i (apply the rotation matrix)
    u = x_rel * cos_i + y_rel * sin_i       #forward axis  u>0 = j in front of i, u<0 = j behind i
    z = x_rel * -sin_i + y_rel * cos_i       #side axis     z>0 = j is to left of i, z<0 = j to right of i

    #eccentricity per agent i
    e = eccentricity[:, None]     # (N,1)
    e = np.repeat(e, N, axis=1)   # (N,N)

    #Equation 13 in Hemelrijk and Kunz 2003 (to find distance)
    dist_ellipse = np.sqrt(((u**2) / (e)) + e*(z**2))

    bearing = np.arctan2(y_rel, x_rel)

    return dist_ellipse, bearing, vec, dist_line



##### weights of different the types of interactions #####

def repel(bearing, distance_array, scale_factor):
    bearing = np.atleast_1d(bearing)
    distance_array = np.atleast_1d(distance_array)
    #signs of bearing: decide left/right
    signs = np.sign(bearing)
    signs[signs == 0] = np.random.choice([-1, 1], size=(signs == 0).sum())  #if signs is 0, pick a random choice of positive or negative for the interaction
    repel_weight = ((0.05 * scale_factor) / (distance_array + epsilon))
    repel_rotate = (-signs * repel_weight )#* turning_rate)
    return np.mean(repel_rotate)


def align(distance_array, target_angles, source_angle, scale_factor, align_range, repel_range):
    distance_array = np.atleast_1d(distance_array)
    target = np.atleast_1d(target_angles)
    angle_difference = angular_difference_signed(target, source_angle)
    align_weight = (scale_factor *  np.exp(-((distance_array - 0.5 * (align_range + repel_range))/(align_range - repel_range))**2))
    align_rotate = angle_difference * align_weight * turning_rate
    return np.mean(align_rotate)


def attract(bearing, distance_array, scale_factor, attract_range, align_range):
    bearing = np.atleast_1d(bearing)
    distance_array = np.atleast_1d(distance_array)
    sign_to_j = np.sign(bearing)
    sign_to_j[sign_to_j == 0] = np.random.choice([-1, 1], size=(sign_to_j == 0).sum())
    attract_rotate = bearing #* turning_rate  #heading_vals * sign_to_j
    attract_weight = (0.2 * scale_factor * np.exp(-((distance_array - 0.5 * (attract_range + align_range))/(attract_range - align_range))**2))
    return np.mean(attract_rotate * attract_weight)

##########################################################




class HemelrijkSimulation:        #vvv change to N=None for static plots (but give a number for animation)
    def __init__(self, mode=None, N=None, size='large'):

        #assert mode in ("point", "line", "ellipse")
        # normalize mode to lowercase so step() comparisons are case-insensitive
        self.mode = mode.lower() if isinstance(mode, str) else mode
        self.N = 50
        self.D = D

        #random spawn across domain
        #self.pos = np.random.uniform(0, D, size=(N, 2))
        #self.angle = np.random.uniform(-np.pi, np.pi, size=N)

        #one school (2.5 x 2.5m box, with angles up to 90deg)
        self.pos = np.random.uniform(1, 3.5, size=(self.N, 2))
        self.angle = np.random.uniform(0, np.pi/2, size=self.N)

        self.v0 = v0
        self.v_SD = v_SD
        self.v = np.random.normal(loc=self.v0, scale=self.v_SD, size=self.N)
        self.see_pred = np.zeros(self.N, dtype=bool)   #whether each fish sees the predator or not (1 if yes, 0 if no)


        ############### use for animations, mix of large and small ###############
        
        #Size array (0 = small, 1 = large) 
        ratio = self.N   #number of small fish
        num_predators = 1
        #size_s = np.zeros(ratio, dtype=int)
        #size_l = np.ones(self.N - ratio, dtype=int)
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


        self.pred_pos = np.full((num_predators, 2), fill_value=D/3)
        self.pred_angle = np.random.uniform(-np.pi, np.pi, size=num_predators)
        self.pred_angle_SD = np.pi/72
        self.pred_speed_cruise = 0.2
        self.pred_speed_hunt = 0.5
        self.pred_speed_SD = 0.05
        self.pred_mask_distance = 5.0
        self.pred_stop_duration = 50
        self.pred_stop_counter = np.zeros(num_predators, dtype=int)
        self.pred_turning_rate = turning_rate / 20
        self.pred_turning_rate_SD = self.pred_turning_rate / 5
        self.pred_stepsize = stepsize * 1
        self.fish_caught = 0
        self.pred_attack_motion = False

        self.repulsion_range = np.where(self.size == 0, repulsion_range_s, repulsion_range_l)
        self.aligning_range = np.where(self.size == 0, aligning_range_s, aligning_range_l)
        self.attraction_range = np.full(self.N, attraction_range)

        #lengths for line agents
        self.lengths = np.where(self.size == 0, length_small, length_large)     #when self.size == 0, gives value of self.lengths 0.1m (length of small fish)
                                                                                #when not == 0, gives length 0.2m

        #lists to store statistics for plots
        self.centre_dist = []
        self.near1 = []
        self.near2 = []
        self.Nearest_Neighbours_ = []
        self.pol_parame = []
        self.group_vel = []
        self.prev_cg = None

        self.centre_dist_threat = []
        self.near1_threat = []
        self.near2_threat = []
        self.Nearest_Neighbours_threat = []
        self.pol_parame_threat = []
        self.group_vel_threat = []
        self.prev_cg_threat = None

    #function for random speed element
    def speed(self):
        return np.random.normal(loc=self.v0, scale=self.v_SD, size=self.N)
    
    def speed_pred_cruise(self):
        return np.random.normal(loc=self.pred_speed_cruise, scale=self.pred_speed_SD, size=self.pred_pos.shape[0])
    
    def speed_pred_hunt(self):
        return np.random.normal(loc=self.pred_speed_hunt, scale=self.pred_speed_SD, size=self.pred_pos.shape[0])
    

    

    #centre of gravity of school
    def c_grav(self):
        return np.mean(self.pos, axis=0)

    #average centre distance (c^t in Hemelrijk and Kunz 2003/5)
    def avg_c_dist(self):
        diff = self.c_grav() - self.pos         
        distances = np.linalg.norm(diff, axis=1) 

        #only distances less than 2
        filtered_distances = [x for x in distances if x < 2]

        return np.mean(filtered_distances)  


    #nearest neighbour distance
    def near_n(self):
        dist_x = self.pos[:, None, 0] - self.pos[None, :, 0]
        dist_y = self.pos[:, None, 1] - self.pos[None, :, 1]
        dist = np.sqrt(dist_x**2 + dist_y**2)

        np.fill_diagonal(dist, np.inf)

        #sort distances, shortest to farthest
        sort_dist = np.sort(dist, axis=1)

        filtered_sort_dist = [x for x in sort_dist[:, 0] if x < 1]

        n1 = np.mean(filtered_sort_dist)   #average distance to nearest neighbour (over all fish)
        n2 = np.mean(sort_dist[:, 1])   #average distance to second nearest neighbour``

        return n1, n2
    
    #polar order parameter
    def pol_param(self):
        vx = np.mean(np.cos(self.angle))
        vy = np.mean(np.sin(self.angle))
        return np.sqrt(vx**2 + vy**2)
    

    def Nearest_Neighbours(self):
        n1, n2 = self.near_n()
        return n1 / n2
    
    def group_velocity(self):

        cg = self.c_grav()  # current centre of gravity (2,)

        if self.prev_cg is None:
            # no velocity at first step
            self.prev_cg = cg.copy()
            return 0.0

        # displacement with periodic wrapping
        delta = cg - self.prev_cg

        # minimum-image convention for periodic boundaries
        delta = (delta + self.D / 2) % self.D - self.D / 2

        speed = np.linalg.norm(delta) / stepsize

        self.prev_cg = cg.copy()

        if speed >= 0.5:
            return 0

        return speed


    #function to see when fish are moving to avoid predator (0 dont see, 1 see and move away)
    #use for looking at differences in motion when avoiding pred and when moving normally (e.g. for looking at differences in stats when avoiding pred vs not)
    def under_threat(self):

        check = self.see_pred == True
        
        if check.sum() > self.N//2:  #if more than half the fish see the predator, count as under threat
            return 1
        else:
            return 0

    

    #all stats in one function
    def stats(self):

        #stats during normal motion
        if self.under_threat() == 0:

            c = self.avg_c_dist()
            n1, n2 = self.near_n()
            h = n1 / n2
            p = self.pol_param()
            gv = self.group_velocity()

            return {"center_distance": c,
                "nearest_neighbor": n1,
                "second_neighbor": n2,
                "Nearest_Neighbours": h,
                "polar order parameter": p,
                "group velocity": gv}
        
        #stats when any fish see predator
        else:

            c_threat = self.avg_c_dist()
            n1_threat, n2_threat = self.near_n()
            h_threat = n1_threat / n2_threat
            p_threat = self.pol_param()
            gv_threat = self.group_velocity()

            return {"center_distance_threat": c_threat,
                "nearest_neighbor_threat": n1_threat,
                "second_neighbor_threat": n2_threat,
                "Nearest_Neighbours_threat": h_threat,
                "polar order parameter_threat": p_threat,
                "group velocity_threat": gv_threat}
    


    def repel_mask(self, rel_bearing, dist):
        LOS_mask = (np.abs(rel_bearing) <= (repel_LOS/2)) #& (rel_bearing >= -repel_LOS/2)
        range_repel = self.repulsion_range[:, None] 
        mask_distance = dist <= range_repel
        mask = mask_distance & LOS_mask
        return mask
    
    def attract_mask(self, rel_bearing, dist, rel_bearing2, dist2):
        LOS_mask = (np.abs(rel_bearing) <= (attract_LOS/2)) #& ((rel_bearing) >= (-attract_LOS / 2)) 
        range_attract = self.attraction_range[:, None] 
        range_align = self.aligning_range[:, None] 
        range_repel = self.repulsion_range[:, None] 
        mask_distance = (dist <= range_attract) & (dist2 > range_align) & (dist2 > range_repel)
        mask = (mask_distance & LOS_mask) #not (self.repel_mask(rel_bearing, dist) & self.align_mask(rel_bearing, dist))
        return mask
    
    def align_mask(self, rel_bearing, dist):
        LOS_mask = ((np.abs(rel_bearing) > (repel_LOS/2)) & (np.abs(rel_bearing) <= (attract_LOS/2)))# & (((rel_bearing) < (-repel_LOS/2)) & ((rel_bearing) >= (-attract_LOS/2)))
        range_align = self.aligning_range[:, None] 
        mask_distance = dist <= range_align
        mask = mask_distance & LOS_mask
        return mask



    #Divide field of view into a number of sectors. Only the closeset neighbour in that sector can be perceived. If neighbour covers multiple zones,
    #it is only counted once (it is counted as the closest neighbour for both zones.), hence focal agent may interact with less fish than there are zones.
    def LOS_block(self, Number_sectors, distance, bearing):

        lengths = self.lengths
        fov = attract_LOS
        sectors = Number_sectors

        #dis, bearing, vec, dist_line = ellipse_distance_hemlrijk(positions, angles, eccentricity, lengths)

        half_len = lengths / 2 
        angular_half_width = np.arctan2(half_len, distance)

        #max and min angles the line segment can be seen at
        theta_max = bearing + angular_half_width
        theta_min = bearing - angular_half_width 

        #angles of the sectors
        sector_edges = np.linspace(-fov/2, fov/2, sectors + 1)

        #sets up array for the mask
        sector_mask = np.zeros((self.N, self.N), dtype=bool)

        #
        theta_max = theta_max[..., None]
        theta_min = theta_min[..., None]

        a0 = sector_edges[:-1][None, None, :]
        a1 = sector_edges[1:][None, None, :]


        overlaps = (theta_max >= a0) & (theta_min <= a1)

        overlaps[np.arange(self.N), np.arange(self.N), :] = False

        dist_ex = distance[..., None]
        mask_dis = np.where(overlaps, dist_ex, np.inf)

        j_min = np.argmin(mask_dis, axis=1)

        valid_sector = np.any(overlaps, axis=1)

        i_idx = np.repeat(np.arange(self.N), sectors)
        s_idx = np.tile(np.arange(sectors), self.N)
        j_idx = j_min.reshape(-1)

        valid = valid_sector.reshape(-1)

        sector_mask[i_idx[valid], j_idx[valid]] = True

        return sector_mask
    
    
    def pred_mask_fish(self, angl2pred):
        # angl2pred is expected shape (N, num_pred)
        angl2pred = np.atleast_2d(angl2pred)

        # ensure distances shape matches angl2pred
        distances = scipy.spatial.distance.cdist(self.pos, self.pred_pos)  # (N, num_pred)
        if distances.shape != angl2pred.shape:
            # try transpose if shapes swapped
            if distances.T.shape == angl2pred.shape:
                angl2pred = angl2pred.T
            else:
                raise RuntimeError(f"Shape mismatch in pred_mask_fish: distances {distances.shape}, angl2pred {angl2pred.shape}")

        LOS_mask = (np.abs(angl2pred) <= (attract_LOS / 2.0))

        range_i = np.atleast_1d(self.attraction_range)[:, None]  # (N,1)
        mask_distance = distances <= range_i

        mask = mask_distance & LOS_mask
        return mask
    

    def pred_dist(self):
        dist = scipy.spatial.distance.cdist(self.pos, self.pred_pos)
        #dist_matrix = scipy.spatial.distance.squareform(dist)
        return dist


    #force eqs, fish turn away from predator
    def prey_scatter(self, rel_bearing_to_pred):#, distance_):
        #dist = self.pred_dist()
        signs = np.sign(rel_bearing_to_pred)
        signs[signs == 0] = np.random.choice([-1, 1], size=(signs == 0).sum())  #if signs is 0, pick a random choice of positive or negative for the interaction
        rotate = -turning_rate * signs
        rotate_weight = 2

        return np.mean(rotate * rotate_weight)
    


    #def prey_


    #check fucntion
    def pred_vision_areas(self, distance2prey, bearing_of_prey):

        fov = attract_LOS
        sectors, areas = 5, 3
        sector_edges = np.linspace(-fov/2, fov/2, sectors + 1)
        area_edges = np.linspace(0, self.pred_mask_distance, areas + 1)

        #for prey with body length
        half_len = self.lengths / 2 
        angular_half_width = np.arctan2(half_len, distance2prey)

        #max and min angles the line segment can be seen at
        theta_max = bearing_of_prey + angular_half_width
        theta_min = bearing_of_prey - angular_half_width 

        #max and min distances fish can be seen 
        x_max = distance2prey * np.cos(theta_max)
        x_min = distance2prey * np.cos(theta_min)
        y_max = distance2prey * np.sin(theta_max)
        y_min = distance2prey * np.sin(theta_min)

        #max and min positions as a vector (N, num_pred, 2)
        prey_max_pos = np.stack([x_max, y_max], axis=-1)
        prey_min_pos = np.stack([x_min, y_min], axis=-1)

        #positions relative to pred
        rel_max_pos = prey_max_pos - self.pred_pos[None, :, :]  # (N, num_pred, 2)
        rel_min_pos = prey_min_pos - self.pred_pos[None, :, :]  # (N, num_pred, 2)

        #angles relative to pred frame
        rel_max_angle = angle_wrap(theta_max - self.pred_angle[None, :])
        rel_min_angle = angle_wrap(theta_min - self.pred_angle[None, :])

        #max and min sector and area edges (the different zones the prey can be in)
        a0 = sector_edges[:-1][None, None, :]
        a1 = sector_edges[1:][None, None, :]
        
        b0 = area_edges[:-1][None, None, :]
        b1 = area_edges[1:][None, None, :]

        #find what zone the prey is in
        in_sector = (rel_max_angle >= a0) & (rel_min_angle <= a1)
        in_area = (rel_max_pos[..., 0] >= b0) & (rel_min_pos[..., 0] <= b1) & (rel_max_pos[..., 1] >= b0) & (rel_min_pos[..., 1] <= b1)
        in_zone = in_sector & in_area

        #number counter for fish in each zone
        zone_counts = np.sum(in_zone, axis=0)  # (num_pred, sectors, areas)

        return in_zone, zone_counts


    #pred move straight at closeset fish
    def pred_rotate(self, rel_bearing_to_pred, distance2prey, bearing_of_prey):
        dist = self.pred_dist()
        #closest_fish = np.argmin(dist, axis=1)
        #direction_to_fish = self.pos[closest_fish] - self.pred_pos
        #angle_to_fish = np.arctan2(direction_to_fish[:, 1], direction_to_fish[:, 0])
        #direction = angle_wrap(angle_to_fish - self.pred_angle)
        #sign = np.sign(direction)
        #sign[sign == 0] = np.random.choice([-1, 1], size=(sign == 0).sum())
        #rotate = self.pred_turning_rate * sign


        ### add random choosing element ###
        #confusion to which fish to target
        crowdedness = np.sum(self.pred_mask_fish(rel_bearing_to_pred), axis=0) #number of fish visible to each predator


        #crowedness of each fish (number of fish in its zone and the zone directly behind it)
        in_zone, zone_counts = self.pred_vision_areas(distance2prey, bearing_of_prey)
        crowdedness = np.sum(zone_counts, axis=(1,2))  #sum over sectors and areas for each fish

        attractivness = np.zeros(self.pred_pos.shape[0])  #attractiveness of each fish, if >0.1 then candidate to be chased
        chased = np.zeros(self.pred_pos.shape[0])  #whether the fish was the one targeted in the last frame (1 if yes, 0 if no)

        for i in range(self.pred_pos.shape[0]):
            if crowdedness[i] > 3:
                                                                                                          #vvv if that was the one targetted in the last frame (reduces constant switching of target fish)
                attractivness[i] = (1 - dist[i] / self.pred_mask_distance[i]) * (0.5 / crowdedness[i]**2) * 5 if chased[i] == 1 else (1 - dist[i] / self.pred_mask_distance[i]) * (0.5 / crowdedness[i]**2)

        candidate = np.argmax(attractivness)
        chased[candidate] = 1   

        #pick fish (highest attractivness, above threshold)
        target = candidate if attractivness[candidate] > 0.1 else None

        direction_to_fish = self.pos[target] - self.pred_pos
        angle_to_fish = np.arctan2(direction_to_fish[:, 1], direction_to_fish[:, 0])
        #direction = angle_wrap(angle_to_fish - self.pred_angle)
        #sign = np.sign(direction)
        #sign[sign == 0] = np.random.choice([-1, 1], size=(sign == 0).sum())
        rotate2 = self.pred_turning_rate * angle_to_fish

        return rotate2
    

    def pred_mask(self):
        #LOS_mask = (np.abs(rel_bearing_to_pred) <= (attract_LOS/1.5))
        dist_mask = self.pred_dist() <= self.pred_mask_distance
        mask = dist_mask #& LOS_mask
        return mask
        



    #function for one step of simulation
    def step(self):


        ##### predator movement #####

        # compute vectors from each fish to each predator: shape (N, num_pred, 2)
        rel_pred = self.pos[:, None, :] - self.pred_pos[None, :, :]
        bearing_to_pred = np.arctan2(rel_pred[:, :, 1], rel_pred[:, :, 0])
        rel_bearing_to_pred = angle_wrap(bearing_to_pred - self.angle[:, None])
        pred_mask = self.pred_mask_fish(rel_bearing_to_pred)

        #shape (N, num_pred)
        pred_distances = self.pred_dist()

        #compute vectors from each fish to each predator (N, num_pred, 2)
        #pred_rel = self.pos[:, None, :] - self.pred_pos[None, :, :]
        #pred_bearing_to_pred = np.arctan2(pred_rel[:, :, 1], pred_rel[:, :, 0])

        #relative bearing in pred frame (N, num_pred)
        pred_rel_bearing_to_pred = angle_wrap(self.angle[:, None] - bearing_to_pred)

        #mask of fish that predators can see (N, num_pred)
        pred_mask = self.pred_mask_fish(pred_rel_bearing_to_pred)

        num_pred = self.pred_pos.shape[0]
        pred_update_angle = np.array(self.pred_angle, copy=True)
        pred_move_mask = np.zeros(num_pred, dtype=bool)

        #choose nearest visible fish per predator and aim at it

        #caught counter
        #fish_caught = 0

        for j in range(num_pred):
            candidates = np.where(pred_mask[:, j])[0]
            if candidates.size == 0:
                continue

            #nearest fish
            local_dists = pred_distances[candidates, j]
            sorted_candidates = np.sort(local_dists).argsort()  #sort candidates by distance
            #chosen = candidates[np.argmin(local_dists)]
            close_five = sorted_candidates[:5]  #indices of the five closest candidates
            prob_dist = np.exp(-local_dists[close_five])  #convert distances to probabilities (closer fish more likely to be chosen)
            prob_dist /= np.sum(prob_dist)  #normalize to sum to 1
            chosen = np.random.choice(close_five, p=prob_dist)#closest five candidates


            #direction to chosen fish
            direction_to_fish = self.pos[chosen] - self.pred_pos[j]
            angle_to_fish = np.arctan2(direction_to_fish[1], direction_to_fish[0])

            #set predator heading directly at fish 
            pred_update_angle[j] = np.random.normal(angle_to_fish, self.pred_angle_SD)

            #update angle using function
            #pred_update_angle[j] = np.random.normal(self.pred_rotate(rel_bearing_to_pred[:, j], pred_distances[:, j], pred_bearing_to_pred[:, j]), self.pred_angle_SD)

            pred_move_mask[j] = True


            if pred_distances[chosen] <= 0.1:
                print("caught")
                self.fish_caught +=1
                self.pred_stop_counter[j] = self.pred_stop_duration

                #delete fish
                self.N -= 1
                self.pos = np.array([x for i, x in enumerate(self.pos) if i != chosen])
                self.angle = np.array([x for i, x in enumerate(self.angle) if i != chosen])
                self.size = np.array([x for i, x in enumerate(self.size) if i != chosen])
                self.eccentricity = np.array([x for i, x in enumerate(self.eccentricity) if i != chosen])
                self.lengths = np.array([x for i, x in enumerate(self.lengths) if i != chosen])
                self.repulsion_range = np.array([x for i, x in enumerate(self.repulsion_range) if i != chosen])
                self.aligning_range = np.array([x for i, x in enumerate(self.aligning_range) if i != chosen])
                self.attraction_range = np.array([x for i, x in enumerate(self.attraction_range) if i != chosen])
                self.v = np.array([x for i, x in enumerate(self.v) if i != chosen])
                self.see_pred = np.array([x for i, x in enumerate(self.see_pred) if i != chosen])


                #predator stops moving for some frames, then starts moving again at random angle
                #for _ in range(51): 
                    #self.pred_angle[j] = np.random.uniform(-1, 1) * turning_rate * 10
                    #self.pred_speed = 0 if t < 50 else self.speed_pred()
                    #self.pred_pos[j] = (self.pred_pos[j] + np.array([np.cos(self.pred_angle[j]), np.sin(self.pred_angle[j])]) * self.pred_speed * stepsize) % self.D    
                pass

        #no fish in mask
        no_move = ~pred_move_mask
        if np.any(no_move):
            pred_update_angle[no_move] = np.random.normal(self.pred_angle[no_move], self.pred_turning_rate_SD)

        pred_speeds_cruise = self.speed_pred_cruise()
        pred_speeds_hunt = self.speed_pred_hunt()

        pred_speeds = pred_speeds_cruise if not np.any(pred_move_mask) else pred_speeds_hunt
        self.pred_attack_motion = True if np.any(pred_move_mask) else False

        #freeze stopped predators
        pred_speeds[self.pred_stop_counter > 0] = 0.0
        pred_update_angle[self.pred_stop_counter > 0] = np.random.uniform(-1, 1) * turning_rate * 10

        # limit predator turning per step
        if not pred_update_angle[self.pred_stop_counter > 0].size == 1:
            delta = angle_wrap(pred_update_angle - self.pred_angle)
            # if `self.pred_turning_rate` is radians per step use directly,
            # otherwise if it's radians per unit time multiply by step: * self.pred_stepsize
            max_turn = self.pred_turning_rate
            delta = np.clip(delta, -max_turn, max_turn)
            pred_update_angle = angle_wrap(self.pred_angle + delta)

        pred_cos = np.cos(pred_update_angle)
        pred_sin = np.sin(pred_update_angle)

        pred_vx = pred_cos * pred_speeds * self.pred_stepsize
        pred_vy = pred_sin * pred_speeds * self.pred_stepsize

        self.pred_pos[:, 0] = (self.pred_pos[:, 0] + pred_vx) % self.D
        self.pred_pos[:, 1] = (self.pred_pos[:, 1] + pred_vy) % self.D
        self.pred_angle = pred_update_angle

                # decrement counters
        self.pred_stop_counter[self.pred_stop_counter > 0] -= 1

        ##### end of pred section #####






        pos = self.pos
        angle = self.angle
        N = self.N

             
        #different sims for the three different modes in described in Hemelrijk and Kunz 2003
        if self.mode == "point":
            dist, bearing, vec = point_to_point_distances(pos)

        elif self.mode == "line":
            #min_pt, max_pt = segment_endpoints(pos, angle, self.lengths)
            dist, bearing, vec = vectorized_point_to_segment_distances(pos, angle, self.lengths)

        elif self.mode == "ellipse":
            dist, bearing, vec, dist_line = ellipse_distance_hemlrijk(pos, angle, self.eccentricity, self.lengths)
            #min_pt, max_pt = segment_endpoints(pos, angle, self.lengths)
            dist_at, bearing_at, vec_at = vectorized_point_to_segment_distances(pos, angle, self.lengths)

        else:
            raise ValueError("Unknown mode")
        

        #dist_line1, bearing1, vec1 = vectorized_point_to_segment_distances(pos, angle, self.lengths)


        #for repel and align terms
        #dist, bearing, vec, dist_line = ellipse_distance_hemlrijk(pos, angle, self.eccentricity, self.lengths)        
        
        #for attraction terms, different as this segment is not elliptical
        #min_pt, max_pt = segment_endpoints(self.pos, self.angle, self.lengths)
        #dist_at, bearing_at, vec_at = vectorized_point_to_segment_distances(pos, angle, self.lengths)

        #dist_pred, rotate_pred = self.pred_dist(), self.predator()


        #fill diagonal of distance matrix so the boids dont interact with themselves
        if dist.ndim == 2 and dist.shape[0] == dist.shape[1]:
            np.fill_diagonal(dist, np.inf)
        #if dist_at.ndim == 2 and dist_at.shape[0] == dist_at.shape[1]:
            #np.fill_diagonal(dist_at, np.inf)
        else:
            raise RuntimeError(f"Distance matrix not square! dist shape={dist_at.shape}")
        
        #relative angle of j from frame of i (how agent i views j)
        #bearing = angle from i to j (in world frame)

        rel_bearing = angle_wrap(bearing - angle[:, None]) 
        #rel_bearing_at = angle_wrap(bearing_at - angle[:, None])

        #rel_bearing = (np.arctan2(bearing, angle))
        #rel_bearing_at = (np.arctan2(bearing_at, angle))


        #repel_mask, align_mask, attract_mask = self.LOS_masks(rel_bearing, dist) 
        #repel_mask_at, align_mask_at, attract_mask_at = self.LOS_masks(rel_bearing_at, dist_at) 


        #np.full((N, N), False) #

        
        # compute vectors from each fish to each predator: shape (N, num_pred, 2)
        rel_pred = self.pos[:, None, :] - self.pred_pos[None, :, :]
        bearing_to_pred = np.arctan2(rel_pred[:, :, 1], rel_pred[:, :, 0])
        rel_bearing_to_pred = angle_wrap(bearing_to_pred - self.angle[:, None])
        pred_mask = self.pred_mask_fish(rel_bearing_to_pred)

        #shape (N, num_pred)
        pred_distances = self.pred_dist()



        repel_mask = self.repel_mask(rel_bearing, dist)
        align_mask = self.align_mask(rel_bearing, dist)
        attract_mask = self.attract_mask(rel_bearing, dist, rel_bearing, dist)

        fish_pred_mask = self.pred_mask_fish(rel_bearing_to_pred)


        ### For LOS blocking ###
        #sector_mask = self.LOS_block(10, dist, bearing)
        #sector_mask_at = self.LOS_block(10, dist_at, bearing_at)
        #repel_mask = sector_mask & repel_mask
        #align_mask  = sector_mask & align_mask
        #attract_mask = sector_mask_at & attract_mask

        rotation = np.zeros(N)
        rotation_repel = np.zeros(N)

        #find rotation at each time step for all agents i
        for i in range(N):

            self.see_pred[i] = 1 if bool(np.any(fish_pred_mask[i])) else 0

            if np.any(self.see_pred[i]) == 0:

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
                        
                    attract_contribution[k] = attract(rel_bearing[i, j], dist[i, j], scale_factor, self.attraction_range[i], self.aligning_range[i])
            

                    #self.attract(dist_at[i, j], scale_factor)    
                    

                rotation_align = 0 if align_.size == 0 else np.mean(align_contribution)
                rotation_attract = 0 if attract_.size == 0 else np.mean(attract_contribution)

                if (align_.size > 0) or (attract_.size > 0):
                    rotation[i] = np.mean(rotation_align) + np.mean(rotation_attract) 

                else:
                    rotation[i] = 0

            ##### response to pred #####
            else:
                pred_ = np.where(fish_pred_mask[i])[0]
                pred_contribution = np.zeros(len(pred_))

                #for j in enumerate(pred_):

                #fish school disperse away from predator, so rotation is in opposite direction to the contribution of the predator
                pred_contribution = -self.prey_scatter(rel_bearing_to_pred[i])

                rotation_pred = 0 if pred_.size == 0 else np.mean(pred_contribution)
                #self.see_pred[i] = 1 if pred_.size > 0 else 0

                if pred_.size > 0:
                    rotation[i] = np.mean(rotation_pred)
                else:
                    rotation[i] = 0
            

            
            


        #adding rotation to original angle
        mean_heading = angle + rotation * stepsize
                                                      #vvvvvvvvvvv = noise
        update_angle = np.random.normal(mean_heading, direction_SD)     

        #self.v = self.speed()

        change_in_angle = angle_wrap(update_angle - angle)
        max_turn = turning_rate 
        change_in_angle = np.clip(change_in_angle, -max_turn, max_turn)
        update_angle = angle_wrap(angle + change_in_angle)

        cos = np.cos(update_angle)
        sin = np.sin(update_angle)
        vx = cos * self.v * stepsize   #speed in x-direction
        vy = sin * self.v * stepsize   #speed in y-direction

        #update state
        self.pos[:, 0] = (self.pos[:, 0] + vx) % self.D
        self.pos[:, 1] = (self.pos[:, 1] + vy) % self.D
        self.angle = update_angle

        stats = self.stats()

        #if np.all(self.see_pred == 0):

        check = self.see_pred == True

        # if more than half the fish see the predator, count as under threat
        if check.sum() > self.N // 2:
            # append threat stats (use non-threat as fallback)
            self.centre_dist_threat.append(stats.get("center_distance_threat", stats.get("center_distance")))
            self.near1_threat.append(stats.get("nearest_neighbor_threat", stats.get("nearest_neighbor")))
            self.near2_threat.append(stats.get("second_neighbor_threat", stats.get("second_neighbor")))
            self.Nearest_Neighbours_threat.append(stats.get("Nearest_Neighbours_threat", stats.get("Nearest_Neighbours")))
            self.pol_parame_threat.append(stats.get("polar order parameter_threat", stats.get("polar order parameter")))
            self.group_vel_threat.append(stats.get("group velocity_threat", stats.get("group velocity")))
        else:
            # append normal stats (use threat as fallback)
            self.centre_dist.append(stats.get("center_distance", stats.get("center_distance_threat")))
            self.near1.append(stats.get("nearest_neighbor", stats.get("nearest_neighbor_threat")))
            self.near2.append(stats.get("second_neighbor", stats.get("second_neighbor_threat")))
            self.Nearest_Neighbours_.append(stats.get("Nearest_Neighbours", stats.get("Nearest_Neighbours_threat")))
            self.pol_parame.append(stats.get("polar order parameter", stats.get("polar order parameter_threat")))
            self.group_vel.append(stats.get("group velocity", stats.get("group velocity_threat")))



        return self.pos, cos, sin, self.pred_pos, pred_cos, pred_sin, self.see_pred#, t#, self.centre_dist.copy(), self.near1.copy(), self.near2.copy(), self.Nearest_Neighbours_.copy()
    

#for animation

if __name__ == "__main__":
    sim = HemelrijkSimulation(mode=MODE)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, D)
    ax.set_ylim(0, D)
    ax.set_xticks([D])
    ax.set_yticks([D])
    ax.set_aspect('equal', adjustable='box')
    #ax.set_xlabel('x-axis', fontsize='14')
    #ax.set_ylabel('y-axis', fontsize='14')
    # colors by size
    colors = np.where(sim.size == 0, 'black', 'black')

    quiv = ax.quiver(sim.pos[:, 0], sim.pos[:, 1], np.cos(sim.angle), np.sin(sim.angle),
                     angles='xy', scale_units='xy', pivot='mid', color=colors)
    pred = ax.quiver(sim.pred_pos[:, 0], sim.pred_pos[:, 1], np.cos(sim.pred_angle), np.sin(sim.pred_angle), 
                     angles='xy', scale_units='xy', pivot='mid', color='red')
    pred_range = plt.Circle((sim.pred_pos[:, 0], sim.pred_pos[:, 1]), 5, color='red', fill=False)
    ax.add_patch(pred_range)

    #plt.title(f"Hemelrijk-style simulation — mode: {MODE}", fontsize=14)
    #plt.title('(c)', fontsize='14')

    def animate(frame):
        #print('frame:', frame)
        global quiv
        pos, cos, sin, pred_pos, pred_cos, pred_sin, b = sim.step()

        # If the number of agents has changed (e.g. one was removed by predator),
        # recreate the quiver so the number of arrow positions matches the data.
        current_N = pos.shape[0]
        quiv_N = getattr(quiv, 'N', None)   #get named attribute from object, return None if not found

        if quiv_N is None:
            try:
                quiv_N = quiv.X.size
            except Exception:
                quiv_N = None

        if quiv_N is None or current_N != quiv_N:
            try:
                quiv.remove()
            except Exception:
                pass
            colors_now = np.where(sim.size == 0, 'black', 'black')
            quiv = ax.quiver(pos[:, 0], pos[:, 1], cos, sin,
                             angles='xy', scale_units='xy', pivot='mid', color=colors_now)
        else:
            quiv.set_offsets(pos)
            quiv.set_UVC(cos, sin)

        # Update predator arrows and range
        pred.set_offsets(pred_pos)
        pred.set_UVC(pred_cos, pred_sin)
        pred_range.center = pred_pos[0]
        print(b)
        return (quiv, pred, pred_range)

    anim = FuncAnimation(fig, animate, frames=1000, interval=20, blit=False, repeat=False)
    #anim.save("moving predator, stops when fish caught, increase speed for hunt, prey escape.gif", dpi=300)
    plt.show()





    '''

    def pred_step(self):
        #shape (N, num_pred)
        pred_distances = self.pred_dist()

        #compute vectors from each fish to each predator (N, num_pred, 2)
        pred_rel = self.pos[:, None, :] - self.pred_pos[None, :, :]
        pred_bearing_to_pred = np.arctan2(pred_rel[:, :, 1], pred_rel[:, :, 0])

        #relative bearing in pred frame (N, num_pred)
        pred_rel_bearing_to_pred = angle_wrap(self.angle[:, None] - bearing_to_pred)

        #mask of fish that predators can see (N, num_pred)
        pred_mask = self.pred_mask_fish(pred_rel_bearing_to_pred)

        num_pred = self.pred_pos.shape[0]
        pred_update_angle = np.array(self.pred_angle, copy=True)
        pred_move_mask = np.zeros(num_pred, dtype=bool)

        #choose nearest visible fish per predator and aim at it
        for j in range(num_pred):
            candidates = np.where(pred_mask[:, j])[0]
            if candidates.size == 0:
                continue

            #nearest fish
            local_dists = pred_distances[candidates, j]
            chosen = candidates[np.argmin(local_dists)]

            #direction to chosen fish
            direction_to_fish = self.pos[chosen] - self.pred_pos[j]
            angle_to_fish = np.arctan2(direction_to_fish[1], direction_to_fish[0])

            #set predator heading directly at fish 
            pred_update_angle[j] = np.random.normal(angle_to_fish, self.pred_angle_SD)
            pred_move_mask[j] = True


            if pred_distances[chosen] <= 0.1:
                #print("caught")

                #predator stops moving for some frames, then starts moving again at random position
                for _ in range(10):
                    self.pred_pos[j] = self.pos[chosen]  #stay on top of caught fish
                    self.pred_angle[j] = np.random.uniform(-np.pi, np.pi)

                pass

        #no fish in mask
        no_move = ~pred_move_mask
        if np.any(no_move):
            pred_update_angle[no_move] = np.random.normal(self.pred_angle[no_move], self.pred_angle_SD)

        pred_speeds = self.speed_pred()
        pred_cos = np.cos(pred_update_angle)
        pred_sin = np.sin(pred_update_angle)

        pred_vx = pred_cos * pred_speeds * stepsize
        pred_vy = pred_sin * pred_speeds * stepsize

        self.pred_pos[:, 0] = (self.pred_pos[:, 0] + pred_vx) % self.D
        self.pred_pos[:, 1] = (self.pred_pos[:, 1] + pred_vy) % self.D
        self.pred_angle = pred_update_angle

        return self.pred_pos.copy(), pred_cos, pred_sin
        '''
    
#sim=HemelrijkSimulation(mode=MODE)
#print(sim.pred_step()[0])
#print(sim.pred_step()[1])
#print(sim.pred_step()[2])

def run_sim(N_vals, fish_size, repeats=25):
    modes = ['point', 'line', 'ellipse']

    #avg_centre = {m: [] for m in modes}
    #avg_near_n = {m: [] for m in modes}

    #avg_centre_norm = {m: [] for m in modes}
    #avg_near_n_norm = {m: [] for m in modes}
    

    avg_polar_order = {m: [] for m in modes}
    avg_group_vel = {m: [] for m in modes}

    avg_polar_order_norm = {m: [] for m in modes}
    avg_group_vel_norm = {m: [] for m in modes}


    #avg_Nearest_Neighbours = {m: [] for m in modes}

    for mode in modes:
        print(f"\n=== {mode} ===")

        for N in N_vals:
            print(f"  N = {N}")

            #centre_runs = []
            #near_runs = []

            polar_order_runs = []
            group_vel_runs = []

            #Nearest_Neighbours_runs = []

            for r in range(repeats):
                sim = HemelrijkSimulation(mode=mode, N=N, size=fish_size)

                sim.pos = np.random.uniform(5, 7.5, size=(N, 2))
                sim.angle = np.random.uniform(0, np.pi/2, size=N)

                print(f"    run {r}: ")

                
                for t in range(T):
                    sim.step()

                #last 1000 time steps
                #centre_runs.append(np.mean(sim.centre_dist[-1000:]))
                #near_runs.append(np.mean(sim.near1[-1000:]))

                polar_order_runs.append(np.mean(sim.pol_parame[-1000:]))
                group_vel_runs.append(np.mean(sim.group_vel[-1000:]))

                #Nearest_Neighbours_runs.append(np.mean(sim.Nearest_Neighbours_[-1000:]))

            
            #avg_centre[mode].append(np.mean(centre_runs))
            #avg_near_n[mode].append(np.mean(near_runs))

            #avg_centre_norm[mode] = avg_centre[mode] / np.linalg.norm(avg_centre[mode])
            #avg_near_n_norm[mode] = avg_near_n[mode] / np.linalg.norm(avg_near_n[mode])


            avg_polar_order[mode].append(np.mean(polar_order_runs))
            avg_group_vel[mode].append(np.mean(group_vel_runs))

            avg_polar_order_norm[mode] = avg_polar_order[mode] / np.linalg.norm(avg_polar_order[mode])
            avg_group_vel_norm[mode] = avg_group_vel[mode] / np.linalg.norm(avg_group_vel[mode])

            #avg_Nearest_Neighbours[mode].append(np.mean(Nearest_Neighbours_runs))

    return avg_polar_order_norm, avg_group_vel_norm #avg_centre_norm, avg_near_n_norm#, avg_polar_order




plt.rcParams["font.family"] = "Times New Roman"



####### STATIC PLOTS #######

N_vals = [3, 4, 6, 10, 25, 50, 75, 100]

#print("\n=== Running SMALL ===")
#centre_s, near_s = run_sim(N_vals, 'small')
#pol_order_s, group_vel_s = run_sim(N_vals, 'small')
#Nearest_Neighbours_s = run_sim(N_vals, 'small')

#print("\n=== Running LARGE ===")
#centre_l, near_l = run_sim(N_vals, 'large')
#pol_order_l, group_vel_l = run_sim(N_vals, 'large')
#Nearest_Neighbours_l = run_sim(N_vals, 'large')



modes = ['point', 'line', 'ellipse']
linestyles = {'point':':', 'line':'--', 'ellipse':'-'}
markers = {'point':'o', 'line':'square', 'ellipse':'thin_diamond'}


### static plot of fish ###




### Nearest_Neighbours plot ###
'''
fig, axs = plt.subplots(2, 1, figsize=(6, 6))
ax = axs[0, 0]
for m in modes:
    ax.plot(N_vals, Nearest_Neighbours_s[m], 'o', linestyle=linestyles[m], label=m)
ax.set_title('(a)', fontsize='14')
ax.set_xlabel('Group Size', fontsize='14')
ax.set_ylabel('Nearest_Neighbours', fontsize='14')
ax.legend(fontsize='14')

ax = axs[0, 1]
for m in modes:
    ax.plot(N_vals, Nearest_Neighbours_l[m], 'o', linestyle=linestyles[m], label=m)
ax.set_title('(b)', fontsize='14')
ax.set_xlabel('Group Size', fontsize='14')
ax.set_ylabel('Nearest_Neighbours', fontsize='14')
ax.legend(fontsize='14')
'''

### for polar order parameter and group velocity plot ###
'''
fig, axs = plt.subplots(2, 2, figsize=(6, 6))

ax = axs[0, 0]
for m in modes:
    ax.plot(N_vals, pol_order_s[m], 'o', linestyle=linestyles[m], label=m)
ax.set_title('(a)', fontsize='14')
ax.set_xlabel('Group Size', fontsize='14')
ax.set_ylabel('Polar Order Parameter', fontsize='14')
ax.legend(fontsize='14')

ax = axs[1, 0]
for m in modes:
    ax.plot(N_vals, pol_order_l[m], 'o', linestyle=linestyles[m])
ax.set_title('(b)', fontsize='14')
ax.set_xlabel('Group Size', fontsize='14')
ax.set_ylabel('Polar Order Parameter', fontsize='14')
ax.legend()

ax = axs[0, 1]
for m in modes:
    ax.plot(N_vals, group_vel_s[m], 'o', linestyle=linestyles[m], label=m)
ax.set_title('(c)', fontsize='14')
ax.set_xlabel('Group Size', fontsize='14')
ax.set_ylabel('Group Velocity', fontsize='14')
ax.legend(fontsize='14')

ax = axs[1, 1]
for m in modes:
    ax.plot(N_vals, group_vel_l[m], 'o', linestyle=linestyles[m])
ax.set_title('(d)', fontsize='14')
ax.set_xlabel('Group Size', fontsize='14')
ax.set_ylabel('Group Velocity', fontsize='14')
ax.legend()
'''

### for 2x2 distance to centre of nearest neighbour distance ###
'''
fig, axs = plt.subplots(2, 2, figsize=(6, 6))
ax = axs[0, 0]
for m in modes:
    ax.plot(N_vals, centre_s[m], '-o', linestyle=linestyles[m], marker=markers[m], label=m)
ax.set_title('Average centre distance (small fish)')
ax.set_xlabel('group size')
ax.set_ylabel('centre distance')
ax.legend()

ax = axs[0, 1]
for m in modes:
    ax.plot(N_vals, centre_l[m], '-o', linestyle=linestyles[m], marker=markers[m])
ax.set_title('Average centre distance (large fish)')
ax.set_xlabel('group size')
ax.set_ylabel('centre distance')
ax.legend()

ax = axs[1, 0]
for m in modes:
    ax.plot(N_vals, near_s[m], '-o', linestyle=linestyles[m], marker=markers[m])
ax.set_title('nearest neighbour distance (small fish)')
ax.set_xlabel('group size')
ax.set_ylabel('nearest neighbour')
ax.legend()

ax = axs[1, 1]
for m in modes:
    ax.plot(N_vals, near_l[m], '-o', linestyle=linestyles[m], marker=markers[m])
ax.set_title('nearest neighbour distance (large fish)')
ax.set_xlabel('group size')
ax.set_ylabel('nearest neighbour')
ax.legend()


plt.tight_layout()
plt.savefig('Hemelrijk 2003 Near neighbour and distance to centre plot.png', dpi=400)
'''

    
#sim = HemelrijkSimulation(mode=MODE)
#print(sim.step()[3])







'''
def plot_agent_LOS_debug(sim, i=0, N_seCTORS=10, max_range=None):
    """
    Visualise FOV and sector blocking for agent i
    using only existing simulation quantities.
    """

    pos = sim.pos
    ang = sim.angle
    lengths = sim.lengths
    N = sim.N
    eccentricity = sim.eccentricity

    dist, bearing, vec, dist_line = ellipse_distance_hemlrijk(pos, ang, eccentricity, lengths)        
    dist_at, bearing_at, vec_at = vectorized_point_to_segment_distances(pos, ang, lengths)


    # Run LOS once
    sector_mask = sim.LOS_block(N_seCTORS, dist, bearing)
    sector_mask_at = sim.LOS_block(N_seCTORS, dist_at, bearing_at)

    # Relative positions
    #rel = pos - pos[i]
    #dist = np.linalg.norm(rel, axis=1)

    # Plot range (purely visual)
    if max_range is None:
        max_range = np.percentile(dist[dist > 0], 90)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_aspect("equal")

    # --- Visible vs blocked neighbours ---
    visible = sector_mask[i]
    #visible = sector_mask_at[i]
    visible[i] = False

    ax.prey_scatter(
        pos[~visible, 0],
        pos[~visible, 1],
        s=40,
        facecolors="none",
        edgecolors="red",
        label="Blocked"
    )

    ax.prey_scatter(
        pos[visible, 0],
        pos[visible, 1],
        s=40,
        color="green",
        label="Visible"
    )

    # --- Focal agent ---
    ax.prey_scatter(pos[i, 0], pos[i, 1], c="black", s=80, zorder=5)

    # Heading arrow
    ax.plot(
        [pos[i, 0], pos[i, 0] + np.cos(ang[i])],
        [pos[i, 1], pos[i, 1] + np.sin(ang[i])],
        color="black",
        linewidth=2
    )

    # --- Field of view ---
    fov = attract_LOS
    fov_angles = np.array([
        ang[i] - fov / 2,
        ang[i] + fov / 2
    ])

    for a in fov_angles:
        ax.plot(
            [pos[i, 0], pos[i, 0] + max_range * np.cos(a)],
            [pos[i, 1], pos[i, 1] + max_range * np.sin(a)],
            color="grey",
            linewidth=2,
            alpha=0.8
        )

    # --- Sector boundaries ---
    sector_edges = np.linspace(-fov / 2, fov / 2, N_seCTORS + 1)
    for a in sector_edges:
        ax.plot(
            [pos[i, 0], pos[i, 0] + max_range * np.cos(ang[i] + a)],
            [pos[i, 1], pos[i, 1] + max_range * np.sin(ang[i] + a)],
            linestyle="--",
            color="black",
            alpha=0.4
        )

    # --- Optional: draw body segments (you already use line agents) ---
    try:
        min_pt, max_pt = segment_endpoints(pos, ang, lengths)
        ax.plot(
            [min_pt[:, 0], max_pt[:, 0]],
            [min_pt[:, 1], max_pt[:, 1]],
            color="blue",
            alpha=0.3
        )
    except Exception:
        pass

    # --- Formatting ---
    ax.legend()
    ax.set_title(f"Agent {i}: FOV and Sector LOS")
    ax.set_xlim(pos[i, 0] - max_range, pos[i, 0] + max_range)
    ax.set_ylim(pos[i, 1] - max_range, pos[i, 1] + max_range)

    plt.show()


sim = HemelrijkSimulation(mode=MODE)
sim.step()

#plot_agent_LOS_debug(sim, i=0, N_seCTORS=10)

'''




def plot_agent_interaction_regions(sim, i=0, max_range=6.0):
    """
    Static plot of interaction regions for focal agent i.
    """

    pos = sim.pos
    ang = sim.angle
    lengths = sim.lengths
    ecc = sim.eccentricity
    N = sim.N

    # distances & bearings
    dist, bearing, vec, dist_line = ellipse_distance_hemlrijk(
        pos, ang, ecc, lengths
    )
    dist_at, bearing_at, vec_at = vectorized_point_to_segment_distances(
        pos, ang, lengths
    )

    # relative bearings (FIXED, correct version)
    rel_bearing = angle_wrap(bearing - ang[:, None])
    rel_bearing_at = angle_wrap(bearing_at - ang[:, None])

    # masks
    repel_mask = sim.repel_mask(rel_bearing, dist)
    align_mask = sim.align_mask(rel_bearing, dist)
    attract_mask = sim.attract_mask(rel_bearing_at, dist_at)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_aspect("equal")

    # focal agent
    ax.prey_scatter(pos[i, 0], pos[i, 1], s=120, c="black", zorder=5)
    ax.arrow(
        pos[i, 0], pos[i, 1],
        np.cos(ang[i]), np.sin(ang[i]),
        width=0.03, color="black", zorder=6
    )

    # neighbours by interaction type
    ax.prey_scatter(
        pos[repel_mask[i], 0],
        pos[repel_mask[i], 1],
        c="red", label="Repulsion", s=60
    )

    ax.prey_scatter(
        pos[align_mask[i], 0],
        pos[align_mask[i], 1],
        c="orange", label="Alignment", s=60
    )

    #ax.prey_scatter(
    #    pos[attract_mask[i], 0],
    #    pos[attract_mask[i], 1],
    #    c="blue", label="Attraction", s=60
    #)

    # non-interacting neighbours
    none_mask = ~(repel_mask[i] | align_mask[i] | attract_mask[i])
    none_mask[i] = False
    ax.prey_scatter(
        pos[none_mask, 0],
        pos[none_mask, 1],
        facecolors="none", edgecolors="grey",
        label="No interaction"
    )

    # --- LOS boundaries ---
    for a in [-repel_LOS/2, repel_LOS/2]:
        ax.plot(
            [pos[i, 0], pos[i, 0] + max_range * np.cos(ang[i] + a)],
            [pos[i, 1], pos[i, 1] + max_range * np.sin(ang[i] + a)],
            linestyle="--", color="red", alpha=0.6
        )

    for a in [-attract_LOS/2, attract_LOS/2]:
        ax.plot(
            [pos[i, 0], pos[i, 0] + max_range * np.cos(ang[i] + a)],
            [pos[i, 1], pos[i, 1] + max_range * np.sin(ang[i] + a)],
            linestyle="--", color="blue", alpha=0.4
        )

    # --- draw body segments ---
    min_pt, max_pt = segment_endpoints(pos, ang, lengths)
    ax.plot(
        [min_pt[:, 0], max_pt[:, 0]],
        [min_pt[:, 1], max_pt[:, 1]],
        color="black", alpha=0.25
    )

    ax.set_xlim(pos[i, 0] - max_range, pos[i, 0] + max_range)
    ax.set_ylim(pos[i, 1] - max_range, pos[i, 1] + max_range)

    ax.set_title(f"Interaction regions for agent {i}")
    ax.legend()
    plt.show()


sim = HemelrijkSimulation(mode=MODE)
sim.step()
#plot_agent_interaction_regions(sim, i=0)
