

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
MODE = "ellipse"  


np.random.seed(3)




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

    dist_line = dist_line[None, :]  ###### NOT SURE ABOUT THIS ######

    #x and y headings of each agent
    cos_i = np.cos(angles)[:, None]   # (N,1)
    sin_i = np.sin(angles)[:, None]   # (N,1)

    #expand to (N,N)
    cos_i = np.repeat(cos_i, N, axis=1)
    sin_i = np.repeat(sin_i, N, axis=1)

    #body-frame of agent i
    x_rel = dist_line[:, :, 0]
    y_rel = dist_line[:, :, 1]

    #rotate relative vectors into body frame of agent i (apply the rotation matrix)
    u = x_rel * cos_i + y_rel * sin_i       #forward axis  u>0 = j in front of i, u<0 = j behind i
    z = x_rel * -sin_i + y_rel * cos_i       #side axis     z>0 = j is to left of i, z<0 = j to right of i

    #eccentricity per agent i
    e = eccentricity[:, None]     # (N,1)
    e = np.repeat(e, N, axis=1)   # (N,N)

    #Equation 13 in Hemelrijk and Kunz 2003 (to find distance)
    dist_ellipse = np.sqrt(((u**2) / (e)) + e*(z**2))

    #Bearings (not in agent i's frame)
    bearing = np.arctan2(rel[:, :, 1], rel[:, :, 0])

    return dist_ellipse, bearing, rel, dist_line



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
    attract_weight = (0.2 * scale_factor * np.exp(-((distance_array - 0.5 * (attract_range + align_range))/(attract_range - align_range, epsilon))**2))
    return np.mean(attract_rotate * attract_weight)

##########################################################




class HemelrijkSimulation:          #vvv change to N=None for static plots (but give a number for animation)
    def __init__(self, mode=None, N=50, size='small'):

        #assert mode in ("point", "line", "ellipse")
        self.mode = mode
        self.N = N
        self.D = D

        #random spawn across domain
        #self.pos = np.random.uniform(0, D, size=(N, 2))
        #self.angle = np.random.uniform(-np.pi, np.pi, size=N)

        #one school (2.5 x 2.5m box, with angles up to 90deg)
        self.pos = np.random.uniform(1.5, 3.5, size=(N, 2))
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
        self.attraction_range = np.full(N, attraction_range)

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
        return speed

    

    #all stats in one function
    def stats(self):
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
    


    def repel_mask(self, rel_bearing, dist):
        LOS_mask = np.abs(rel_bearing) <= (repel_LOS)
        range_i = self.repulsion_range[:, None] 
        mask_distance = dist <= range_i
        mask = mask_distance & LOS_mask
        return mask
    
    def attract_mask(self, rel_bearing, dist):
        LOS_mask = np.abs(rel_bearing) <= (attract_LOS)
        range_i = self.attraction_range[:, None] 
        mask_distance = dist <= range_i
        mask = mask_distance & LOS_mask
        return mask
    
    def align_mask(self, rel_bearing, dist):
        LOS_mask = (np.abs(rel_bearing) > (repel_LOS)) & (np.abs(rel_bearing) <= (attract_LOS))
        range_i = self.aligning_range[:, None] 
        mask_distance = dist <= range_i
        mask = mask_distance & LOS_mask
        return mask


    
    '''
    def LOS_masks(self, rel_bearing, dist):

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

        return repel_mask, align_mask, attract_mask
    '''


    #Divide field of view into a number of sectors. Only the closeset neighbour in that sector can be perceived. If neighbour covers multiple zones,
    #it is only counted once (it is counted as the closest neighbour for both zones.), hence focal agent may interact with less fish than there are zones.
    def LOS_block(self, Number_sectors):

        positions = self.pos
        angles = self.angle
        lengths = self.lengths
        eccentricity = self.eccentricity
        fov = attract_LOS
        sectors = Number_sectors
        sector_width = fov/sectors

        zone_angle = attract_LOS / sectors

        min_point, max_point = segment_endpoints(positions, angles, lengths)

        #neighbours = np.zeros(len=10)

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
        u = x_rel * cos_i + y_rel * sin_i       #forward axis  u>0 = j in front of i, u<0 = j behind i             y?
        z = x_rel * -sin_i + y_rel * cos_i       #side axis     z>0 = j is to left of i, z<0 = j to right of i     x?

        #angle of j in frame of i, (theta=0 is directily infront of i)
        angle = np.arctan2(z, u)

        dis, bearing, vec, dist_line = ellipse_distance_hemlrijk(positions, angles, eccentricity, lengths)

        half_len = lengths / 2
        angular_half_width = np.arctan2(half_len, dis + epsilon)

        theta_max = angle + angular_half_width
        theta_min = angle - angular_half_width 

        sector_edges = np.linspace(-fov/2, fov/2, sectors + 1)

        sector_mask = np.zeros((N, N), dtype=bool)


        theta_max = theta_max[..., None]
        theta_min = theta_min[..., None]

        a0 = sector_edges[:-1][None, None, :]
        a1 = sector_edges[1:][None, None, :]

        overlaps = (theta_max >= a0) & (theta_min <= a1)

        overlaps[np.arange(N), np.arange(N), :] = False

        dist_ex = dis[..., None]
        mask_dis = np.where(overlaps, dist_ex, np.inf)

        j_min = np.argmin(mask_dis, axis=1)

        valid_sector = np.any(overlaps, axis=1)

        i_idx = np.repeat(np.arange(N), sectors)
        s_idx = np.tile(np.arange(sectors), N)
        j_idx = j_min.reshape(-1)

        valid = valid_sector.reshape(-1)

        sector_mask[i_idx[valid], j_idx[valid]] = True

        '''
        for i in range(N):
            for s in range(sectors):
                a0 = sector_edges[s]
                a1 = sector_edges[s+1]

                overlap = ((theta_max[i] >= a0) & (theta_min[i] <= a1))

                if not np.any(overlap):
                    continue

                j_closest = np.argmin(dis[i, overlap])
                j_idx = np.where(overlap)[0][j_closest]

                sector_mask[i, j_idx] = True        
        '''

        return sector_mask
    

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

        # Run LOS once
        sector_mask = sim.LOS_block(N_seCTORS)

        # Relative positions
        rel = pos - pos[i]
        dist = np.linalg.norm(rel, axis=1)

        # Plot range (purely visual)
        if max_range is None:
            max_range = np.percentile(dist[dist > 0], 90)

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.set_aspect("equal")

        # --- Visible vs blocked neighbours ---
        visible = sector_mask[i]
        visible[i] = False

        ax.scatter(
            pos[~visible, 0],
            pos[~visible, 1],
            s=40,
            facecolors="none",
            edgecolors="red",
            label="Blocked"
        )

        ax.scatter(
            pos[visible, 0],
            pos[visible, 1],
            s=40,
            color="green",
            label="Visible"
        )

        # --- Focal agent ---
        ax.scatter(pos[i, 0], pos[i, 1], c="black", s=80, zorder=5)

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
        '''
    

    def predator():
        return
    

    def current():
        return
    


    #function for one step of simulation
    def step(self):
        pos = self.pos
        angle = self.angle
        N = self.N

        '''        
        #different sims for the three different modes in described in Hemelrijk and Kunz 2003
        if self.mode == "point":
            dist, bearing, vec = point_to_point_distances(pos)

        elif self.mode == "line":
            min_pt, max_pt = segment_endpoints(pos, angle, self.lengths)
            dist, bearing, vec = vectorized_point_to_segment_distances(pos, min_pt, max_pt)

        elif self.mode == "ellipse":
            dist, bearing, vec, dist_line = ellipse_distance_hemlrijk(self.pos, self.angle, self.eccentricity, self.lengths)
            min_pt, max_pt = segment_endpoints(pos, angle, self.lengths)
            dist_at, bearing_at, vec_at = vectorized_point_to_segment_distances(pos, min_pt, max_pt)

        else:
            raise ValueError("Unknown mode")
        '''

        #for repel and align terms
        dist, bearing, vec, dist_line = ellipse_distance_hemlrijk(pos, angle, self.eccentricity, self.lengths)        
        
        #for attraction terms, different as this segment is not elliptical
        #min_pt, max_pt = segment_endpoints(self.pos, self.angle, self.lengths)
        dist_at, bearing_at, vec_at = vectorized_point_to_segment_distances(pos, angle, self.lengths)



        #fill diagonal of distance matrix so the boids dont interact with themselves
        if dist.ndim == 2 and dist.shape[0] == dist.shape[1]:
            np.fill_diagonal(dist, np.inf)
        if dist_at.ndim == 2 and dist_at.shape[0] == dist_at.shape[1]:
            np.fill_diagonal(dist_at, np.inf)
        else:
            raise RuntimeError(f"Distance matrix not square! dist shape={dist_at.shape}")
        
        #relative angle of j from frame of i (how agent i views j)
        #bearing = angle from i to j (in world frame)

        #rel_bearing = angle_wrap(bearing - angle[:, None]) <<<<<<< old method, not good
        rel_bearing = (np.arctan2(bearing, angle))
        rel_bearing_at = (np.arctan2(bearing_at, angle))


        #repel_mask, align_mask, attract_mask = self.LOS_masks(rel_bearing, dist) 
        #repel_mask_at, align_mask_at, attract_mask_at = self.LOS_masks(rel_bearing_at, dist_at) 

        sector_mask = self.LOS_block(Number_sectors=10)


        repel_mask = self.repel_mask(rel_bearing, dist)
        align_mask = self.align_mask(rel_bearing, dist)
        attract_mask = self.attract_mask(rel_bearing_at, dist_at)


        #repel_mask = sector_mask & repel_mask
        #align_mask  = sector_mask & align_mask
        #attract_mask_at = sector_mask & attract_mask_at

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
                    
                attract_contribution[k] = attract(rel_bearing_at[i, j], dist_at[i, j], scale_factor, self.attraction_range[i], self.aligning_range[i])
            

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
        self.Nearest_Neighbours_.append(stats["Nearest_Neighbours"])
        self.pol_parame.append(stats["polar order parameter"])
        self.group_vel.append(stats["group velocity"])
        t = self.LOS_block(10)

        return self.pos.copy(), cos, sin, dist_line#, t#, self.centre_dist.copy(), self.near1.copy(), self.near2.copy(), self.Nearest_Neighbours_.copy()
    
def run_sim(N_vals, fish_size, repeats=25):
    modes = ['point', 'line', 'ellipse']

    avg_centre = {m: [] for m in modes}
    avg_near_n = {m: [] for m in modes}
    #avg_polar_order = {m: [] for m in modes}
    #avg_group_vel = {m: [] for m in modes}
    #avg_Nearest_Neighbours = {m: [] for m in modes}

    for mode in modes:
        #print(f"\n=== {mode} ===")

        for N in N_vals:
            #print(f"  N = {N}")

            centre_runs = []
            near_runs = []
            #polar_order_runs = []
            #group_vel_runs = []
            #Nearest_Neighbours_runs = []

            for r in range(repeats):
                sim = HemelrijkSimulation(mode=mode, N=N, size=fish_size)

                sim.pos = np.random.uniform(5, 7.5, size=(N, 2))
                sim.angle = np.random.uniform(0, np.pi/2, size=N)

                #print(f"    run {r}: ")

                
                for t in range(T):
                    sim.step()

                #last 1000 time steps
                centre_runs.append(np.mean(sim.centre_dist[-1000:]))
                near_runs.append(np.mean(sim.near1[-1000:]))
                #polar_order_runs.append(np.mean(sim.pol_parame[-1000:]))
                #group_vel_runs.append(np.mean(sim.group_vel[-1000:]))
                #Nearest_Neighbours_runs.append(np.mean(sim.Nearest_Neighbours_[-1000:]))

            
            avg_centre[mode].append(np.mean(centre_runs))
            avg_near_n[mode].append(np.mean(near_runs))
            #avg_polar_order[mode].append(np.mean(polar_order_runs))
            #avg_group_vel[mode].append(np.mean(group_vel_runs))
            #avg_Nearest_Neighbours[mode].append(np.mean(Nearest_Neighbours_runs))

    return avg_centre, avg_near_n #avg_Nearest_Neighbours #avg_polar_order, group_vel_runs #avg_centre, avg_near_n, avg_polar_order




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

    #plt.title(f"Hemelrijk-style simulation — mode: {MODE}", fontsize=14)
    #plt.title('(c)', fontsize='14')

    def animate(frame):
        print('frame:', frame)
        pos, cos, sin, dist = sim.step()
        quiv.set_offsets(pos)
        quiv.set_UVC(cos, sin)
        #print(dist)
        return (quiv,)

    anim = FuncAnimation(fig, animate, frames=T, interval=20, blit=False, repeat=False)
    #anim.save("All large, N=50, random spawn.gif", dpi=400)
    plt.show()






def plot_agent_LOS_debug(sim, i=0, N_seCTORS=10, max_range=None):
    """
    Visualise FOV and sector blocking for agent i
    using only existing simulation quantities.
    """

    pos = sim.pos
    ang = sim.angle
    lengths = sim.lengths
    N = sim.N

    # Run LOS once
    sector_mask = sim.LOS_block(N_seCTORS)

    # Relative positions
    rel = pos - pos[i]
    dist = np.linalg.norm(rel, axis=1)

    # Plot range (purely visual)
    if max_range is None:
        max_range = np.percentile(dist[dist > 0], 90)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_aspect("equal")

    # --- Visible vs blocked neighbours ---
    visible = sector_mask[i]
    visible[i] = False

    ax.scatter(
        pos[~visible, 0],
        pos[~visible, 1],
        s=40,
        facecolors="none",
        edgecolors="red",
        label="Blocked"
    )

    ax.scatter(
        pos[visible, 0],
        pos[visible, 1],
        s=40,
        color="green",
        label="Visible"
    )

    # --- Focal agent ---
    ax.scatter(pos[i, 0], pos[i, 1], c="black", s=80, zorder=5)

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

