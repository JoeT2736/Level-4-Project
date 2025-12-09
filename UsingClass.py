'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import scipy.spatial
import scipy.constants



class Boid:
    def __init__(self, position, angle, size, length,
                 v0, direction_SD,
                 repel_LOS, attract_LOS,
                 repulsion_range,
                 aligning_range, attraction_range,
                 turning_rate, eccentricity,
                 repel_scalefactor, align_scalefactor,
                 attract_scalefactor, active_sort_repel,
                 active_sort_align, active_sort_attract,
                 risk_avoidance):

        self.pos = np.asarray(position, dtype=float)
        self.angle = float(angle)
        self.size = float(size)
        self.length = float(length)

        # --- Individual parameters ---
        self.v0 = v0
        self.direction_SD = direction_SD
        self.repel_LOS = repel_LOS
        self.attract_LOS = attract_LOS
        self.repulsion_range = repulsion_range
        self.aligning_range = aligning_range
        self.attraction_range = attraction_range
        self.turning_rate = turning_rate
        self.eccentricity = eccentricity
        self.repel_scalefactor = repel_scalefactor
        self.align_scalefactor = align_scalefactor
        self.attract_scalefactor = attract_scalefactor
        self.active_sort_repel = active_sort_repel
        self.active_sort_align = active_sort_align
        self.active_sort_attract = active_sort_attract
        self.risk_avoidance = risk_avoidance



class Predator:
    def __init__(self, position, angle, v0, v_sd, interaction_range, los):
        self.pos = np.asarray(position, dtype=float)
        self.angle = float(angle)
        self.v0 = v0
        self.v_sd = v_sd
        self.range = interaction_range
        self.los = los




class BoidSimulation:
    def __init__(self):

        self.N = 100
        self.D = 20

        start_pos = np.random.uniform(0, self.D, size=(self.N, 2))
        start_angle = np.random.uniform(0, np.pi/2, size=self.N)

        size_s = np.zeros(25)
        size_l = np.ones(75)
        sizes = np.concatenate((size_s, size_l))

        self.boids = [Boid(pos[i], angle[i], sizes[i]) for i in range(self.N)]


    

    @staticmethod
    def angle_wrap(a):
        return (a + np.pi) % (2 * np.pi) - np.pi

    def angular_difference_signed(self, target, source):
        return self.angle_wrap(target - source)

    
    def repel(self, bearing, distance_array, scale_factor):
        bearing = np.atleast_1d(bearing)
        distance_array = np.atleast_1d(distance_array)
        signs = np.sign(bearing)
        signs[signs == 0] = np.random.choice([-1, 1], size=(signs == 0).sum())
        repel_weight = ((0.05*scale_factor)/distance_array)
        repel_rotate = (-self.turning_rate * signs * repel_weight)
        return np.mean(repel_rotate)

    def align(self, distance_array, target, source, scale_factor, align_range, repel_range):
        distance_array = np.atleast_1d(distance_array)
        target = np.atleast_1d(target)
        source = np.atleast_1d(source)
        angle_difference = self.angular_difference_signed(target, source)
        align_weight = (scale_factor * np.exp(-(distance_array - 0.5*(align_range+repel_range)/(align_range-repel_range))**2))
        align_rotate = self.turning_rate * angle_difference * align_weight
        return np.mean(align_rotate)

    def attract(self, bearing, distance_array, scale_factor, attract_range, align_range):
        bearing = np.atleast_1d(bearing)
        distance_array = np.atleast_1d(distance_array)
        heading_vals = np.cos(bearing)
        sign_to_j = np.sign(bearing)
        sign_to_j[sign_to_j == 0] = 1
        attract_rotate = self.turning_rate * heading_vals * sign_to_j
        attract_weight = (0.2*scale_factor*np.exp(-(distance_array - 0.5*(attract_range+align_range)/(attract_range-align_range))**2))
        return np.mean(attract_rotate * attract_weight)

    

    def update(self):
        # Extract arrays from objects
        pos = np.array([b.pos for b in self.boids])
        angle = np.array([b.angle for b in self.boids])
        size = np.array([b.size for b in self.boids])

        
        DistanceMatrix = scipy.spatial.distance.pdist(pos)
        DistanceMatrix = scipy.spatial.distance.squareform(DistanceMatrix)

        vec_ij = pos[None, :, :] - pos[:, None, :]
        dist = np.linalg.norm(vec_ij, axis=2)
        np.fill_diagonal(dist, np.inf)

        bearing_ij = np.arctan2(vec_ij[:, :, 1], vec_ij[:, :, 0])
        rel_bearing_ij = self.angle_wrap(bearing_ij - angle[:, None])

        # Masks (identical)
        repel_mask_distance_s = dist <= self.repulsion_range_s
        align_mask_distance_s = (dist > self.repulsion_range_s) & (dist <= self.aligning_range_s)
        attract_mask_distance_s = (dist > self.aligning_range_s) & (dist <= self.attraction_range)

        repel_mask_distance_l = dist <= self.repulsion_range_l
        align_mask_distance_l = (dist > self.repulsion_range_l) & (dist <= self.aligning_range_l)
        attract_mask_distance_l = (dist > self.aligning_range_l) & (dist <= self.attraction_range)

        repel_mask_LOS = np.abs(rel_bearing_ij) <= (self.repel_LOS/2)
        align_mask_LOS = (np.abs(rel_bearing_ij) > (self.repel_LOS/2)) & (np.abs(rel_bearing_ij) <= (np.deg2rad(90) + self.repel_LOS/2))
        attract_mask_LOS = np.abs(rel_bearing_ij) <= (self.attract_LOS/2)

        repel_mask_s = repel_mask_distance_s & repel_mask_LOS
        align_mask_s = align_mask_distance_s & align_mask_LOS
        attract_mask_s = attract_mask_distance_s & attract_mask_LOS

        repel_mask_l = repel_mask_distance_l & repel_mask_LOS
        align_mask_l = align_mask_distance_l & align_mask_LOS
        attract_mask_l = attract_mask_distance_l & attract_mask_LOS

        rotation_s = np.zeros(self.N)
        rotation_l = np.zeros(self.N)

        

        for i in range(self.N):

            if size[i] == 0:   # Small fish
                repel_s = np.where(repel_mask_s[i])[0]
                repel_contribution = np.zeros(len(repel_s))

                if repel_s.size > 0:
                    for j in range(len(repel_s)):
                        if size[repel_s][j] == 0:
                            repel_contribution[j] = self.repel(rel_bearing_ij[i, repel_s[j]], DistanceMatrix[i, repel_s[j]], self.repel_scalefactor_s/self.active_sort_repel)
                        else:
                            repel_contribution[j] = self.repel(rel_bearing_ij[i, repel_s[j]], DistanceMatrix[i, repel_s[j]], self.repel_scalefactor_s*self.active_sort_repel*self.risk_avoidance)

                    rotation_s[i] = np.mean(repel_contribution)

                else:
                    # alignment
                    align_s = np.where(align_mask_s[i])[0]
                    align_contribution_s = np.zeros(len(align_s))
                    if align_s.size > 0:
                        for j in range(len(align_s)):
                            if size[align_s][j] == 0:
                                align_contribution_s[j] = self.align(DistanceMatrix[i, align_s[j]], angle[align_s[j]], angle[i],
                                                                      self.align_scalefactor*self.active_sort_align,
                                                                      self.aligning_range_s, self.repulsion_range_s)
                            else:
                                align_contribution_s[j] = self.align(DistanceMatrix[i, align_s[j]], angle[align_s[j]], angle[i],
                                                                      self.align_scalefactor/self.active_sort_align,
                                                                      self.aligning_range_s, self.repulsion_range_s)

                    # attraction
                    attract_s = np.where(attract_mask_s[i])[0]
                    attract_contribution_s = np.zeros(len(attract_s))
                    if attract_s.size > 0:
                        for j in range(len(attract_s)):
                            if size[attract_s][j] == 0:
                                attract_contribution_s[j] = self.attract(rel_bearing_ij[i, attract_s[j]], DistanceMatrix[i, attract_s[j]],
                                                                          self.attract_scalefactor*self.active_sort_attract,
                                                                          self.attraction_range, self.aligning_range_s)
                            else:
                                attract_contribution_s[j] = self.attract(rel_bearing_ij[i, attract_s[j]], DistanceMatrix[i, attract_s[j]],
                                                                          self.attract_scalefactor/self.active_sort_attract,
                                                                          self.attraction_range, self.aligning_range_s)

                    rotation_s[i] = (0 if align_s.size == 0 else np.mean(align_contribution_s)) + \
                                    (0 if attract_s.size == 0 else np.mean(attract_contribution_s))

        
            else:
                repel_l = np.where(repel_mask_l[i])[0]
                repel_contribution = np.zeros(len(repel_l))

                if repel_l.size > 0:
                    for j in range(len(repel_l)):
                        if size[repel_l][j] == 0:
                            repel_contribution[j] = self.repel(rel_bearing_ij[i, repel_l[j]], DistanceMatrix[i, repel_l[j]],
                                                               self.repel_scalefactor_l*self.active_sort_repel)
                        else:
                            repel_contribution[j] = self.repel(rel_bearing_ij[i, repel_l[j]], DistanceMatrix[i, repel_l[j]],
                                                               self.repel_scalefactor_l/self.active_sort_repel)
                    rotation_l[i] = np.mean(repel_contribution)

                else:
                    align_l = np.where(align_mask_l[i])[0]
                    align_contribution_l = np.zeros(len(align_l))

                    if align_l.size > 0:
                        for j in range(len(align_l)):
                            if size[align_l][j] == 0:
                                align_contribution_l[j] = self.align(DistanceMatrix[i, align_l[j]], angle[align_l[j]], angle[i],
                                                                     self.align_scalefactor/self.active_sort_align,
                                                                     self.aligning_range_l, self.repulsion_range_l)
                            else:
                                align_contribution_l[j] = self.align(DistanceMatrix[i, align_l[j]], angle[align_l[j]], angle[i],
                                                                     self.align_scalefactor*self.active_sort_align,
                                                                     self.aligning_range_l, self.repulsion_range_l)

                    attract_l = np.where(attract_mask_l[i])[0]
                    attract_contribution_l = np.zeros(len(attract_l))

                    if attract_l.size > 0:
                        for j in range(len(attract_l)):
                            if size[attract_l][j] == 0:
                                attract_contribution_l[j] = self.attract(rel_bearing_ij[i, attract_l[j]], DistanceMatrix[i, attract_l[j]],
                                                                          self.attract_scalefactor/self.active_sort_attract,
                                                                          self.attraction_range, self.aligning_range_l)
                            else:
                                attract_contribution_l[j] = self.attract(rel_bearing_ij[i, attract_l[j]], DistanceMatrix[i, attract_l[j]],
                                                                          self.attract_scalefactor*self.active_sort_attract,
                                                                          self.attraction_range, self.aligning_range_l)

                    rotation_l[i] = (0 if align_l.size == 0 else np.mean(align_contribution_l)) + \
                                    (0 if attract_l.size == 0 else np.mean(attract_contribution_l))

        
        mean_heading = angle + (rotation_s + rotation_l) * self.stepsize
        update_angle = np.random.normal(mean_heading, self.direction_SD)

        cos = np.cos(update_angle)
        sin = np.sin(update_angle)

        vx = cos * self.v0 * self.stepsize
        vy = sin * self.v0 * self.stepsize

        pos[:, 0] += vx
        pos[:, 1] += vy
        pos = np.mod(pos, self.D)

        # Write back to objects
        for i in range(self.N):
            self.boids[i].pos = pos[i]
            self.boids[i].angle = update_angle[i]

        return pos, cos, sin



sim = BoidSimulation()

fig, ax = plt.subplots()
ax.set_xlim([0, sim.D])
ax.set_ylim([0, sim.D])

def qlook(A, B):
    return np.concatenate((np.full(25, A), np.full(75, B)))

pos = np.array([b.pos for b in sim.boids])
angle = np.array([b.angle for b in sim.boids])

animated_plot_quiver = ax.quiver(
    pos[:, 0], pos[:, 1],
    np.cos(angle), np.sin(angle),
    clim=[-np.pi, np.pi],
    angles='xy',
    scale_units='xy',
    scale=qlook(1.5, 1),
    pivot='mid',
    color=qlook('black', 'green'),
)

ax.set_aspect('equal')

def Animate_quiver(frame):
    pos, cos, sin = sim.update()
    animated_plot_quiver.set_offsets(pos)
    animated_plot_quiver.set_UVC(cos, sin)
    return (animated_plot_quiver,)

anim = FuncAnimation(fig, Animate_quiver, interval=1, frames=sim.T, blit=False, repeat=False)
plt.show()
'''



# rewritten_hemlrijk_sim.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import scipy.spatial

# -------------------------
# Simulation parameters
# -------------------------
N = 100         # number of agents
D = 40          # domain size (square [0, D] x [0, D])
T = 1000        # number of frames (animation)
stepsize = 0.2  # s per update
eta = 0.15      # angle noise (not used directly; replaced by direction_SD)
v0 = 0.3        # base speed (m/s)
scale = 3
epsilon = 1e-6
v_SD = 0.03
direction_SD = np.pi/72
turning_rate = (np.pi/2)  # rad / s

# size-specific radii (as in your original variables)
repulsion_range_s = 0.3   # small fish
repulsion_range_l = 0.6   # large fish
aligning_range_s = 1.0
aligning_range_l = 2.0
attraction_range = 5.0    # same for all sizes (vision)
eccentricity_s = 2
eccentricity_l = 4

repel_LOS = np.deg2rad(60)    # used to select LOS for repulsion
attract_LOS = np.deg2rad(300) # vision cone for attraction

repel_scalefactor_s = 1.0
repel_scalefactor_l = 2.0
align_scalefactor = 1.0
attract_scalefactor = 1.0

active_sort_repel = 2
active_sort_align = 2
active_sort_attract = 2

risk_avoidance = 20.0

# line & ellipse geometry lengths
length_small = 0.2
length_large = 0.4

# select mode: "point", "line", or "ellipse"
MODE = "ellipse"  # change to "point" or "line" as required

# random seed for reproducibility
np.random.seed(3)



def angle_wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi

def angular_difference_signed(target, source):
    return angle_wrap(target - source)



def point_to_point_distances(positions):
    #pairwise distance for point agents
    vec = positions[None, :, :] - positions[:, None, :]  # (N,N,2), vec[i,j] = pos_j - pos_i
    dist = np.linalg.norm(vec, axis=2)
    bearing = np.arctan2(vec[:, :, 1], vec[:, :, 0])  # angle from i to j
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
    """
    Vectorized distance from each agent center (point) to every other agent's segment (line segment).
    We'll compute a matrix dist[i,j] = distance between pos[i] (point) and segment j.
    Returns dist matrix and bearing from i to closest point on segment j (approximate but good).
    """
    # Expand dims to (N, N, 2) where first index is point i and second is segment j
    P = positions[:, None, :]   # (N,1,2)
    A = min_point[None, :, :]   # (1,N,2)
    B = max_point[None, :, :]   # (1,N,2)

    AB = B - A                  # (1,N,2)
    AP = P - A                  # (N,N,2)

    AB_dot_AB = np.sum(AB * AB, axis=2)  # (1,N)
    # avoid divide-by-zero
    AB_dot_AB = np.where(AB_dot_AB == 0, epsilon, AB_dot_AB)

    t = np.sum(AP * AB, axis=2) / AB_dot_AB  # (N,N), projection factor for each pair
    t_clamped = np.clip(t, 0.0, 1.0)

    closest = A + (t_clamped[..., None] * AB)  # (N,N,2)
    diff = P - closest
    dist = np.linalg.norm(diff, axis=2)
    bearing = np.arctan2(closest[:, :, 1] - P[:, :, 1], closest[:, :, 0] - P[:, :, 0])  # angle from point i to closest point on j
    vec = closest - P  # from i to closest on j
    return dist, bearing, vec


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


def ellipse_distance_hemlrijk(positions, angles, eccentricity):
    """
    Implements Equation (13) from Hemelrijk & Hildenbrandt:
    e_ij = sqrt( (u^2) / e^2 + z^2 )
    where (u, z) are coordinates of j in the body-frame of i.
    Returns dist (N,N), bearing (N,N), vec (N,N,2).
    """

    N = positions.shape[0]

    # relative vectors
    rel = positions[None, :, :] - positions[:, None, :]   # (N,N,2)

    # focal agent headings
    cos_i = np.cos(angles)[:, None]   # (N,1)
    sin_i = np.sin(angles)[:, None]   # (N,1)

    # expand to (N,N)
    cos_i = np.repeat(cos_i, N, axis=1)
    sin_i = np.repeat(sin_i, N, axis=1)

    # body-frame of agent i
    x_rel = rel[:, :, 0]
    y_rel = rel[:, :, 1]

    u =  x_rel * cos_i + y_rel * sin_i       # forward axis
    z = -x_rel * sin_i + y_rel * cos_i       # side axis

    # eccentricity per agent i
    e = eccentricity[:, None]     # (N,1)
    e = np.repeat(e, N, axis=1)   # (N,N)

    # Equation 13
    dist = np.sqrt((u**2) / (e**2 + 1e-9) + z**2)

    # Bearings in world frame (for attraction direction etc.)
    bearing = np.arctan2(rel[:, :, 1], rel[:, :, 0])

    return dist, bearing, rel




# -------------------------
# Interaction functions (repel / align / attract / predator)
# These keep the behavioural forms you used, but vectorized usage happens in update.
# -------------------------
def repel(bearing, distance_array, scale_factor):
    """Return mean turning contribution from repulsion per neighbor set (bearing array & distance array)."""
    bearing = np.atleast_1d(bearing)
    distance_array = np.atleast_1d(distance_array)
    # signs of bearing: decide left/right
    signs = np.sign(bearing)
    signs[signs == 0] = np.random.choice([-1, 1], size=(signs == 0).sum())

    repel_weight = ((0.05 * scale_factor) / (distance_array + epsilon))
    repel_rotate = (-turning_rate * signs * repel_weight)
    return np.mean(repel_rotate)


def align(distance_array, target_angles, source_angle, scale_factor, align_range, repel_range):
    """Align: turning proportional to signed angular difference scaled by distance weight."""
    distance_array = np.atleast_1d(distance_array)
    target = np.atleast_1d(target_angles)
    angle_difference = angular_difference_signed(target, source_angle)
    # Use a gaussian-like weight (similar to your original)
    mid = 0.5 * (align_range + repel_range) / max(align_range - repel_range, epsilon)
    align_weight = (scale_factor * np.exp(-(distance_array - mid)**2))
    align_rotate = turning_rate * angle_difference * align_weight
    return np.mean(align_rotate)


def attract(bearing, distance_array, scale_factor, attract_range, align_range):
    """Attraction turns towards heading indicated by bearing (cos component)."""
    bearing = np.atleast_1d(bearing)
    distance_array = np.atleast_1d(distance_array)

    heading_vals = np.cos(bearing)
    sign_to_j = np.sign(bearing)
    sign_to_j[sign_to_j == 0] = 1

    attract_rotate = turning_rate * heading_vals * sign_to_j
    mid = 0.5 * (attract_range + align_range) / max(attract_range - align_range, epsilon)
    attract_weight = (0.2 * scale_factor * np.exp(-(distance_array - mid)**2))
    return np.mean(attract_rotate * attract_weight)


# -------------------------
# Main Simulation Class
# -------------------------
class HemelrijkSimulation:
    def __init__(self, mode="point"):
        assert mode in ("point", "line", "ellipse")
        self.mode = mode
        self.N = N
        self.D = D
        # State
        self.pos = np.random.uniform(0, D, size=(N, 2))
        self.angle = np.random.uniform(-np.pi, np.pi, size=N)
        self.v0 = v0
        # Size array (0 = small, 1 = large) keeping your split 50/50 by default
        half = N // 2
        size_s = np.zeros(half, dtype=int)
        size_l = np.ones(N - half, dtype=int)
        self.size = np.concatenate((size_s, size_l))
        self.eccentricity = np.where(self.size == 0, eccentricity_s, eccentricity_l)


        # Precompute size-dependent geometry arrays
        self.repulsion_range = np.where(self.size == 0, repulsion_range_s, repulsion_range_l)
        self.aligning_range = np.where(self.size == 0, aligning_range_s, aligning_range_l)
        # Attraction is uniform
        self.attraction_range = attraction_range

        # lengths for line agents
        self.lengths = np.where(self.size == 0, length_small, length_large)

        # For ellipse agents compute axes a (long axis) and b (short axis)
        # We'll base ellipse axes on repulsion and aligning ranges and eccentricity
        self.a_axis = np.where(self.size == 0, repulsion_range_s * eccentricity_s, repulsion_range_l * eccentricity_l)
        self.b_axis = np.where(self.size == 0, repulsion_range_s, repulsion_range_l)  # simpler choice

    def step(self):
        """Compute one update and return pos, cos_sin used for plotting."""
        pos = self.pos
        angle = self.angle
        N = self.N

        # compute pairwise distances and bearings according to mode
        if self.mode == "point":
            dist, bearing, vec = point_to_point_distances(pos)
        elif self.mode == "line":
            # endpoints for segments (min_point,max_point)
            min_pt, max_pt = segment_endpoints(pos, angle, self.lengths)
            dist, bearing, vec = vectorized_point_to_segment_distances(pos, min_pt, max_pt)
        elif self.mode == "ellipse":
            # use ellipse distance transform relative to j's ellipse axes
            #dist, bearing, vec = ellipse_distance_transform(pos, angle, self.a_axis, self.b_axis)
            dist, bearing, vec = ellipse_distance_hemlrijk(self.pos, self.angle, self.eccentricity)


        else:
            raise ValueError("Unknown mode")

        # Ensure dist is square before filling the diagonal
        if dist.ndim == 2 and dist.shape[0] == dist.shape[1]:
            np.fill_diagonal(dist, np.inf)
        else:
            raise RuntimeError(f"Distance matrix not square! dist shape={dist.shape}")

        # Compute rel_bearing = bearing_ij - heading_i (how agent i views j)
        # bearing is angle from i to j (same orientation), subtract angle_i to get relative bearing in agent i frame
        rel_bearing = angle_wrap(bearing - angle[:, None])

        # LOS masks (based on your earlier scheme)
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

        # combine with LOS for final masks
        repel_mask = repel_mask_distance & repel_LOS_mask
        align_mask = align_mask_distance & align_LOS_mask
        attract_mask = attract_mask_distance & attract_LOS_mask

        # compute rotation contributions per agent
        rotation = np.zeros(N)

        # For each focal agent i compute rotation
        for i in range(N):
            # neighbors in each zone
            repel_idx = np.where(repel_mask[i])[0]
            if repel_idx.size > 0:
                # compute contributions per neighbor (respecting neighbor sizes when needed)
                contributions = []
                for j in repel_idx:
                    # decide scale factor depending on neighbor size and active sorting
                    if self.size[j] == 0:
                        sf = repel_scalefactor_s / active_sort_repel
                    else:
                        sf = repel_scalefactor_l * active_sort_repel * risk_avoidance
                    contributions.append(repel(rel_bearing[i, j], dist[i, j], sf))
                rotation[i] = np.mean(contributions)
                continue  # repulsion has priority

            # if no repulsion neighbors, compute align+attract
            align_idx = np.where(align_mask[i])[0]
            attract_idx = np.where(attract_mask[i])[0]

            align_contribs = []
            for j in align_idx:
                if self.size[j] == 0:
                    sf = align_scalefactor * active_sort_align
                else:
                    sf = align_scalefactor / active_sort_align
                align_contribs.append(align(dist[i, j], self.angle[j], self.angle[i], sf, self.aligning_range[i], self.repulsion_range[i]))

            attract_contribs = []
            for j in attract_idx:
                if self.size[j] == 0:
                    sf = attract_scalefactor * active_sort_attract
                else:
                    sf = attract_scalefactor / active_sort_attract
                attract_contribs.append(attract(rel_bearing[i, j], dist[i, j], sf, self.attraction_range, self.aligning_range[i]))

            rot_align = 0.0 if len(align_contribs) == 0 else np.mean(align_contribs)
            rot_attract = 0.0 if len(attract_contribs) == 0 else np.mean(attract_contribs)
            rotation[i] = rot_align + rot_attract

        # integrate heading with noise
        mean_heading = angle + rotation * stepsize
        update_angle = np.random.normal(mean_heading, direction_SD)

        cos = np.cos(update_angle)
        sin = np.sin(update_angle)
        vx = cos * self.v0 * stepsize
        vy = sin * self.v0 * stepsize

        # update state
        self.pos[:, 0] = (self.pos[:, 0] + vx) % self.D
        self.pos[:, 1] = (self.pos[:, 1] + vy) % self.D
        self.angle = update_angle

        return self.pos.copy(), cos, sin


# -------------------------
# Quick test + animation
# -------------------------
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

    #plt.title(f"Hemelrijk-style simulation — mode: {MODE}", fontsize=14)

    def animate(frame):
        pos, cos, sin = sim.step()
        quiv.set_offsets(pos)
        quiv.set_UVC(cos, sin)
        return (quiv,)

    anim = FuncAnimation(fig, animate, frames=T, interval=20, blit=False, repeat=False)
    plt.show()
