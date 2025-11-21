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
fake_wall = 1
Radius_circle = D/2
centre = np.array([Radius_circle, Radius_circle])
epsilon = 1e-6

pos = np.random.uniform(0+0.3, D-0.3, size=(N, 2))
angle = np.random.uniform(0, 2*np.pi, size=N)

### random positions in circular area ###
#length = (np.random.uniform(-1, 1, size=(N, 2))) * Radius_circle
#pos = centre + length 
#print(pos)



        ### could maybe change wall_force to include cos and sin of direction to see the x and y directions of the boids ??????

def wall_force(position, wall, direction, choice):
    #if wall == wall:
    #    dist = wall - position
    #elif wall == wall:
    #    dist = position
    
    dist = wall - position

    if abs(dist) < fake_wall:      #if close enough then...
        force_wall = 1/(abs(dist))**scale      #potential due to wall
        #force_wall = np.exp(dist**scale)
        if direction >= choice:      #choice = angle where boid turns either up/down or left/right based on its direction
            force_wall = force_wall
        elif direction < choice:
            force_wall = -force_wall
    else:
        force_wall = 0

    return force_wall


def wall_force_circle(position):
    vector = position - centre
    distance = np.linalg.norm(vector)
    s = Radius_circle - distance
    if s <= fake_wall: 
        force_wall = 1/(Radius_circle - distance)**scale
    else:
        force_wall = 0
    return force_wall

def wrap_to_pi(a):
    """Wrap angle(s) to [-pi, pi]."""
    return (a + np.pi) % (2 * np.pi) - np.pi


def wall_torque_single(pos_i, ang_i, wall_orientation, wall_coord):
    """
    Compute scalar torque (signed) from a single wall on a single boid.
    - pos_i: (x,y)
    - ang_i: scalar angle (radians)
    - wall_orientation: 'vertical' or 'horizontal'
    - wall_coord: coordinate of the wall (0 or D)
    Behavior:
      - Only exerts torque if the wall is within fake_wall distance AND the wall is
        ahead in the boid's heading (i.e. in the boid's line-of-sight).
      - Chooses the tangent direction (two possible: up/down for vertical, left/right for horizontal)
        that requires the *least rotation* from the current heading and produces a signed torque
        to rotate toward that tangent direction.
      - Torque magnitude grows as distance -> 0 (via 1/(dist+eps)**scale).
    """
    h = np.array([np.cos(ang_i), np.sin(ang_i)])  # heading vector

    if wall_orientation == 'vertical':
        # for vertical wall x = wall_coord
        dist_to_wall = abs(pos_i[0] - wall_coord)
        # vector from boid to wall (points toward the wall)
        vec_to_wall = np.array([wall_coord - pos_i[0], 0.0])
        # project onto heading to see if wall lies ahead
        dist_along_heading = np.dot(vec_to_wall, h)
        # tangential directions along wall: up or down
        tangent_angles = np.array([np.pi / 2, -np.pi / 2])
    else:  # 'horizontal'
        dist_to_wall = abs(pos_i[1] - wall_coord)
        vec_to_wall = np.array([0.0, wall_coord - pos_i[1]])
        dist_along_heading = np.dot(vec_to_wall, h)
        # tangential directions along wall: right or left
        tangent_angles = np.array([0.0, np.pi])

    # only influence if close enough AND wall is in front of boid (line-of-sight)
    if (dist_to_wall < fake_wall) and (dist_along_heading > 0):
        # pick the tangent (two choices) that requires the least rotation from current angle
        diffs = wrap_to_pi(tangent_angles - ang_i)
        # choose index with minimum absolute angle difference
        idx = np.argmin(np.abs(diffs))
        target_angle = tangent_angles[idx]
        # signed minimal rotation from current to target (positive = rotate ccw)
        signed_rotation = wrap_to_pi(target_angle - ang_i)
        sign = np.sign(signed_rotation) if abs(signed_rotation) > 1e-12 else 0.0

        # magnitude (potential-like); larger when closer
        mag = 1.0 / ((dist_to_wall + epsilon) ** scale)
        torque = sign * mag
        return torque
    else:
        return 0.0
    

def wall_force_vector(pos_i, ang_i):
    """
    Returns a continuous steering torque produced by all 4 walls
    based on repulsive potential gradients.

    The force is a vector; steering torque = cross(h, F) = h_x*F_y - h_y*F_x
    """
    h = np.array([np.cos(ang_i), np.sin(ang_i)])  # heading
    F = np.zeros(2)

    # distance to walls
    dist_left = pos_i[0]         # x = 0
    dist_right = D - pos_i[0]    # x = D
    dist_bot = pos_i[1]          # y = 0
    dist_top = D - pos_i[1]      # y = D

    # normal vectors pointing *away* from walls into the domain
    normals = [
        (np.array([1.0, 0.0]), dist_left),       # from left wall
        (np.array([-1.0, 0.0]), dist_right),     # from right wall
        (np.array([0.0, 1.0]), dist_bot),        # from bottom wall
        (np.array([0.0, -1.0]), dist_top)        # from top wall
    ]

    for norm_vec, dist in normals:
        if dist < fake_wall:
            mag = 1.0 / ((dist + epsilon)**scale)
            F += mag * norm_vec   # vector force pointing away from wall

    # convert force vector into steering torque
    # torque = 2D cross product sign → how much force pushes us to turn CCW (+) or CW (–)
    torque = h[0]*F[1] - h[1]*F[0]

    return torque



def Vicsek():

    global pos
    global angle

    Meandirection = np.zeros(N, )
    noise = np.random.uniform(-eta/2, eta/2, size=(N))
    force_wall_left = np.zeros(N)
    force_wall_right = np.zeros(N)
    force_wall_bot = np.zeros(N)
    force_wall_top = np.zeros(N)
    dist_bot = np.zeros(N)
    dist_top = np.zeros(N)
    total_wall_torque = np.zeros(N)

    DistanceMatrix = scipy.spatial.distance.pdist(pos)  #Scipy function to calculate distance between two agents
    DistanceMatrix = scipy.spatial.distance.squareform(DistanceMatrix)   #matrix of form [i, j] => distance between agents i and j
        #returns the distance of each agent to all other agents, the array is of size [N, N]

    for i in range(N):

        Neighbours = DistanceMatrix <= R #Gives array of True/False, if distance less than R, this returns True
        
        Meandirection[i] = (np.arctan2(np.mean(np.sin(angle[Neighbours[:, i]])), np.mean(np.cos(angle[Neighbours[:, i]])))) + noise[i]
                                                        #^^^^^Angles of the agents within R, the True values in 'Neighbours'
        #Equation as in Vicsek 1995 to get the average angle of all the neighbours

        #### Instead of closest distance to wall to find force, 
        # force should be based off of distance to wall in the LOS of the boid ####

        #force_wall_left[i] = wall_force(pos[i, 0], 0, Meandirection[i], np.pi)
        #force_wall_right[i] = wall_force(pos[i, 0], D, Meandirection[i], 0)
        #force_wall_bot[i] = wall_force(pos[i, 1], 0, Meandirection[i], 3*np.pi/2)   #always a forces the boid to the left wall
        #force_wall_top[i] = wall_force(pos[i, 1], D, Meandirection[i], np.pi/2)

        #if Meandirection[i] >= 3*np.pi/2:
        #    force_wall_bot[i] = -force_wall_bot[i]
        #elif Meandirection[i] < 3*np.pi/2:
        #    force_wall_bot[i] = force_wall_bot[i]

        '''
        dist_bot[i] = pos[i, 1] - 0
        if abs(dist_bot[i]) <= fake_wall:
            force_wall_bot[i] = 1/(abs(dist_bot[i])**scale)

            if np.sin(Meandirection[i]) <= 0:
                force_wall_bot[i] = force_wall_bot[i]
            else:
                force_wall_bot[i] = 0#-force_wall_bot[i]
        else:
            force_wall_bot[i] = 0

        dist_top[i] = pos[i, 1] - 0
        if abs(dist_top[i]) <= fake_wall:
            force_wall_top[i] = 1/(abs(dist_top[i])**scale)

            if np.sin(Meandirection[i]) >= 0:
                force_wall_top[i] = -force_wall_top[i]
            else:
                force_wall_top[i] = 0#force_wall_top[i]
        else:
            force_wall_top[i] = 0
        '''

        #force_wall_left[i] = wall_force_circle(pos[i, 0])
        #force_wall_right[i] = wall_force_circle(pos[i, 0])
        #force_wall_bot[i] = wall_force_circle(pos[i, 1])
        #force_wall_top[i] = wall_force_circle(pos[i, 1])

    #force_wall = force_wall_left + force_wall_right + force_wall_bot + force_wall_top
    #force_wall = force_wall_bot + force_wall_top 

            # compute torque contributions from walls (signed)
        #force_wall_left[i] = wall_torque_single(pos[i], Meandirection[i], 'vertical', 0.0)
        #force_wall_right[i] = wall_torque_single(pos[i], Meandirection[i], 'vertical', D)
        #force_wall_bot[i] = wall_torque_single(pos[i], Meandirection[i], 'horizontal', 0.0)
        #force_wall_top[i] = wall_torque_single(pos[i], Meandirection[i], 'horizontal', D)

    # sum torques and scale
    #total_torque = (force_wall_left + force_wall_right + force_wall_bot + force_wall_top) * force_scale
    #Meandirection = Meandirection + total_torque

        wall_torque = wall_force_vector(pos[i], Meandirection[i])
        total_wall_torque[i] = wall_torque
    Meandirection = Meandirection + total_wall_torque * force_scale


    #Meandirection = Meandirection + force_wall*force_scale

    cos = np.cos(Meandirection)
    sin = np.sin(Meandirection)

    vx = cos * v0 * stepsize
    vy = sin * v0 * stepsize

    #Updating the position of the agents
    pos[:, 0] += vx
    pos[:, 1] += vy

    pos[:, 0] = np.clip(pos[:, 0], 0.0, D)
    pos[:, 1] = np.clip(pos[:, 1], 0.0, D)

    angle = wrap_to_pi(Meandirection)

    #angle = Meandirection

    #pos = np.mod(pos, D)

    return pos, cos, sin

#print(Vicsek()[3])



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
#ax.spines['top'].set_linewidth(3)
#ax.spines['bottom'].set_linewidth(3)

#circle = plt.Circle((centre), radius=Radius_circle, fill=False)
#ax.add_artist(circle)


def Animate_quiver(frame):
    pos, cos, sin = Vicsek()
    animated_plot_quiver.set_offsets(pos)
    animated_plot_quiver.set_UVC(cos, sin)
    return (animated_plot_quiver,)
#Animate_quiver

anim = FuncAnimation(fig = fig, func = Animate_quiver, interval = 1, frames = T, blit = False, repeat=False)

#anim.save(f"Hemelrijk, Noise Level = {eta}, N={N}, D={D}, potential scale = {force_scale}.gif", dpi=400)
#plt.savefig("2DVicsekAnimation.png", dpi=400)
plt.show()


