import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import scipy.spatial
import scipy.constants


N=10  #Number of agents
D=5  #Size of domain
T=6000   #Total number of time steps (frames) in simulation
stepsize=1  #change in time between calculation of position and angle
eta=0.15   #Random noise added to angles
v0=0.03   #Starting velocity
R=1    #Interaction radius
scale = 2
force_scale = 0.005
w = np.pi/2   #FOV
fake_wall = 1

pos = np.random.uniform(0, D, size=(N, 2))
angle = np.random.uniform(0, 2*np.pi, size=N)

def wall_force(position, wall, direction, choice):
    dist = wall - position
    '''
    if direction <= np.pi/2 and direction >= 0:
        dist = (wall - position) * np.cos(direction)
    
    if direction > np.pi/2 and direction <= np.pi:
        dist = (wall - position) * np.sin(direction)

    if direction > np.pi and direction <= 3*np.pi/2:
        dist = (wall - position) * np.cos(direction - np.pi)
    
    else:
        dist = (wall - position) * np.sin(direction - np.pi)
        '''
    if abs(dist) < fake_wall:      #if close enough then...
        force_wall = 1/(abs(dist))**scale      #potential due to wall
        if direction >= choice:      #choice = angle where boid turns either up/down or left/right based on its direction
            force_wall = force_wall
        elif direction < choice:
            force_wall = -force_wall
    else:
        force_wall = 0

    return force_wall

R_circ = D/2
centre = np.array([D/2, D/2])
def wall_force_circle(position, centre, R_circ, scale):
    vec = pos - centre
    dist = np.linalg.norm(vec)
    if dist >= R_circ:
        return np.zeros(2)

    # outward normal
    n = vec / dist

    strength = 1 / ((R_circ - dist)**scale)
    return strength * n


def Vicsek():
    global pos
    global angle

    Meandirection = np.zeros(N, )
    noise = np.random.uniform(-eta/2, eta/2, size=(N))
    force_wall_left = np.zeros(N)
    force_wall_right = np.zeros(N)
    force_wall_bot = np.zeros(N)
    force_wall_top = np.zeros(N)

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


        force_wall_left[i] = wall_force(pos[i, 0], 0, Meandirection[i], np.pi)
        #force_wall_left[i] = wall_force_circle(pos[i, 0], centre, R_circ, scale, )
        force_wall_right[i] = wall_force(pos[i, 0], D, Meandirection[i], 0)
        #force_wall_left[i] = wall_force_circle(pos[i, 0], centre, R_circ, scale)
        force_wall_bot[i] = wall_force(pos[i, 1], 0, Meandirection[i], 3*np.pi/2)
        #force_wall_left[i] = wall_force_circle(pos[i, 0], centre, R_circ, scale)
        force_wall_top[i] = wall_force(pos[i, 1], D, Meandirection[i], np.pi/2)
        #force_wall_left[i] = wall_force_circle(pos[i, 0], centre, R_circ, scale)

    force_wall = force_wall_left + force_wall_right + force_wall_bot + force_wall_top

    Meandirection = Meandirection + force_wall*force_scale

    cos = np.cos(Meandirection)
    sin = np.sin(Meandirection)

    vx = cos * v0 * stepsize
    vy = sin * v0 * stepsize

    #Updating the position of the agents
    pos[:, 0] += vx
    pos[:, 1] += vy

    angle = Meandirection

    return pos, cos, sin, vector, angle_difference

print(Vicsek()[3])
print(Vicsek()[4])


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
    pos, cos, sin, dots, t = Vicsek()
    animated_plot_quiver.set_offsets(pos)
    animated_plot_quiver.set_UVC(cos, sin)
    return (animated_plot_quiver,)
#Animate_quiver

anim = FuncAnimation(fig = fig, func = Animate_quiver, interval = 1, frames = T, blit = False, repeat=False)

#anim.save(f"Hemelrijk, Noise Level = {eta}, N={N}, D={D}, potential scale = {force_scale}.gif", dpi=400)
#plt.savefig("2DVicsekAnimation.png", dpi=400)
plt.show()




