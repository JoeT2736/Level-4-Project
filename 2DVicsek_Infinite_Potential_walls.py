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


pos = np.random.uniform(0+0.1, D-0.1, size=(N, 2))
angle = np.random.uniform(0, 2*np.pi, size=N)

def wall_force(position, wall, direction, choice):
    if D == D:
        dist = wall - position
    elif D == 0:
        dist = position

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

        Neighbours = DistanceMatrix <= R #Gives array of True/False, if distance less than R, this returns True
        
        Meandirection[i] = (np.arctan2(np.mean(np.sin(angle[Neighbours[:, i]])), np.mean(np.cos(angle[Neighbours[:, i]])))) + noise[i]
                                                        #^^^^^Angles of the agents within R, the True values in 'Neighbours'
        #Equation as in Vicsek 1995 to get the average angle of all the neighbours

        #### Instead of closest distance to wall to find force, 
        # force should be based off of distance to wall in the LOS of the boid ####

        force_wall_left[i] = wall_force(pos[i, 0], 0, Meandirection[i], np.pi)
        force_wall_right[i] = wall_force(pos[i, 0], D, Meandirection[i], 0)
        force_wall_bot[i] = wall_force(pos[i, 1], 0, Meandirection[i], 3*np.pi/2)
        force_wall_top[i] = wall_force(pos[i, 1], D, Meandirection[i], np.pi/2)

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

    return pos, cos, sin, force_wall_left

print(Vicsek()[3])



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
    pos, cos, sin, k = Vicsek()
    animated_plot_quiver.set_offsets(pos)
    animated_plot_quiver.set_UVC(cos, sin)
    return (animated_plot_quiver,)
#Animate_quiver

anim = FuncAnimation(fig = fig, func = Animate_quiver, interval = 1, frames = T, blit = False, repeat=False)

#anim.save(f"Hemelrijk, Noise Level = {eta}, N={N}, D={D}, potential scale = {force_scale}.gif", dpi=400)
#plt.savefig("2DVicsekAnimation.png", dpi=400)
plt.show()


