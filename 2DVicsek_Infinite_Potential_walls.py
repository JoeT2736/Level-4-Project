import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import scipy.spatial
import scipy.constants


N=30  #Number of agents
D=5  #Size of domain
T=600   #Total number of time steps (frames) in simulation
stepsize=1  #change in time between calculation of position and angle
eta=0.15   #Random noise added to angles
v0=0.03   #Starting velocity
R=1    #Interaction radius
scale = 2
force_scale = 0.005

pos = np.random.uniform(0+0.5, D-0.5, size=(N, 2))
angle = np.random.uniform(0, 2*np.pi, size=N)


def Vicsek():
    global pos
    global angle

    MeanAngle = np.zeros(N, )
    noise = np.random.uniform(-eta/2, eta/2, size=(N))
    #cos = np.zeros(N)
    #sin = np.zeros(N)
    #vx = np.zeros(N)
    #vy = np.zeros(N)
    force_wall = np.zeros(N)
    force_wall_left = np.zeros(N)
    force_wall_right = np.zeros(N)
    force_wall_bot = np.zeros(N)
    force_wall_top = np.zeros(N)
    dx_left = np.zeros(N)
    dx_right = np.zeros(N)
    dy_bot = np.zeros(N)
    dy_top = np.zeros(N)

    DistanceMatrix = scipy.spatial.distance.pdist(pos)  #Scipy function to calculate distance between two agents
    DistanceMatrix = scipy.spatial.distance.squareform(DistanceMatrix)   #matrix of form [i, j] => distance between agents i and j
        #returns the distance of each agent to all other agents, the array is of size [N, N]

    for i in range(N):

        Neighbours = DistanceMatrix <= R #Gives array of True/False, if distance less than R, this returns True
        
        MeanAngle[i] = (np.arctan2(np.mean(np.sin(angle[Neighbours[:, i]])), np.mean(np.cos(angle[Neighbours[:, i]])))) + noise[i]
                                                        #^^^^^Angles of the agents within R, the True values in 'Neighbours'
        #Equation as in Vicsek 1995 to get the average angle of all the neighbours


        dx_left[i] = pos[i, 0]
        dx_right[i] = D - pos[i, 0]
        dy_bot[i] = pos[i, 1]
        dy_top[i] = D - pos[i, 1]

        force_wall_left[i] = 1/(dx_left[i])**scale
        force_wall_right[i] = 1/(dx_right[i])**scale
        force_wall_bot[i] = 1/(dy_bot[i])**scale    
        force_wall_top[i] = 1/(dy_top[i])**scale

        if MeanAngle[i] >= 0:
            force_wall_left[i] = -force_wall_left[i]
        elif MeanAngle[i] < 0:
            force_wall_left[i] = force_wall_left[i]

        if MeanAngle[i] >= 0:
            force_wall_right[i] = force_wall_right[i]
        elif MeanAngle[i] < 0:
            force_wall_right[i] = -force_wall_right[i]

        if MeanAngle[i] >= 3*np.pi/2:
            force_wall_bot[i] = force_wall_bot[i]
        elif MeanAngle[i] < 3*np.pi/2:
            force_wall_bot[i] = -force_wall_bot[i]

        if MeanAngle[i] >= np.pi/2:
            force_wall_top[i] = force_wall_top[i]
        elif MeanAngle[i] < np.pi/2:
            force_wall_top[i] = -force_wall_top[i]

        force_wall[i] = force_wall_left[i] + force_wall_right[i] + force_wall_bot[i] + force_wall_top[i]

    MeanAngle = MeanAngle + force_wall*force_scale

    cos = np.cos(MeanAngle)
    sin = np.sin(MeanAngle)

    vx = cos * v0 * stepsize
    vy = sin * v0 * stepsize

    #Updating the position of the agents
    pos[:, 0] += vx
    pos[:, 1] += vy

    angle = MeanAngle

    return pos, cos, sin, MeanAngle, vx



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
    pos, cos, sin, MeanAngle, vx = Vicsek()
    animated_plot_quiver.set_offsets(pos)
    animated_plot_quiver.set_UVC(cos, sin)
    return (animated_plot_quiver,)
#Animate_quiver

anim = FuncAnimation(fig = fig, func = Animate_quiver, interval = 1, frames = T, blit = False, repeat=False)

anim.save(f"Potential Wall, Noise Level = {eta}, N={N}, D={D}, potential scale = {force_scale}.gif", dpi=400)
#plt.savefig("2DVicsekAnimation.png", dpi=400)
plt.show()








        ### Potential Walls ###
'''
        scale = 2
        force_wall[i] = 1/(dx_left[i])**scale + 1/(dx_right[i])**scale + 1/(dy_bot[i])**scale + 1/(dy_top[i])**scale
        #forces should be seperated to x and y directions ???

        if dx_left[i] < 1:
            fwx_left[i] = 1/(dx_left[i])**scale
        if dx_right[i] < 1:
            fwx_right[i] = 1/(dx_right[i]**scale)

        if dy_top[i] < 1:
            fwy_top[i] = 1/(dy_top[i])**scale
        if dy_bot[i] < 1:
            fwy_bot[i] = 1/(dy_bot[i]**scale)

        force_wall_x[i] = fwx_left[i] + fwx_right[i]
        force_wall_y[i] = fwy_bot[i] + fwy_top[i]
        
    #Seperate x and y forces converted to angle
    wall_angle = np.arctan2(np.sin(force_wall_y), np.cos(force_wall_x))
    #Angle goes to pi then stops, and stays at pi     
    
    #MeanAngle += force_wall    #agents walk around close to edge of domain
    '''