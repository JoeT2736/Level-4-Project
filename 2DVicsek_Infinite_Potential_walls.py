import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import scipy.spatial
import scipy.constants


N=20  #Number of agents
D=5  #Size of domain
T=600   #Total number of time steps (frames) in simulation
stepsize=1  #change in time between calculation of position and angle
eta=0.15   #Random noise added to angles
v0=0.01   #Starting velocity
R=1    #Interaction radius
Q = 1   #Potential of agents
QW = 1  #Potential of walls
L = 100     #Number of points in wall
Repel_distance = 0.2

pos = np.random.uniform(0+0.5, D-0.5, size=(N, 2))
#angle = np.random.uniform(0, 2*np.pi, size=N)
angle = np.random.uniform(0, 2*np.pi, size=N)

'''
D1 = np.full(shape=(10), fill_value=D)
Dist = np.linspace(0, D, 10)
#Dist = Dist.reshape(10, 1)
#print(np.vstack((Dist, D1)).T)

wall_l = np.vstack((Dist, D1)).T
qw = scipy.spatial.distance.cdist(pos, wall_l)
rw = scipy.spatial.distance.pdist(pos)
print(rw)
print(qw)

#qw = scipy.spatial.distance.squareform(qw)
rw = scipy.spatial.distance.squareform(rw)
#print(rw)
'''
#print(pos)
#print(wall_l)

#print(scipy.spatial.distance.cdist(pos, wall_l))
#print(qw)

def Wall_force(Distance_Wall):
    return (Q * QW)/(4 * np.pi * Distance_Wall * scipy.constants.epsilon_0)



def Vicsek():
    #print(i)
    global pos
    global angle

    MeanAngle = np.zeros(N, )
    noise = np.random.uniform(-eta/2, eta/2, size=(N))
    force_l = np.zeros(N, )
    force_r = np.zeros(N, )
    force_b = np.zeros(N, )
    t = np.zeros(N, )
    dist_wall_xd = np.zeros(N)
    dist_wall_x0 = np.zeros(N)
    dist_wall_yd = np.zeros(N)
    dist_wall_y0 = np.zeros(N)
    force_wall_x = np.zeros(N)
    force_wall_y = np.zeros(N)
    force_wall = np.zeros(N)
    repel_force = np.zeros(N)
    repel_angle=np.full(shape=(N,), fill_value=np.pi/12)
    r=np.zeros(N,)

    DistanceMatrix = scipy.spatial.distance.pdist(pos)  #Scipy function to calculate distance between two agents
    DistanceMatrix = scipy.spatial.distance.squareform(DistanceMatrix)   #matrix of form [i, j] => distance between agents i and j
        #returns the distance of each agent to all other agents, the array is of size [N, N]

    for i in range(N):

        Neighbours = DistanceMatrix <= R #Gives array of True/False, if distance less than R, this returns True
        #repel = DistanceMatrix = np.logical_and(DistanceMatrix <= R/2, DistanceMatrix > 0)

        #r[i] = np.mean(repel_angle[repel[:, i]])

        MeanAngle[i] = (np.arctan2(np.mean(np.sin(angle[Neighbours[:, i]])), np.mean(np.cos(angle[Neighbours[:, i]])))) #+ r[i]
                                                        #^^^^^Angles of the agents within R, the True values in 'Neighbours'
        #Equation as in Vicsek 1995 to get the average angle of all the neighbours

        #MeanAngl = DistanceMatrix[Repel[:, i]]

        #repel_force = (1/DistanceMatrix[Repel[:, i]])

        #Infinite potential walls => agents stay within domain
        dist_wall_xd[i] = D - pos[:, 0][i]
        dist_wall_x0[i] = pos[:, 0][i]
        dist_wall_yd[i] = D - pos[:, 1][i]
        dist_wall_y0[i] = pos[:, 1][i]
        
        scale = 2
        force_wall[i] = 1/abs(dist_wall_xd[i])**scale + 1/abs(dist_wall_x0[i])**scale + 1/abs(dist_wall_yd[i])**scale + 1/abs(dist_wall_y0[i])**scale
        #forces should be seperated to x and y directions ???

#add a random element to the force

        #if MeanAngle[i] > 0:
        #    MeanAngle[i] += force[i]
        #if MeanAngle[i] < 0:
        #    MeanAngle[i] -= force[i]

        #MeanAngle[i] += force_wall[i]

    ### Makes agents 'stick' to outside of domain ###

    MeanAngle += force_wall

    MeanAngle = MeanAngle + noise   #Adding random noise element to the angle
    MeanAngle = np.mod(MeanAngle, 2*np.pi)

        #x and y directions accoring to new angle
    cos = (np.cos(MeanAngle))   
    sin = (np.sin(MeanAngle))

        #Updating the position of the agents
    pos[:, 0] += cos * v0 * stepsize #+ 0.5 * (force) * stepsize**2  
    pos[:, 1] += sin * v0 * stepsize #+ 0.5 * (force) * stepsize**2 

    #pos = np.mod(pos, D)    #Agent appears on other side of domain when goes off one end

    #frame = []
    #frame += 1
    #print(frame)

    return pos, cos, sin, MeanAngle, force_wall, Neighbours
print(Vicsek()[5])
#print(Vicsek()[5])


fig, ax = plt.subplots()
ax.set_xlim([0, D])
ax.set_ylim([0, D])
animated_plot_quiver = ax.quiver(pos[:, 0], pos[:, 1], np.cos(angle), np.sin(angle), clim=[-np.pi, np.pi])#, angle)


plt.rcParams["font.family"] = "Times New Roman"
plt.title(f'Noise level = {eta}', fontsize=16)
plt.xlabel('X-axis', fontsize=14)
plt.ylabel('Y-axis', fontsize=14)
plt.setp(ax.get_xticklabels(), visible='False')
plt.setp(ax.get_yticklabels(), visible='False')
plt.locator_params(axis='y', nbins=1)
plt.locator_params(axis='x', nbins=1)
ax.set_aspect('equal', adjustable='box')


def Animate_quiver(frame):
    animated_plot_quiver.set_offsets(Vicsek()[0])
    animated_plot_quiver.set_UVC(Vicsek()[1], Vicsek()[2])#, Vicsek()[3])
    return (animated_plot_quiver, )

anim = FuncAnimation(fig = fig, func = Animate_quiver, interval = 1, frames = T, blit = False, repeat=False)

#anim.save(f"Infinite Potential Wall, Noise Level = {eta}, D={D}.gif", dpi=400)
#plt.savefig("2DVicsekAnimation.png", dpi=400)
plt.show()








