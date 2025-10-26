import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import scipy.spatial


####### Looks like only some agents are being updated at each step, where the others are moving unfazed
####### Some might be moving faster than others



#pos=np.random.uniform(0, 5, size=(10, 2))
#print(pos[:, 1])
#angle = np.random.uniform(-np.pi, np.pi, size=10)
#print(0.03 * np.array([np.cos(angle), np.sin(angle)]).reshape(10, 2))
#print((np.random.uniform(0, 5, size=(10, 2)) * (0.03 * np.array([np.cos(angle), np.sin(angle)]).reshape(10, 2)))[:, 0])

#print(scipy.spatial.distance.pdist(pos))
#print(scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(pos)))
#n = (scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(pos))<=1)

#print(angle[n[:, 2]])

#fig = plt.figure(figsize=(8, 8))

N=300  #Number of agents
D=5  #Size of domain
T=600   #Total number of time steps (frames) in simulation
stepsize=1  #change in time between calculation of position and angle
eta=0.1   #Random noise added to angles
v0=0.00003    #Starting velocity
R=1    #Interaction radius

pos = np.random.uniform(0, D, size=(N, 2))
angle = np.random.uniform(-np.pi, np.pi, size=N)

#print(pos)
#print(pos.reshape(2, N))

def Vicsek():

    global pos
    global angle

    MeanAngle = np.zeros(N, )

    noise = np.random.uniform(-eta/2, eta/2, size=(N))

    for i in range(N):

        DistanceMatrix = scipy.spatial.distance.pdist(pos)  #Scipy function to calculate distance between two agents
        DistanceMatrix = scipy.spatial.distance.squareform(DistanceMatrix)   #matrix of form [i, j] => distance between agents i and j
        Neighbours = DistanceMatrix <= R #Gives array of True/False

        MeanAngle[i] = (np.arctan2(np.sum(np.sin(angle[Neighbours[:, i]])), np.sum(np.cos(angle[Neighbours[:, i]])))) #+ noise[Neighbours[:, i]]

        cos = (np.cos(MeanAngle + noise))
        sin = (np.sin(MeanAngle + noise))

        g = np.array([cos, sin])

        k = np.transpose(g)

        vel = v0 * k

        pos = pos + vel * stepsize

    pos = np.mod(pos, D)    #Agent appears on other side of domain when goes off one end

    #angle = MeanAngle + noise

    #noise = np.random.uniform(-eta/2, eta/2, size=N)
    #angle = np.mod(angle + MeanAngle + noise, 2*np.pi)
    return pos, k#, MeanAngle

#print(Vicsek()[1][:, 0])

#animated_plot = plt.quiver([], [], [], [], [])

#plt.xlim(0, D)
#plt.ylim(0, D)

fig, ax = plt.subplots()
ax.set_xlim([0, D])
ax.set_ylim([0, D])
animated_plot, = ax.plot([], [], '.')
#animated_plot_quiver, = ax.quiver([], [], [], [])


plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 12})
plt.title(f'Noise level = {eta}', fontsize=16)
plt.xlabel('X-axis', fontsize=14)
plt.ylabel('Y-axis', fontsize=14)
plt.setp(ax.get_xticklabels(), visible='False')
plt.setp(ax.get_yticklabels(), visible='False')
ax.set_aspect('equal', adjustable='box')


#def Animate_quiver(frame):
#    animated_plot_quiver.set_data()
#    return


def Animation(frame):
    animated_plot.set_data([Vicsek()[0][:, 0], Vicsek()[0][:, 1]])
    return 

anim = FuncAnimation(fig = fig, func = Animation, interval = 1,frames = T, blit = False)

anim.save("2DVicsekAnimation.gif", dpi=400)
plt.savefig("2DVicsekAnimation.png", dpi=400)
plt.show()


        







