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

N=200  #Number of agents
D=10  #Size of domain
T=600   #Total number of time steps (frames) in simulation
stepsize=1  #change in time between calculation of position and angle
eta=0.1   #Random noise added to angles
v0=0.001   #Starting velocity
R=1    #Interaction radius

pos = np.random.uniform(0, D, size=(N, 2))
angle = np.random.uniform(0, 2*np.pi, size=N)


def Vicsek():

    global pos
    global angle

    MeanAngle = np.zeros(N, )

    for i in range(N):

        noise = np.random.uniform(-eta/2, eta/2, size=(N))

        DistanceMatrix = scipy.spatial.distance.pdist(pos)  #Scipy function to calculate distance between two agents
        DistanceMatrix = scipy.spatial.distance.squareform(DistanceMatrix)   #matrix of form [i, j] => distance between agents i and j
        #returns the distance of each agent to all other agents, the array is of size [N, N]

        Neighbours = DistanceMatrix <= R #Gives array of True/False, if distance less than R, this returns True

        MeanAngle[i] = (np.arctan2(np.sum(np.sin(angle[Neighbours[:, i]])), np.sum(np.cos(angle[Neighbours[:, i]]))))
                                                        #^^^^^Angles of the agents within R, the True values in 'Neighbours'
        #Equation as in Vicsek 1995 to get the average angle of all the neighbours

        MeanAngle = MeanAngle + noise   #Adding random noise element to the angle
        MeanAngle = np.mod(MeanAngle, 2*np.pi)

        #x and y directions accoring to new angle
        cos = (np.cos(MeanAngle))   
        sin = (np.sin(MeanAngle))

        #Updating the position of the agents
        pos[:, 0] += cos * v0 * stepsize
        pos[:, 1] += sin * v0 * stepsize

    pos = np.mod(pos, D)    #Agent appears on other side of domain when goes off one end

    return pos, cos, sin, MeanAngle

fig, ax = plt.subplots()
ax.set_xlim([0, D])
ax.set_ylim([0, D])
animated_plot_quiver = ax.quiver(pos[:, 0], pos[:, 1], np.cos(angle), np.sin(angle), angle)


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
    animated_plot_quiver.set_UVC(Vicsek()[1], Vicsek()[2], Vicsek()[3])
    return (animated_plot_quiver, )

anim = FuncAnimation(fig = fig, func = Animate_quiver, interval = 1, frames = T, blit = False, repeat=False)

#anim.save("2DVicsekAnimation.gif", dpi=400)
#plt.savefig("2DVicsekAnimation.png", dpi=400)
plt.show()








