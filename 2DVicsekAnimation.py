import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import scipy.spatial


pos=np.random.uniform(0, 5, size=(10, 2))
#print(pos[:, 1])
angle = np.random.uniform(-np.pi, np.pi, size=10)
#print(0.03 * np.array([np.cos(angle), np.sin(angle)]).reshape(10, 2))
print((np.random.uniform(0, 5, size=(10, 2)) * (0.03 * np.array([np.cos(angle), np.sin(angle)]).reshape(10, 2)))[:, 0])

#print(scipy.spatial.distance.pdist(pos))
#print(scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(pos)))
n = (scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(pos))<=1)

#print(angle[n[:, 2]])

fig = plt.figure(figsize=(8, 8))

T=6000   #Total number of time steps (frames) in simulation
D=5  #Size of domain


def Vicsek():

    N=300   #Number of agents
    D=5  #Size of domain
    T=6000   #Total number of time steps (frames) in simulation
    stepsize=1  #change in time between calculation of position and angle
    eta=0.1   #Random noise added to angles
    v0=0.03    #Starting velocity
    R=1    #Interaction radius


    pos = np.random.uniform(0, D, size=(N, 2))
    angle = np.random.uniform(-np.pi, np.pi, size=N)

    #noise = np.random.uniform(-eta/2, eta/2, size=N)

    #quiver = plt.quiver(pos[:, 0], pos[:, 1], angle[:, 0], angle[:, 1])

    vel = v0 * np.array([np.cos(angle), np.sin(angle)]).reshape(N, 2)

    #DistanceMatrix = scipy.spatial.distance.pdist(pos)  #Scipy function to calculate distance between two agents
    #DistanceMatrix = scipy.spatial.distance.squareform(DistanceMatrix)   #matrix of form [i, j] => distance between agents i and j
    #Neighbours = DistanceMatrix <= R 

    for i in range(N):

        #pos += pos + vel * stepsize

        DistanceMatrix = scipy.spatial.distance.pdist(pos)  #Scipy function to calculate distance between two agents
        DistanceMatrix = scipy.spatial.distance.squareform(DistanceMatrix)   #matrix of form [i, j] => distance between agents i and j
        Neighbours = DistanceMatrix <= R #Gives array of True/False

        MeanAngle = np.arctan2(np.cos(angle[Neighbours[:, i]]), np.sin(angle[Neighbours[:, i]])) + np.random.uniform(-eta/2, eta/2, size=N)

        vel = v0 * np.array([np.cos(MeanAngle), np.sin(MeanAngle)]).reshape(N, 2)

        pos += pos + vel * stepsize

        pos = np.mod(pos, D)

    return pos#, vel, MeanAngle


#animated_plot = plt.quiver([], [], [], [], [])

#plt.xlim(0, D)
#plt.ylim(0, D)

fig, ax = plt.subplots()
ax.set_xlim([0, D])
ax.set_ylim([0, D])
animated_plot, = ax.plot([], [], 'o')



def Animation(frame):
    animated_plot.set_data(Vicsek())
    return

anim = FuncAnimation(fig = fig, func = Animation, interval = 1,frames = T)

anim.save("Test.gif")

plt.show()


        







