import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import scipy.spatial

#print(np.random.uniform(-np.pi, np.pi, size=(20, 1)))
#print(np.random.rand(20, 1) * 5)

####### Only spawns the agents in facing top right hand corner or bottom left hand corner, no inbetween #######


def Vicsek():

    N=300   #Number of agents
    D=5  #Size of domain
    T=6000   #Total number of time steps (frames) in simulation
    stepsize=1  #  
    eta=0.1   #Random noise added to angles
    v0=0.03    #Starting velocity
    R=1    #Interaction radius
    plotRealTime = True


    #np.random.seed(4)

    #Set agent positions at start to random locations within the domain
    x = np.random.rand(N, 1) * D
    y = np.random.rand(N, 1) * D

    #position = np.random(N, 2) * D

    #Staring velocities
    angle = np.random.uniform(-np.pi, np.pi, size=(N, 1))
    vx = v0 * np.cos(angle)
    vy = v0 * np.sin(angle)

    fig = plt.figure(figsize=(8, 8))
    ax = plt.gca()

    for i in range(T):
        #Move agents
        x += vx * stepsize
        y += vy * stepsize

        #If agents goes outside of the domain, it will reappear on the other side
        x[x>D] -= D
        x[x<0] += D

        y[y>D] -= D
        y[y<0] += D

        MeanNeighbourAngle = angle
        
        for j in range(N):
            neighbours = (x - x[j])**2 + (y - y[j])**2 < R**2
            sx = np.sum(np.cos(angle[neighbours]))
            sy = np.sum(np.cos(angle[neighbours]))
            MeanNeighbourAngle[j] = np.arctan2(sy, sx)

        angle = MeanNeighbourAngle + (eta/2) * (np.random.rand(N, 1))

        vx = v0 * np.cos(angle)
        vy = v0 * np.sin(angle)

        if plotRealTime or (i == T - 1):
            plt.cla()
            plt.quiver(x, y, vx, vy, angle)
            ax.set(xlim=(0, D), ylim=(0, D))
            ax.set_aspect("equal")
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            plt.pause(0.001)
    
    plt.savefig("CollectiveMotion.png", dpi=240)
    plt.show()

    return 0



if __name__ == "__main__":
    Vicsek()



