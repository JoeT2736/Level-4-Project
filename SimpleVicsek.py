import numpy as np
import scipy
import scipy.spatial
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots(figsize=(10, 10))


#Setup matrix of all agents in simulation
#N = number of agents
#D = Size of the domain

def InitialiseAgents(N, D=25):
    state0 = np.zeros( shape=(N*3,))
    state0[0::3] = np.random.rand(N,) * 2 * np.pi   #Every third element => agent angle
    state0[1::3] = np.random.rand(N,) * D   #x-Direction
    state0[2::3] = np.random.rand(N,) * D   #y-Direction
    return state0


#Create plot of all agents showing there position and direction of motion using an arrow
'''
def PlotAgents(statevector, D=25):
    ax = plt.axes()
    ax.set_xlim(0, D)
    ax.set_ylim(0, D)
    arcScale = D/250        # // = integer division
    N = np.size(statevector) // 3 #Number of cells from size of statevector

    for n in range(N):
        theta, x, y = statevector[np.array([0, 1, 2]) + 3*n]   #Coordinates of n_th agent
        ax.arrow(x, y, 5*arcScale*np.cos(theta), 5*arcScale*np.sin(theta), \
                                                            head_width = 2*arcScale, head_length = arcScale, fc='k', ec='k')   #Arrow head
        
    #plt.show()
    return ax
'''

from Vicsek_Try1 import updateRule5
from Vicsek_Try1 import test5


anim = animation.FuncAnimation(fig = fig, func = test5(N = 20, T = 1000, R = 3, D = 25, eta = 0.1, stepsize = 1), interval = 30)

anim.save("")

plt.show()

