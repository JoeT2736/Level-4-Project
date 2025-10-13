##### Vicsek Model, constant speed, new direction = average of neighbours #####



import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


N = 5   #Number of Agents
D = 25   #Size of domain

state0 = np.zeros(shape=(N*3, ))
state0[0::3] = np.random.rand(N, ) * 2 * np.pi   #Selects every third element of state0 (angles) making them random
state0[1::3] = np.random.rand(N, ) * D   #Random Horizontal coordinates
state0[2::3] = np.random.rand(N, ) * D   #Random Vertical coordinates

'''
ax = plt.axes()
ax.set_xlim(0, D)
ax.set_ylim(0, D)

for n in range(N):
    theta, x, y = state0[np.array([0, 1, 2]) + 3*n]   #Coordinates of n_th agent
    ax.arrow(x, y, 0.5*np.cos(theta), 0.5*np.sin(theta), \
             head_width = 0.5, head_length = 0.1, fc = 'k', ec = 'k')   #Arrow heads
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
        
    plt.show()
    return ax

#ax = PlotAgents(state0)

def initialiseCells(N, D=25):
    state0 = np.zeros( shape=(N*3,))
    state0[0::3] = np.random.rand(N,) * 2 * np.pi
    state0[1::3] = np.random.rand(N,) * D
    state0[2::3] = np.random.rand(N,) * D
    return state0

N = 5
state0 = initialiseCells(N)
T = 10   #Number of Time Steps

#PlotAgents(state0)


trajectory = np.zeros(shape=(N*3, T))
trajectory[:, 0] = state0
np.set_printoptions(precision=1)   #Prints to 2 decimal places

#print(Trajectory)


def updateRule1(present, stepsize = 1):
    future = present   #
    N = np.size(present) // 3
    for n in range(N):
        theta = present[3*n]
        v = np.array([np.cos(theta), np.sin(theta)])
        future[np.array([1, 2]) + 3*n] = present[np.array([1, 2]) + 3*n] + stepsize*v
    return(future)

'''
for t in range(T-1):
    trajectory[:, t+1] = updateRule1(trajectory[:, t], 3)   #Repeatedly applies update rule to columns of the trajecotry matrix#

#for t in range(3):
#    PlotAgents(trajectory[:, t])

def PlotTrajectories(trajectory, D = 25):
    ax = plt.axes()
    ax.set_xlim(0, D)
    ax.set_ylim(0, D)
    plt.plot(trajectory[1::3, :].T, trajectory[2::3, :].T, '.')
    plt.show()

PlotTrajectories(trajectory)
'''


def updateRule2(present, stepsize = 1):
    future = present   #
    N = np.size(present) // 3
    for n in range(N):
        theta = present[3*n]
        v = np.array([np.cos(theta), np.sin(theta)])
        future[np.array([1, 2]) + 3*n] = present[np.array([1, 2]) + 3*n] + stepsize*v
        future[np.array([1, 2]) + 3*n] = np.mod(future[np.array([1, 2]) + 3*n], D)   #Added line so that if agent goes offscreen, it reappears on the other side
    return(future)

for t in range(T-1):
    trajectory[:, t+1] = updateRule2(trajectory[:, t], 3)   #Repeatedly applies update rule to columns of the trajecotry matrix#

def PlotTrajectories(trajectory, D = 25):
    ax = plt.axes()
    ax.set_xlim(0, D)
    ax.set_ylim(0, D)
    plt.plot(trajectory[1::3, :].T, trajectory[2::3, :].T, '.')
    plt.show()

PlotTrajectories(trajectory)











'''

agent_position = np.
agent_direction = np.
v = np. #Distance travelled by each agent during time taken to update
range = #Range of interaction
nearest_neighbours = np. #Number of neighbours each agent interacts with
FOV = 
plane = #Size of plane agents move on
noise = 
vector_motion = #velocity of each agent
time_step = 


def movement(agent_position, agent_direction, v, plane):
    move = (np.cos(agent_direction), np.sin(agent_direction)) * v
    agent_position += move
    agent_position[(agent_position < 0)] += plane
    agent_position %= plane

    return agent_position, move


def New_Direction(pos, dir, eta, neighbour)
'''