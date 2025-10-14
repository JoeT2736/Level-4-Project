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
        
    #plt.show()
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
    #plt.show()

#PlotTrajectories(trajectory)


#Add noise to angle (trajectories become curved slightly)
def updateRule3(present, stepsize = 1, eta = 0.1, D = 25, v0 = 0.03):
    future = present   #
    N = np.size(present) // 3
    for n in range(N):
        theta = present[3*n]
        v = v0 * np.array([np.cos(theta), np.sin(theta)])
        future[np.array([1, 2]) + 3*n] = present[np.array([1, 2]) + 3*n] + stepsize*v
        future[np.array([1, 2]) + 3*n] = np.mod(future[np.array([1, 2]) + 3*n], D)

    noise = (-eta/2 + np.random.rand(N, ) * eta/2) 
    future[0::3] = present[0::3] + noise

    return(future)

T = 10
trajectory = np.zeros(shape = (N*3, T))
trajectory[:, 0] = state0
for t in range(T-1):
    trajectory[:, t+1] = updateRule3(trajectory[:, t], v0 = 1)

#PlotTrajectories(trajectory)



#Adding neighbour terms
#Cycle through list of agent, compute distance, if close enough then add angle to list of angles to average
def updateRule4(present, stepsize = 1, eta = 0.1, D = 25, R = 1, v0 = 0.03):
    future = present                                  #R = radius of interaction
    N = np.size(present) // 3
    MeanNeighbourAngles = np.zeros(N, )
    for n in range(N):
        theta = present[3*n]
        position = present[np.array([1, 2]) + 3*n]

        angleList = []  #List of angles to average
        
        for m in range(N):  #Loop through every agent
            Pos2 = present[np.array([1, 2]) + 3*m]
            if (np.linalg.norm(position - Pos2) < R):  #Distance between vectors less than R (Distance to be considered a neighbour)
                angleList.append(present[3*m])   #Add to list

        MeanNeighbourAngles[n] = np.sum(np.array(angleList)) / len(angleList)

        v = v0 * np.array([np.cos(theta), np.sin(theta)])
        future[np.array([1, 2]) + 3*n] = present[np.array([1, 2]) + 3*n] + stepsize*v
        future[np.array([1, 2]) + 3*n] = np.mod(future[np.array([1, 2]) + 3*n], D)

    noise = (-eta/2 + np.random.rand(N, ) * eta/2) 
    future[0::3] = present[0::3] + noise

    return(future)

N = 30
state0 = initialiseCells(N)
T = 100
trajectory = np.zeros(shape = (N*3, T))
trajectory[:, 0] = state0

for t in range(T - 1):
    trajectory[:, t + 1] = updateRule4(trajectory[:, t], stepsize = 5, eta = 0.1, R = 1)

'''
PlotTrajectories(trajectory)
PlotAgents(trajectory[:, 0]) #plot first step
PlotAgents(trajectory[:, -1]) #plot final step
plt.show()
'''


#Shortening of neighbour calculations by using scipy

import scipy.spatial
def updateRule5(present, stepsize = 1, eta = 0.1, D = 25, R = 1, v0 = 0.03):
    future = present
    N = np.size(present) // 3
    MeanNeighbourAngles = np.zeros(N, )

    #Matrix, each row = one agent (columns = x, y coordinates)
    positions = np.zeros((N, 2))
    positions[:, 0] = present[1::3]
    positions[:, 1] = present[2::3]
    angles = present[0::3]

    #Identify neighbours
    DistanceMatrix = scipy.spatial.distance.pdist(positions)
    DistanceMatrix = scipy.spatial.distance.squareform(DistanceMatrix)   #matrix of form [i, j] => distance between agents i and j
    Neighbours = DistanceMatrix <= R   #If distance between i and j less than R, update neighbour list

    for n in range(N):
        theta = angles[n]   #angle of agent
        pos2 = positions[n, :]   #agent position

        selection = DistanceMatrix[:, n] < R   #selection vector - true when distance smaller than R
        MeanNeighbourAngles[n] = np.sum(angles[Neighbours[:, n]]) / np.sum(Neighbours[:, n])   #Mean neighbour angle

        v = v0 * np.array([np.cos(theta), np.sin(theta)])
        future[np.array([1, 2]) + 3*n] = present[np.array([1, 2]) + 3*n] + stepsize * v
        future[np.array([1, 2]) + 3*n] = np.mod(future[np.array([1, 2]) + 3*n], D)

    noise = (-eta/2 + np.random.rand(N, ) * eta/2)
    future[0::3] = np.mod(present[0::3] + MeanNeighbourAngles + noise, 2*np.pi)

    return(future)


def test(N = 20, T = 1000, R = 5, D = 20, eta = 0.1, stepsize = 1):
    state0 = initialiseCells(N, D=D)
    trajectory = np.zeros(shape = (N*3, T))
    trajectory[:, 0] = state0

    for t in range(T-1):
        trajectory[:, t+1] = updateRule5(trajectory[:, t], stepsize=stepsize, eta=eta, R=R, D=D)

        plt.figure(0)
        PlotTrajectories(trajectory, D=D)

        #plt.figure(1)
        #PlotAgents(trajectory[:, 0], D=D)

        #plt.figure(2)
        #PlotAgents(trajectory[:, -1], D=D)

        
    
    #plt.show()

test()
plt.show()
        







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