import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import scipy.spatial
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation

N=50
D=25
T=6000
stepsize=1
eta=0.1
v0=0.5
R=1



def PlotAgents(statevector, D):
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

def initialiseAgents(N, D):
    state0 = np.zeros( shape=(N*3,))
    state0[0::3] = np.random.rand(N, ) * 2 * np.pi   #Selects every third element of state0 (angles) making them random
    state0[1::3] = np.random.rand(N, ) * D   #Random Horizontal coordinates
    state0[2::3] = np.random.rand(N, ) * D   #Random Vertical coordinates
    return state0




#Shortening of neighbour calculations by using scipy
#present => (random 3D array describing the angle, x and y directions)
#Stepsize = time between calculations
#eta = noise element for new averaged angle of direction
#D = size of domain agents are in
#R = radius of interaction
#v0 = starting velocity (speed is constant in vicsek model)


def updateRule(present, stepsize, eta, D, R, v0):
    future = present
    N = np.size(present) // 3   
    MeanNeighbourAngles = np.zeros(N, )   #Gives array of the length of the total number of elements in "present"/state0 divided by 3 (3 because state0 
    #has x, y, and angle in it) 

    #Matrix, each row = one agent (columns = x, y coordinates)
    positions = np.zeros((N, 2))               # <<<=== This might be the part to fix????         Maybe not actually???
            # vvvv => all of the 0th column gets changed
    positions[:, 0] = present[1::3] #x-position updated => first column of 'positions', second column of 'present'
    positions[:, 1] = present[2::3] #y-position updated => second column of 'positions', third column of 'present'
    angles = present[0::3] #seperate list updated => first column 'present'

    #Identify neighbours
    DistanceMatrix = scipy.spatial.distance.pdist(positions)  #Scipy function to calculate distance between two agents
    DistanceMatrix = scipy.spatial.distance.squareform(DistanceMatrix)   #matrix of form [i, j] => distance between agents i and j
    Neighbours = DistanceMatrix <= R   #If distance between i and j less than R, update neighbour list of the direction of motion of those agents
    
    #^^^^ Make it so "Neighbours" can only have a certain amount of elements in it (should be the closeset neighbours)
    #Also should be only those that the agent can 'sense'

    for n in range(N):  
        theta = angles[n]   #angle of agent
        
        MeanNeighbourAngles[n] = np.sum(angles[Neighbours[:, n]]) / np.sum(Neighbours[:, n])   #Mean neighbour angle

        v = v0 * np.array([np.cos(theta), np.sin(theta)])  #Velocity change after new direction calculated
        future[np.array([1, 2]) + 3*n] = present[np.array([1, 2]) + 3*n] + stepsize * v  #New x and y positions after update
        future[np.array([1, 2]) + 3*n] = np.mod(future[np.array([1, 2]) + 3*n], D)  #If agent goes off axis, it reappears on the other side

    noise = (-eta/2 + np.random.rand(N, ) * eta/2)
    future[0::3] = np.mod(present[0::3] + MeanNeighbourAngles + noise, 2*np.pi)  
   #after 0th element, but before the third element (i.e. the first and second elements)
    return(future)




state0 = initialiseAgents(N, D=D)                                  # vvvvv same with this??
trajectory = np.zeros(shape = (N*3, T))  #N*3 columns, T rows      Change so trajecory can be continuosly updated / size can keep increasing??
trajectory[:, 0] = state0   #: means every column, 0th row

for t in range(T-1):
    trajectory[:, t+1] = updateRule(trajectory[:, t], stepsize=stepsize, eta=eta, R=R, D=D, v0=v0)
    #Repeatedly applies update rule to columns of the trajecotry matrix


fig, ax = plt.subplots()
ax.set_xlim([0, 25])
ax.set_ylim([0, 25])

animated_plot, = ax.plot([], [], 'o')

def Animation(frame):
                                               # vvvvvvvv => for every column, each row is updated by 'frame'
    animated_plot.set_data(updateRule(trajectory[:, frame], stepsize=stepsize, eta=eta, R=R, D=D, v0 = v0)[1::3], 
                           updateRule(trajectory[:, frame], stepsize=stepsize, eta=eta, R=R, D=D, v0 = v0)[2::3])

    return 
                                    # ^^^^^^^^^ problem, trajecotry only has 100 elements limit, when animation reaches the 101st "frame" there is no
                                    # more data points to read off. Need to update "trajectory" or "updateRule" so that its size can keep growing
                                    # alongside the animation.
 

#After set number of frames, the animation changes, either the noise increases or another parameter changes each time.

anim = FuncAnimation(fig = fig, func = Animation, interval = 1,frames = T,)

anim.save("Update_Animation.gif")


plt.show()