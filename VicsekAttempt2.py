import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import scipy.spatial 
from scipy.spatial import ckdtree



N=15
D=25
T=100
stepsize=1
eta=0.1
v0=0.5
R=3
M=3 #Max amount of neighbours


fig, ax = plt.subplots(figsize=(8, 8))


StartPositions = np.random.uniform(0, D, size=(N, 2))
StaringDirection = np.random.uniform(-np.pi, np.pi, size=N)

quiver = ax.quiver(StartPositions[:, 0], StartPositions[:, 1], np.cos(StaringDirection[0], np.sin(StaringDirection)))


'''
tree = scipy.spatial.KDTree(StartPositions, boxsize=[D, D])
distance = tree.sparse_distance_matrix(tree, max_distance=R, output_type='coo_matrix')


data = (StaringDirection[distance.col])  #column index format of direction of nearest neighbours

n = scipy.sparse.coo_matrix ((data, (distance.row, distance.col)),
                             shape = distance.get_shape())
s = np.squeeze(np.asarray(n.tocsr().sum(axis=1)))


print(s)
'''


#print(data)
#print(n)
#print(s)



def UpdateRule():

    global StaringDirection

    #Create KDTree 
    tree = scipy.spatial.KDTree(StartPositions, boxsize=[D, D])

    #Gives distance between points that are within neighbour region
    #If distance is larger than region, value is given 0
    #Queries all points with every other point, only if they are within a set domain
    #that is deemed close enough to calculate further (this reduces run time comapared
    #to other more brute force methods)
    #Each distance, is given alongside the relative i,j values of the positions in the matrix
    #This returns a sparse matrix in coordiante form ('ijv', or 'triplet' format)
    distance = tree.sparse_distance_matrix(tree, max_distance=R, output_type='coo_matrix')

    #column index format of direction of nearest neighbours
    data = StaringDirection[distance.col]

    #Takeout direction of neighbours with value 0 in coo_matrix ('distance')
    neighbours = scipy.sparse.coo_matrix ((data, (distance.row, distance.col)),
                             shape = distance.get_shape())
    
    sum = np.squeeze(np.asarray(neighbours.tocsr().sum(axis=1)))

    
    angle = sum + eta*np.ranom.uniform(-np.pi, np.pi, size=N)

    StartPositions[:, 0] += np.cos(angle) * v0
    StartPositions[:, 1] += np.sin(angle) * v0

    quiver.set_offsets(StartPositions)
    quiver.set_UVC(np.cos(angle), np.sin(angle), angle)

    return quiver,

anim = FuncAnimation(fig, UpdateRule, np.arange(1, T), interval=1, blit=True)
plt.show()







#Intitate agents at with random positions and directions of motion 
#within the domain
#Returns list of length 3 of x, y, and angle states
'''
def SpawnAgents(NumberOfAgents, AreaOfDomain):
    position = 1
    initialAngle = np.random.rand(NumberOfAgents, ) * 2 * np.pi
    State0 = (initialx, initialy, initialAngle)
    return list(State0) 
'''




'''
def Update_x(NumberOfAgents, AreaOfDomain, StepSize, Noise,
              RadiusOfInteraction, InitialVelocity, Initial,
              NumberOfNeighbours):

    #Update new variable each time step with new positions
    Future_x = Initial[0]
    Future_y = Initial[1]
    Future_angle = Initial[2]

    Positions = (Future_x, Future_y)
    points = np.asarray(Future_x, Future_y)

    #Find Neighbours using KDTree
    tree = scipy.spatial.ckdtree(points, boxsize=[D, D])
    NN_distance, NN_indices = tree.query(tree, r=RadiusOfInteraction)

    distance = tree.sparse_distance_matrix(tree, max_distance=RadiusOfInteraction) 


    

    #Update direction of motion to be the averge of their neighours
    #np.arctan(np.sin(theta) / np.cos(theta)) + Noise

    for i in range(NumberOfAgents):
        Future_x[i] = Future_x[i] + (InitialVelocity * StepSize)
        Future_y[i] = Future_y[i] + (InitialVelocity * StepSize)

        

        #Future_angle[i] = np.sum(Future_angle[Neighbours[:, i]]) / np.sum(Neighbours[:, i])


    Future = (Future_x, Future_y, Future_angle)
    return(Future)
'''

#print(Update_x(N, D, stepsize, eta, R, v0, SpawnAgents(N, D)))

'''
plt.figure()
plt.scatter(SpawnAgents(N, D)[0], SpawnAgents(N, D)[1])
plt.scatter(Update_x(N, D, stepsize, eta, R, v0, SpawnAgents(N, D)[0]), SpawnAgents(N, D)[1])
plt.show()
'''

