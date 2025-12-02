
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import scipy.spatial
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform

#For infinite potential wall version -> if agent within some range of a wall, add some angle to its direction at each time step.


plt.rcParams["font.family"] = "Times New Roman"

#N=100  #Number of agents
#D=10  #Size of domain
T=600   #Total number of time steps (frames) in simulation
stepsize=1  #change in time between calculation of position and angle
#eta=0.15   #Random noise added to angles
v0=0.03   #Starting velocity
R=1    #Interaction radius


def Vicsek_pol(N, eta, D):

    average_pol =[]

    for k in eta:

        pol = []
    
        for _ in range(4):

            pos = np.random.uniform(0, D, size=(N, 2))
            angle = np.random.uniform(0, 2*np.pi, size=N)
            pol_time_step = []

            for _ in range(T):
            
                DistanceMatrix = squareform(pdist(pos))
                noise = np.random.uniform(-k/2, k/2, size=(N))
                MeanAngle = np.zeros(N,)

                for i in range(N):

                    Neighbours = DistanceMatrix <= R #Gives array of True/False, if distance less than R, this returns True

                    MeanAngle[i] = (np.arctan2(np.sum(np.sin(angle[Neighbours[:, i]])), np.sum(np.cos(angle[Neighbours[:, i]])))) + noise[i]
                                                        #^^^^^Angles of the agents within R, the True values in 'Neighbours'
                #Equation as in Vicsek 1995 to get the average angle of all the neighbour

        #x and y directions accoring to new angle
                cos = (np.cos(MeanAngle))   
                sin = (np.sin(MeanAngle))

        #Updating the position of the agents 
                pos[:, 0] += cos * v0 * stepsize
                pos[:, 1] += sin * v0 * stepsize

                pos = np.mod(pos, D)    #Agent appears on other side of domain when goes off one end

                angle = MeanAngle

                vx = np.mean(cos)
                vy = np.mean(sin)

                pol_time_step.append(np.sqrt(vx**2 + vy**2))
        
            pol.append(np.mean(pol_time_step))

        average_pol.append(np.mean(pol))

    return average_pol



Populations = [40, 100, 400]
Size = [3.1, 5, 10]
eta = np.linspace(0, 10, 50)
shapes = ['o', '^', 's']
#print(Vicsek_pol(40, eta, 3.1))

fig, ax = plt.subplots(figsize=(7, 7))

for N, D, s in zip(Populations, Size, shapes):
    polarisation = Vicsek_pol(N, eta, D)
    plt.plot(eta, polarisation, marker=s, linestyle='', label=f'N={N}')

ax.plot(0, 1, color='white')
ax.plot(0, 0, color='white')
ax.set_xlabel('Noise (η)', fontsize=14)
ax.set_ylabel('Polar Order Parameter', fontsize=14)
ax.set_title('Periodic boundary conditions')
ax.tick_params(direction='out', length=4, width=1, labelsize=12, top=False, right=False)
ax.legend(fontsize=14)
#ax.minorticks_on()
#plt.savefig(f"Polar Order Parameter, periodic boundary conditions Plot.png", dpi=400)
plt.show()





'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import scipy.spatial
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform

#For infinite potential wall version -> if agent within some range of a wall, add some angle to its direction at each time step.


plt.rcParams["font.family"] = "Times New Roman"

#N=100  #Number of agents
#D=10  #Size of domain
T=600   #Total number of time steps (frames) in simulation
stepsize=1  #change in time between calculation of position and angle
#eta=0.15   #Random noise added to angles
v0=0.03   #Starting velocity
R=1    #Interaction radius
scale = 2
force_scale = 0.005
fake_wall = 1
epsilon = 1e-6


def wall_force_vector(position, angle):
                         # x and y direction of the boid
    direction = np.array([np.cos(angle), np.sin(angle)])  
    Force = np.zeros(2)     # setting array for force due to walls

    # distance to walls
    dist_left = position[0]         # x = 0
    dist_right = D - position[0]    # x = D
    dist_bot = position[1]          # y = 0
    dist_top = D - position[1]      # y = D

    # normal vectors pointing away from each wall, alongside the boids respective distance to that wall
    normals = [
        (np.array([1.0, 0.0]), dist_left),       
        (np.array([-1.0, 0.0]), dist_right),    
        (np.array([0.0, 1.0]), dist_bot),        
        (np.array([0.0, -1.0]), dist_top)        
    ]

    # calculate force due to wall, and update 'Force', with the force and its direction using the normal vectors
    for norm_vec, dist in normals:
        if dist < fake_wall:
            mag = 1.0 / ((dist + epsilon)**scale)
            Force += mag * norm_vec   

    # changes the force from the wall, to a force that changes the direction of the boid, not the position
    torque = direction[0]*Force[1] - direction[1]*Force[0]

    return torque


def Vicsek_pol(N, eta, D):

    pol = []
    
    for k in eta:

        pos = np.random.uniform(0, D, size=(N, 2))
        angle = np.random.uniform(0, 2*np.pi, size=N)
        pol_time_step = []

        for _ in range(T):
            
            DistanceMatrix = squareform(pdist(pos))
            noise = np.random.uniform(-k/2, k/2, size=(N))
            MeanAngle = np.zeros(N,)
            total_wall_torque = np.zeros(N)

            for i in range(N):

                Neighbours = DistanceMatrix <= R #Gives array of True/False, if distance less than R, this returns True

                MeanAngle[i] = (np.arctan2(np.sum(np.sin(angle[Neighbours[:, i]])), np.sum(np.cos(angle[Neighbours[:, i]])))) + noise[i]
                                                        #^^^^^Angles of the agents within R, the True values in 'Neighbours'
                #Equation as in Vicsek 1995 to get the average angle of all the neighbour

                total_wall_torque[i] = wall_force_vector(pos[i], MeanAngle[i])
    
            MeanAngle = MeanAngle + total_wall_torque * force_scale  

        #x and y directions accoring to new angle
            cos = (np.cos(MeanAngle))   
            sin = (np.sin(MeanAngle))

        #Updating the position of the agents 
            pos[:, 0] += cos * v0 * stepsize
            pos[:, 1] += sin * v0 * stepsize

            pos = np.mod(pos, D)    #Agent appears on other side of domain when goes off one end

            angle = MeanAngle

            vx = np.mean(cos)
            vy = np.mean(sin)

            pol_time_step.append(np.sqrt(vx**2 + vy**2))

        
        pol.append(np.mean(pol_time_step))


    return pol



Populations = [40, 100, 400]
Size = [3.1, 5, 10]
eta = np.linspace(0, 10, 50)
shapes = ['o', '^', 's']
#print(Vicsek_pol(40, eta, 3.1))

fig, ax = plt.subplots(figsize=(7, 7))

for N, D, s in zip(Populations, Size, shapes):
    polarisation = Vicsek_pol(N, eta, D)
    plt.plot(eta, polarisation, marker=s, linestyle='', label=f'N={N}')

ax.plot(0, 1, color='white')
ax.plot(0, 0, color='white')
ax.set_xlabel('Noise (η)', fontsize=14)
ax.set_ylabel('Polar Order Parameter', fontsize=14)
ax.set_title('Infinite boundary conditions')
ax.tick_params(direction='out', length=4, width=1, labelsize=12, top=False, right=False)
ax.legend(fontsize=14)
#ax.minorticks_on()
plt.savefig(f"Polar Order Parameter, Infinite bounday conditions Plot.png", dpi=400)
plt.show()
'''



