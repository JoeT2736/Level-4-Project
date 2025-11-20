
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

    pol = []
    
    for k in eta:

        pos = np.random.uniform(0, D, size=(N, 2))
        angle = np.random.uniform(0, 2*np.pi, size=N)
        pol_time_step = []

        for _ in range(T):
            
            DistanceMatrix = squareform(pdist(pos))
            noise = np.random.uniform(-k/2, k/2, size=(N))
            MeanAngle = np.zeros(N,)

            for i in range(N):

                Neighbours = DistanceMatrix <= R #Gives array of True/False, if distance less than R, this returns True

                MeanAngle[i] = (np.arctan2(np.mean(np.sin(angle[Neighbours[:, i]])), np.mean(np.cos(angle[Neighbours[:, i]])))) + noise[i]
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


    return pol



Populations = [40, 100, 400]
Size = [3.1, 5, 10]
eta = np.linspace(0, 5, 50)
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
plt.savefig(f"Polar Order Parameter, periodic boundary conditions Plot.png", dpi=400)
plt.show()






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
            force_wall = np.zeros(N)
            force_wall_left = np.zeros(N)
            force_wall_right = np.zeros(N)
            force_wall_bot = np.zeros(N)
            force_wall_top = np.zeros(N)
            dx_left = np.zeros(N)
            dx_right = np.zeros(N)
            dy_bot = np.zeros(N)
            dy_top = np.zeros(N)

            for i in range(N):

                Neighbours = DistanceMatrix <= R #Gives array of True/False, if distance less than R, this returns True

                MeanAngle[i] = (np.arctan2(np.mean(np.sin(angle[Neighbours[:, i]])), np.mean(np.cos(angle[Neighbours[:, i]])))) + noise[i]
                                                        #^^^^^Angles of the agents within R, the True values in 'Neighbours'
                #Equation as in Vicsek 1995 to get the average angle of all the neighbour

                dx_left[i] = pos[i, 0]
                dx_right[i] = D - pos[i, 0]
                dy_bot[i] = pos[i, 1]
                dy_top[i] = D - pos[i, 1]

                force_wall_left[i] = 1/(dx_left[i])**scale
                force_wall_right[i] = 1/(dx_right[i])**scale
                force_wall_bot[i] = 1/(dy_bot[i])**scale    
                force_wall_top[i] = 1/(dy_top[i])**scale

                if MeanAngle[i] >= 0:
                    force_wall_left[i] = -force_wall_left[i]
                elif MeanAngle[i] < 0:
                    force_wall_left[i] = force_wall_left[i]

                if MeanAngle[i] >= 0:
                    force_wall_right[i] = force_wall_right[i]
                elif MeanAngle[i] < 0:
                    force_wall_right[i] = -force_wall_right[i]

                if MeanAngle[i] >= 3*np.pi/2:
                    force_wall_bot[i] = force_wall_bot[i]
                elif MeanAngle[i] < 3*np.pi/2:
                    force_wall_bot[i] = -force_wall_bot[i]

                if MeanAngle[i] >= np.pi/2:
                    force_wall_top[i] = force_wall_top[i]
                elif MeanAngle[i] < np.pi/2:
                    force_wall_top[i] = -force_wall_top[i]

                force_wall[i] = force_wall_left[i] + force_wall_right[i] + force_wall_bot[i] + force_wall_top[i]

            MeanAngle = MeanAngle + force_wall*force_scale    

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
eta = np.linspace(0, 5, 50)
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




