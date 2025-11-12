
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import scipy.spatial
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform

#For infinite potential wall version -> if agent within some range of a wall, add some angle to its direction at each time step.


plt.rcParams["font.family"] = "Times New Roman"

#N=100  #Number of agents
D=10  #Size of domain
T=600   #Total number of time steps (frames) in simulation
stepsize=1  #change in time between calculation of position and angle
#eta=0.15   #Random noise added to angles
v0=0.03   #Starting velocity
R=1    #Interaction radius


def Vicsek_pol(N, eta):

    pol = []
    
    for k in eta:

        pos = np.random.uniform(0, D, size=(N, 2))
        angle = np.random.uniform(0, 2*np.pi, size=N)
        pol_time_step = []

        for _ in range(T):
            
            DistanceMatrix = squareform(pdist(pos))
            NewAngle = np.zeros_like(angle)

            for i in range(N):

                Neighbours = DistanceMatrix <= R #Gives array of True/False, if distance less than R, this returns True

                avgsin = np.mean(np.sin(angle[Neighbours[:, i]]))
                avgcos = np.mean(np.cos(angle[Neighbours[:, i]]))

                MeanAngle = np.arctan2(avgsin, avgcos) 
                                                        #^^^^^Angles of the agents within R, the True values in 'Neighbours'
                #Equation as in Vicsek 1995 to get the average angle of all the neighbours

                NewAngle[i] = MeanAngle + np.random.uniform(-k/2, k/2)

            angle = NewAngle    #Adding random noise element to the angle
        
        #x and y directions accoring to new angle
            cos = (np.cos(angle))   
            sin = (np.sin(angle))

        #Updating the position of the agents 
            pos[:, 0] += cos * v0 * stepsize
            pos[:, 1] += sin * v0 * stepsize

            pos = np.mod(pos, D)    #Agent appears on other side of domain when goes off one end

            vx = np.mean(cos)
            vy = np.mean(sin)
            pol_time_step.append(np.sqrt(vx**2 + vy**2))
        
        pol.append(np.mean(pol_time_step))


    return pol



Populations = [40, 100, 400]
eta = np.linspace(0, 5, 300)
#print(Vicsek_pol(40, eta))

fig, ax = plt.subplots(figsize=(7, 7))

for N in Populations:
    polarisation = Vicsek_pol(N, eta)
    plt.plot(eta, polarisation, '-o', label=f'N={N}')

ax.plot(0, 1, color='white')
ax.plot(0, 0, color='white')
ax.set_xlabel('Noise (η)', fontsize=14)
ax.set_ylabel('Polarisation', fontsize=14)
ax.tick_params(direction='out', length=4, width=1, labelsize=12, top=False, right=False)
ax.legend(fontsize=14)
#ax.minorticks_on()
plt.savefig(f"Polarisation Plot.png", dpi=400)
plt.show()



'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

def vicsek_model(N, eta_values, T=200, D=10, R=1, v0=0.03, stepsize=1):
    """
    Simulates the Vicsek model and returns the average polarization for each eta.
    """
    pol_values = []

    for eta in eta_values:
        # Initialize positions and angles
        pos = np.random.uniform(0, D, size=(N, 2))
        angle = np.random.uniform(0, 2 * np.pi, size=N)
        polarisation_time = []

        for _ in range(T):
            # Compute pairwise distances
            DistanceMatrix = squareform(pdist(pos))

            # Compute new angles based on neighbors
            new_angle = np.zeros_like(angle)
            for i in range(N):
                neighbors = DistanceMatrix[i] <= R
                avg_sin = np.mean(np.sin(angle[neighbors]))
                avg_cos = np.mean(np.cos(angle[neighbors]))
                mean_dir = np.arctan2(avg_sin, avg_cos)
                # Add noise
                new_angle[i] = mean_dir + np.random.uniform(-eta/2, eta/2)
            
            angle = new_angle

            # Update positions
            pos[:, 0] += v0 * np.cos(angle) * stepsize
            pos[:, 1] += v0 * np.sin(angle) * stepsize
            pos = np.mod(pos, D)  # periodic boundaries

            # Compute polarization
            vx = np.mean(np.cos(angle))
            vy = np.mean(np.sin(angle))
            polarisation_time.append(np.sqrt(vx**2 + vy**2))
        
        pol_values.append(np.mean(polarisation_time))
    
    return pol_values


# --- Parameters ---
eta_values = np.linspace(0, 5, 15)
N_values = [50, 100, 200, 400]
T = 300

# --- Run simulations ---
plt.figure(figsize=(8, 6))
for N in N_values:
    pol = vicsek_model(N, eta_values, T=T)
    plt.plot(eta_values, pol, '-o', label=f'N = {N}')

# --- Plot results ---
plt.title("Vicsek Model: Average Polarization vs Noise η")
plt.xlabel("Noise (η)")
plt.ylabel("Average Polarization")
plt.legend()
plt.grid(True)
plt.show()
'''

