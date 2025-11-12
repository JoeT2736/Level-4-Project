import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import scipy.spatial

#For infinite potential wall version -> if agent within some range of a wall, add some direction to its direction at each time step.


plt.rcParams["font.family"] = "Times New Roman"

N=3  #Number of agents
D=5  #Size of domain
T=600   #Total number of time steps (frames) in simulation
timestep=1  #change in time between calculation of position and direction
eta=0.15   #Random noise added to directions
v0=0.03   #Starting velocity
v_SD = 0.003   #Standard deviation for gaussian of speed
R=1    #Interaction radius
SD = np.pi/72   #Standard deviation for gaussian distribution of new directions
w_def = np.pi/2     #Default turning rate (per second)
blind_back = 2*np.pi/3     #Area behind fish where it cant see
blind_front = 2*np.pi/3     #Area in front of fish, where it does not align
attraction_scale = 1 
repulsion_scale = 1
aligning_scale = 1
attraction_range = 3
repulsion_range = 1
aligning_range = 2   
eccentricity = 2



pos = np.random.uniform(0, D, size=(N, 2))
#direction = np.random.uniform(0, 2*np.pi, size=N)
direction = np.random.uniform(0, 2*np.pi, size=N)

def speed():
    global v0
    vel0 = np.full(shape=N, fill_value=v0)
    return np.random.normal(vel0, scale=v_SD)


def Vicsek():
    #print(i)
    global pos
    global direction

    Meandirection = np.zeros(N, )
    noise = np.random.uniform(-eta/2, eta/2, size=(N))
    #repel_direction=np.full(shape=(N,), fill_value=np.pi/12)
    #r=np.zeros(N,)
    align_angle = np.zeros(N,)
    repel_angle = np.zeros(N,)
    attract_angle = np.zeros(N,)
    w_a = np.zeros(N,)
    w_r = np.zeros(N,)
    w_p = np.zeros(N,)
    rotation = np.zeros(N,)
    weight_a = np.zeros(N,)
    weight_r = np.zeros(N,)
    weight_p = np.zeros(N,)

    DistanceMatrix = scipy.spatial.distance.pdist(pos)  #Scipy function to calculate distance between two agents
    DistanceMatrix = scipy.spatial.distance.squareform(DistanceMatrix)   #matrix of form [i, j] => distance between agents i and j
    #returns the distance of each agent to all other agents, the array is of size [N, N]

    for i in range(N):

        #Gives array of True/False, if distance less than R, this returns True
        attract = np.logical_and(DistanceMatrix <= attraction_range, DistanceMatrix > aligning_range, DistanceMatrix > repulsion_range) 
        repel = np.logical_and(DistanceMatrix <= repulsion_range, DistanceMatrix > 0)
        align = np.logical_and(DistanceMatrix <= aligning_range, DistanceMatrix > repulsion_range)

        x = pos[:, 0]
        y = pos[:, 1]
        
        ### repulsion rotation ###
        #angle between heading direction of i and the direction of i to j
        #repel_angle[i] = np.angle((np.sqrt((x[repel[:, i+1]] - x[repel[:, i]])**2 - (y[repel[:, i+1]] - y[repel[:, i]])**2), direction[repel[:, i]]))
        
        #if repel_angle[i] > 0:
        #    w_r[i] = -w_def[i]
        #if repel_angle[i] < 0:
        #    w_r[i] = w_def[i]
        
        ###                    ###

        '''
        #attraction rotation
        attract_angle[i] = np.angle((np.sqrt((x[attract[:, i + 1]] - x[attract[:, i]])**2 - (y[attract[:, i + 1]] - y[attract[:, i]])**2) , direction[attract[:, i]]))
        w_a[i] = w_def * attract_angle[i]

        #alignment rotation
        #difference between angle of diections of i and j
        align_angle[i] = direction[align[:, i]] - direction[align[:, i + 1]]
        w_p[i] = w_def * align_angle[i]

        #weight of repulsion
        weight_r[i] = np.min(0.005 * repulsion_scale / DistanceMatrix[repel[:, i]]**3, 10)

        #weight of attraction
        weight_a[i] = 0.2 * attraction_scale * np.exp(-((DistanceMatrix[attract[:, i]] - 0.5*(attraction_range + aligning_range))/(attraction_range - aligning_range))**2)

        #weight of alignment
        weight_p[i] = aligning_scale * np.exp(-((DistanceMatrix[attract[:, i]] - 0.5*(aligning_range + repulsion_range))/(aligning_range - repulsion_range))**2)

        #behavioural reaction = weighted sum
        rotation[i] = weight_r*w_r[i] + weight_a*w_a[i] + weight_p*w_p[i]

        #new direction of ith agent
        direction[i] += np.random.normal(direction[i] + rotation[i]*timestep, scale=SD)
        '''

    #x and y directions accoring to new direction
    cos = (np.cos(direction))   
    sin = (np.sin(direction))

        #Updating the position of the agents
    pos[:, 0] += cos * speed() * timestep
    pos[:, 1] += sin * speed() * timestep

    pos = np.mod(pos, D)    #Agent appears on other side of domain when goes off one end

    return pos, cos, sin, Meandirection, w_r, attract

#print(Vicsek()[4])
print(Vicsek()[5])

fig, ax = plt.subplots()
ax.set_xlim([0, D])
ax.set_ylim([0, D])
colours=['g', 'r', 'b']
ax.quiver(pos[:, 0], pos[:, 1], np.cos(direction), np.sin(direction), clim=[-np.pi, np.pi], color=colours)
plt.show()


### Code to get animation ###

fig, ax = plt.subplots()
ax.set_xlim([0, D])
ax.set_ylim([0, D])
animated_plot_quiver = ax.quiver(pos[:, 0], pos[:, 1], np.cos(direction), np.sin(direction), clim=[-np.pi, np.pi])#, direction)


plt.rcParams["font.family"] = "Times New Roman"
plt.title(f'Noise level = {eta}', fontsize=16)
plt.xlabel('X-axis', fontsize=14)
plt.ylabel('Y-axis', fontsize=14)
plt.setp(ax.get_xticklabels(), visible='False')
plt.setp(ax.get_yticklabels(), visible='False')
plt.locator_params(axis='y', nbins=1)
plt.locator_params(axis='x', nbins=1)
ax.set_aspect('equal', adjustable='box')


def Animate_quiver(frame):
    animated_plot_quiver.set_offsets(Vicsek()[0])
    animated_plot_quiver.set_UVC(Vicsek()[1], Vicsek()[2])#, Vicsek()[3])
    return (animated_plot_quiver, )

#anim = FuncAnimation(fig = fig, func = Animate_quiver, interval = 1, frames = T, blit = False, repeat=False)

#anim.save(f"Hemelrijk, Pop={N}.gif", dpi=400)
#plt.savefig("2DVicsekAnimation.png", dpi=400)
#plt.show()