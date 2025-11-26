'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import scipy.spatial

#For infinite potential wall version -> if agent within some range of a wall, add some angle to its direction at each time step.


plt.rcParams["font.family"] = "Times New Roman"

N=300  #Number of agents
D=5  #Size of domain
T=600   #Total number of time steps (frames) in simulation
stepsize=1  #change in time between calculation of position and angle
eta=0.1   #Random noise added to angles
v0=0.03   #Starting velocity
R=1    #Interaction radius



pos = np.random.uniform(0, D, size=(N, 2))
angle = np.random.uniform(0, 2*np.pi, size=N)


def Vicsek():
    #print(i)
    global pos
    global angle

    MeanAngle = np.zeros(N, )
    #MeanAngle2 = np.zeros(N, )
    noise = np.random.uniform(-eta/2, eta/2, size=(N))

    DistanceMatrix = scipy.spatial.distance.pdist(pos)  #Scipy function to calculate distance between two agents
    DistanceMatrix = scipy.spatial.distance.squareform(DistanceMatrix)   #matrix of form [i, j] => distance between agents i and j
        #returns the distance of each agent to all other agents, the array is of size [N, N]

    for i in range(N):
        Neighbours = DistanceMatrix <= R #Gives array of True/False, if distance less than R, this returns True

        MeanAngle[i] = (np.arctan2(np.sum(np.sin(angle[Neighbours[:, i]])), np.sum(np.cos(angle[Neighbours[:, i]]))))
        #MeanAngle2[i] = (np.arctan2(np.mean(np.sin(angle[Neighbours[:, i]])), np.mean(np.cos(angle[Neighbours[:, i]]))))
                                                        #^^^^^Angles of the agents within R, the True values in 'Neighbours'
        #Equation as in Vicsek 1995 to get the average angle of all the neighbours

    MeanAngle = MeanAngle + noise   #Adding random noise element to the angle
    

    #x and y directions accoring to new angle
    cos = (np.cos(MeanAngle))   
    sin = (np.sin(MeanAngle))

        #Updating the position of the agents
    pos[:, 0] += cos * v0 * stepsize
    pos[:, 1] += sin * v0 * stepsize

    pos = np.mod(pos, D)    #Agent appears on other side of domain when goes off one end

    angle = MeanAngle

    return pos, cos, sin, MeanAngle, Neighbours#, MeanAngle2

#print(Vicsek()[3])
#print(Vicsek()[5])
### Code to get animation ###

fig, ax = plt.subplots()
ax.set_xlim([0, D])
ax.set_ylim([0, D])
animated_plot_quiver = ax.quiver(pos[:, 0], pos[:, 1], np.cos(angle), np.sin(angle), clim=[-np.pi, np.pi])#, angle)


plt.rcParams["font.family"] = "Times New Roman"
plt.title(f'Noise level = {eta}', fontsize=16)
plt.xlabel('X-axis', fontsize=14)
plt.ylabel('Y-axis', fontsize=14)
plt.setp(ax.get_xticklabels(), visible='False')
plt.setp(ax.get_yticklabels(), visible='False')
plt.locator_params(axis='y', nbins=1)
plt.locator_params(axis='x', nbins=1)
ax.set_aspect('equal', adjustable='box')

magic_value = 10

def Animate_quiver(frame):
    animated_plot_quiver.set_offsets(Vicsek()[0])
    animated_plot_quiver.set_UVC(Vicsek()[1], Vicsek()[2])#, Vicsek()[3])
    
    return (animated_plot_quiver, )

anim = FuncAnimation(fig = fig, func = Animate_quiver, interval = stepsize, frames = T, blit = False, repeat=False)

anim.save(f"Noise Level = {eta}, D={D}, V0={v0}, N={N}.gif", dpi=400)
fig.savefig(f"2DVicsek, Noise Level = {eta}, D={D}, V0={v0}, N={N}.gif.png", dpi=400)
#plt.show()
'''






import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import scipy.spatial
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform

#For infinite potential wall version -> if agent within some range of a wall, add some angle to its direction at each time step.


plt.rcParams["font.family"] = "Times New Roman"

N=300  #Number of agents
#D=10  #Size of domain
T=600   #Total number of time steps (frames) in simulation
stepsize=1  #change in time between calculation of position and angle
#eta=0.15   #Random noise added to angles
v0=0.03   #Starting velocity
R=1    #Interaction radius

np.random.seed(10)

def Vicsek_pol(eta, D, t):
    
    pos = np.random.uniform(0, D, size=(N, 2))
    pos_2 = pos.copy()
    angle = np.random.uniform(0, 2*np.pi, size=N)
    trajectory=[]

    for _ in range(t):
            
        DistanceMatrix = squareform(pdist(pos))
        noise = np.random.uniform(-eta/2, eta/2, size=(N))
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
        #pos[:, 0] += cos * v0 * stepsize
        #pos[:, 1] += sin * v0 * stepsize

        pos_2[:, 0] += cos * v0 * stepsize
        pos_2[:, 1] += sin * v0 * stepsize        

        pos = np.mod(pos_2, D)

        trajectory.append(pos_2.copy())    #Agent appears on other side of domain when goes off one end

        angle = MeanAngle.copy()
    
    l = pos 
    c = np.cos(angle)
    s = np.sin(angle)

    return l, c, s, np.array(trajectory[-20:])

print((Vicsek_pol(0.1, 5, 100)[3]))


def break_trajectory(traj, D):
    segments = []   #make new list of trajectories on other side of domain
    current = [traj[0]]

    for i in range(1, len(traj)):
        diff = np.abs(traj[i] - traj[i-1])
        if np.any(diff > D/2):      #if there is a large jump between two consecutive values in trajectory (agent jumps accross domain) 
            segments.append(np.array(current))      #add those elements to the new list
            current = [traj[i]]
        else:
            current.append(traj[i])

    segments.append(np.array(current))
    return segments


#         vv = eta         vv = domain       vv = total time of simulation
params = [(2, 7, 0), (0.1, 25, 200), (2, 7, 200), (0.1, 5, 200)]


fig, ax = plt.subplots(2,2, figsize=(6,6))

for axs, (eta, D, t) in zip(ax.reshape(-1), params):

    pos, cx, sy, traj = Vicsek_pol(eta, D, t)


    axs.set_xlim(0, D)
    axs.set_ylim(0, D)
    axs.set_aspect("equal")
    axs.set_xticks([])
    axs.set_yticks([])
    axs.tick_params(left=False, bottom=False)


    if traj.ndim == 3:      #makes sure trajecory is 3D (if not error comes out in the first plot where there is no trajectory to plot)
        wrapped_traj = np.mod(traj, D)
        for i in range(N):
            agent_traj = wrapped_traj[:, i, :]      #gives x and y points for all agents seperately
            segments = break_trajectory(agent_traj, D)

            for seg in segments:
                axs.plot(seg[:, 0], seg[:, 1], lw=0.5, color='black')


    axs.quiver(pos[:, 0], pos[:, 1], cx, sy,
    angles='xy',
    scale_units='xy',
    scale=9,          
    width=0.003,
    headlength=10,       
    headwidth=8,
    headaxislength=8,
    minlength=0,         
    minshaft=0,
    pivot='tip',
    color='black')

    axs.set_title(f"η={eta}, D={D}")

plt.tight_layout()
plt.savefig("2x2 Vicsek.png", dpi=600)
plt.show()


'''

#print(Vicsek_pol(0.1, 5, 100)[3])

noise = np.array([2, 0.1, 2, 0.1])
size = np.array([7, 25, 7, 5])
time = np.array([0, 100, 100, 100])

fig, ax = plt.subplots(2,2)
#ax.set_xlim([0, D])
#ax.set_ylim([0, D])


pos1, cos1, sin1, line1 = Vicsek_pol(2, 7, 0)
line1 = np.array([line1])
ax[0, 0].quiver(pos1[:, 0], pos1[:, 1], cos1, sin1, clim=[-np.pi, np.pi], pivot='middle')


pos2, cos2, sin2, line2 = Vicsek_pol(0.1, 25, 100)
line2 = np.array([line2]).reshape(20, 5, 2)
#ax[0, 1].quiver(pos2[:, 0], pos2[:, 1], cos2, sin2, clim=[-np.pi, np.pi], pivot='middle')
ax[0, 1].annotate("", xytext=(pos2[:, 0], pos2[:, 1]), xy = (cos2, sin2), arrowprops=dict(arrowstyle='->', color='black'))
#print(np.shape(line2))
#print(line2)
#print(line2[:, :, 0])
ax[0, 1].plot(line2[:, :, 0], line2[:, :, 1], color='black')


pos3, cos3, sin3, line3 = Vicsek_pol(2, 7, 100)
line3 = np.array([line3]).reshape(20, 5, 2)
ax[1, 0].quiver(pos3[:, 0], pos3[:, 1], cos3, sin3, clim=[-np.pi, np.pi], pivot='middle')
ax[1, 0].plot(line3[:, :, 0], line3[:, :, 1], color='black')


pos4, cos4, sin4, line4 = Vicsek_pol(0.1, 5, 100)
line4 = np.array([line4]).reshape(20, 5, 2)
ax[1, 1].quiver(pos4[:, 0], pos4[:, 1], cos4, sin4, clim=[-np.pi, np.pi], pivot='middle')
ax[1, 1].plot(line4[:, :, 0], line4[:, :, 1], color='black')


#plt.savefig(f"2x2 Vicsek.png", dpi=400)
plt.show()

'''









'''
# do it for different population and noise levels like in vicsek paper


#Use for polarisation eqs           Works but gives weird plot due to boundary conditions????
#If set starting angles in smaller range, better plot
def Vicsek_pol(N):

    eta = np.linspace(0, 5, 10)
    polarisation = np.zeros(T)
    pol = []
    
    for k in eta:

        for j in range(T):

            pos = np.random.uniform(0, D, size=(N, 2))
            angle = np.random.uniform(0, 2*np.pi, size=N)

            MeanAngle = np.zeros(N, )
            noise = np.random.uniform(-k/2, k/2, size=(N))

            for i in range(N):

                DistanceMatrix = scipy.spatial.distance.pdist(pos)  #Scipy function to calculate distance between two agents
                DistanceMatrix = scipy.spatial.distance.squareform(DistanceMatrix)   #matrix of form [i, j] => distance between agents i and j
        #returns the distance of each agent to all other agents, the array is of size [N, N]

                Neighbours = DistanceMatrix <= R #Gives array of True/False, if distance less than R, this returns True

                MeanAngle[i] = (np.arctan2(np.mean(np.sin(angle[Neighbours[:, i]])), np.mean(np.cos(angle[Neighbours[:, i]])))) 
                                                        #^^^^^Angles of the agents within R, the True values in 'Neighbours'
        #Equation as in Vicsek 1995 to get the average angle of all the neighbours

            MeanAngle = MeanAngle + noise   #Adding random noise element to the angle
        #MeanAngle = np.mod(MeanAngle, 2*np.pi)


        #x and y directions accoring to new angle
            cos = (np.cos(MeanAngle))   
            sin = (np.sin(MeanAngle))

        #Updating the position of the agents 
            pos[:, 0] += cos * v0 * stepsize
            pos[:, 1] += sin * v0 * stepsize

            pos = np.mod(pos, D)    #Agent appears on other side of domain when goes off one end

            polarisation[j] = (abs(np.sum((MeanAngle)/abs(MeanAngle))) / (N))
        
        pol = np.mean(polarisation)

    return pos, cos, sin, MeanAngle, pol, eta, polarisation, noise#, p


print(Vicsek_pol(40)[4])
print(len(Vicsek_pol(40)[7]))


#print(Vicsek_pol()[4])



### Polarisation plot ###

#Polarisatio = Vicsek()[4]
#for i in range(T):
#    Polarisation[i] = np.sum((Vicsek()[3]/abs(Vicsek()[3])))/(N)
    #Polarisation[i] = Vicsek()[3]

time = np.linspace(0, T, T)

#print(Polarisation)

#avg = np.full(shape=len(time), fill_value=np.mean(Vicsek_pol()[4]))
#noise = np.linspace(1, 5, 50)

fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(Vicsek_pol(40)[5], Vicsek_pol(40)[4], color='black', marker='o', label='N=40')
ax.scatter(Vicsek_pol(100)[5], Vicsek_pol(100)[4], color='black', marker='v', label='N=100')
ax.scatter(Vicsek_pol(400)[5], Vicsek_pol(400)[4], color='black', marker='s', label='N=400')

ax.plot(0, 1, color='white')
ax.plot(0, 0, color='white')
#ax.plot(time, avg, color='red', label='Mean Polarisation', linewidth=1.5, linestyle=':')
#ax.set_ylim(-0.01, 1.01)
ax.set_xlabel('Noise', fontsize=14)
ax.set_ylabel('Polarisation', fontsize=14)
ax.tick_params(direction='out', length=4, width=1, labelsize=12, top=False, right=False)
ax.legend(fontsize=14)
#ax.minorticks_on()
#plt.savefig(f"Polarisation Plot: {N} Agents.pdf" dpi=400)
#plt.show()
#plt.savefig(f"Polarisation Plot Agents, Spawn angle 0_2pi.png", dpi=400)
plt.show()
'''








