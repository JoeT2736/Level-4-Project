import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import scipy.spatial

#For infinite potential wall version -> if agent within some range of a wall, add some angle to its direction at each time step.


plt.rcParams["font.family"] = "Times New Roman"

N=100  #Number of agents
D=10  #Size of domain
T=600   #Total number of time steps (frames) in simulation
stepsize=1  #change in time between calculation of position and angle
eta=0.15   #Random noise added to angles
v0=0.03   #Starting velocity
R=1    #Interaction radius



pos = np.random.uniform(0, D, size=(N, 2))
angle = np.random.uniform(0, 2*np.pi, size=N)


def Vicsek():
    #print(i)
    global pos
    global angle

    MeanAngle = np.zeros(N, )
    noise = np.random.uniform(-eta/2, eta/2, size=(N))

    DistanceMatrix = scipy.spatial.distance.pdist(pos)  #Scipy function to calculate distance between two agents
    DistanceMatrix = scipy.spatial.distance.squareform(DistanceMatrix)   #matrix of form [i, j] => distance between agents i and j
        #returns the distance of each agent to all other agents, the array is of size [N, N]

    for i in range(N):
        Neighbours = DistanceMatrix <= R #Gives array of True/False, if distance less than R, this returns True

        MeanAngle[i] = (np.arctan2(np.mean(np.sin(angle[Neighbours[:, i]])), np.mean(np.cos(angle[Neighbours[:, i]]))))
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

    return pos, cos, sin, MeanAngle, Neighbours

print(Vicsek()[4])
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


def Animate_quiver(frame):
    animated_plot_quiver.set_offsets(Vicsek()[0])
    animated_plot_quiver.set_UVC(Vicsek()[1], Vicsek()[2])#, Vicsek()[3])
    return (animated_plot_quiver, )

anim = FuncAnimation(fig = fig, func = Animate_quiver, interval = stepsize, frames = T, blit = False, repeat=False)

#anim.save(f"Noise Level = {eta}, D={D}.gif", dpi=400)
#plt.savefig("2DVicsekAnimation.png", dpi=400)
plt.show()







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








