import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import scipy.spatial

plt.rcParams["font.family"] = "Times New Roman"

N=40  #Number of agents
D=5  #Size of domain
T=600   #Total number of time steps (frames) in simulation
stepsize=1  #change in time between calculation of position and angle
eta=0.15   #Random noise added to angles
v0=0.03   #Starting velocity
R=1    #Interaction radius



pos = np.random.uniform(0, D, size=(N, 2))
#angle = np.random.uniform(0, 2*np.pi, size=N)
angle = np.random.uniform(0, 2*np.pi, size=N)


def Vicsek():
    #print(i)
    global pos
    global angle

    MeanAngle = np.zeros(N, )
    noise = np.random.uniform(-eta/2, eta/2, size=(N))

    for i in range(N):

        DistanceMatrix = scipy.spatial.distance.pdist(pos)  #Scipy function to calculate distance between two agents
        DistanceMatrix = scipy.spatial.distance.squareform(DistanceMatrix)   #matrix of form [i, j] => distance between agents i and j
        #returns the distance of each agent to all other agents, the array is of size [N, N]

        Neighbours = DistanceMatrix <= R #Gives array of True/False, if distance less than R, this returns True

        MeanAngle[i] = (np.arctan2(np.sum(np.sin(angle[Neighbours[:, i]])), np.sum(np.cos(angle[Neighbours[:, i]]))))
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

    #frame = []
    #frame += 1
    #print(frame)
    
    #Polarisation = np.zeros(T)
    #for k in range(T):
        #Polarisation = np.zeros(N)
    #Polarisation[k] = abs(np.sum(MeanAngle/abs(MeanAngle)))/N
    
    #p = Polarisation


    return pos, cos, sin, MeanAngle#, p






#Use for polarisation eqs           Works but gives weird plot due to boundary conditions????
#If set starting angles in smaller range, better plot
def Vicsek_pol():
    #print(i)
    #global pos
    #global angle

    polarisation = np.zeros(T)

    for j in range(T):

        pos = np.random.uniform(0, D, size=(N, 2))
        angle = np.random.uniform(0, 2*np.pi, size=N)

        MeanAngle = np.zeros(N, )
        noise = np.random.uniform(-eta/2, eta/2, size=(N))

        for i in range(N):

            DistanceMatrix = scipy.spatial.distance.pdist(pos)  #Scipy function to calculate distance between two agents
            DistanceMatrix = scipy.spatial.distance.squareform(DistanceMatrix)   #matrix of form [i, j] => distance between agents i and j
        #returns the distance of each agent to all other agents, the array is of size [N, N]

            Neighbours = DistanceMatrix <= R #Gives array of True/False, if distance less than R, this returns True

            MeanAngle[i] = (np.arctan2(np.sum(np.sin(angle[Neighbours[:, i]])), np.sum(np.cos(angle[Neighbours[:, i]]))))
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

        polarisation[j] = abs(np.sum((MeanAngle)/abs(MeanAngle))) / (N)

    return pos, cos, sin, MeanAngle, polarisation, abs(np.sum(MeanAngle))


print(Vicsek_pol()[4])

#print(Vicsek_pol()[4])



### Polarisation plot ###

#Polarisatio = Vicsek()[4]
#for i in range(T):
#    Polarisation[i] = np.sum((Vicsek()[3]/abs(Vicsek()[3])))/(N)
    #Polarisation[i] = Vicsek()[3]

time = np.linspace(0, T, T)

#print(Polarisation)

fig, ax = plt.subplots(figsize=(7, 7))
ax.plot(time, Vicsek_pol()[4])
ax.set_xlabel('Time')
ax.set_ylabel('Polarisation')
plt.show()











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

anim = FuncAnimation(fig = fig, func = Animate_quiver, interval = 1, frames = T, blit = False, repeat=False)

#anim.save(f"Noise Level = {eta}, D={D}.gif", dpi=400)
#plt.savefig("2DVicsekAnimation.png", dpi=400)
plt.show()








