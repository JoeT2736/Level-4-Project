import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from matplotlib.patches import Circle

inpath = "birdloc.txt" #the file that contains the bird data
outpath = "video.mp4" #the output video file
N = 400 #how many birds are in the simulation
tmax = 100 #how many time steps should be plotted
L = 10.0 #what is the dimension of the 2D field

lw = 1.0 #plot line width
lc = 'mediumblue' #plot line color

fig = plt.figure()
plt.suptitle("Running %d time steps with %d birds" % (tmax, N))
ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
ax2 = plt.subplot2grid((3, 1), (2, 0), rowspan=1)

xdata2 = np.linspace(0, tmax, tmax)
ydata2 = np.zeros((tmax, 1))
xdata2[:] = np.nan
ydata2[:] = np.nan

#reads the data for single time step, very clumsy implementation
def read_data(i):
    data = np.genfromtxt(inpath, dtype=np.float32, skip_header=i*N, max_rows=N)
    data = np.reshape(data, (N, 4))
    x = data[:, 1]
    y = data[:, 2]
    vx = np.cos(data[:, 3])
    vy = np.sin(data[:, 3])
    mva = np.sqrt(np.mean(vx)**2.0+np.mean(vy)**2.0)
    return x, y, vx, vy, mva

#plots the quiver plot of birds for each time step and the order parameter as a function of time
def update(i):
    x, y, vx, vy, mva = read_data(i)
    print(i, mva)

    #bird plotting
    ax1.clear()
    ax1.add_artist(Circle((L/2.0, L/2.0), 1.0, color='grey', fill=False))
    ln = ax1.quiver(x, y, vx, vy, pivot='mid', scale=40.0)
    ax1.set_aspect('equal', 'box')
    ax1.set_xlim((0, L))
    ax1.set_ylim((0, L))

    #order parameter plotting
    ax2.clear()
    xdata2[i] = i
    ydata2[i] = mva
    ax2.plot(xdata2, ydata2, linewidth=lw, color=lc)
    ax2.set_xlim((0, tmax))
    ax2.set_ylim((0, 1))
    ax2.set_xlabel('time step')
    ax2.set_ylabel('order parameter')



Writer = anim.writers['ffmpeg']
writer = Writer(fps=15,metadata=dict(artist='Me'),bitrate=3600)

ani = anim.FuncAnimation(fig, update, frames=np.arange(0, tmax, dtype=np.int), repeat=False, interval=0.01)
ani.save(outpath, writer=writer)


plt.show()
