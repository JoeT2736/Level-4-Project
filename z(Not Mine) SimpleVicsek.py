#Francesco Turci code (online)

'''
import numpy as np
import scipy as sp
from scipy import sparse
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
 
 
L = 32.0
rho = 3.0
N = int(rho*L**2)
print(" N",N)
 
r0 = 1.0
deltat = 1.0
factor =0.5
v0 = r0/deltat*factor
iterations = 10000
eta = 0.15
 
 
pos = np.random.uniform(0,L,size=(N,2))
orient = np.random.uniform(-np.pi, np.pi,size=N)
 
fig, ax= plt.subplots(figsize=(6,6))
 
qv = ax.quiver(pos[:,0], pos[:,1], np.cos(orient[0]), np.sin(orient), orient, clim=[-np.pi, np.pi])
 
def animate(i):
    #print(i)
 
    global orient
    tree = cKDTree(pos,boxsize=[L,L])
    dist = tree.sparse_distance_matrix(tree, max_distance=r0,output_type='coo_matrix')
 
    #important 3 lines: we evaluate a quantity for every column j
    data = np.exp(orient[dist.col]*1j)
    # construct  a new sparse marix with entries in the same places ij of the dist matrix
    neigh = sparse.coo_matrix((data,(dist.row,dist.col)), shape=dist.get_shape())
    # and sum along the columns (sum over j)
    S = np.squeeze(np.asarray(neigh.tocsr().sum(axis=1)))
     
     
    orient = np.angle(S)+eta*np.random.uniform(-np.pi, np.pi, size=N)
 
 
    cos, sin= np.cos(orient), np.sin(orient)
    pos[:,0] += cos*v0
    pos[:,1] += sin*v0
 
    pos[pos>L] -= L
    pos[pos<0] += L
 
    qv.set_offsets(pos)
    qv.set_UVC(cos, sin,orient)
    return qv,
 
anim = FuncAnimation(fig,animate,np.arange(1, 200),interval=1, blit=True)
plt.show()
'''


import matplotlib.pyplot as plt
import numpy as np

"""
Create Your Own Active Matter Simulation (With Python)
Philip Mocz (2021) Princeton University, @PMocz

Simulate Viscek model for flocking birds

"""


def main():
    """Finite Volume simulation"""

    # Simulation parameters
    v0 = 1.0  # velocity
    eta = 0.5  # random fluctuation in angle (in radians)
    L = 10  # size of box
    R = 1  # interaction radius
    dt = 0.2  # time step
    Nt = 200  # number of time steps
    N = 500  # number of birds
    plotRealTime = True

    # Initialize
    np.random.seed(17)  # set the random number generator seed

    # bird positions
    x = np.random.rand(N, 1) * L
    y = np.random.rand(N, 1) * L

    # bird velocities
    theta = 2 * np.pi * np.random.rand(N, 1)
    vx = v0 * np.cos(theta)
    vy = v0 * np.sin(theta)

    # Prep figure
    fig = plt.figure(figsize=(4, 4), dpi=80)
    ax = plt.gca()

    # Simulation Main Loop
    for i in range(Nt):
        # move
        x += vx * dt
        y += vy * dt

        # apply periodic BCs
        x = x % L
        y = y % L

        # find mean angle of neighbors within R
        mean_theta = theta
        for b in range(N):
            neighbors = (x - x[b]) ** 2 + (y - y[b]) ** 2 < R**2
            sx = np.sum(np.cos(theta[neighbors]))
            sy = np.sum(np.sin(theta[neighbors]))
            mean_theta[b] = np.arctan2(sy, sx)

        # add random perturbations
        theta = mean_theta + eta * (np.random.rand(N, 1) - 0.5)

        # update velocities
        vx = v0 * np.cos(theta)
        vy = v0 * np.sin(theta)

        # plot in real time
        if plotRealTime or (i == Nt - 1):
            plt.cla()
            plt.quiver(x, y, vx, vy)
            ax.set(xlim=(0, L), ylim=(0, L))
            ax.set_aspect("equal")
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            plt.pause(0.001)

    # Save figure
    plt.savefig("activematter.png", dpi=240)
    plt.show()

    return 0


if __name__ == "__main__":
    main()
