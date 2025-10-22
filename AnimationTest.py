import matplotlib.pylab as plt
from matplotlib.animation import FuncAnimation 
import numpy as np


t = np.linspace(0, 10, 100)
y = np.sin(t)

fig, axis = plt.subplots()
axis.set_xlim([min(t), max(t)])
axis.set_ylim([-2, 2])

animated_plot, = axis.plot([], [])

#print(animated_plot)

def update(frame):

    animated_plot.set_data(t[:frame], y[:frame])

    return


animation = FuncAnimation(
    fig = fig, 
    func = update,
    frames = len(t),
    interval = 25,
    repeat = False,
)

animation.save("Animation_sin.gif")

plt.show()