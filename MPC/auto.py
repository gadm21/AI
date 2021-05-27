import matplotlib.pyplot as plt
import numpy as np

num_rows = 10
num_cols = 1
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
labels = ["Label {}".format(i+1) for i in range(num_rows)]

def myplot(i, ax):
    ax.plot(np.arange(10), np.arange(10)**i, color=colors[i])
    ax.set_ylabel(labels[i])


fig, axs = plt.subplots(num_rows, num_cols, sharex=True)
for i in range(num_rows):
     myplot(i, axs[i])


def on_click(event):
    axes = event.inaxes
    print("axes:", axes)
    if not axes: return
    inx = list(fig.axes).index(axes)
    fig2 = plt.figure()
    ax = fig2.add_subplot(111)
    myplot(inx, ax)
    fig2.show()

fig.canvas.mpl_connect('button_press_event', on_click)

plt.show()