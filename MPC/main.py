#!/usr/bin/env python3

"""
2D Controller Class to be used for the CARLA waypoint follower demo.
"""

from utils import *




x = np.linspace(0, 200,200)
y = (np.sin(x) * 2) + 4

x = np.arange(-20, 20, 1)
y = 3*x #np.ones(len(x))*5


v = np.ones(len(x))
waypoints = np.array([x, y])
waypoints = np.moveaxis(waypoints, 0, -1)

init_state = state(path = waypoints, X=5, Y=-5, th=50, V=15, deg = True)

# print(waypoints[:,0][:50])
controller = Controller2D(waypoints, init_state)

journey = 100
log_every = 2
plt.plot(waypoints[:, 0], waypoints[:, 1])

for i in range(journey):
    if i % log_every == 0 :
        print(controller)
        # render((waypoints[:, 0], waypoints[:, 1]), player = [controller.state.x, controller.state.y])

        plt.scatter(controller.state.x, controller.state.y, color = 'r', s = 3)

        plt.show(block = False)
        plt.pause(0.1)

    controller.update_controls()
plt.savefig(datetime.now().strftime("%H_%M_%S") + ".png")


