#!/usr/bin/env python3

"""
2D Controller Class to be used for the CARLA waypoint follower demo.
"""

from utils import *




x = np.linspace(0, 200,200)
y = (np.sin(x) * 2) + 4

x = np.arange(-20, 20, 1)
y = x**2 #np.ones(len(x))*5


v = np.ones(len(x))
waypoints = np.array([x, y])
waypoints = np.moveaxis(waypoints, 0, -1)

init_state = State(path = waypoints, X=5, Y=-5, th=50, V=5, deg = True)

# print(waypoints[:,0][:50])
controller = Controller2D(waypoints, init_state)

journey = 200
log_every = 1

fig = figure()

for i in range(journey):
    if i % log_every == 0 :
        if waypoints[-1, 0] < int(controller.state.x) : break

        print(controller)
        # render((waypoints[:, 0], waypoints[:, 1]), player = [controller.state.x, controller.state.y])

        render(waypoints, controller, fig, show = True, arrow = True, clear_first = True, pause = 0.01)



        # plt.close()

    controller.update_controls()
plt.savefig(datetime.now().strftime("%H_%M_%S") + ".png")


