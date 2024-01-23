# TODO: Think of a better name for this file

import time

import matplotlib.pyplot as plt
import numpy as np


class PointPlot:
    def __init__(self, x_lim, y_lim, z_lim, c=[[1, 0, 0]], m="o"):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(projection="3d")
        self.c = c
        self.m = m

        self.ax.set_xlim(x_lim)
        self.ax.set_ylim(y_lim)
        self.ax.set_zlim(z_lim)

        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_zlabel("z")

        self.point_list = []

    def draw_point(self, point):
        self.point_list.append(self.ax.scatter(point[0], point[1], point[2], c=self.c))
        plt.draw()

    def remove_points(self):
        if len(self.point_list) > 0:
            for point in self.point_list:
                point.remove()

            self.point_list = []

    def remove_oldest_point(self):
        if len(self.point_list) > 0:
            self.point_list.pop(0).remove()


def draw_test():
    import time

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    # Make the X, Y meshgrid.
    xs = np.linspace(-1, 1, 50)
    ys = np.linspace(-1, 1, 50)
    X, Y = np.meshgrid(xs, ys)

    # Set the z axis limits, so they aren't recalculated each frame.
    ax.set_zlim(-1, 1)

    # Begin plotting.
    wframe = None
    tstart = time.time()
    for phi in np.linspace(0, 180.0 / np.pi, 100):
        # If a line collection is already remove it before drawing.
        if wframe:
            wframe.remove()
        # Generate data.
        Z = np.cos(2 * np.pi * X + phi) * (1 - np.hypot(X, Y))
        # Plot the new wireframe and pause briefly before continuing.
        wframe = ax.plot_wireframe(X, Y, Z, rstride=2, cstride=2)
        plt.pause(0.001)

    print("Average FPS: %f" % (100 / (time.time() - tstart)))


def blit_test():
    # See https://stackoverflow.com/a/15724978
    # See https://matplotlib.org/stable/users/explain/animations/blitting.html
    # See https://stackoverflow.com/a/38126963
    tic = time.time()
    frame_count = 500

    x = np.linspace(0, 2 * np.pi, 5)
    # x = 0
    # y = 0

    fig, ax = plt.subplots()

    # animated=True tells matplotlib to only draw the artist when we
    # explicitly request it
    (ln,) = ax.plot(x, np.sin(x), "o", animated=True)

    # make sure the window is raised, but the script keeps going
    plt.show(block=False)

    # stop to admire our empty window axes and ensure it is rendered at
    # least once.
    #
    # We need to fully draw the figure at its final size on the screen
    # before we continue on so that :
    #  a) we have the correctly sized and drawn background to grab
    #  b) we have a cached renderer so that ``ax.draw_artist`` works
    # so we spin the event loop to let the backend process any pending operations
    plt.pause(0.1)

    # get copy of entire figure (everything inside fig.bbox) sans animated artist
    bg = fig.canvas.copy_from_bbox(fig.bbox)
    # draw the animated artist, this uses a cached renderer
    ax.draw_artist(ln)
    # show the result to the screen, this pushes the updated RGBA buffer from the
    # renderer to the GUI framework so you can see it
    fig.canvas.blit(fig.bbox)

    for j in range(frame_count):
        # reset the background back in the canvas state, screen unchanged
        fig.canvas.restore_region(bg)
        # update the artist, neither the canvas state nor the screen have changed
        ln.set_ydata(np.sin(x + (j / 100) * np.pi))
        # re-render the artist, updating the canvas state, but not the screen
        ax.draw_artist(ln)
        # copy the image to the GUI state, but screen might not be changed yet
        fig.canvas.blit(fig.bbox)
        # flush any pending GUI events, re-painting the screen if needed
        fig.canvas.flush_events()
        # you can put a pause in if you want to slow things down
        # plt.pause(.1)

    print(f"Average FPS: {frame_count / (time.time() - tic)}")


def spiral_test():
    lim = [-2, 2]
    pp = PointPlot(lim, lim, lim)
    theta = 0
    theta_step = 0.1

    tracked_points = 5

    while True:
        x = np.cos(theta)
        y = np.sin(theta)
        z = theta - np.pi

        if theta > 2 * np.pi:
            theta = theta % 2 * np.pi

        pp.draw_point([x, y, z])
        plt.pause(0.001)

        if tracked_points > 1:
            tracked_points -= 1
        else:
            pp.remove_oldest_point()

        theta += theta_step


if __name__ == "__main__":
    # draw_test()
    blit_test()
