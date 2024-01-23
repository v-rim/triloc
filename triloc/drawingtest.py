import time

import matplotlib.pyplot as plt
import numpy as np


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
    frame_count = 5000

    x = np.array([0])  # np.linspace(0, 2 * np.pi, 5)
    # x = 0
    # y = 0

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    # animated=True tells matplotlib to only draw the artist when we
    # explicitly request it
    (ln,) = ax.plot(x, np.sin(x), 0, "o", animated=True)

    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    ax.set_zlim([-0.5, 3.5])

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

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
        ln.set_data(np.cos(x + (j / 100) * np.pi), np.sin(x + (j / 100) * np.pi))
        ln.set_3d_properties(x + 3 * (j % 200.0) / 200.0, "z")
        # re-render the artist, updating the canvas state, but not the screen
        ax.draw_artist(ln)
        # copy the image to the GUI state, but screen might not be changed yet
        fig.canvas.blit(fig.bbox)
        # flush any pending GUI events, re-painting the screen if needed
        fig.canvas.flush_events()
        # you can put a pause in if you want to slow things down
        # plt.pause(.1)

    print(f"Average FPS: {frame_count / (time.time() - tic)}")


if __name__ == "__main__":
    # draw_test()
    blit_test()
