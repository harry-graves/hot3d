import matplotlib.pyplot as plt
import time
import numpy as np
from matplotlib import animation

def plot_movement_timeline(objects_moving, object_library):
    global_start_time = min(min(data["timestamps"]) for data in objects_moving.values())
    global_end_time = max(max(data["timestamps"]) for data in objects_moving.values())
    duration = (global_end_time - global_start_time) / 1e9

    fig, ax = plt.subplots()

    for object_uid, data in objects_moving.items():
        object_name = object_library.object_id_to_name_dict[object_uid]
        timestamps = np.array(data["timestamps"])
        timestamps = (timestamps - global_start_time) / 1e9
        statuses = data["statuses"]
        ax.plot(timestamps, statuses, label=object_name, linewidth=5)

    # Labels and title
    ax.set_xlabel("Timestamps (s)")
    ax.set_yticks([1, 2])
    ax.set_yticklabels(['Left', 'Right'])
    ax.set_title("HOT3D: Hand-Object Interactions throughout a sequence")
    ax.legend()
    ax.set_ylim([0, 3])

    # Vertical red line
    vl = ax.axvline(0, ls='-', color='r', lw=2, zorder=10)
    ax.set_xlim(0, duration)

    playing = [False]  # Mutable variable to track play state
    start_time = [None]  # Store the actual wall-clock start time

    def on_press(event):
        if event.key == ' ':
            if playing[0]:
                playing[0] = False
            else:
                playing[0] = True
                start_time[0] = time.time() - (vl.get_xdata()[0])  # Sync to current position

    fig.canvas.mpl_connect('key_press_event', on_press)

    def animate(_):
        if playing[0]:
            elapsed = time.time() - start_time[0]  # Compute real-time elapsed
            elapsed = min(elapsed, duration)  # Stop at duration limit
            vl.set_xdata([elapsed, elapsed])

    ani = animation.FuncAnimation(fig, animate, interval=10)  # Frequent updates for smoothness

    plt.show()