import os
from dataset_api import Hot3dDataProvider
from data_loaders.loader_object_library import load_object_library
from data_loaders.mano_layer import MANOHandModel
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

def plot_movement_timeline(left_hand_interactions, left_hand_timestamps, right_hand_interactions, right_hand_timestamps, object_library):
    # Create y-values for both left and right hand interactions
    left_line = [1 if obj is not None else np.nan for obj in left_hand_interactions]
    right_line = [2 if obj is not None else np.nan for obj in right_hand_interactions]

    # Convert timestamps to seconds
    left_hand_timestamps = (np.array(left_hand_timestamps) - left_hand_timestamps[0]) / 1e9
    right_hand_timestamps = (np.array(right_hand_timestamps) - right_hand_timestamps[0]) / 1e9

    # Plot the left and right hand interactions
    plt.plot(left_hand_timestamps, left_line, label='Left Hand Interaction', color = "blue", linewidth=3)
    plt.plot(right_hand_timestamps, right_line, label='Right Hand Interaction', color = "red", linewidth=3)
    
    # Adding labels and title
    plt.xlabel("Timestamps (s)")
    plt.yticks([1, 2], ['Left', 'Right'])
    plt.title("HOT3D: Hand-Object Interactions throughout a sequence")
    #plt.legend()
    plt.ylim([0,3])
    plt.show()

def calculate_3d_velocity(current_position, previous_position, time2, time1):

    if previous_position is None or time1 is None or time2 is None or time2 == time1:
        return None

    # Convert timestamps to seconds to unit is m/s
    time1 = time1/1e9
    time2 = time2/1e9
    
    velocity = np.linalg.norm(current_position - previous_position) / np.abs(time2 - time1)

    return velocity

# Threshold for movement
VELOCITY_THRESHOLD = 0.05 # m/s

# Store previous positions and timestamps
previous_positions = {}
previous_timestamps = {}

# Store movement status for each object
object_interactions = {}
left_hand_interactions = []
left_hand_timestamps = []
right_hand_interactions = []
right_hand_timestamps = []

home = os.path.expanduser("~")
hot3d_dataset_path = home + "/workspace/hot3d/hot3d/dataset"
sequence_path = os.path.join(hot3d_dataset_path, "P0003_c701bd11")
object_library_path = os.path.join(hot3d_dataset_path, "assets")
mano_hand_model_path = os.path.join(hot3d_dataset_path, "mano_v1_2/models")

if not os.path.exists(sequence_path) or not os.path.exists(object_library_path):
    print("Invalid input sequence or library path.")
    raise ValueError("Please update the path to valid values for your system.")

# Initialize object library and hand model
object_library = load_object_library(object_library_folderpath=object_library_path)
mano_hand_model = MANOHandModel(mano_hand_model_path) if mano_hand_model_path else None

# Initialize HOT3D data providers
hot3d_data_provider = Hot3dDataProvider(
    sequence_folder=sequence_path,
    object_library=object_library,
    mano_hand_model=mano_hand_model,
)
device_pose_provider = hot3d_data_provider.device_pose_data_provider
device_data_provider = hot3d_data_provider.device_data_provider
hand_data_provider = hot3d_data_provider.mano_hand_data_provider if hot3d_data_provider.mano_hand_data_provider is not None else hot3d_data_provider.umetrack_hand_data_provider
object_pose_data_provider = hot3d_data_provider.object_pose_data_provider

# Sample timestamps at 30Hz
timestamps = device_data_provider.get_sequence_timestamps()
sampling_frequency = 30  # Hz
sampling_interval_ns = int(1e9 / sampling_frequency)  # Convert Hz to nanoseconds
selected_timestamps = [timestamps[0]]  # Start with the first timestamp
for ts in timestamps:
    if ts >= selected_timestamps[-1] + sampling_interval_ns:
        selected_timestamps.append(ts)

# Initialize parameters for moving average smoothing
window_size = 30  # The number of past velocity readings to average
velocity_history = {}  # Store velocity history for each object

# For each object, apply moving average smoothing to velocity
for timestamp in tqdm(selected_timestamps):
    object_poses_with_dt = object_pose_data_provider.get_pose_at_timestamp(
        timestamp_ns=timestamp,
        time_query_options=TimeQueryOptions.CLOSEST,
        time_domain=TimeDomain.TIME_CODE,
    )
    headset_pose3d_with_dt = device_pose_provider.get_pose_at_timestamp(
        timestamp_ns=timestamp,
        time_query_options=TimeQueryOptions.CLOSEST,
        time_domain=TimeDomain.TIME_CODE,
    )
    hand_poses_with_dt = hand_data_provider.get_pose_at_timestamp(
        timestamp_ns=timestamp,
        time_query_options=TimeQueryOptions.CLOSEST,
        time_domain=TimeDomain.TIME_CODE,
    )

    if object_poses_with_dt is None or headset_pose3d_with_dt is None or hand_poses_with_dt is None:
        continue

    objects_pose3d_collection = object_poses_with_dt.pose3d_collection
    hand_pose_collection = hand_poses_with_dt.pose3d_collection



    left_hand_interactions.append(None)
    left_hand_timestamps.append(timestamp)
    right_hand_interactions.append(None)
    right_hand_timestamps.append(timestamp)

    for object_uid, object_pose3d in objects_pose3d_collection.poses.items():
        object_name = object_library.object_id_to_name_dict[object_uid]

        object_position = object_pose3d.T_world_object.translation()
        hand_object_distance = np.inf
        for hand_pose_data in hand_pose_collection.poses.values():
            # Retrieve the handedness of the hand (i.e Left or Right)
            handedness_label = hand_pose_data.handedness_label()
            wrist_position = hand_pose_data.wrist_pose.translation()
            new_distance = np.linalg.norm(object_position - wrist_position)
            if new_distance < hand_object_distance:
                hand_object_distance = new_distance
                handedness = handedness_label

        # Get previous position and timestamp
        previous_position = previous_positions.get(object_uid)
        previous_timestamp = previous_timestamps.get(object_uid)

        # Calculate velocity
        velocity = calculate_3d_velocity(object_position, previous_position, timestamp, previous_timestamp)

        if velocity is not None:
            # Store the velocity history
            if object_uid not in velocity_history:
                velocity_history[object_uid] = []

            # Append current velocity to history
            velocity_history[object_uid].append(velocity)

            # Keep only the last 'window_size' velocities
            if len(velocity_history[object_uid]) > window_size:
                velocity_history[object_uid].pop(0)

            # Calculate moving average of velocity
            smoothed_velocity = np.mean(velocity_history[object_uid])

            # Set movement status based on smoothed velocity
            is_moving = smoothed_velocity > VELOCITY_THRESHOLD

            # Limit interactions to one per hand, hence the break
            if is_moving and handedness == "left":
                left_hand_interactions[-1] = object_name
                break
            elif is_moving and handedness == "right":
                right_hand_interactions[-1] = object_name
                break

        # Update previous position and timestamp
        previous_positions[object_uid] = object_position
        previous_timestamps[object_uid] = timestamp

# Call the function to plot the timeline
plot_movement_timeline(left_hand_interactions, left_hand_timestamps, right_hand_interactions, right_hand_timestamps, object_library)