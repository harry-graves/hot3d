import os
import numpy as np
import rerun as rr
from tqdm import tqdm

from dataset_api import Hot3dDataProvider
from data_loaders.mano_layer import MANOHandModel
from data_loaders.loader_object_library import ObjectLibrary
from data_loaders.loader_object_library import load_object_library
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions

from eval_modules.rerun import log_pose
from eval_modules.plotting import plot_movement_timeline

def calculate_3d_velocity(current_position, previous_position, time2, time1):

    if previous_position is None or time1 is None or time2 is None or time2 == time1:
        return None

    # Convert timestamps to seconds such that velocity unit is m/s
    time1 = time1/1e9
    time2 = time2/1e9
    
    velocity = np.linalg.norm(current_position - previous_position) / np.abs(time2 - time1)

    return velocity

def main(hot3d_dataset_path, velocity_threshold, window_size):
    """
    velocity_threshold - threshold for the average velocity of the window for an object to be considered as moving
    window_size - number of frames (running at 30fps) to calculate the average velocity over
    """
    sequence_path = os.path.join(hot3d_dataset_path, "P0001_550ea2ac")
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
    #device_pose_provider = hot3d_data_provider.device_pose_data_provider
    device_data_provider = hot3d_data_provider.device_data_provider
    hand_data_provider = hot3d_data_provider.mano_hand_data_provider if hot3d_data_provider.mano_hand_data_provider is not None else hot3d_data_provider.umetrack_hand_data_provider
    object_pose_data_provider = hot3d_data_provider.object_pose_data_provider

    #image_stream_ids = device_data_provider.get_image_stream_ids()

    # Sample timestamps at 30Hz
    timestamps = device_data_provider.get_sequence_timestamps()
    sampling_frequency = 30  # Hz
    sampling_interval_ns = int(1e9 / sampling_frequency)  # Convert Hz to nanoseconds
    selected_timestamps = [timestamps[0]]  # Start with the first timestamp
    for ts in timestamps:
        if ts >= selected_timestamps[-1] + sampling_interval_ns:
            selected_timestamps.append(ts)

    # Store previous object positions, timestamps and movement statuses
    previous_positions = {}
    previous_timestamps = {}
    objects_status = {}
    object_cache_status = {}
    velocity_history = {}

    rr.init("Moving Objects")

    # For each object, apply moving average smoothing to velocity
    for timestamp in tqdm(selected_timestamps):

        rr.set_time_nanos("synchronization_time", int(timestamp))
        rr.set_time_sequence("timestamp", timestamp)

        object_poses_with_dt = object_pose_data_provider.get_pose_at_timestamp(
            timestamp_ns=timestamp,
            time_query_options=TimeQueryOptions.CLOSEST,
            time_domain=TimeDomain.TIME_CODE,
        )
        hand_poses_with_dt = hand_data_provider.get_pose_at_timestamp(
            timestamp_ns=timestamp,
            time_query_options=TimeQueryOptions.CLOSEST,
            time_domain=TimeDomain.TIME_CODE,
        )

        if object_poses_with_dt is None or hand_poses_with_dt is None:
            continue

        objects_pose3d_collection = object_poses_with_dt.pose3d_collection
        hand_pose_collection = hand_poses_with_dt.pose3d_collection

        for hand_pose_data in hand_pose_collection.poses.values():
            # Retrieve the handedness of the hand (i.e Left or Right)
            handedness_label = hand_pose_data.handedness_label()

            # Using the triangular mesh representation
            if hand_pose_data.is_left_hand():
                hand_mesh_vertices = hand_data_provider.get_hand_mesh_vertices(hand_pose_data)
                hand_triangles, hand_vertex_normals = hand_data_provider.get_hand_mesh_faces_and_normals(hand_pose_data)
                
                rr.log(
                    f"world/{handedness_label}/mesh_faces",
                    rr.Mesh3D(
                        vertex_positions=hand_mesh_vertices,
                        vertex_normals=hand_vertex_normals,
                        triangle_indices=hand_triangles,
                    ),
                )

            if hand_pose_data.is_right_hand():
                hand_mesh_vertices = hand_data_provider.get_hand_mesh_vertices(hand_pose_data)
                hand_triangles, hand_vertex_normals = hand_data_provider.get_hand_mesh_faces_and_normals(hand_pose_data)
                
                rr.log(
                    f"world/{handedness_label}/mesh_faces",
                    rr.Mesh3D(
                        vertex_positions=hand_mesh_vertices,
                        vertex_normals=hand_vertex_normals,
                        triangle_indices=hand_triangles,
                    ),
                )

            # Keep a mapping to know what object has been seen, and which one has not
        object_uids = object_pose_data_provider.object_uids_with_poses
        logging_status = {x: False for x in object_uids}

        for (
            object_uid,
            object_pose3d,
        ) in objects_pose3d_collection.poses.items():

            object_name = object_library.object_id_to_name_dict[object_uid]
            object_name = object_name + "_" + str(object_uid)
            object_cad_asset_filepath = ObjectLibrary.get_cad_asset_path(
                object_library_folderpath=object_library.asset_folder_name,
                object_id=object_uid,
            )

            log_pose(pose=object_pose3d.T_world_object, label=f"world/objects/{object_name}")
            
            # Mark object has been seen (enable to know which object has been logged or not)
            # I.E and object not logged, has not been seen and will have its entity cleared for rerun
            logging_status[object_uid] = True

            # Link the corresponding 3D object to the pose
            if object_uid not in object_cache_status.keys():
                object_cache_status[object_uid] = True
                rr.log(
                    f"world/objects/{object_name}",
                    rr.Asset3D(
                        path=object_cad_asset_filepath,
                    ),
                )

        # Rerun specifics (if an entity is disapearing, the last status is shown)
        # To compensate that , if some objects are not visible, we clear the entity
        for object_uid, displayed in logging_status.items():
            if not displayed:
                object_name = object_library.object_id_to_name_dict[object_uid]
                object_name = object_name + "_" + str(object_uid)
                rr.log(
                    f"world/objects/{object_name}",
                    rr.Clear.recursive(),
                )
                if object_uid in object_cache_status.keys():
                    del object_cache_status[object_uid]  # We will log the mesh again

        for object_uid, object_pose3d in objects_pose3d_collection.poses.items():

            # Determine which hand is being used
            object_position = object_pose3d.T_world_object.translation()
            hand_object_distance = np.inf
            for hand_pose_data in hand_pose_collection.poses.values():
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
                is_moving = smoothed_velocity > velocity_threshold

                if object_uid not in objects_status:
                    objects_status[object_uid] = {"statuses": [], "timestamps": []}
                objects_status[object_uid]["timestamps"].append(timestamp)

                # Limit interactions to one per hand, hence the break
                if is_moving:
                    if handedness == "left":
                        objects_status[object_uid]["statuses"].append(1)
                    elif handedness == "right":
                        objects_status[object_uid]["statuses"].append(2)
                else:
                    objects_status[object_uid]["statuses"].append(np.nan)

            # Update previous position and timestamp
            previous_positions[object_uid] = object_position
            previous_timestamps[object_uid] = timestamp

    # Show the rerun viewer
    rr.spawn()

    # Plot the timeline
    plot_movement_timeline(objects_status, object_library)

hot3d_dataset_path = "/home/harry/workspace/hot3d/hot3d/dataset"
main(hot3d_dataset_path, velocity_threshold=0.03, window_size=45)