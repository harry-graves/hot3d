import numpy as np
import rerun as rr
from projectaria_tools.core.sophus import SE3
from projectaria_tools.utils.rerun_helpers import ToTransform3D

def log_pose(
    pose: SE3,
    label: str,
    static=False
) -> None:
    rr.log(label, ToTransform3D(pose, False), static=static)

def log_image(
    image: np.array,
    label: str,
    static=False
) -> None:
    rr.log(label, rr.Image(image), static=static)