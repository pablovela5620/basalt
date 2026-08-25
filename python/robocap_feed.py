"""RoboCap -> VIT feed contract shared by both drivers.

Owns the pieces where the file-fed gate (drive_files) and the catalog driver
(drive_catalog) must agree exactly: the camera-to-IMU clock offset, frameset
assembly mirroring basalt's src/io/dataset_io_robocap.cpp, and the IMU-lead
feed discipline. Dependency-light on purpose — the file-fed gate must not
import the catalog stack (rerun/pyarrow/torch).
"""

from __future__ import annotations

from jaxtyping import Float64
from numpy import ndarray

from vit_binding import Tracker

CAMERA_TO_IMU_OFFSET_NS = 14_902_432  # basalt kCameraToImuOffsetNs (dataset_io_robocap.cpp)
FRAMESET_TOLERANCE_NS = 1_000_000

ImuSamples = list[tuple[int, Float64[ndarray, "3"], Float64[ndarray, "3"]]]
"""Paired (t_ns, gyro_xyz rad/s, accel_xyz m/s^2) rows on the tracker clock."""


def build_framesets(camera_frames: list[list[tuple[int, int]]]) -> list[tuple[int, list[int]]]:
    """Mirror the basalt reader: anchor camera 0, nearest match per camera, median + offset.

    Each camera's frames are (opaque_id, timestamp_ns) in time order; the opaque
    id is not used here (a pts for file feeds, a decode index for the catalog feed).
    Returns (timestamp_ns_on_imu_clock, per-camera frame index) per complete frameset.
    """
    framesets: list[tuple[int, list[int]]] = []
    cursors: list[int] = [0] * len(camera_frames)
    for anchor_index, (_, anchor_ns) in enumerate(camera_frames[0]):
        selected: list[int] = [anchor_index]
        complete: bool = True
        for camera in range(1, len(camera_frames)):
            frames: list[tuple[int, int]] = camera_frames[camera]
            index: int = cursors[camera]
            if index >= len(frames):
                complete = False
                break
            while index + 1 < len(frames) and abs(frames[index + 1][1] - anchor_ns) <= abs(frames[index][1] - anchor_ns):
                index += 1
            if abs(frames[index][1] - anchor_ns) > FRAMESET_TOLERANCE_NS:
                if frames[index][1] < anchor_ns:
                    cursors[camera] = index + 1
                complete = False
                break
            cursors[camera] = index
            selected.append(index)
        if not complete:
            continue
        times: list[int] = sorted(camera_frames[camera][selected[camera]][1] for camera in range(len(camera_frames)))
        middle: int = len(times) // 2
        median: int = times[middle] if len(times) % 2 == 1 else times[middle - 1] + (times[middle] - times[middle - 1]) // 2
        framesets.append((median + CAMERA_TO_IMU_OFFSET_NS, selected))
    return framesets


def feed_imu_lead(tracker: Tracker, imu: ImuSamples, cursor: int, frame_t_ns: int) -> int:
    """Push IMU samples until one at/after frame_t_ns went in; return the new cursor.

    The backend integrates up to the frame time, and in deterministic mode
    push_img blocks until it can — feeding only samples <= frame_t_ns deadlocks
    on the very first frameset.
    """
    while cursor < len(imu) and (cursor == 0 or imu[cursor - 1][0] <= frame_t_ns):
        t_ns, gyro_xyz, accel_xyz = imu[cursor]
        tracker.push_imu(int(t_ns), gyro_xyz, accel_xyz)
        cursor += 1
    return cursor
