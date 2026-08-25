"""RoboCap -> VIT feed contract shared by both drivers.

Owns the pieces where the file-fed gate (drive_files) and the catalog driver
(drive_catalog) must agree exactly: the camera-to-IMU clock offset, frameset
assembly mirroring basalt's src/io/dataset_io_robocap.cpp, and the IMU-lead
feed discipline. Dependency-light on purpose — the file-fed gate must not
import the catalog stack (rerun/pyarrow/torch).
"""

from __future__ import annotations

from typing import NamedTuple, TypeAlias

from jaxtyping import Float64
from numpy import ndarray

from vit_binding import Tracker

CAMERA_TO_IMU_OFFSET_NS = 14_902_432  # basalt kCameraToImuOffsetNs (dataset_io_robocap.cpp)
FRAMESET_TOLERANCE_NS = 1_000_000


class FrameStamp(NamedTuple):
    """One camera frame as seen by the synchronizer."""

    opaque_id: int
    """Per-camera handle for fetching the pixels later: a pts for file feeds, a decode index for the catalog feed."""
    t_ns: int
    """Capture timestamp on the camera clock."""


class Frameset(NamedTuple):
    """One synchronized multi-camera capture."""

    t_ns: int
    """Median camera timestamp shifted onto the IMU clock."""
    frame_indices: list[int]
    """Index into each camera's frame list, in camera order."""


class ImuSample(NamedTuple):
    """One paired IMU reading on the tracker clock."""

    t_ns: int
    """Sample timestamp."""
    gyro_xyz: Float64[ndarray, "3"]
    """Angular velocity, rad/s."""
    accel_xyz: Float64[ndarray, "3"]
    """Linear acceleration (interpolated onto the gyro timestamp), m/s^2."""


ImuSamples: TypeAlias = list[ImuSample]
"""Time-ordered paired IMU rows."""


def build_framesets(camera_frames: list[list[FrameStamp]]) -> list[Frameset]:
    """Mirror the basalt reader: anchor camera 0, nearest match per camera, median + offset.

    Each camera's frames must be in time order. A frameset is complete only when
    every camera has a frame within FRAMESET_TOLERANCE_NS of the anchor's.
    """
    framesets: list[Frameset] = []
    cursors: list[int] = [0] * len(camera_frames)
    for anchor_index, anchor in enumerate(camera_frames[0]):
        selected: list[int] = [anchor_index]
        complete: bool = True
        for camera in range(1, len(camera_frames)):
            frames: list[FrameStamp] = camera_frames[camera]
            index: int = cursors[camera]
            if index >= len(frames):
                complete = False
                break
            while index + 1 < len(frames) and abs(frames[index + 1].t_ns - anchor.t_ns) <= abs(frames[index].t_ns - anchor.t_ns):
                index += 1
            if abs(frames[index].t_ns - anchor.t_ns) > FRAMESET_TOLERANCE_NS:
                if frames[index].t_ns < anchor.t_ns:
                    cursors[camera] = index + 1
                complete = False
                break
            cursors[camera] = index
            selected.append(index)
        if not complete:
            continue
        times: list[int] = sorted(camera_frames[camera][selected[camera]].t_ns for camera in range(len(camera_frames)))
        middle: int = len(times) // 2
        median: int = times[middle] if len(times) % 2 == 1 else times[middle - 1] + (times[middle] - times[middle - 1]) // 2
        framesets.append(Frameset(median + CAMERA_TO_IMU_OFFSET_NS, selected))
    return framesets


def feed_imu_lead(tracker: Tracker, imu: ImuSamples, cursor: int, frame_t_ns: int) -> int:
    """Push IMU samples until one at/after frame_t_ns went in; return the new cursor.

    The backend integrates up to the frame time, and in deterministic mode
    push_img blocks until it can — feeding only samples <= frame_t_ns deadlocks
    on the very first frameset.
    """
    while cursor < len(imu) and (cursor == 0 or imu[cursor - 1].t_ns <= frame_t_ns):
        sample: ImuSample = imu[cursor]
        tracker.push_imu(sample.t_ns, sample.gyro_xyz, sample.accel_xyz)
        cursor += 1
    return cursor
