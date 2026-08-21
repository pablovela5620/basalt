"""ctypes binding for the VIT C API exported by libbasalt.so.

Mirrors thirdparty/vit/vit_interface.h. Only the surface a batch driver needs:
create/start/stop, push imu/img, programmatic calibration, pop poses.
"""

from __future__ import annotations

import ctypes
from collections.abc import Iterator
from enum import IntEnum
from pathlib import Path

import numpy as np

DISTORTION_MAX_COUNT = 32


class Result(IntEnum):
    SUCCESS = 0
    ERROR_INVALID_VERSION = -1
    ERROR_INVALID_VALUE = -2
    ERROR_ALLOCATION_FAILURE = -3
    ERROR_NOT_SUPPORTED = -4
    ERROR_NOT_ENABLED = -5


class ImageFormat(IntEnum):
    L8 = 1
    L16 = 2
    R8G8B8 = 3


class CameraDistortion(IntEnum):
    NONE = 0
    RT4 = 1
    RT5 = 2
    RT8 = 3
    KB4 = 4


class Config(ctypes.Structure):
    _fields_ = [
        ("file", ctypes.c_char_p),
        ("cam_count", ctypes.c_uint32),
        ("imu_count", ctypes.c_uint32),
        ("show_ui", ctypes.c_bool),
    ]


class ImuSample(ctypes.Structure):
    _fields_ = [
        ("timestamp", ctypes.c_int64),
        ("ax", ctypes.c_float),
        ("ay", ctypes.c_float),
        ("az", ctypes.c_float),
        ("wx", ctypes.c_float),
        ("wy", ctypes.c_float),
        ("wz", ctypes.c_float),
    ]


class Mask(ctypes.Structure):
    _fields_ = [("x", ctypes.c_float), ("y", ctypes.c_float), ("w", ctypes.c_float), ("h", ctypes.c_float)]


class ImgSample(ctypes.Structure):
    _fields_ = [
        ("cam_index", ctypes.c_uint32),
        ("timestamp", ctypes.c_int64),
        ("data", ctypes.POINTER(ctypes.c_uint8)),
        ("width", ctypes.c_uint32),
        ("height", ctypes.c_uint32),
        ("stride", ctypes.c_uint32),
        ("size", ctypes.c_uint32),
        ("format", ctypes.c_int),
        ("mask_count", ctypes.c_uint32),
        ("masks", ctypes.POINTER(Mask)),
    ]


class PoseData(ctypes.Structure):
    _fields_ = [
        ("timestamp", ctypes.c_int64),
        ("px", ctypes.c_float),
        ("py", ctypes.c_float),
        ("pz", ctypes.c_float),
        ("ox", ctypes.c_float),
        ("oy", ctypes.c_float),
        ("oz", ctypes.c_float),
        ("ow", ctypes.c_float),
        ("vx", ctypes.c_float),
        ("vy", ctypes.c_float),
        ("vz", ctypes.c_float),
    ]


class CameraCalibration(ctypes.Structure):
    _fields_ = [
        ("camera_index", ctypes.c_uint32),
        ("width", ctypes.c_int),
        ("height", ctypes.c_int),
        ("frequency", ctypes.c_double),
        ("fx", ctypes.c_double),
        ("fy", ctypes.c_double),
        ("cx", ctypes.c_double),
        ("cy", ctypes.c_double),
        ("model", ctypes.c_int),
        ("distortion_count", ctypes.c_uint32),
        ("distortion", ctypes.c_double * DISTORTION_MAX_COUNT),
        ("transform", ctypes.c_double * 16),
    ]


class InertialCalibration(ctypes.Structure):
    _fields_ = [
        ("transform", ctypes.c_double * 9),
        ("offset", ctypes.c_double * 3),
        ("bias_std", ctypes.c_double * 3),
        ("noise_std", ctypes.c_double * 3),
    ]


class ImuCalibration(ctypes.Structure):
    _fields_ = [
        ("imu_index", ctypes.c_uint32),
        ("frequency", ctypes.c_double),
        ("accel", InertialCalibration),
        ("gyro", InertialCalibration),
    ]


def load(lib_path: Path) -> ctypes.CDLL:
    """Load libbasalt and declare the vit_* signatures the Tracker uses."""
    lib = ctypes.CDLL(str(lib_path))
    tracker_p = ctypes.c_void_p
    pose_p = ctypes.c_void_p
    signatures: dict[str, tuple[list, object]] = {
        "vit_api_get_version": ([ctypes.POINTER(ctypes.c_uint32)] * 3, ctypes.c_int),
        "vit_tracker_create": ([ctypes.POINTER(Config), ctypes.POINTER(tracker_p)], ctypes.c_int),
        "vit_tracker_destroy": ([tracker_p], None),
        "vit_tracker_start": ([tracker_p], ctypes.c_int),
        "vit_tracker_stop": ([tracker_p], ctypes.c_int),
        "vit_tracker_push_imu_sample": ([tracker_p, ctypes.POINTER(ImuSample)], ctypes.c_int),
        "vit_tracker_push_img_sample": ([tracker_p, ctypes.POINTER(ImgSample)], ctypes.c_int),
        "vit_tracker_add_camera_calibration": ([tracker_p, ctypes.POINTER(CameraCalibration)], ctypes.c_int),
        "vit_tracker_add_imu_calibration": ([tracker_p, ctypes.POINTER(ImuCalibration)], ctypes.c_int),
        "vit_tracker_pop_pose": ([tracker_p, ctypes.POINTER(pose_p)], ctypes.c_int),
        "vit_pose_get_data": ([pose_p, ctypes.POINTER(PoseData)], ctypes.c_int),
        "vit_pose_destroy": ([pose_p], None),
    }
    for name, (argtypes, restype) in signatures.items():
        function = getattr(lib, name)
        function.argtypes = argtypes
        function.restype = restype
    return lib


def check(result: int, operation: str) -> None:
    if result != Result.SUCCESS:
        raise RuntimeError(f"{operation} failed: {Result(result).name}")


class Tracker:
    """Thin ownership wrapper over one vit_tracker_t."""

    def __init__(self, lib_path: Path, config_file: str | None, cam_count: int) -> None:
        self.lib = load(lib_path)
        self._config = Config(
            file=None if config_file is None else config_file.encode(),
            cam_count=cam_count,
            imu_count=1,
            show_ui=False,
        )
        self._handle = ctypes.c_void_p()
        check(self.lib.vit_tracker_create(ctypes.byref(self._config), ctypes.byref(self._handle)), "create")

    def add_camera_calibration(
        self,
        index: int,
        *,
        width: int,
        height: int,
        frequency: float,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        distortion: list[float],
        T_imu_cam: np.ndarray,
    ) -> None:
        calibration = CameraCalibration(
            camera_index=index,
            width=width,
            height=height,
            frequency=frequency,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            model=CameraDistortion.KB4,
            distortion_count=len(distortion),
        )
        calibration.distortion[: len(distortion)] = distortion
        calibration.transform[:] = np.asarray(T_imu_cam, dtype=np.float64).reshape(16).tolist()
        check(self.lib.vit_tracker_add_camera_calibration(self._handle, ctypes.byref(calibration)), "add_camera_calibration")

    def add_imu_calibration(
        self,
        *,
        frequency: float,
        gyro_noise_std: float,
        gyro_bias_std: float,
        accel_noise_std: float,
        accel_bias_std: float,
    ) -> None:
        identity = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        calibration = ImuCalibration(imu_index=0, frequency=frequency)
        for channel, noise, bias in ((calibration.gyro, gyro_noise_std, gyro_bias_std), (calibration.accel, accel_noise_std, accel_bias_std)):
            channel.transform[:] = identity
            channel.offset[:] = [0.0] * 3
            channel.noise_std[:] = [noise] * 3
            channel.bias_std[:] = [bias] * 3
        check(self.lib.vit_tracker_add_imu_calibration(self._handle, ctypes.byref(calibration)), "add_imu_calibration")

    def start(self) -> None:
        check(self.lib.vit_tracker_start(self._handle), "start")

    def stop(self) -> None:
        check(self.lib.vit_tracker_stop(self._handle), "stop")

    def push_imu(self, t_ns: int, gyro_xyz: np.ndarray, accel_xyz: np.ndarray) -> None:
        sample = ImuSample(
            timestamp=t_ns,
            ax=accel_xyz[0],
            ay=accel_xyz[1],
            az=accel_xyz[2],
            wx=gyro_xyz[0],
            wy=gyro_xyz[1],
            wz=gyro_xyz[2],
        )
        check(self.lib.vit_tracker_push_imu_sample(self._handle, ctypes.byref(sample)), "push_imu")

    def push_img(self, cam_index: int, t_ns: int, data_ptr: int, *, width: int, height: int, stride: int) -> None:
        sample = ImgSample(
            cam_index=cam_index,
            timestamp=t_ns,
            data=ctypes.cast(ctypes.c_void_p(data_ptr), ctypes.POINTER(ctypes.c_uint8)),
            width=width,
            height=height,
            stride=stride,
            size=stride * height,
            format=ImageFormat.L8,
            mask_count=0,
            masks=None,
        )
        check(self.lib.vit_tracker_push_img_sample(self._handle, ctypes.byref(sample)), "push_img")

    def poses(self) -> Iterator[tuple[int, float, float, float, float, float, float, float]]:
        """Drain currently available poses as (t_ns, px, py, pz, qw, qx, qy, qz)."""
        while True:
            pose = ctypes.c_void_p()
            check(self.lib.vit_tracker_pop_pose(self._handle, ctypes.byref(pose)), "pop_pose")
            if not pose:  # empty queue: SUCCESS with a null pose
                return
            data = PoseData()
            check(self.lib.vit_pose_get_data(pose, ctypes.byref(data)), "pose_get_data")
            self.lib.vit_pose_destroy(pose)
            yield (data.timestamp, data.px, data.py, data.pz, data.ow, data.ox, data.oy, data.oz)

    def close(self) -> None:
        if self._handle:
            self.lib.vit_tracker_destroy(self._handle)
            self._handle = ctypes.c_void_p()
