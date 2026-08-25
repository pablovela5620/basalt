#!/usr/bin/env python3

"""Convert RoboCap factory Kalibr files to Basalt calibrations.

One run writes two Basalt cereal JSON files: the four-camera coverage
calibration at ``--output``, and the two-camera front-stereo calibration
beside it (``<stem>-stereo.json`` unless ``--stereo-output`` overrides it).
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tyro
import yaml
from serde import serde, to_dict


@dataclass(frozen=True, slots=True)
class CliArgs:
    """Command-line arguments for RoboCap calibration conversion."""

    factory_calibration: Path
    """Directory containing the unmodified RoboCap factory calibration."""
    output: Path
    """Destination Basalt cereal JSON file for the four coverage cameras."""
    stereo_output: Path | None = None
    """Destination for the front-stereo pair; defaults to `<output stem>-stereo.json`."""
    downscale: int = 2
    """Integer factor the RoboCap reader downscales frames by; intrinsics and
    resolution are scaled to match (1 keeps the native calibration)."""


@dataclass(frozen=True, slots=True)
class CameraSource:
    """One camera in the fixed coverage-camera order."""

    relative_path: Path
    """Kalibr camera-chain path relative to the factory directory."""
    camera_key: str
    """Camera key inside the Kalibr YAML document."""


@serde
@dataclass(frozen=True)
class Transform:
    """Basalt ``T_imu_cam``: one camera's pose in the IMU frame."""

    px: float
    """Translation x, metres. py/pz likewise."""
    py: float
    """Translation y, metres."""
    pz: float
    """Translation z, metres."""
    qx: float
    """Quaternion x. qy/qz/qw likewise (normalized, w last in the JSON)."""
    qy: float
    """Quaternion y."""
    qz: float
    """Quaternion z."""
    qw: float
    """Quaternion w."""


@serde
@dataclass(frozen=True)
class Kb4Intrinsics:
    """Kannala-Brandt (kb4) pinhole intrinsics, pixels."""

    fx: float
    """Focal length x."""
    fy: float
    """Focal length y."""
    cx: float
    """Principal point x."""
    cy: float
    """Principal point y."""
    k1: float
    """KB4 distortion coefficient 1. k2..k4 likewise (resolution-invariant)."""
    k2: float
    """KB4 distortion coefficient 2."""
    k3: float
    """KB4 distortion coefficient 3."""
    k4: float
    """KB4 distortion coefficient 4."""


@serde
@dataclass(frozen=True)
class CameraModel:
    """One camera entry in basalt's ``intrinsics`` array."""

    camera_type: str
    """Basalt camera model tag; always ``kb4`` here."""
    intrinsics: Kb4Intrinsics
    """The model's parameters."""


@serde
@dataclass(frozen=True)
class BasaltCalibration:
    """The ``value0`` payload of a basalt cereal calibration JSON."""

    T_imu_cam: list[Transform]
    """Per-camera pose in the IMU frame, coverage order."""
    intrinsics: list[CameraModel]
    """Per-camera model, same order."""
    resolution: list[list[int]]
    """Per-camera [width, height] after downscale."""
    vignette: list[float]
    """Unused; basalt accepts an empty array."""
    calib_accel_bias: list[float]
    """Accel bias + scale misalignment, zeros (factory calib carries none)."""
    calib_gyro_bias: list[float]
    """Gyro bias + scale misalignment, zeros."""
    imu_update_rate: float
    """IMU rate, Hz."""
    accel_noise_std: list[float]
    """Accelerometer noise density, replicated per axis."""
    gyro_noise_std: list[float]
    """Gyroscope noise density, replicated per axis."""
    accel_bias_std: list[float]
    """Accelerometer bias random walk, replicated per axis."""
    gyro_bias_std: list[float]
    """Gyroscope bias random walk, replicated per axis."""
    cam_time_offset_ns: int
    """Camera-to-IMU time offset; the reader applies its own constant instead."""


@serde
@dataclass(frozen=True)
class BasaltCalibrationFile:
    """Root of the cereal JSON: basalt archives wrap the payload as ``value0``."""

    value0: BasaltCalibration
    """The calibration payload."""


_FRONT_CAMCHAIN: Path = Path("imus_cam_lr_front_extrinsic/imus_cam_lr_front_extrinsic-camchain-imucam.yaml")
COVERAGE_CAMERA_SOURCES: tuple[CameraSource, ...] = (
    CameraSource(Path("imus_cam_l_extrinsic/imus_cam_l_extrinsic-camchain-imucam.yaml"), "cam0"),
    CameraSource(_FRONT_CAMCHAIN, "cam0"),
    CameraSource(_FRONT_CAMCHAIN, "cam1"),
    CameraSource(Path("imus_cam_r_extrinsic/imus_cam_r_extrinsic-camchain-imucam.yaml"), "cam0"),
)
STEREO_CAMERA_SOURCES: tuple[CameraSource, ...] = (
    CameraSource(_FRONT_CAMCHAIN, "cam0"),
    CameraSource(_FRONT_CAMCHAIN, "cam1"),
)
IMU_RELATIVE_PATH: Path = Path("imus_intrinsic/imu_mid_0.yaml")


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML mapping from ``path``.

    Args:
        path: YAML file to read.

    Returns:
        Parsed top-level mapping.
    """
    document: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return document


def _rotation_to_quaternion(rotation: list[list[float]]) -> tuple[float, float, float, float]:
    """Convert a 3×3 rotation matrix to an ``x, y, z, w`` quaternion."""
    trace: float = rotation[0][0] + rotation[1][1] + rotation[2][2]
    if trace > 0.0:
        scale: float = math.sqrt(trace + 1.0) * 2.0
        quaternion: tuple[float, float, float, float] = (
            (rotation[2][1] - rotation[1][2]) / scale,
            (rotation[0][2] - rotation[2][0]) / scale,
            (rotation[1][0] - rotation[0][1]) / scale,
            0.25 * scale,
        )
    else:
        diagonal_index: int = max(range(3), key=lambda index: rotation[index][index])
        if diagonal_index == 0:
            scale = math.sqrt(1.0 + rotation[0][0] - rotation[1][1] - rotation[2][2]) * 2.0
            quaternion = (
                0.25 * scale,
                (rotation[0][1] + rotation[1][0]) / scale,
                (rotation[0][2] + rotation[2][0]) / scale,
                (rotation[2][1] - rotation[1][2]) / scale,
            )
        elif diagonal_index == 1:
            scale = math.sqrt(1.0 + rotation[1][1] - rotation[0][0] - rotation[2][2]) * 2.0
            quaternion = (
                (rotation[0][1] + rotation[1][0]) / scale,
                0.25 * scale,
                (rotation[1][2] + rotation[2][1]) / scale,
                (rotation[0][2] - rotation[2][0]) / scale,
            )
        else:
            scale = math.sqrt(1.0 + rotation[2][2] - rotation[0][0] - rotation[1][1]) * 2.0
            quaternion = (
                (rotation[0][2] + rotation[2][0]) / scale,
                (rotation[1][2] + rotation[2][1]) / scale,
                0.25 * scale,
                (rotation[1][0] - rotation[0][1]) / scale,
            )
    norm: float = math.sqrt(sum(component * component for component in quaternion))
    return tuple(component / norm for component in quaternion)  # type: ignore[return-value]


def _invert_kalibr_transform(matrix: list[list[float]]) -> Transform:
    """Invert Kalibr's camera-from-IMU transform for Basalt."""
    rotation_cam_imu: list[list[float]] = [[float(matrix[row][column]) for column in range(3)] for row in range(3)]
    rotation_imu_cam: list[list[float]] = [[rotation_cam_imu[column][row] for column in range(3)] for row in range(3)]
    translation_cam_imu: list[float] = [float(matrix[row][3]) for row in range(3)]
    translation_imu_cam: list[float] = [
        -sum(rotation_imu_cam[row][column] * translation_cam_imu[column] for column in range(3)) for row in range(3)
    ]
    quaternion_xyzw: tuple[float, float, float, float] = _rotation_to_quaternion(rotation_imu_cam)
    return Transform(
        px=translation_imu_cam[0],
        py=translation_imu_cam[1],
        pz=translation_imu_cam[2],
        qx=quaternion_xyzw[0],
        qy=quaternion_xyzw[1],
        qz=quaternion_xyzw[2],
        qw=quaternion_xyzw[3],
    )


def _camera_calibration(
    factory_calibration: Path, source: CameraSource, downscale: int = 1
) -> tuple[Transform, CameraModel, list[int]]:
    """Convert one fixed RoboCap camera from Kalibr to Basalt fields.

    Args:
        factory_calibration: Directory holding the Kalibr factory calibration.
        source: Camera-chain file and camera key to convert.
        downscale: Integer factor the reader shrinks frames by. Focal lengths
            divide by it; the principal point follows the pixel-center
            convention ``c' = (c + 0.5)/downscale - 0.5``. The kb4 distortion
            coefficients are resolution-invariant.
    """
    document: dict[str, Any] = _load_yaml(factory_calibration / source.relative_path)
    camera: dict[str, Any] = document[source.camera_key]
    matrix: list[list[float]] = camera["T_cam_imu"]
    intrinsics: list[float] = [float(value) for value in camera["intrinsics"]]
    distortion: list[float] = [float(value) for value in camera["distortion_coeffs"]]
    resolution: list[int] = [int(value) // downscale for value in camera["resolution"]]
    camera_model: CameraModel = CameraModel(
        camera_type="kb4",
        intrinsics=Kb4Intrinsics(
            fx=intrinsics[0] / downscale,
            fy=intrinsics[1] / downscale,
            cx=(intrinsics[2] + 0.5) / downscale - 0.5,
            cy=(intrinsics[3] + 0.5) / downscale - 0.5,
            k1=distortion[0],
            k2=distortion[1],
            k3=distortion[2],
            k4=distortion[3],
        ),
    )
    return _invert_kalibr_transform(matrix), camera_model, resolution


def convert(
    factory_calibration: Path,
    camera_sources: tuple[CameraSource, ...] = COVERAGE_CAMERA_SOURCES,
    downscale: int = 1,
) -> BasaltCalibration:
    """Convert a RoboCap factory directory into a typed Basalt calibration."""
    transforms: list[Transform] = []
    intrinsics: list[CameraModel] = []
    resolutions: list[list[int]] = []
    for source in camera_sources:
        camera_result: tuple[Transform, CameraModel, list[int]] = _camera_calibration(factory_calibration, source, downscale)
        transforms.append(camera_result[0])
        intrinsics.append(camera_result[1])
        resolutions.append(camera_result[2])

    imu: dict[str, Any] = _load_yaml(factory_calibration / IMU_RELATIVE_PATH)
    accel_noise: float = float(imu["accelerometer_noise_density"])
    gyro_noise: float = float(imu["gyroscope_noise_density"])
    accel_walk: float = float(imu["accelerometer_random_walk"])
    gyro_walk: float = float(imu["gyroscope_random_walk"])
    return BasaltCalibration(
        T_imu_cam=transforms,
        intrinsics=intrinsics,
        resolution=resolutions,
        vignette=[],
        calib_accel_bias=[0.0] * 9,
        calib_gyro_bias=[0.0] * 12,
        imu_update_rate=float(imu["update_rate"]),
        accel_noise_std=[accel_noise] * 3,
        gyro_noise_std=[gyro_noise] * 3,
        accel_bias_std=[accel_walk] * 3,
        gyro_bias_std=[gyro_walk] * 3,
        cam_time_offset_ns=0,
    )


def main(args: CliArgs) -> None:
    """Write the coverage and front-stereo Basalt calibration files."""
    stereo_output: Path = (
        args.stereo_output
        if args.stereo_output is not None
        else args.output.with_name(f"{args.output.stem}-stereo{args.output.suffix}")
    )
    for output_path, camera_sources in ((args.output, COVERAGE_CAMERA_SOURCES), (stereo_output, STEREO_CAMERA_SOURCES)):
        document: BasaltCalibrationFile = BasaltCalibrationFile(value0=convert(args.factory_calibration, camera_sources, args.downscale))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # json.dumps over serde's to_dict, not serde's to_json: cereal-compatible
        # files must keep the exact "key": value formatting of the originals.
        output_path.write_text(json.dumps(to_dict(document), indent=4) + "\n", encoding="utf-8")
        print(output_path)


if __name__ == "__main__":
    main(tyro.cli(CliArgs))
