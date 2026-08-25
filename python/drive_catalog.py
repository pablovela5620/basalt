"""Catalog driver: Rerun base-layer rrd -> NVDEC -> VIT tracker -> trajectory csv.

Everything comes from the catalog: encoded H.264 samples (decoded on the GPU via
torchcodec, BT.601 luma + exact box average on-GPU), KB4 calibration from the
pinhole statics, and IMU scalars re-paired with the reader's accel-onto-gyro
interpolation. No raw-file access. This is the prototype for the dataforge
``slam`` verb.

TODO(slam-verb): the per-camera sample fetch materialises every encoded sample
twice (bytes list + in-memory MP4) — stream the mux per chunk before pointing
this at multi-hour sessions on small-RAM hosts.

Run from the repository root:
    pixi run -e driver python python/drive_catalog.py --rrd <base rrd>
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from fractions import Fraction
from io import BytesIO
from pathlib import Path

import av
import numpy as np
import pyarrow as pa
import rerun as rr
import tyro
from jaxtyping import Float64, Int64, UInt8
from numpy import ndarray

sys.path.insert(0, str(Path(__file__).resolve().parent))

from vit_binding import Tracker  # noqa: E402

CAMERA_TO_IMU_OFFSET_NS = 14_902_432  # basalt kCameraToImuOffsetNs (dataset_io_robocap.cpp)
FRAMESET_TOLERANCE_NS = 1_000_000
COVERAGE_NAMES = ("left", "left_front", "right_front", "right")
# Factory IMU intrinsics for device f408193e6447b3b0, from the Kalibr yaml
# imus_intrinsic/imu_mid_0.yaml (noise/random-walk densities, update rate).
# TODO(slam-verb): log these onto the rrd's imu node and read them, like the
# camera intrinsics already are — hardcoding mis-tunes any other device.
IMU_NOISE = {"gyro_noise_std": 0.0007300442812547, "gyro_bias_std": 3.445397083168e-05, "accel_noise_std": 0.005955224218014, "accel_bias_std": 0.0001963150489218}
IMU_UPDATE_RATE = 200.0


def build_framesets(camera_frames: list[list[tuple[int, int]]]) -> list[tuple[int, list[int]]]:
    """Mirror the basalt reader: anchor camera 0, nearest match per camera, median + offset.

    Each camera's frames are (opaque_id, timestamp_ns) in time order; the opaque
    id is not used here (a pts for file feeds, a decode index for the catalog feed).
    Returns (timestamp_ns_on_imu_clock, per-camera frame index) per complete frameset.
    """
    framesets: list[tuple[int, list[int]]] = []
    cursors = [0] * len(camera_frames)
    for anchor_index, (_, anchor_ns) in enumerate(camera_frames[0]):
        selected = [anchor_index]
        complete = True
        for camera in range(1, len(camera_frames)):
            frames = camera_frames[camera]
            index = cursors[camera]
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
        times = sorted(camera_frames[camera][selected[camera]][1] for camera in range(len(camera_frames)))
        middle = len(times) // 2
        median = times[middle] if len(times) % 2 == 1 else times[middle - 1] + (times[middle] - times[middle - 1]) // 2
        framesets.append((median + CAMERA_TO_IMU_OFFSET_NS, selected))
    return framesets


def wrap_mp4(samples: list[bytes], keyframes: list[bool], fps: int) -> bytes:
    """Mux pre-encoded H.264 samples into an in-memory MP4 (simplecv's add_mux_stream technique)."""
    buffer = BytesIO()
    with av.open(buffer, "w", format="mp4", options={"video_track_timescale": str(fps)}) as container:
        stream = container.add_mux_stream("h264", rate=fps, width=16, height=16)
        stream.time_base = Fraction(1, fps)
        for index, (sample, is_keyframe) in enumerate(zip(samples, keyframes, strict=True)):
            packet = av.Packet(sample)
            packet.pts = packet.dts = index
            packet.duration = 1
            packet.time_base = stream.time_base
            packet.stream = stream
            packet.is_keyframe = is_keyframe
            container.mux(packet)
    return buffer.getvalue()


def static_value(dataset, entity: str, column: str) -> np.ndarray | str:
    """One static component, explicitly typed: list columns -> float array (or a lone string), else str."""
    table = dataset.filter_contents(entity).reader(index=None).select(f"{entity}:{column}").to_arrow_table()
    if table.num_rows == 0:
        raise ValueError(f"no static row for {entity}:{column}")
    cell = table[0][0]
    if pa.types.is_list(table[0].type) or pa.types.is_large_list(table[0].type):
        values = cell.values.to_pylist()
        if len(values) == 1 and isinstance(values[0], str):
            return values[0]
        return np.asarray(values, dtype=np.float64).ravel()
    return str(cell.as_py())


def scalar_column(dataset, entity: str, component: str) -> tuple[np.ndarray, np.ndarray]:
    """Temporal scalars sorted by video_time, deduplicated keep-first (mirrors the reader's sort_and_deduplicate)."""
    table = dataset.filter_contents(entity).reader(index="video_time").select("video_time", f"{entity}:{component}").sort("video_time").to_arrow_table()
    times = np.array([t.value for t in table[0]], dtype=np.int64)
    values = np.array(table[1].combine_chunks().to_pylist(), dtype=np.float64).reshape(len(times), -1)
    keep = np.ones(len(times), dtype=bool)
    keep[1:] = times[1:] != times[:-1]
    return times[keep], values[keep]


@dataclass
class Config:
    """Run the catalog-fed VIT driver on one base-layer rrd."""

    rrd: Path
    """Base-layer rrd holding the session (exoego:v2 schema)."""
    lib: Path = Path("build-norerun/libbasalt.so")
    """libbasalt built without the rerun backend (its arrow clashes with pyarrow in-process)."""
    vit_config: Path = Path("python/robocap_vit.toml")
    """Unified basalt TOML (VIO tuning + determinism; camera/IMU calibration is programmatic)."""
    downscale: int = 3
    """Integer downscale factor (1920x1080 -> 640x360 at 3)."""
    output: Path = Path("datasets/vit_catalog_trajectory.csv")
    """Trajectory csv: ns timestamps, position, w-first quaternion."""


def main(args: Config) -> None:
    import torch
    from torchcodec.decoders import VideoDecoder  # deferred: slow import

    server = rr.server.Server(datasets={"drive": [str(args.rrd)]})
    dataset = server.client().get_dataset("drive")

    num_cameras = int(np.asarray(static_value(dataset, "/world/rig_00", "num_cameras")).ravel()[0])
    cam_by_name: dict[str, int] = {}
    for cam in range(num_cameras):
        name = str(static_value(dataset, f"/world/rig_00/cam_{cam:02d}", "name")).replace("-", "_")
        if name in cam_by_name:
            raise ValueError(f"duplicate camera name {name!r} at cam_{cam:02d} and cam_{cam_by_name[name]:02d}")
        cam_by_name[name] = cam
    missing = [name for name in COVERAGE_NAMES if name not in cam_by_name]
    if missing:
        raise ValueError(f"coverage cameras missing from the recording: {missing} (found {sorted(cam_by_name)})")

    tracker = Tracker(args.lib, str(args.vit_config), cam_count=len(COVERAGE_NAMES))
    decoders, sample_times = [], []
    for index, coverage_name in enumerate(COVERAGE_NAMES):
        entity = f"/world/rig_00/cam_{cam_by_name[coverage_name]:02d}"
        distortion_model = static_value(dataset, f"{entity}/pinhole", "simplecv.components.DistortionModel")
        if distortion_model != "kannala_brandt":
            raise ValueError(f"{entity}: unsupported distortion model {distortion_model!r} (driver maps only kannala_brandt -> KB4)")
        distortion = np.asarray(static_value(dataset, f"{entity}/pinhole", "simplecv.components.DistortionCoefficients"), dtype=np.float64)
        if not np.allclose(distortion[4:], 0.0):
            raise ValueError(f"{entity}: KB4 supports 4 coefficients, got non-zero tail {distortion[4:]}")
        relation = int(np.asarray(static_value(dataset, entity, "Transform3D:relation")).ravel()[0])
        # log_pinhole writes from_parent=True (TransformRelation.ChildFromParent = 2);
        # the inversion below is only correct for that convention — refuse anything else.
        if relation != 2:
            raise ValueError(f"{entity}: Transform3D relation {relation} is not ChildFromParent(2); refusing to invert blindly")
        k_matrix: Float64[ndarray, "3 3"] = np.asarray(static_value(dataset, f"{entity}/pinhole", "Pinhole:image_from_camera"), dtype=np.float64).reshape(3, 3, order="F")
        resolution = np.asarray(static_value(dataset, f"{entity}/pinhole", "Pinhole:resolution"), dtype=np.float64).ravel()
        cam_R_imu: Float64[ndarray, "3 3"] = np.asarray(static_value(dataset, entity, "Transform3D:mat3x3"), dtype=np.float64).reshape(3, 3, order="F")
        cam_t_imu: Float64[ndarray, "3"] = np.asarray(static_value(dataset, entity, "Transform3D:translation"), dtype=np.float64).ravel()
        T_imu_cam: Float64[ndarray, "4 4"] = np.eye(4)
        T_imu_cam[:3, :3] = cam_R_imu.T
        T_imu_cam[:3, 3] = -cam_R_imu.T @ cam_t_imu
        d = args.downscale
        tracker.add_camera_calibration(
            index,
            width=int(resolution[0]) // d,
            height=int(resolution[1]) // d,
            frequency=30.0,
            fx=k_matrix[0, 0] / d,
            fy=k_matrix[1, 1] / d,
            cx=(k_matrix[0, 2] + 0.5) / d - 0.5,  # converter's half-pixel convention (basalt_convert_robocap_calib)
            cy=(k_matrix[1, 2] + 0.5) / d - 0.5,
            distortion=distortion[:4].tolist(),
            T_imu_cam=T_imu_cam,
        )

        video = f"{entity}/pinhole/video"
        table = (
            dataset.filter_contents(video)
            .reader(index="video_time")
            .select("video_time", f"{video}:VideoStream:sample", f"{video}:VideoStream:is_keyframe")
            .sort("video_time")
            .to_arrow_table()
        )
        times_ns: Int64[ndarray, "n_samples"] = np.array([t.value for t in table[0]], dtype=np.int64)
        # large_list: a long session's samples exceed 2 GiB per camera, overflowing
        # the default int32 offsets on combine_chunks.
        blobs = table[1].cast(pa.list_(pa.large_list(pa.uint8()))).combine_chunks().flatten()
        data = memoryview(blobs.flatten().buffers()[1])
        offsets = blobs.offsets.to_pylist()
        samples = [bytes(data[start:end]) for start, end in zip(offsets[:-1], offsets[1:], strict=True)]
        keyframes = [bool(flag) for flag in table[2].combine_chunks().flatten().to_pylist()]
        decoders.append(VideoDecoder(wrap_mp4(samples, keyframes, 30), device="cuda", seek_mode="exact", num_ffmpeg_threads=0))
        sample_times.append(times_ns)
        print(f"cam {index} ({coverage_name}): {len(times_ns)} samples", flush=True)

    tracker.add_imu_calibration(frequency=IMU_UPDATE_RATE, **IMU_NOISE)

    # IMU back onto the raw device clock, accel interpolated onto gyro times.
    gyro_t, gyro = scalar_column(dataset, "/world/rig_00/imu_00/gyro", "Scalars:scalars")
    accel_t, accel = scalar_column(dataset, "/world/rig_00/imu_00/accel", "Scalars:scalars")
    gyro_t = gyro_t + CAMERA_TO_IMU_OFFSET_NS
    accel_t = accel_t + CAMERA_TO_IMU_OFFSET_NS
    assert np.all(np.diff(gyro_t) > 0), "gyro timestamps must be strictly increasing after dedup"
    accel_interp = np.column_stack([np.interp(gyro_t, accel_t, accel[:, axis]) for axis in range(3)])
    inside = (gyro_t >= accel_t[0]) & (gyro_t < accel_t[-1])
    imu = list(zip(gyro_t[inside].tolist(), gyro[inside], accel_interp[inside], strict=True))
    print(f"{len(imu)} paired IMU samples", flush=True)

    framesets = build_framesets([[(i, int(t)) for i, t in enumerate(times)] for times in sample_times])
    print(f"{len(framesets)} complete framesets", flush=True)

    tracker.start()
    poses = []
    imu_cursor = 0
    started = time.monotonic()
    for count, (timestamp_ns, selected) in enumerate(framesets):
        # IMU leads the frame by one sample: the backend integrates up to the frame
        # time and (in deterministic mode) push_img blocks until it can.
        while imu_cursor < len(imu) and (imu_cursor == 0 or imu[imu_cursor - 1][0] <= timestamp_ns):
            t_ns, gyro_xyz, accel_xyz = imu[imu_cursor]
            tracker.push_imu(int(t_ns), gyro_xyz, accel_xyz)
            imu_cursor += 1
        for camera, frame_index in enumerate(selected):
            rgb = decoders[camera].get_frame_at(frame_index).data.float()  # uint8 CHW on cuda
            # BT.601 full-range luma: matches swscale's gray8 conversion to ~0.5 LSB
            # (the streams carry real chroma; taking a raw channel is ~28 LSB off).
            luma = (0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]).unsqueeze(0).unsqueeze(0)
            gray = torch.nn.functional.avg_pool2d(luma, args.downscale).round().clamp(0, 255).to(torch.uint8)  # exact box average == SWS_AREA
            image_hw: UInt8[ndarray, "h w"] = np.ascontiguousarray(gray[0, 0].cpu().numpy())
            tracker.push_img(camera, timestamp_ns, image_hw)
        poses.extend(tracker.poses())
        if count % 100 == 0:
            print(f"frameset {count}/{len(framesets)}, {len(poses)} poses, {time.monotonic() - started:.1f}s", flush=True)

    # Under the gate config (deterministic=1) states pop in-loop; a non-deterministic
    # config delivers the tail asynchronously — drain until complete either way, and
    # treat a deficit as a hard failure rather than writing a short trajectory.
    for _ in range(600):
        if len(poses) >= len(framesets):
            break
        drained = list(tracker.poses())
        poses.extend(drained)
        if not drained:
            time.sleep(0.05)
    if len(poses) != len(framesets):
        print(f"FATAL: {len(poses)} poses for {len(framesets)} framesets", flush=True)
        os._exit(1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as sink:
        sink.write("#timestamp [ns], p_x, p_y, p_z, q_w, q_x, q_y, q_z\n")
        for t_ns, px, py, pz, qw, qx, qy, qz in poses:
            sink.write(f"{t_ns},{px},{py},{pz},{qw},{qx},{qy},{qz}\n")
    print(f"{len(poses)} poses -> {args.output} ({time.monotonic() - started:.1f}s)", flush=True)
    # tracker.stop() segfaults in non-deterministic mode (known upstream teardown
    # bug, see the TODO in vit_tracker.cpp) — the output is on disk, exit hard.
    os._exit(0)


if __name__ == "__main__":
    main(tyro.cli(Config))
