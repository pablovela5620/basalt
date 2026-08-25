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

from robocap_feed import CAMERA_TO_IMU_OFFSET_NS, build_framesets, feed_imu_lead  # noqa: E402
from vit_binding import Tracker  # noqa: E402

COVERAGE_NAMES = ("left", "left_front", "right_front", "right")
# Factory IMU intrinsics for device f408193e6447b3b0, from the Kalibr yaml
# imus_intrinsic/imu_mid_0.yaml (noise/random-walk densities, update rate).
# TODO(slam-verb): log these onto the rrd's imu node and read them, like the
# camera intrinsics already are — hardcoding mis-tunes any other device.
IMU_NOISE = {"gyro_noise_std": 0.0007300442812547, "gyro_bias_std": 3.445397083168e-05, "accel_noise_std": 0.005955224218014, "accel_bias_std": 0.0001963150489218}
IMU_UPDATE_RATE = 200.0


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


def _static_cell(dataset, entity: str, column: str) -> pa.Scalar:
    """The single static cell for one component, or ValueError when absent."""
    table = dataset.filter_contents(entity).reader(index=None).select(f"{entity}:{column}").to_arrow_table()
    if table.num_rows == 0:
        raise ValueError(f"no static row for {entity}:{column}")
    return table[0][0]


def static_array(dataset, entity: str, column: str) -> Float64[ndarray, "n"]:
    """One static list component as a flat float64 array (matrices arrive column-major)."""
    cell = _static_cell(dataset, entity, column)
    return np.asarray(cell.values.to_pylist(), dtype=np.float64).ravel()


def static_str(dataset, entity: str, column: str) -> str:
    """One static string component (bare, or wrapped in a single-element list)."""
    cell = _static_cell(dataset, entity, column)
    value = cell.values.to_pylist()[0] if isinstance(cell, (pa.ListScalar, pa.LargeListScalar)) else cell.as_py()
    if not isinstance(value, str):
        raise ValueError(f"{entity}:{column} is not a string: {value!r}")
    return value


def scalar_column(dataset, entity: str, component: str) -> tuple[Int64[ndarray, "n"], Float64[ndarray, "n 3"]]:
    """Temporal scalars sorted by video_time, deduplicated keep-first (mirrors the reader's sort_and_deduplicate)."""
    table = dataset.filter_contents(entity).reader(index="video_time").select("video_time", f"{entity}:{component}").sort("video_time").to_arrow_table()
    times: Int64[ndarray, "n"] = table[0].combine_chunks().cast(pa.int64()).to_numpy()
    flat = table[1].combine_chunks().flatten().to_numpy(zero_copy_only=False)
    values: Float64[ndarray, "n 3"] = np.asarray(flat, dtype=np.float64).reshape(len(times), -1)
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

    num_cameras = int(static_array(dataset, "/world/rig_00", "num_cameras")[0])
    cam_by_name: dict[str, int] = {}
    for cam in range(num_cameras):
        name = static_str(dataset, f"/world/rig_00/cam_{cam:02d}", "name").replace("-", "_")
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
        distortion_model = static_str(dataset, f"{entity}/pinhole", "simplecv.components.DistortionModel")
        if distortion_model != "kannala_brandt":
            raise ValueError(f"{entity}: unsupported distortion model {distortion_model!r} (driver maps only kannala_brandt -> KB4)")
        distortion = static_array(dataset, f"{entity}/pinhole", "simplecv.components.DistortionCoefficients")
        if not np.allclose(distortion[4:], 0.0):
            raise ValueError(f"{entity}: KB4 supports 4 coefficients, got non-zero tail {distortion[4:]}")
        relation = int(static_array(dataset, entity, "Transform3D:relation")[0])
        # log_pinhole writes from_parent=True (TransformRelation.ChildFromParent = 2);
        # the inversion below is only correct for that convention — refuse anything else.
        if relation != 2:
            raise ValueError(f"{entity}: Transform3D relation {relation} is not ChildFromParent(2); refusing to invert blindly")
        k_matrix: Float64[ndarray, "3 3"] = static_array(dataset, f"{entity}/pinhole", "Pinhole:image_from_camera").reshape(3, 3, order="F")
        resolution = static_array(dataset, f"{entity}/pinhole", "Pinhole:resolution")
        cam_R_imu: Float64[ndarray, "3 3"] = static_array(dataset, entity, "Transform3D:mat3x3").reshape(3, 3, order="F")
        cam_t_imu: Float64[ndarray, "3"] = static_array(dataset, entity, "Transform3D:translation")
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
        times_ns: Int64[ndarray, "n_samples"] = table[0].combine_chunks().cast(pa.int64()).to_numpy()
        # large_list: a long session's samples exceed 2 GiB per camera, overflowing
        # the default int32 offsets on combine_chunks.
        blobs = table[1].cast(pa.list_(pa.large_list(pa.uint8()))).combine_chunks().flatten()
        data = memoryview(blobs.flatten().buffers()[1])
        offsets = blobs.offsets.to_pylist()
        samples = [bytes(data[start:end]) for start, end in zip(offsets[:-1], offsets[1:], strict=True)]
        keyframes = [bool(flag) for flag in table[2].combine_chunks().flatten().to_pylist()]
        decoder = VideoDecoder(wrap_mp4(samples, keyframes, 30), device="cuda", seek_mode="exact", num_ffmpeg_threads=0)
        if decoder.cpu_fallback:
            raise RuntimeError(f"cam {index} ({coverage_name}): torchcodec fell back to CPU decode; NVDEC is required")
        decoders.append(decoder)
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
        imu_cursor = feed_imu_lead(tracker, imu, imu_cursor, timestamp_ns)
        for camera, frame_index in enumerate(selected):
            rgb = decoders[camera].get_frame_at(frame_index).data.float()  # uint8 CHW on cuda
            # BT.601 full-range luma: matches swscale's gray8 conversion to ~0.5 LSB
            # (the streams carry real chroma; taking a raw channel is ~28 LSB off).
            luma = (0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]).unsqueeze(0).unsqueeze(0)
            gray = torch.nn.functional.avg_pool2d(luma, args.downscale).round().clamp(0, 255).to(torch.uint8)  # exact box average == SWS_AREA
            image_hw: UInt8[ndarray, "h w"] = np.ascontiguousarray(gray[0, 0].cpu().numpy())
            tracker.push_img(camera, timestamp_ns, image_hw)
        poses.extend(tracker.drain_poses())
        if count % 100 == 0:
            print(f"frameset {count}/{len(framesets)}, {len(poses)} poses, {time.monotonic() - started:.1f}s", flush=True)

    # Under the gate config (deterministic=1) states pop in-loop; a non-deterministic
    # config delivers the tail asynchronously — drain until complete either way, and
    # treat a deficit as a hard failure rather than writing a short trajectory.
    for _ in range(600):
        if len(poses) >= len(framesets):
            break
        drained = list(tracker.drain_poses())
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
