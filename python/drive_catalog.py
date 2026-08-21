"""Step-B driver: Rerun catalog (base-layer rrd) -> NVDEC -> VIT tracker -> trajectory csv.

Everything comes from the catalog: encoded H.264 samples (decoded on the GPU via
torchcodec, box-averaged 3x on-GPU — the exact SWS_AREA arithmetic), KB4
calibration from the pinhole statics, and IMU scalars re-paired with the same
accel-onto-gyro interpolation the file reader uses. No raw-file access.

Run from the repository root:
    pixi run -e driver python python/drive_catalog.py --rrd <base rrd>
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from fractions import Fraction
from io import BytesIO
from pathlib import Path

import av
import numpy as np
import rerun as rr
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from drive_files import CAMERA_TO_IMU_OFFSET_NS, build_framesets  # noqa: E402
from vit_binding import Tracker  # noqa: E402

COVERAGE_NAMES = ("left", "left_front", "right_front", "right")
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


def statics(dataset, entity: str, columns: list[str]) -> list:
    table = dataset.filter_contents(entity).reader(index=None).select(*[f"{entity}:{c}" for c in columns]).to_arrow_table()
    return [table[i][0].values.to_pylist() if hasattr(table[i][0], "values") else table[i][0].as_py() for i in range(len(columns))]


def scalar_column(dataset, entity: str, component: str) -> tuple[np.ndarray, np.ndarray]:
    table = dataset.filter_contents(entity).reader(index="video_time").select("video_time", f"{entity}:{component}").sort("video_time").to_arrow_table()
    times = np.array([t.value for t in table[0]], dtype=np.int64)
    values = np.array(table[1].combine_chunks().to_pylist(), dtype=np.float64).reshape(len(times), -1)
    return times, values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rrd", type=Path, required=True)
    parser.add_argument("--lib", type=Path, default=Path("build/libbasalt.so"))
    parser.add_argument("--config", type=Path, default=Path("python/robocap_vit.toml"))
    parser.add_argument("--downscale", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("datasets/vit_catalog_trajectory.csv"))
    args = parser.parse_args()
    from torchcodec.decoders import VideoDecoder  # after argparse: slow import

    server = rr.server.Server(datasets={"drive": [str(args.rrd)]})
    dataset = server.client().get_dataset("drive")

    # Coverage cameras by their AnyValues name, in the calibration's fixed order.
    cam_by_name: dict[str, int] = {}
    for cam in range(6):
        entity = f"/world/rig_00/cam_{cam:02d}"
        (name,) = statics(dataset, entity, ["name"])
        cam_by_name[str(name[0] if isinstance(name, list) else name).replace("-", "_")] = cam

    tracker = Tracker(args.lib, str(args.config), cam_count=len(COVERAGE_NAMES))
    decoders, sample_times = [], []
    for index, coverage_name in enumerate(COVERAGE_NAMES):
        cam = cam_by_name[coverage_name]
        entity = f"/world/rig_00/cam_{cam:02d}"
        k_matrix, resolution, distortion = statics(dataset, f"{entity}/pinhole", ["Pinhole:image_from_camera", "Pinhole:resolution", "simplecv.components.DistortionCoefficients"])
        translation, mat3x3 = statics(dataset, entity, ["Transform3D:translation", "Transform3D:mat3x3"])
        k = np.array(k_matrix, dtype=np.float64).ravel().reshape(3, 3, order="F")  # column-major storage
        cam_R_imu = np.array(mat3x3, dtype=np.float64).ravel().reshape(3, 3, order="F")
        resolution = np.array(resolution, dtype=np.float64).ravel()
        distortion = np.array(distortion, dtype=np.float64).ravel()
        T_imu_cam = np.eye(4)
        T_imu_cam[:3, :3] = cam_R_imu.T
        T_imu_cam[:3, 3] = -cam_R_imu.T @ np.array(translation, dtype=np.float64).ravel()
        d = args.downscale
        tracker.add_camera_calibration(
            index,
            width=int(resolution[0]) // d,
            height=int(resolution[1]) // d,
            frequency=30.0,
            fx=k[0, 0] / d,
            fy=k[1, 1] / d,
            cx=(k[0, 2] + 0.5) / d - 0.5,
            cy=(k[1, 2] + 0.5) / d - 0.5,
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
        times = np.array([t.value for t in table[0]], dtype=np.int64)
        # large_list: a long session's samples exceed 2 GiB per camera, overflowing
        # the default int32 list offsets on combine_chunks.
        import pyarrow as pa
        blobs = table[1].cast(pa.list_(pa.large_list(pa.uint8()))).combine_chunks().flatten()
        data = memoryview(blobs.flatten().buffers()[1])
        offsets = blobs.offsets.to_pylist()
        samples = [bytes(data[start:end]) for start, end in zip(offsets[:-1], offsets[1:], strict=True)]
        keyframes = [bool(flag) for flag in table[2].combine_chunks().flatten().to_pylist()]
        decoders.append(VideoDecoder(wrap_mp4(samples, keyframes, 30), device="cuda", seek_mode="exact", num_ffmpeg_threads=0))
        sample_times.append(times)
        print(f"cam {index} ({coverage_name}): {len(times)} samples", flush=True)

    tracker.add_imu_calibration(frequency=IMU_UPDATE_RATE, **IMU_NOISE)

    # IMU back onto the raw device clock, accel interpolated onto gyro times.
    gyro_t, gyro = scalar_column(dataset, "/world/rig_00/imu_00/gyro", "Scalars:scalars")
    accel_t, accel = scalar_column(dataset, "/world/rig_00/imu_00/accel", "Scalars:scalars")
    gyro_t = gyro_t + CAMERA_TO_IMU_OFFSET_NS
    accel_t = accel_t + CAMERA_TO_IMU_OFFSET_NS
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
        while imu_cursor < len(imu) and (imu_cursor == 0 or imu[imu_cursor - 1][0] <= timestamp_ns):
            t_ns, gyro_xyz, accel_xyz = imu[imu_cursor]
            tracker.push_imu(int(t_ns), gyro_xyz, accel_xyz)
            imu_cursor += 1
        for camera, frame_index in enumerate(selected):
            rgb = decoders[camera].get_frame_at(frame_index).data.float()  # uint8 CHW on cuda
            # BT.601 full-range luma: matches swscale's gray8 conversion to ~0.5 LSB
            # (the streams carry real chroma; taking a raw channel is ~28 LSB off).
            luma = (0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]).unsqueeze(0).unsqueeze(0)
            gray = torch.nn.functional.avg_pool2d(luma, args.downscale).round().clamp(0, 255).to(torch.uint8)  # exact 3x box average == SWS_AREA
            image = np.ascontiguousarray(gray[0, 0].cpu().numpy())
            tracker.push_img(camera, timestamp_ns, image.ctypes.data, width=image.shape[1], height=image.shape[0], stride=image.shape[1])
        poses.extend(tracker.poses())
        if count % 100 == 0:
            print(f"frameset {count}/{len(framesets)}, {len(poses)} poses, {time.monotonic() - started:.1f}s", flush=True)
    # Non-deterministic mode delivers the last states asynchronously: poll-drain.
    for _ in range(100):
        if len(poses) >= len(framesets):
            break
        drained = list(tracker.poses())
        poses.extend(drained)
        if not drained:
            time.sleep(0.05)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as sink:
        sink.write("#timestamp [ns], p_x, p_y, p_z, q_w, q_x, q_y, q_z\n")
        for t_ns, px, py, pz, qw, qx, qy, qz in poses:
            sink.write(f"{t_ns},{px},{py},{pz},{qw},{qx},{qy},{qz}\n")
    print(f"{len(poses)} poses -> {args.output} ({time.monotonic() - started:.1f}s)", flush=True)
    # tracker.stop() segfaults in non-deterministic mode (known upstream shutdown
    # bug, see the TODO in vit_tracker.cpp) — the output is on disk, exit hard.
    os._exit(0)


if __name__ == "__main__":
    main()
