/**
Rerun (rerun.io) visualization backend for the Basalt VIO runner.

All use of the Rerun C++ SDK is confined to `rerun_export.cpp` via a PIMPL, so
no other translation unit needs to include `<rerun.hpp>`. This file is only
compiled/linked when the `BASALT_ENABLE_RERUN` CMake option is set (which
defines `BASALT_RERUN`).

The exporter mirrors the data the Pangolin GUI draws onto a Rerun recording so a
C++ run and a Mojo-port run of the same dataset can be loaded together in one
viewer (see docs / the integration plan). This C0 skeleton owns the recording
stream lifecycle and the static world coordinate frame; per-frame logging
(trajectory, landmarks, images, scalars) is added in later checkpoints.
*/
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <sophus/se3.hpp>

namespace basalt {

/// Per-camera static calibration, extracted from `basalt::Calibration` at the
/// call site so this header stays free of the templated calibration type.
struct CamCalib {
  Sophus::SE3d T_i_c;  ///< rig/IMU <- camera extrinsic
  float fx, fy, cx, cy;
  int width, height;
};

/// Thin owner of a `rerun::RecordingStream`. Non-copyable.
class RerunExporter {
 public:
  /// Create a recording for `app_id`. If `rrd_path` is non-empty the stream is
  /// saved to that file; otherwise, if `spawn` is true, a Rerun viewer is
  /// spawned and the stream connects to it; otherwise, if `connect_url` is
  /// non-empty, the stream connects to that URL. On success the `/world` view
  /// coordinates (right-handed, Z up — Basalt's world frame) are logged static.
  RerunExporter(const std::string& app_id, const std::string& rrd_path, bool spawn,
                const std::string& connect_url = "");
  ~RerunExporter();

  RerunExporter(const RerunExporter&) = delete;
  RerunExporter& operator=(const RerunExporter&) = delete;

  /// Log the static camera rig: for each camera, a `Transform3D` at
  /// `/world/rig_0/cam_{i}` (the rig<-cam extrinsic) and a `Pinhole` at
  /// `/world/rig_0/cam_{i}/pinhole` (linear intrinsics + resolution). Static.
  void log_static_calib(const std::vector<CamCalib>& cams);

  /// Log the ground-truth trajectory as a static green polyline at
  /// `/world/rig_0_path`, with start/end markers at `/world/rig_0_path/endpoints`.
  /// No-op if `gt_positions` is empty.
  void log_gt_path(const std::vector<Eigen::Vector3d>& gt_positions);

  /// Log the full raw IMU stream (gyro rad/s, accel m/s²) as two
  /// multi-component `Scalars` at `/world/rig_0/imu_0/{gyro,accel}` on the
  /// `sensor_time` timeline. `t_ns`, `gyro`, `accel` are parallel arrays.
  void log_imu(const std::vector<int64_t>& t_ns, const std::vector<Eigen::Vector3d>& gyro,
               const std::vector<Eigen::Vector3d>& accel, int64_t start_t_ns);

  /// Log one live IMU sample on the `sensor_time` timeline. Intended for
  /// streaming paths that do not have the full IMU sequence up front.
  void log_imu_sample(int64_t t_ns, const Eigen::Vector3d& gyro, const Eigen::Vector3d& accel,
                      int64_t start_t_ns);

  // --- per-frame logging (call begin_frame, then any log_* below). begin_frame
  // takes an explicit frame index (derived from t_ns) so the state-queue and
  // vis-queue consumer threads stamp the same physical frame identically. ---

  /// Set the `frame` (= frame_idx) + `sensor_time` timelines for the frame.
  void begin_frame(int64_t frame_idx, int64_t t_ns, int64_t start_t_ns);

  /// Per-frame rig pose at `/world/rig_0` + append to `/world/runs/basalt/trajectory`.
  void log_pose(const Sophus::SE3d& T_w_i);

  /// Per-camera input image at `/world/rig_0/cam_{cam}/pinhole/image`. `data` is
  /// Basalt's 16-bit grayscale buffer (8-bit sources are stored `<< 8`); it is
  /// downconverted to 8-bit and JPEG-encoded before logging as `EncodedImage`.
  void log_image(int cam, const uint16_t* data, int width, int height, size_t pitch_bytes);

  /// Per-frame VIO scalars: velocity at `/world/metrics/velocity` and the IMU
  /// bias estimates at `/world/rig_0/imu_0/bias_{gyro,accel}`.
  void log_metrics(const Eigen::Vector3d& vel, const Eigen::Vector3d& bias_gyro,
                   const Eigen::Vector3d& bias_accel);

  /// 2D tracked keypoints for one camera at `/world/rig_0/cam_{cam}/pinhole/keypoints`
  /// (`Points2D`). `rgba` holds one packed `0xRRGGBBAA` color per point (sampled
  /// from the image at the keypoint pixel). `uv` and `rgba` are parallel.
  void log_keypoints(int cam, const std::vector<Eigen::Vector2f>& uv,
                     const std::vector<uint32_t>& rgba);

  /// 2D reprojections of the 3D landmarks for one camera at
  /// `/world/rig_0/cam_{cam}/pinhole/observations` (`Points2D`). These should
  /// land on the tracked keypoints (small reprojection error) — the visual proof
  /// the 3D map and the 2D features correspond.
  void log_observations(int cam, const std::vector<Eigen::Vector2f>& uv, uint32_t rgba);

  /// 3D landmark cloud at `/world/landmarks` (`Points3D`, world frame), one
  /// packed `0xRRGGBBAA` color per point. `points` and `rgba` are parallel.
  void log_landmarks(const std::vector<Eigen::Vector3f>& points, const std::vector<uint32_t>& rgba);

  /// True if the underlying recording stream was created and a sink attached.
  bool good() const;

  /// Flush all pending data to the sink (blocking). Safe to call repeatedly.
  void flush();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace basalt
