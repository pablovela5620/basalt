/**
Rerun (rerun.io) visualization backend for the Basalt VIO runner.

All use of the Rerun C++ SDK is confined to `rerun_export.cpp` via a PIMPL, so
no other translation unit in `basalt_vio` needs to include `<rerun.hpp>`. This
file is only compiled/linked into the `basalt_vio` executable, and only when the
`BASALT_ENABLE_RERUN` CMake option is set (which defines `BASALT_RERUN`).

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
  /// spawned and the stream connects to it. On success the `/world` view
  /// coordinates (right-handed, Z up — Basalt's world frame) are logged static.
  RerunExporter(const std::string& app_id, const std::string& rrd_path, bool spawn);
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

  /// Log one estimated state at `t_ns`. Sets the `frame` (monotonic per call)
  /// and `sensor_time` (= (t_ns - start_t_ns) seconds) timelines, then logs the
  /// per-frame rig pose at `/world/rig_0` and appends to the growing estimated
  /// trajectory polyline at `/world/runs/basalt/trajectory`. No-op if !good().
  void log_state(const Sophus::SE3d& T_w_i, int64_t t_ns, int64_t start_t_ns);

  /// True if the underlying recording stream was created and a sink attached.
  bool good() const;

  /// Flush all pending data to the sink (blocking). Safe to call repeatedly.
  void flush();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace basalt
