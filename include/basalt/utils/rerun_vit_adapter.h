/**
Rerun integration boundary for the live VIT tracker.

This header contains the VIT-facing configuration and adapter. Rerun SDK types
remain private to the implementation.
*/
#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include <Eigen/Core>

namespace basalt {

template <class Scalar>
struct Calibration;

template <class Scalar>
struct PoseVelBiasState;

struct RerunVitConfig {
  bool enabled = false;
  bool spawn = false;
  bool log_imu = false;
  std::string connect_url;
  std::string rrd_path;
  int image_stride = 1;

  static RerunVitConfig from_environment();
};

/// Owns all Rerun-specific state for the live VIT path. The tracker passes its
/// existing calibration, IMU samples, and states by reference; no Basalt data
/// is copied at this boundary.
class RerunVitAdapter {
 public:
  explicit RerunVitAdapter(RerunVitConfig config = RerunVitConfig::from_environment());
  ~RerunVitAdapter();

  RerunVitAdapter(const RerunVitAdapter&) = delete;
  RerunVitAdapter& operator=(const RerunVitAdapter&) = delete;

  bool enabled() const;
  bool active() const;
  bool requires_pose_features() const;

  void start(const Calibration<double>& calibration);
  void note_frame_timestamp(int64_t t_ns);
  void log_imu_sample(int64_t t_ns, const Eigen::Vector3d& gyro,
                      const Eigen::Vector3d& accel);
  void log_state(const PoseVelBiasState<double>& state);
  void flush();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace basalt
