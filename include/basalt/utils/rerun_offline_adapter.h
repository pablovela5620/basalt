/** Rerun integration boundary for the offline Basalt VIO runner. */
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include <Eigen/Core>

namespace basalt {

template <class Scalar>
struct Calibration;

template <class Scalar>
struct PoseVelBiasState;

class VioDataset;
struct VioVisualizationData;

struct RerunOfflineConfig {
  bool enabled = false;
  bool spawn = true;
  std::string app_id = "basalt_vio";
  std::string rrd_path;
};

/// Owns all Rerun-specific state for `basalt_vio`. Existing Basalt objects are
/// passed by reference, so the integration boundary adds no dataset or frame
/// copies. Serialization-owned buffers remain inside the Rerun backend.
class RerunOfflineAdapter {
 public:
  explicit RerunOfflineAdapter(RerunOfflineConfig config);
  ~RerunOfflineAdapter();

  RerunOfflineAdapter(const RerunOfflineAdapter&) = delete;
  RerunOfflineAdapter& operator=(const RerunOfflineAdapter&) = delete;

  bool enabled() const;
  bool active() const;

  void start(const Calibration<double>& calibration, VioDataset& dataset,
             int64_t start_t_ns);
  void log_state(const PoseVelBiasState<double>& state);
  void log_visualization(const VioVisualizationData& data);
  void log_ground_truth(const Eigen::Vector3d* positions, size_t count);
  void flush();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace basalt
