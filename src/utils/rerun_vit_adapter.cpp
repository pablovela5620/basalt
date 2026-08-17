/** Rerun integration boundary for the live VIT tracker. */
#include <basalt/utils/rerun_vit_adapter.h>

#include <basalt/calibration/calibration.hpp>
#include <basalt/optical_flow/optical_flow.h>
#include <basalt/utils/rerun_export.h>

#include <cerrno>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

namespace basalt {
namespace {

int read_positive_int_env(const char* name, int default_value) {
  const char* raw = std::getenv(name);
  if (raw == nullptr || raw[0] == '\0') return default_value;

  char* end = nullptr;
  errno = 0;
  const long value = std::strtol(raw, &end, 10);
  if (errno != 0 || end == raw || *end != '\0' || value < 1 || value > std::numeric_limits<int>::max()) {
    std::cerr << "[rerun] ignoring invalid " << name << "=" << raw << std::endl;
    return default_value;
  }
  return static_cast<int>(value);
}

bool read_bool_env(const char* name, bool default_value) {
  const char* raw = std::getenv(name);
  if (raw == nullptr || raw[0] == '\0') return default_value;

  const std::string value(raw);
  return !(value == "0" || value == "false" || value == "FALSE" || value == "no" || value == "NO");
}

}  // namespace

RerunVitConfig RerunVitConfig::from_environment() {
  RerunVitConfig config;
  const char* sink = std::getenv("BASALT_VIT_RERUN");
  if (sink == nullptr || sink[0] == '\0') return config;

  config.enabled = true;
  const std::string sink_value(sink);
  if (sink_value == "spawn") {
    config.spawn = true;
  } else if (sink_value.size() > 4 && sink_value.compare(sink_value.size() - 4, 4, ".rrd") == 0) {
    config.rrd_path = sink_value;
  } else {
    config.connect_url = sink_value;
  }

  config.image_stride = read_positive_int_env("BASALT_VIT_RERUN_IMG_STRIDE", 1);
  config.log_imu = read_bool_env("BASALT_VIT_RERUN_IMU", false);
  return config;
}

struct RerunVitAdapter::Impl {
  explicit Impl(RerunVitConfig config) : config(std::move(config)) {}

  int64_t start_time_for(int64_t t_ns) {
    std::lock_guard<std::mutex> lock(timeline_mutex);
    if (start_t_ns < 0) start_t_ns = t_ns;
    return start_t_ns;
  }

  std::pair<int64_t, int64_t> timeline_for(int64_t t_ns) {
    std::lock_guard<std::mutex> lock(timeline_mutex);
    if (start_t_ns < 0) start_t_ns = t_ns;

    auto it = frame_idx_by_t_ns.find(t_ns);
    if (it == frame_idx_by_t_ns.end()) {
      it = frame_idx_by_t_ns.emplace(t_ns, next_frame_idx++).first;
    }
    return {it->second, start_t_ns};
  }

  RerunVitConfig config;
  int64_t start_t_ns = -1;
  int64_t next_frame_idx = 0;
  std::mutex timeline_mutex;
  std::unordered_map<int64_t, int64_t> frame_idx_by_t_ns;
  std::unique_ptr<RerunExporter> exporter;
};

RerunVitAdapter::RerunVitAdapter(RerunVitConfig config)
    : impl_(std::make_unique<Impl>(std::move(config))) {}

RerunVitAdapter::~RerunVitAdapter() = default;

bool RerunVitAdapter::enabled() const { return impl_->config.enabled; }

bool RerunVitAdapter::active() const {
  return impl_->exporter && impl_->exporter->good();
}

bool RerunVitAdapter::requires_pose_features() const { return active(); }

void RerunVitAdapter::start(const Calibration<double>& calibration) {
  if (!enabled() || impl_->exporter) return;

  impl_->exporter = std::make_unique<RerunExporter>(
      "basalt_vit", impl_->config.rrd_path, impl_->config.spawn,
      impl_->config.connect_url);
  if (!impl_->exporter->good()) {
    std::cerr << "[rerun] VIT recording disabled (no sink attached)" << std::endl;
    impl_->exporter.reset();
    return;
  }

  std::vector<CamCalib> cameras;
  cameras.reserve(calibration.intrinsics.size());
  for (size_t i = 0; i < calibration.intrinsics.size(); ++i) {
    const Eigen::VectorXd intrinsics = calibration.intrinsics[i].getParam();
    const Eigen::Vector2i resolution = calibration.resolution[i];
    cameras.push_back(CamCalib{
        calibration.T_i_c[i], static_cast<float>(intrinsics[0]),
        static_cast<float>(intrinsics[1]), static_cast<float>(intrinsics[2]),
        static_cast<float>(intrinsics[3]), resolution[0], resolution[1]});
  }
  impl_->exporter->log_static_calib(cameras);
}

void RerunVitAdapter::note_frame_timestamp(int64_t t_ns) {
  if (!active()) return;
  (void)impl_->timeline_for(t_ns);
}

void RerunVitAdapter::log_imu_sample(int64_t t_ns,
                                     const Eigen::Vector3d& gyro,
                                     const Eigen::Vector3d& accel) {
  if (!active() || !impl_->config.log_imu) return;
  impl_->exporter->log_imu_sample(t_ns, gyro, accel,
                                  impl_->start_time_for(t_ns));
}

void RerunVitAdapter::log_state(const PoseVelBiasState<double>& state) {
  if (!active()) return;

  const auto [frame_idx, start_t_ns] = impl_->timeline_for(state.t_ns);
  impl_->exporter->begin_frame(frame_idx, state.t_ns, start_t_ns);
  impl_->exporter->log_pose(state.T_w_i);
  impl_->exporter->log_metrics(state.vel_w_i, state.bias_gyro,
                               state.bias_accel);

  if (frame_idx % impl_->config.image_stride != 0 || !state.input_images) {
    return;
  }

  const auto& image_data = state.input_images->img_data;
  const auto& features_per_cam = state.input_images->stats.features_per_cam;
  constexpr uint32_t kFeatureColor = 0x39ff14ffu;

  for (size_t cam = 0; cam < image_data.size(); ++cam) {
    const auto& image = image_data[cam].img;
    if (!image) continue;

    impl_->exporter->log_image(static_cast<int>(cam), image->ptr,
                               static_cast<int>(image->w),
                               static_cast<int>(image->h), image->pitch);

    if (cam >= features_per_cam.size()) continue;
    std::vector<Eigen::Vector2f> positions;
    std::vector<uint32_t> colors;
    positions.reserve(features_per_cam[cam].size());
    colors.reserve(features_per_cam[cam].size());
    for (const vit::PoseFeature& feature : features_per_cam[cam]) {
      positions.emplace_back(feature.u, feature.v);
      colors.push_back(kFeatureColor);
    }
    impl_->exporter->log_keypoints(static_cast<int>(cam), positions, colors);
  }
}

void RerunVitAdapter::flush() {
  if (active()) impl_->exporter->flush();
}

}  // namespace basalt
