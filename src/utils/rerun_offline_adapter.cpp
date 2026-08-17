/** Rerun integration boundary for the offline Basalt VIO runner. */
#include <basalt/utils/rerun_offline_adapter.h>

#include <basalt/calibration/calibration.hpp>
#include <basalt/io/dataset_io.h>
#include <basalt/utils/rerun_export.h>
#include <basalt/vi_estimator/vio_estimator.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

namespace basalt {
namespace {

uint32_t pixel_rgba(const ManagedImage<uint16_t>* image, float u, float v) {
  const int x = static_cast<int>(std::lround(u));
  const int y = static_cast<int>(std::lround(v));
  if (image == nullptr || !image->InBounds(x, y)) return 0xffffffffu;
  const uint32_t intensity =
      static_cast<uint32_t>((*image)(static_cast<size_t>(x),
                                     static_cast<size_t>(y)) >>
                            8);
  return (intensity << 24) | (intensity << 16) | (intensity << 8) | 0xffu;
}

}  // namespace

struct RerunOfflineAdapter::Impl {
  explicit Impl(RerunOfflineConfig config) : config(std::move(config)) {}

  int64_t frame_index(int64_t t_ns) const {
    if (image_timestamps == nullptr) return 0;
    return static_cast<int64_t>(
        std::lower_bound(image_timestamps->begin(), image_timestamps->end(),
                         t_ns) -
        image_timestamps->begin());
  }

  void log_imu_through(int64_t t_ns) {
    std::lock_guard<std::mutex> lock(imu_mutex);
    if (gyro == nullptr || accel == nullptr || exporter == nullptr) return;

    const size_t sample_count = std::min(gyro->size(), accel->size());
    while (imu_cursor < sample_count &&
           (*gyro)[imu_cursor].timestamp_ns <= t_ns) {
      const auto& gyro_sample = (*gyro)[imu_cursor];
      exporter->log_imu_sample(
          gyro_sample.timestamp_ns, gyro_sample.data,
          (*accel)[imu_cursor].data, start_t_ns,
          frame_index(gyro_sample.timestamp_ns));
      ++imu_cursor;
    }
  }

  RerunOfflineConfig config;
  int64_t start_t_ns = -1;
  const std::vector<int64_t>* image_timestamps = nullptr;
  const Eigen::aligned_vector<GyroData>* gyro = nullptr;
  const Eigen::aligned_vector<AccelData>* accel = nullptr;
  size_t imu_cursor = 0;
  std::mutex imu_mutex;
  bool reprojection_checked = false;
  std::unique_ptr<RerunExporter> exporter;
};

RerunOfflineAdapter::RerunOfflineAdapter(RerunOfflineConfig config)
    : impl_(std::make_unique<Impl>(std::move(config))) {}

RerunOfflineAdapter::~RerunOfflineAdapter() = default;

bool RerunOfflineAdapter::enabled() const { return impl_->config.enabled; }

bool RerunOfflineAdapter::active() const {
  return impl_->exporter && impl_->exporter->good();
}

void RerunOfflineAdapter::start(const Calibration<double>& calibration,
                                VioDataset& dataset, int64_t start_t_ns) {
  if (!enabled() || impl_->exporter) return;

  impl_->exporter = std::make_unique<RerunExporter>(
      impl_->config.app_id, impl_->config.rrd_path, impl_->config.spawn);
  if (!impl_->exporter->good()) {
    std::cerr << "[rerun] recording disabled (no sink attached)" << std::endl;
    impl_->exporter.reset();
    return;
  }

  impl_->start_t_ns = start_t_ns;
  impl_->image_timestamps = &dataset.get_image_timestamps();

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

  impl_->gyro = &dataset.get_gyro_data();
  impl_->accel = &dataset.get_accel_data();
}

void RerunOfflineAdapter::log_state(
    const PoseVelBiasState<double>& state) {
  if (!active()) return;

  impl_->log_imu_through(state.t_ns);
  impl_->exporter->begin_frame(impl_->frame_index(state.t_ns), state.t_ns,
                               impl_->start_t_ns);
  impl_->exporter->log_pose(state.T_w_i);
  impl_->exporter->log_metrics(state.vel_w_i, state.bias_gyro,
                               state.bias_accel);

  if (!state.input_images) return;
  for (size_t cam = 0; cam < state.input_images->img_data.size(); ++cam) {
    const auto& image = state.input_images->img_data[cam].img;
    if (!image) continue;
    impl_->exporter->log_image(static_cast<int>(cam), image->ptr,
                               static_cast<int>(image->w),
                               static_cast<int>(image->h), image->pitch);
  }
}

void RerunOfflineAdapter::log_visualization(
    const VioVisualizationData& data) {
  if (!active()) return;

  impl_->log_imu_through(data.t_ns);
  impl_->exporter->begin_frame(impl_->frame_index(data.t_ns), data.t_ns,
                               impl_->start_t_ns);

  const auto& optical_flow = data.opt_flow_res;
  if (optical_flow) {
    constexpr uint32_t kKeypointColor = 0x39ff14ffu;
    for (size_t cam = 0; cam < optical_flow->keypoints.size(); ++cam) {
      std::vector<Eigen::Vector2f> positions;
      std::vector<uint32_t> colors;
      positions.reserve(optical_flow->keypoints[cam].size());
      colors.reserve(optical_flow->keypoints[cam].size());
      for (const auto& keypoint : optical_flow->keypoints[cam]) {
        positions.push_back(keypoint.second.translation());
        colors.push_back(kKeypointColor);
      }
      impl_->exporter->log_keypoints(static_cast<int>(cam), positions, colors);
    }
  }

  if (data.projections) {
    for (size_t cam = 0; cam < data.projections->size(); ++cam) {
      const auto& projections = (*data.projections)[cam];
      std::vector<Eigen::Vector2f> positions;
      positions.reserve(projections.size());
      for (const auto& projection : projections) {
        positions.emplace_back(static_cast<float>(projection[0]),
                               static_cast<float>(projection[1]));
      }
      impl_->exporter->log_observations(static_cast<int>(cam), positions,
                                        0xff00ffffu);
    }

    if (!impl_->reprojection_checked && optical_flow) {
      impl_->reprojection_checked = true;
      for (size_t cam = 0;
           cam < data.projections->size() &&
           cam < optical_flow->keypoints.size();
           ++cam) {
        double error_sum = 0.0;
        double max_error = 0.0;
        size_t observation_count = 0;
        for (const auto& projection : (*data.projections)[cam]) {
          const auto keypoint = optical_flow->keypoints[cam].find(
              static_cast<size_t>(projection[3]));
          if (keypoint == optical_flow->keypoints[cam].end()) continue;
          const Eigen::Vector2f position = keypoint->second.translation();
          const double error =
              std::hypot(projection[0] - position.x(),
                         projection[1] - position.y());
          error_sum += error;
          max_error = std::max(max_error, error);
          ++observation_count;
        }
        if (observation_count > 0) {
          std::cerr << "[rerun] cam" << cam
                    << " reprojection error vs keypoints: mean="
                    << (error_sum / static_cast<double>(observation_count))
                    << "px max=" << max_error << "px over "
                    << observation_count << " obs\n";
        }
      }
    }
  }

  if (data.points.empty()) return;

  std::unordered_map<int, Eigen::Vector2f> cam0_observations;
  if (data.projections && !data.projections->empty()) {
    const auto& projections = (*data.projections)[0];
    cam0_observations.reserve(projections.size());
    for (const auto& projection : projections) {
      cam0_observations[static_cast<int>(projection[3])] =
          Eigen::Vector2f(static_cast<float>(projection[0]),
                          static_cast<float>(projection[1]));
    }
  }

  const ManagedImage<uint16_t>* cam0_image =
      optical_flow && optical_flow->input_images &&
              !optical_flow->input_images->img_data.empty()
          ? optical_flow->input_images->img_data[0].img.get()
          : nullptr;

  std::vector<Eigen::Vector3f> points;
  std::vector<uint32_t> colors;
  points.reserve(data.points.size());
  colors.reserve(data.points.size());
  for (size_t i = 0; i < data.points.size(); ++i) {
    points.push_back(data.points[i].cast<float>());
    uint32_t color = 0xffffffffu;
    if (i < data.point_ids.size()) {
      const auto observation = cam0_observations.find(data.point_ids[i]);
      if (observation != cam0_observations.end()) {
        color = pixel_rgba(cam0_image, observation->second.x(),
                           observation->second.y());
      }
    }
    colors.push_back(color);
  }
  impl_->exporter->log_landmarks(points, colors);
}

void RerunOfflineAdapter::log_ground_truth(
    const Eigen::Vector3d* positions, size_t count) {
  if (active()) impl_->exporter->log_gt_path(positions, count);
}

void RerunOfflineAdapter::flush() {
  if (active()) impl_->exporter->flush();
}

}  // namespace basalt
