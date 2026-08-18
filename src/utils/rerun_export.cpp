/**
Rerun (rerun.io) visualization backend — implementation.

This is the ONLY translation unit that includes `<rerun.hpp>`; everything else
talks to `basalt::RerunExporter` through the PIMPL declared in
`basalt/utils/rerun_export.h`. Compiled only when `BASALT_RERUN` is defined.
*/
#include "basalt/utils/rerun_export.h"

#include <algorithm>
#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <rerun.hpp>

namespace basalt {

namespace {

// --- Canonical entity-path scheme (kept in one place to avoid drift; a typo
// here would silently detach an entity from its Pinhole/Transform3D parent). ---
constexpr const char* kWorld = "/world";
constexpr const char* kRig = "/world/rig_0";
constexpr const char* kEstTraj = "/world/runs/basalt/trajectory";
constexpr const char* kGtPath = "/world/rig_0_path";
constexpr const char* kGtEndpoints = "/world/rig_0_path/endpoints";
constexpr const char* kLandmarks = "/world/landmarks";
constexpr const char* kVelocity = "/world/metrics/velocity";
constexpr const char* kImuGyro = "/world/rig_0/imu_0/gyro";
constexpr const char* kImuAccel = "/world/rig_0/imu_0/accel";
constexpr const char* kBiasGyro = "/world/rig_0/imu_0/bias_gyro";
constexpr const char* kBiasAccel = "/world/rig_0/imu_0/bias_accel";

std::string cam_base(int i) { return "/world/rig_0/cam_" + std::to_string(i); }
std::string cam_pinhole(int i) { return cam_base(i) + "/pinhole"; }
std::string cam_image(int i) { return cam_pinhole(i) + "/image"; }
std::string cam_keypoints(int i) { return cam_pinhole(i) + "/keypoints"; }
std::string cam_observations(int i) { return cam_pinhole(i) + "/observations"; }

// --- Colors and point sizes ---
const rerun::Color kEstColor(0xff, 0xa5, 0x00);  // estimated trajectory (amber)
const rerun::Color kGtColor(0x3c, 0xb0, 0x43);   // ground-truth path (green)
const rerun::Color kGtStart(0x2e, 0xcc, 0x40);   // GT start marker (green)
const rerun::Color kGtEnd(0xff, 0x41, 0x36);     // GT end marker (red)
constexpr float kLandmarkRadius = 0.02f;
constexpr float kEndpointRadius = 0.02f;
constexpr float kKeypointRadius = 3.0f;
constexpr float kObsRadius = 2.0f;

// Unpack a packed 0xRRGGBBAA color.
rerun::Color color_rgba(uint32_t c) {
  return rerun::Color(uint8_t(c >> 24), uint8_t(c >> 16), uint8_t(c >> 8), uint8_t(c));
}

// Sophus SE3 -> rerun Transform3D (translation + xyzw quaternion).
rerun::Transform3D se3_to_transform3d(const Sophus::SE3d& T) {
  const Eigen::Vector3d p = T.translation();
  const Eigen::Quaterniond q = T.unit_quaternion();
  return rerun::Transform3D()
      .with_translation({float(p.x()), float(p.y()), float(p.z())})
      .with_rotation(rerun::Quaternion::from_xyzw(float(q.x()), float(q.y()), float(q.z()), float(q.w())));
}

}  // namespace

struct RerunExporter::Impl {
  std::unique_ptr<rerun::RecordingStream> rec;
  bool ok = false;
  std::vector<rerun::datatypes::Vec3D> traj;  // accumulated estimated positions (state thread only)
};

RerunExporter::RerunExporter(const std::string& app_id, const std::string& rrd_path, bool spawn,
                             const std::string& connect_url)
    : impl_(std::make_unique<Impl>()) {
  const std::string id = app_id.empty() ? std::string("basalt_vio") : app_id;
  impl_->rec = std::make_unique<rerun::RecordingStream>(id);

  rerun::Error err;
  if (!rrd_path.empty()) {
    err = impl_->rec->save(rrd_path);
  } else if (spawn) {
    err = impl_->rec->spawn();
  } else if (!connect_url.empty()) {
    err = impl_->rec->connect_grpc(connect_url);
  } else {
    // No sink requested — leave the exporter "not good" so callers no-op.
    return;
  }

  if (err.is_err()) {
    std::cerr << "[rerun] failed to attach sink: " << err.description << std::endl;
    return;
  }
  impl_->ok = true;

  // World coordinate frame: right-handed, Z up — matches Basalt's world frame.
  impl_->rec->log_static(kWorld, rerun::ViewCoordinates::RIGHT_HAND_Z_UP);
}

void RerunExporter::log_static_calib(const std::vector<CamCalib>& cams) {
  if (!good()) return;
  for (size_t i = 0; i < cams.size(); ++i) {
    const CamCalib& c = cams[i];
    impl_->rec->log_static(cam_base(int(i)), se3_to_transform3d(c.T_i_c));

    // Linear pinhole approximation (Basalt cameras are KB4/radtan8/...; principal
    // point is centered here — see the integration plan, Risk #5).
    impl_->rec->log_static(
        cam_pinhole(int(i)),
        rerun::Pinhole::from_focal_length_and_resolution(
            {c.fx, c.fy}, {static_cast<float>(c.width), static_cast<float>(c.height)})
            .with_camera_xyz(rerun::components::ViewCoordinates::RDF)
            .with_image_plane_distance(0.1f));
  }
}

void RerunExporter::log_gt_path(const Eigen::Vector3d* gt_positions,
                                size_t count) {
  if (!good() || gt_positions == nullptr || count == 0) return;

  std::vector<rerun::datatypes::Vec3D> pts;
  pts.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    const Eigen::Vector3d& g = gt_positions[i];
    pts.push_back({static_cast<float>(g.x()), static_cast<float>(g.y()), static_cast<float>(g.z())});
  }
  impl_->rec->log_static(kGtPath,
                         rerun::LineStrips3D(rerun::components::LineStrip3D(pts)).with_colors(kGtColor));

  const std::vector<rerun::datatypes::Vec3D> ends = {pts.front(), pts.back()};
  impl_->rec->log_static(kGtEndpoints, rerun::Points3D(ends)
                                           .with_colors({kGtStart, kGtEnd})
                                           .with_radii({kEndpointRadius, kEndpointRadius}));
}

void RerunExporter::log_imu_sample(int64_t t_ns, const Eigen::Vector3d& gyro,
                                   const Eigen::Vector3d& accel,
                                   int64_t start_t_ns,
                                   std::optional<int64_t> frame_idx) {
  if (!good()) return;
  if (frame_idx.has_value()) {
    begin_frame(*frame_idx, t_ns, start_t_ns);
  } else {
    impl_->rec->disable_timeline("frame");
    impl_->rec->set_time_duration_secs(
        "sensor_time", static_cast<double>(t_ns - start_t_ns) * 1e-9);
  }
  impl_->rec->log(kImuGyro, rerun::Scalars({gyro.x(), gyro.y(), gyro.z()}));
  impl_->rec->log(kImuAccel, rerun::Scalars({accel.x(), accel.y(), accel.z()}));
}

void RerunExporter::begin_frame(int64_t frame_idx, int64_t t_ns, int64_t start_t_ns) {
  if (!good()) return;
  impl_->rec->set_time_sequence("frame", frame_idx);
  impl_->rec->set_time_duration_secs("sensor_time", static_cast<double>(t_ns - start_t_ns) * 1e-9);
}

void RerunExporter::log_pose(const Sophus::SE3d& T_w_i) {
  if (!good()) return;

  // Per-frame rig pose (world <- imu/body).
  impl_->rec->log(kRig, se3_to_transform3d(T_w_i));

  // Growing estimated-trajectory polyline (logged in full each frame so that
  // scrubbing the `frame` timeline shows the path up to that frame).
  const Eigen::Vector3d p = T_w_i.translation();
  impl_->traj.push_back({static_cast<float>(p.x()), static_cast<float>(p.y()), static_cast<float>(p.z())});
  impl_->rec->log(kEstTraj,
                  rerun::LineStrips3D(rerun::components::LineStrip3D(impl_->traj)).with_colors(kEstColor));
}

void RerunExporter::log_image(int cam, const uint16_t* data, int width, int height, size_t pitch_bytes) {
  if (!good() || data == nullptr) return;
  // Basalt stores 8-bit sources as (v << 8); recover 8-bit, then JPEG-encode.
  const cv::Mat m16(height, width, CV_16UC1, const_cast<uint16_t*>(data), pitch_bytes);
  cv::Mat m8;
  m16.convertTo(m8, CV_8UC1, 1.0 / 256.0);
  std::vector<uint8_t> jpg;
  if (!cv::imencode(".jpg", m8, jpg)) return;
  impl_->rec->log(cam_image(cam), rerun::EncodedImage::from_bytes(jpg, rerun::components::MediaType::jpeg()));
}

void RerunExporter::log_metrics(const Eigen::Vector3d& vel, const Eigen::Vector3d& bias_gyro,
                                const Eigen::Vector3d& bias_accel) {
  if (!good()) return;
  impl_->rec->log(kVelocity, rerun::Scalars({vel.x(), vel.y(), vel.z()}));
  impl_->rec->log(kBiasGyro, rerun::Scalars({bias_gyro.x(), bias_gyro.y(), bias_gyro.z()}));
  impl_->rec->log(kBiasAccel, rerun::Scalars({bias_accel.x(), bias_accel.y(), bias_accel.z()}));
}

void RerunExporter::log_keypoints(int cam, const std::vector<Eigen::Vector2f>& uv,
                                  const std::vector<uint32_t>& rgba) {
  if (!good() || uv.empty()) return;
  std::vector<rerun::datatypes::Vec2D> pts;
  std::vector<rerun::Color> colors;
  pts.reserve(uv.size());
  colors.reserve(uv.size());
  for (size_t i = 0; i < uv.size(); ++i) {
    pts.push_back({uv[i].x(), uv[i].y()});
    colors.push_back(color_rgba(i < rgba.size() ? rgba[i] : 0xffffffffu));
  }
  impl_->rec->log(cam_keypoints(cam), rerun::Points2D(pts).with_colors(colors).with_radii(kKeypointRadius));
}

void RerunExporter::log_observations(int cam, const std::vector<Eigen::Vector2f>& uv,
                                     const std::vector<float>& radii, const std::vector<uint32_t>& rgba) {
  if (!good() || uv.empty()) return;
  std::vector<rerun::components::Position2D> centers;
  std::vector<rerun::components::HalfSize2D> half_sizes;
  std::vector<rerun::Color> colors;
  centers.reserve(uv.size());
  half_sizes.reserve(uv.size());
  colors.reserve(uv.size());
  for (size_t i = 0; i < uv.size(); ++i) {
    centers.push_back({uv[i].x(), uv[i].y()});
    const float radius = i < radii.size() ? radii[i] : kObsRadius;
    half_sizes.push_back({radius, radius});
    colors.push_back(color_rgba(i < rgba.size() ? rgba[i] : 0xffffffffu));
  }
  // Ring perimeters (Ellipses2D renders outlines), like Pangolin's
  // glDrawCirclePerimeter; the stroke is in UI points so it stays crisp at
  // any zoom while the ring diameter scales with the image.
  impl_->rec->log(cam_observations(cam), rerun::Ellipses2D::from_centers_and_half_sizes(centers, half_sizes)
                                             .with_colors(colors)
                                             .with_line_radii(rerun::components::Radius::ui_points(1.5f)));
}

void RerunExporter::log_landmarks(const std::vector<Eigen::Vector3f>& points,
                                  const std::vector<uint32_t>& rgba) {
  if (!good() || points.empty()) return;
  std::vector<rerun::datatypes::Vec3D> pts;
  std::vector<rerun::Color> colors;
  pts.reserve(points.size());
  colors.reserve(points.size());
  for (size_t i = 0; i < points.size(); ++i) {
    pts.push_back({points[i].x(), points[i].y(), points[i].z()});
    colors.push_back(color_rgba(i < rgba.size() ? rgba[i] : 0xffffffffu));
  }
  // World frame (NOT under the moving /world/rig_0 transform — these points are
  // already in world coordinates; parenting them to the rig would re-apply the
  // camera pose and make the cloud swim with the camera).
  impl_->rec->log(kLandmarks, rerun::Points3D(pts).with_colors(colors).with_radii(kLandmarkRadius));
}

RerunExporter::~RerunExporter() { flush(); }

bool RerunExporter::good() const { return impl_ && impl_->ok; }

void RerunExporter::flush() {
  if (impl_ && impl_->rec && impl_->ok) {
    (void)impl_->rec->flush_blocking();
  }
}

}  // namespace basalt
