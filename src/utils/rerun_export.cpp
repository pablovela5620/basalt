/**
Rerun (rerun.io) visualization backend — implementation.

This is the ONLY translation unit that includes `<rerun.hpp>`; everything else
in `basalt_vio` talks to `basalt::RerunExporter` through the PIMPL declared in
`basalt/utils/rerun_export.h`. Compiled only when `BASALT_RERUN` is defined.
*/
#include "basalt/utils/rerun_export.h"

#include <algorithm>
#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <rerun.hpp>

namespace basalt {

struct RerunExporter::Impl {
  std::unique_ptr<rerun::RecordingStream> rec;
  bool ok = false;
  std::vector<rerun::datatypes::Vec3D> traj;  // accumulated estimated positions (state thread only)

  static rerun::Color color_rgba(uint32_t c) {
    return rerun::Color(uint8_t(c >> 24), uint8_t(c >> 16), uint8_t(c >> 8), uint8_t(c));
  }
};

RerunExporter::RerunExporter(const std::string& app_id, const std::string& rrd_path, bool spawn)
    : impl_(std::make_unique<Impl>()) {
  const std::string id = app_id.empty() ? std::string("basalt_vio") : app_id;
  impl_->rec = std::make_unique<rerun::RecordingStream>(id);

  rerun::Error err;
  if (!rrd_path.empty()) {
    err = impl_->rec->save(rrd_path);
  } else if (spawn) {
    err = impl_->rec->spawn();
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
  impl_->rec->log_static("/world", rerun::ViewCoordinates::RIGHT_HAND_Z_UP);
}

void RerunExporter::log_static_calib(const std::vector<CamCalib>& cams) {
  if (!good()) return;
  for (size_t i = 0; i < cams.size(); ++i) {
    const CamCalib& c = cams[i];
    const Eigen::Vector3d p = c.T_i_c.translation();
    const Eigen::Quaterniond q = c.T_i_c.unit_quaternion();
    const std::string base = "/world/rig_0/cam_" + std::to_string(i);

    impl_->rec->log_static(
        base,
        rerun::Transform3D()
            .with_translation({static_cast<float>(p.x()), static_cast<float>(p.y()), static_cast<float>(p.z())})
            .with_rotation(rerun::Quaternion::from_xyzw(
                static_cast<float>(q.x()), static_cast<float>(q.y()), static_cast<float>(q.z()),
                static_cast<float>(q.w()))));

    // Linear pinhole approximation (Basalt cameras are KB4/radtan8/...; principal
    // point is centered here — see the integration plan, Risk #5).
    impl_->rec->log_static(
        base + "/pinhole",
        rerun::Pinhole::from_focal_length_and_resolution(
            {c.fx, c.fy}, {static_cast<float>(c.width), static_cast<float>(c.height)})
            .with_camera_xyz(rerun::components::ViewCoordinates::RDF)
            .with_image_plane_distance(0.1f));
  }
}

void RerunExporter::log_gt_path(const std::vector<Eigen::Vector3d>& gt_positions) {
  if (!good() || gt_positions.empty()) return;

  std::vector<rerun::datatypes::Vec3D> pts;
  pts.reserve(gt_positions.size());
  for (const auto& g : gt_positions) {
    pts.push_back({static_cast<float>(g.x()), static_cast<float>(g.y()), static_cast<float>(g.z())});
  }
  impl_->rec->log_static(
      "/world/rig_0_path",
      rerun::LineStrips3D(rerun::components::LineStrip3D(pts)).with_colors(rerun::Color(0x3c, 0xb0, 0x43)));

  const std::vector<rerun::datatypes::Vec3D> ends = {pts.front(), pts.back()};
  impl_->rec->log_static(
      "/world/rig_0_path/endpoints",
      rerun::Points3D(ends)
          .with_colors({rerun::Color(0x2e, 0xcc, 0x40), rerun::Color(0xff, 0x41, 0x36)})  // start green / end red
          .with_radii({0.02f, 0.02f}));
}

void RerunExporter::log_imu(const std::vector<int64_t>& t_ns, const std::vector<Eigen::Vector3d>& gyro,
                            const std::vector<Eigen::Vector3d>& accel, int64_t start_t_ns) {
  if (!good()) return;
  const size_t n = std::min(t_ns.size(), std::min(gyro.size(), accel.size()));
  for (size_t i = 0; i < n; ++i) {
    impl_->rec->set_time_duration_secs("sensor_time", static_cast<double>(t_ns[i] - start_t_ns) * 1e-9);
    impl_->rec->log("/world/rig_0/imu_0/gyro", rerun::Scalars({gyro[i].x(), gyro[i].y(), gyro[i].z()}));
    impl_->rec->log("/world/rig_0/imu_0/accel", rerun::Scalars({accel[i].x(), accel[i].y(), accel[i].z()}));
  }
}

void RerunExporter::begin_frame(int64_t frame_idx, int64_t t_ns, int64_t start_t_ns) {
  if (!good()) return;
  impl_->rec->set_time_sequence("frame", frame_idx);
  impl_->rec->set_time_duration_secs("sensor_time", static_cast<double>(t_ns - start_t_ns) * 1e-9);
}

void RerunExporter::log_pose(const Sophus::SE3d& T_w_i) {
  if (!good()) return;
  const Eigen::Vector3d p = T_w_i.translation();
  const Eigen::Quaterniond q = T_w_i.unit_quaternion();

  // Per-frame rig pose (world <- imu/body).
  impl_->rec->log(
      "/world/rig_0",
      rerun::Transform3D()
          .with_translation({static_cast<float>(p.x()), static_cast<float>(p.y()), static_cast<float>(p.z())})
          .with_rotation(rerun::Quaternion::from_xyzw(
              static_cast<float>(q.x()), static_cast<float>(q.y()), static_cast<float>(q.z()),
              static_cast<float>(q.w()))));

  // Growing estimated-trajectory polyline (logged in full each frame so that
  // scrubbing the `frame` timeline shows the path up to that frame).
  impl_->traj.push_back(
      {static_cast<float>(p.x()), static_cast<float>(p.y()), static_cast<float>(p.z())});
  impl_->rec->log(
      "/world/runs/basalt/trajectory",
      rerun::LineStrips3D(rerun::components::LineStrip3D(impl_->traj))
          .with_colors(rerun::Color(0xff, 0xa5, 0x00)));
}

void RerunExporter::log_image(int cam, const uint16_t* data, int width, int height, size_t pitch_bytes) {
  if (!good() || data == nullptr) return;
  // Basalt stores 8-bit sources as (v << 8); recover 8-bit, then JPEG-encode.
  const cv::Mat m16(height, width, CV_16UC1, const_cast<uint16_t*>(data), pitch_bytes);
  cv::Mat m8;
  m16.convertTo(m8, CV_8UC1, 1.0 / 256.0);
  std::vector<uint8_t> jpg;
  if (!cv::imencode(".jpg", m8, jpg)) return;
  const std::string path = "/world/rig_0/cam_" + std::to_string(cam) + "/pinhole/image";
  impl_->rec->log(path, rerun::EncodedImage::from_bytes(jpg, rerun::components::MediaType::jpeg()));
}

void RerunExporter::log_metrics(const Eigen::Vector3d& vel, const Eigen::Vector3d& bias_gyro,
                                const Eigen::Vector3d& bias_accel) {
  if (!good()) return;
  impl_->rec->log("/world/metrics/velocity", rerun::Scalars({vel.x(), vel.y(), vel.z()}));
  impl_->rec->log("/world/rig_0/imu_0/bias_gyro",
                  rerun::Scalars({bias_gyro.x(), bias_gyro.y(), bias_gyro.z()}));
  impl_->rec->log("/world/rig_0/imu_0/bias_accel",
                  rerun::Scalars({bias_accel.x(), bias_accel.y(), bias_accel.z()}));
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
    colors.push_back(Impl::color_rgba(i < rgba.size() ? rgba[i] : 0xffffffffu));
  }
  const std::string path = "/world/rig_0/cam_" + std::to_string(cam) + "/pinhole/keypoints";
  impl_->rec->log(path, rerun::Points2D(pts).with_colors(colors).with_radii(3.0f));
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
    colors.push_back(Impl::color_rgba(i < rgba.size() ? rgba[i] : 0xffffffffu));
  }
  impl_->rec->log("/world/rig_0/landmarks",
                  rerun::Points3D(pts).with_colors(colors).with_radii(0.02f));
}

RerunExporter::~RerunExporter() { flush(); }

bool RerunExporter::good() const { return impl_ && impl_->ok; }

void RerunExporter::flush() {
  if (impl_ && impl_->rec && impl_->ok) {
    (void)impl_->rec->flush_blocking();
  }
}

}  // namespace basalt
