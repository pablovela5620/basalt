/**
Rerun (rerun.io) visualization backend — implementation.

This is the ONLY translation unit that includes `<rerun.hpp>`; everything else
in `basalt_vio` talks to `basalt::RerunExporter` through the PIMPL declared in
`basalt/utils/rerun_export.h`. Compiled only when `BASALT_RERUN` is defined.
*/
#include "basalt/utils/rerun_export.h"

#include <iostream>

#include <rerun.hpp>

namespace basalt {

struct RerunExporter::Impl {
  std::unique_ptr<rerun::RecordingStream> rec;
  bool ok = false;
  int64_t frame_count = 0;
  std::vector<rerun::datatypes::Vec3D> traj;  // accumulated estimated positions
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

void RerunExporter::log_state(const Sophus::SE3d& T_w_i, int64_t t_ns, int64_t start_t_ns) {
  if (!good()) return;

  impl_->rec->set_time_sequence("frame", impl_->frame_count);
  impl_->rec->set_time_duration_secs("sensor_time", static_cast<double>(t_ns - start_t_ns) * 1e-9);

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

  ++impl_->frame_count;
}

RerunExporter::~RerunExporter() { flush(); }

bool RerunExporter::good() const { return impl_ && impl_->ok; }

void RerunExporter::flush() {
  if (impl_ && impl_->rec && impl_->ok) {
    (void)impl_->rec->flush_blocking();
  }
}

}  // namespace basalt
