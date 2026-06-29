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

RerunExporter::~RerunExporter() { flush(); }

bool RerunExporter::good() const { return impl_ && impl_->ok; }

void RerunExporter::flush() {
  if (impl_ && impl_->rec && impl_->ok) {
    (void)impl_->rec->flush_blocking();
  }
}

}  // namespace basalt
