/**
BSD 3-Clause License

This file is part of the Basalt project.
https://gitlab.com/VladyslavUsenko/basalt.git

Copyright (c) 2026, Pablo Vela.
All rights reserved.
*/

#pragma once

#include <basalt/io/dataset_io.h>

namespace basalt {

/// Offline reader for the fixed four-camera RoboCap coverage rig.
class RobocapIO final : public DatasetIoInterface {
 public:
  void read(const std::string& path) override;
  void reset() override;
  VioDatasetPtr get_data() override;

 private:
  VioDatasetPtr data_;
};

}  // namespace basalt
