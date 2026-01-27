#pragma once

#include <AMReX_Array.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_TableData.H>
#include <AMReX_Vector.H>

#include <array>
#include <cstdint>

#include "WeakLibReader_Layout.hpp"
#include "WeakLibReader_AxisTypes.hpp"

namespace WeakLibReader {

enum class Hdf5LoadStatus : std::uint8_t {
  Success = 0,
  FileOpenFailed,
  DatasetOpenFailed,
  DatasetRankInvalid,
  DatasetReadFailed,
  AxisDatasetOpenFailed,
  AxisExtentMismatch,
  AxisNotMonotone,
  AxisInvalidScale,
  AxisReadFailed,
  IncompatibleDatasetExtent
};

struct Hdf5LoadConfig {
  std::string valueDataset = "values";
  std::string axisPrefix = "axis";
  std::string axisScaleAttribute = "scale";
};

struct TableView {
  int nd = 0;
  Layout layout{};
  Axis axes[5]{};
  const double* data = nullptr;
};

struct TableDevice {
  int nd = 0;
  Layout layout{};
  Axis axes[5]{};
  amrex::TableData<double, 4> values{};
  std::array<amrex::Gpu::DeviceVector<double>, 5> axisStorage{};

  [[nodiscard]] TableView View() const noexcept
  {
    TableView view{};
    view.nd = nd;
    view.layout = layout;
    view.data = values.const_table().p;
    for (int dim = 0; dim < 5; ++dim) {
      view.axes[dim] = axes[dim];
    }
    return view;
  }
};

struct Hdf5Table {
  int nd = 0;
  std::array<int, 5> extents{{1, 1, 1, 1, 1}};
  Layout layout{};
  Axis axes[5]{};
  amrex::TableData<double, 4> values{};
  std::array<amrex::Vector<double>, 5> axisStorage{};

  Hdf5Table() = default;
  Hdf5Table(Hdf5Table&&) = default;
  Hdf5Table& operator=(Hdf5Table&&) = default;
  Hdf5Table(const Hdf5Table&) = delete;
  Hdf5Table& operator=(const Hdf5Table&) = delete;

  [[nodiscard]] double* DataPtr() noexcept { return values.table().p; }
  [[nodiscard]] const double* DataPtr() const noexcept { return values.const_table().p; }

  [[nodiscard]] TableView View() const noexcept
  {
    TableView view{};
    view.nd = nd;
    view.layout = layout;
    view.data = values.const_table().p;
    for (int dim = 0; dim < 5; ++dim) {
      view.axes[dim] = axes[dim];
    }
    return view;
  }
};

} // namespace WeakLibReader
