# Cactus Thorn Integration

## Overview

Two thorns in `cactus_interface/`:
- **WeakLibReader** — Library thorn that provides the WeakLibReader capability and declares table file parameters
- **TestWeakLibReader** — Consumer thorn demonstrating EOS + opacity loading and GPU interpolation

**Note:** Thorn source files depend on Cactus-generated headers (`cctk.h`, `cctk_Arguments.h`, `cctk_Parameters.h`) and CarpetX/AMReX headers (`loop_device.hxx`). They will **not** pass standalone clangd analysis — this is expected.

## CCL File Patterns

### Grid Function Declarations (`interface.ccl`)

```ccl
USES INCLUDE HEADER: loop_device.hxx

CCTK_REAL energy TYPE=GF CENTERING={VVV} TAGS="checkpoint='yes'" "description"
```

- `TYPE=GF` — grid function storage
- `CENTERING={VVV}` — vertex-centered (all three dimensions)
- `TAGS="checkpoint='yes'"` — enables checkpoint/restart

### Scheduling (`schedule.ccl`)

```ccl
# Allocate storage for grid functions
STORAGE: energy
STORAGE: opacity_emab

# Global function (runs once, not per patch) — use for I/O and device memory
SCHEDULE MyThorn_LoadTable AT INITIAL
{
  LANG: C
  OPTIONS: global
} "Load table"

# Grid loop function (runs per patch) — use for interpolation
SCHEDULE MyThorn_Init AT INITIAL AFTER MyThorn_LoadTable
{
  LANG: C
  WRITES: energy(interior)
  SYNC: energy
} "Initialize grid function"

# Reading another grid function
SCHEDULE MyThorn_Compute AT INITIAL AFTER MyThorn_Init
{
  LANG: C
  READS: energy(interior)
  WRITES: opacity(interior)
  SYNC: opacity
} "Compute from energy"

# Cleanup at termination
SCHEDULE MyThorn_Cleanup AT TERMINATE
{
  LANG: C
  OPTIONS: global
} "Clean up before AMReX finalizes"
```

Key scheduling points: `AT INITIAL`, `AT TERMINATE`. Use `AFTER` for ordering.

### Parameter Declaration and Sharing (`param.ccl`)

Library thorn declares parameters as `RESTRICTED` for sharing:

```ccl
# In WeakLibReader/param.ccl
RESTRICTED:
STRING eos_table_file "eos table name (hdf5)" STEERABLE=RECOVER
{
  ".*" :: "can be anything"
} "foo.h5"
```

Consumer thorn imports via `SHARES`:

```ccl
# In TestWeakLibReader/param.ccl
SHARES: WeakLibReader
USES STRING eos_table_file
USES STRING opacity_emab_file
```

### Configuration (`configuration.ccl`)

```ccl
# Library thorn
REQUIRES Loop HDF5
PROVIDES WeakLibReader
{
  SCRIPT src/detect.sh
  LANG bash
  OPTIONS WEAKLIBREADER_DIR WEAKLIBREADER_INC_DIR
}

# Consumer thorn
REQUIRES Loop WeakLibReader
```

## C++ Implementation Patterns

### Function Signatures

```cpp
// Global function (I/O, device memory management)
extern "C" void MyThorn_LoadTable(CCTK_ARGUMENTS) {
  DECLARE_CCTK_PARAMETERS;            // Access parameters (e.g., eos_table_file)
  // No DECLARE_CCTK_ARGUMENTSX needed for global functions
}

// Grid loop function
extern "C" void MyThorn_Init(CCTK_ARGUMENTS) {
  DECLARE_CCTK_PARAMETERS;
  DECLARE_CCTK_ARGUMENTSX_MyThorn_Init;  // Access grid functions
}
```

- `DECLARE_CCTK_ARGUMENTSX_FunctionName` — name must match scheduled function exactly
- All scheduled functions use `extern "C"` linkage

### Device Loop Pattern

```cpp
grid.loop_int_device<0, 0, 0>(
    grid.nghostzones,
    [=] CCTK_DEVICE(const Loop::PointDesc &p) CCTK_ATTRIBUTE_ALWAYS_INLINE {
    // p.I = flat index into grid function arrays
    // p.x, p.y, p.z = physical coordinates
    energy(p.I) = WeakLibReader::LogInterpolateSingleVariable3DCustomPoint(
        p.x, p.y, p.z, axes, data, offset);
    });
```

- Template `<0, 0, 0>` = no staggering
- Capture `[=]` by value (device-safe — no host pointers)
- Grid functions accessed as 1D arrays via `p.I`

### Device Memory Management

```cpp
namespace MyThorn {
  // Namespace-scope device tables (persist across scheduled functions)
  WeakLibReader::WeakLibEosTableDevice eos_device;
  WeakLibReader::WeakLibOpacityTableDevice opacity_device;
  amrex::Gpu::DeviceVector<double> iso_slice_device;
}

// Loading: host load → device copy
extern "C" void MyThorn_LoadTable(CCTK_ARGUMENTS) {
  DECLARE_CCTK_PARAMETERS;
  WeakLibReader::WeakLibEosTable host_table;
  auto status = WeakLibReader::LoadWeakLibEosTableFullParallel(eos_table_file, host_table);
  if (status != WeakLibReader::Hdf5LoadStatus::Success) CCTK_ERROR("Failed");
  eos_device = WeakLibReader::MakeDeviceCopy(host_table);
}

// Cleanup: assign empty structures
extern "C" void MyThorn_Cleanup(CCTK_ARGUMENTS) {
  eos_device = WeakLibReader::WeakLibEosTableDevice{};
  iso_slice_device.clear();
}
```

### Extracting Device Pointers for Kernels

```cpp
// Before loop: extract device-safe values
const WeakLibReader::Axis axes[3] = {
  eos_device.axes[0], eos_device.axes[1], eos_device.axes[2]};
const double* data = eos_device.VariableData(iVariable);
const double offset = eos_device.offsets[iVariable];

// Then use in device loop via [=] capture
```

### Logging

- `CCTK_INFO("message")` — informational
- `CCTK_VINFO("format %d", val)` — formatted info
- `CCTK_ERROR("message")` — abort execution
