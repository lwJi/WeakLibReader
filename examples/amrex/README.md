# AMReX CUDA Demo (Placeholder)

This directory contains a placeholder for the AMReX CUDA demonstration.

## Status

This is a minimal scaffold demonstrating the basic usage pattern. Full CUDA kernel integration will be completed in Phase 3 of development (see `AGENTS.md`).

## Planned Features

- GPU kernel example using `InterpLinearND` with AMReX `MultiFab`
- Device memory management with `TableDevice`
- Parallel interpolation across grid cells
- Performance benchmarking utilities

## Building

The example is not yet integrated into the main build system. Once complete, it will be built with:

```bash
cmake -S . -B build -DAMREX_ROOT=/path/to/amrex -DBUILD_EXAMPLES=ON
cmake --build build --target amrex_interp_demo
```

## Usage Example (Conceptual)

```cpp
// Load HDF5 table on host
WeakLibReader::Hdf5Table hostTable;
LoadHdf5Table("eos_table.h5", hostTable);

// Copy to device
auto deviceTable = WeakLibReader::MakeDeviceCopy(hostTable);

// Use in AMReX kernel
amrex::ParallelFor(mf, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
    double coords[3] = {rho[i], temp[j], ye[k]};
    auto view = deviceTable.View();
    double value = WeakLibReader::InterpLinearND(
        view.data, view.layout, view.axes, coords,
        WeakLibReader::InterpConfig{}, view.nd
    );
    mf(i, j, k) = value;
});
```
