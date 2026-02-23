# HDF5 Table Formats

## Native WeakLib EOS Format

Used by `LoadWeakLibEosTableFull()` and `LoadWeakLibEosTableFullParallel()`.

```
/ThermoState/
  LogInterp[3]                      # Axis scale flags (0=linear, 1=log10)
  Dimensions[3]                     # Grid dimensions [nRho, nT, nYe]
  Names[3], Units[3]                # Axis metadata (string arrays)
  Density[nRho]                     # Density axis (Log10 scale typically)
  Temperature[nT]                   # Temperature axis (Log10 scale typically)
  Electron Fraction[nYe]            # Ye axis (Linear scale typically)

/DependentVariables/
  nVariables                        # Number of dependent variables (scalar int)
  Names[nVars], Units[nVars]        # Variable metadata
  Offsets[nVars]                    # Offset values for log-interpolation
  Repaired[nYe, nT, nRho]          # Repair mask (3D integer array)

  # Variable data arrays — one per variable
  {variable_name}[nYe, nT, nRho]   # Fortran order in file, reversed to C order on load

  # Index mappings (1-based in file, converted to 0-based)
  iPressure, iEntropyPerBaryon, iInternalEnergyDensity,
  iElectronChemicalPotential, iProtonChemicalPotential,
  iNeutronChemicalPotential, iProtonMassFraction, iNeutronMassFraction,
  iAlphaMassFraction, iHeavyMassFraction, iHeavyChargeNumber,
  iHeavyMassNumber, iHeavyBindingEnergy, iThermalEnergy, iGamma1
```

## WeakLib Opacity Format

Used by `LoadWeakLibOpacityTableFull()` and `LoadWeakLibOpacityTableFullParallel()`.

### Shared Groups

```
/EnergyGrid/
  Name[1], Unit[1]                  # Grid metadata
  nPoints                           # Number of energy points
  LogInterp                         # Scale flag (0=linear, 1=log10)
  Values[nPoints]                   # Grid values

/ThermoState/                       # Same structure as EOS ThermoState
  LogInterp[3], Dimensions[3], Names[3], Units[3]
  Density[nRho], Temperature[nT], Electron Fraction[nYe]

/EtaGrid/                           # Same structure as EnergyGrid (NES/Pair only)
  Name[1], Unit[1], nPoints, LogInterp, Values[nPoints]
```

### EmAb (Emission/Absorption)

```
/EmAb/                              # or legacy: /EmAb_CorrectedAbsorption/
  nOpacities, Units[nOp], Offsets[nOp]
  Electron Neutrino[nE, nRho, nT, nYe]
  Electron Antineutrino[nE, nRho, nT, nYe]

/EmAb Parameters/                   # Optional process flags
  np_FK, np_FK_inv_n_decay, np_isoenergetic, ...

/EC_table/                          # Optional electron capture table
  Spectrum[nRho, nT, nYe, nE], Rate[nRho, nT, nYe]
```

### Iso (Isoenergetic Scattering)

```
/Scat_Iso_Kernels/
  nOpacities, nMoments
  Offsets[nOpacities, nMoments]     # 2D (Fortran order in file)
  weak_magnetism_corr, ion_ion_corr, many_body_corr, ga_strange  # Optional
  Electron Neutrino[nE, nMom, nRho, nT, nYe]
  Electron Antineutrino[nE, nMom, nRho, nT, nYe]
```

### NES (Neutrino-Electron Scattering)

```
/Scat_NES_Kernels/
  nOpacities, nMoments, NPS         # NPS = optional flag
  Offsets[nOpacities, nMoments]
  Kernels[nE_in, nE_out, nMom, nT, nEta]
```

### Pair (Pair Production)

```
/Scat_Pair_Kernels/
  nOpacities, nMoments
  Offsets[nOpacities, nMoments]
  Kernels[nE_in, nE_out, nMom, nT, nEta]
```

### Brem (Bremsstrahlung)

```
/Scat_Brem_Kernels/
  nOpacities, nMoments
  Offsets[nOpacities, nMoments]
  S_sigma[nE_in, nE_out, nMom, nRho, nT]
```

**Note:** All multi-dimensional arrays are stored in Fortran order in HDF5 files and reversed to column-major (Fortran-style, stride-1 on first dimension) order on load.

## Key Loader Functions

| Function | Purpose |
|----------|---------|
| `LoadWeakLibEosTableFull()` | Load complete EOS table (ThermoState + DependentVariables) |
| `LoadWeakLibEosTableFullParallel()` | Parallel broadcast version of above |
| `LoadWeakLibOpacityTableFull()` | Master loader for all opacity subtables |
| `LoadWeakLibOpacityTableFullParallel()` | Parallel broadcast version of above |
| `MakeDeviceCopy()` | Copy host table to GPU device memory (2 public overloads: EOS and opacity) |
| `ExtractIsoMomentSlice4D()` | Extract contiguous 4D slice from 5D Iso kernel at fixed moment |
