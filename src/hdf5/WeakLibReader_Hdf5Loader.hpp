#pragma once

// Types and internal helpers
#include "hdf5/WeakLibReader_Hdf5Types.hpp"
#include "hdf5/WeakLibReader_Hdf5LoaderDetail.hpp"

// Serial loaders
#include "hdf5/WeakLibReader_Hdf5LoaderTable.hpp"
#include "hdf5/WeakLibReader_Hdf5LoaderEos.hpp"
#include "hdf5/WeakLibReader_Hdf5LoaderOpacityEmAb.hpp"
#include "hdf5/WeakLibReader_Hdf5LoaderOpacityScat.hpp"
#include "hdf5/WeakLibReader_Hdf5LoaderOpacityMain.hpp"

// Device copy
#include "hdf5/WeakLibReader_Hdf5LoaderOpacityDevice.hpp"

// MPI-parallel loaders
#include "hdf5/WeakLibReader_Hdf5LoaderParallel.hpp"
#include "hdf5/WeakLibReader_Hdf5LoaderOpacityParallel.hpp"
