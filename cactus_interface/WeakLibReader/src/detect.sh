#! /bin/bash

################################################################################
# Prepare
################################################################################

# Set up shell
if [ "$(echo ${VERBOSE} | tr '[:upper:]' '[:lower:]')" = 'yes' ]; then
    set -x                      # Output commands
fi
set -e                          # Abort on errors

. $CCTK_HOME/lib/make/bash_utils.sh

THORN=WeakLibReader

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
THORN_DIR="$(cd "${SCRIPT_DIR}/../../../" && pwd)"

# Bundled headers live in the thorn's src/
BUNDLED_INC_DIR="${THORN_DIR}/src"

################################################################################
# Decide header location
################################################################################

WEAKLIBREADER_DIR_INPUT="${WEAKLIBREADER_DIR}"

# Treat DIR=BUILD (or empty) as "use bundled headers"
if [ "$(echo "${WEAKLIBREADER_DIR}" | tr '[:lower:]' '[:upper:]')" = 'BUILD' ] || [ -z "${WEAKLIBREADER_DIR}" ]; then
    echo "BEGIN MESSAGE"
    echo "Using bundled WeakLibReader headers..."
    echo "END MESSAGE"

    WEAKLIBREADER_BUILD=
    WEAKLIBREADER_DIR="${THORN_DIR}"
    WEAKLIBREADER_INC_DIRS="${BUNDLED_INC_DIR}"
else
    # User supplied a location (or NO_BUILD)
    WEAKLIBREADER_BUILD=

    if [ "${WEAKLIBREADER_DIR}" != "NO_BUILD" ]; then
        # If user didn't explicitly set INC_DIRS, pick a reasonable default:
        # prefer <DIR>/include if it exists, otherwise use <DIR> directly.
        if [ -z "${WEAKLIBREADER_INC_DIRS}" ]; then
            if [ -d "${WEAKLIBREADER_DIR}/include" ]; then
                WEAKLIBREADER_INC_DIRS="${WEAKLIBREADER_DIR}/include"
            else
                WEAKLIBREADER_INC_DIRS="${WEAKLIBREADER_DIR}"
            fi
        fi
    fi

    # Header-only: no link libraries
    WEAKLIBREADER_LIB_DIRS=""
    WEAKLIBREADER_LIBS=""
fi

################################################################################
# Configure Cactus
################################################################################

# Pass configuration options to build system
echo "BEGIN MAKE_DEFINITION"
echo "WEAKLIBREADER_BUILD       = ${WEAKLIBREADER_BUILD}"
echo "WEAKLIBREADER_INSTALL_DIR = ${WEAKLIBREADER_INSTALL_DIR}"
echo "END MAKE_DEFINITION"

# Set defaults (again) unless disabled
if [ "${WEAKLIBREADER_DIR}" != "NO_BUILD" ]; then
    : ${WEAKLIBREADER_INC_DIRS="${WEAKLIBREADER_DIR}/include"}
fi
: ${WEAKLIBREADER_LIB_DIRS=""}
: ${WEAKLIBREADER_LIBS=""}

# Normalize include dirs
WEAKLIBREADER_INC_DIRS="$(${CCTK_HOME}/lib/sbin/strip-incdirs.sh $WEAKLIBREADER_INC_DIRS)"

# Pass options to Cactus
echo "BEGIN MAKE_DEFINITION"
echo "WEAKLIBREADER_DIR      = ${WEAKLIBREADER_DIR}"
echo "WEAKLIBREADER_INC_DIRS = ${WEAKLIBREADER_INC_DIRS}"
echo "WEAKLIBREADER_LIB_DIRS = ${WEAKLIBREADER_LIB_DIRS}"
echo "WEAKLIBREADER_LIBS     = ${WEAKLIBREADER_LIBS}"
echo "END MAKE_DEFINITION"

# Emit directives for the build system
echo 'INCLUDE_DIRECTORY $(WEAKLIBREADER_INC_DIRS)'

# Only emit link directives if you ever add libs later
if [ -n "${WEAKLIBREADER_LIB_DIRS}" ]; then
    echo 'LIBRARY_DIRECTORY $(WEAKLIBREADER_LIB_DIRS)'
fi
if [ -n "${WEAKLIBREADER_LIBS}" ]; then
    echo 'LIBRARY           $(WEAKLIBREADER_LIBS)'
fi
