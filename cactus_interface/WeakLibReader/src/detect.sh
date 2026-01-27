#! /bin/bash

################################################################################
# Prepare
################################################################################

# Set up shell
if [ "$(echo ${VERBOSE} | tr '[:upper:]' '[:lower:]')" = 'yes' ]; then
    set -x                      # Output commands
fi
set -e                          # Abort on errors

#. $CCTK_HOME/lib/make/bash_utils.sh

#THORN=WeakLibReader

SCRIPT_DIR="$(cd -P "$(dirname "$0")" && pwd -P)"
INTERFACE_DIR="$(cd -P "${SCRIPT_DIR}/../../" && pwd -P)"
PACKAGE_DIR="$(cd -P "${INTERFACE_DIR}/../" && pwd -P)"

# Bundled headers live in the thorn's src/
BUNDLED_INC_DIR="${PACKAGE_DIR}/src"

################################################################################
# Decide header location
################################################################################

echo "BEGIN MESSAGE"
echo "Using bundled WeakLibReader headers..."
echo "END MESSAGE"

WEAKLIBREADER_INC_DIRS="${BUNDLED_INC_DIR}"

################################################################################
# Configure Cactus
################################################################################

# Normalize include dirs
WEAKLIBREADER_INC_DIRS="$(${CCTK_HOME}/lib/sbin/strip-incdirs.sh $WEAKLIBREADER_INC_DIRS)"

# Emit directives for the build system
echo 'INCLUDE_DIRECTORY ${WEAKLIBREADER_INC_DIRS}'
