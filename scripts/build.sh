#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="${BUILD_DIR:-build}"
JOBS="${JOBS:-$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)}"

# Create temp file for output capture
output_file=$(mktemp)
trap 'rm -f "$output_file"' EXIT

# Run build and capture output
if cmake --build "$BUILD_DIR" -j"$JOBS" >"$output_file" 2>&1; then
    echo "Build ✓"
else
    exit_code=$?
    cat "$output_file"
    exit $exit_code
fi
