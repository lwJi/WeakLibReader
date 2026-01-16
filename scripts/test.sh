#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="${BUILD_DIR:-build}"

# Create temp file for output capture
output_file=$(mktemp)
trap 'rm -f "$output_file"' EXIT

# Run tests and capture output
if ctest --test-dir "$BUILD_DIR" --output-on-failure >"$output_file" 2>&1; then
    echo "Test ✓"
else
    exit_code=$?
    cat "$output_file"
    exit $exit_code
fi
