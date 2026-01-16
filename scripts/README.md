# Build Scripts

Wrapper scripts that provide quiet output on success.

## Usage

```bash
# Build (assumes build/ directory is already configured)
./scripts/build.sh

# Run tests
./scripts/test.sh

# Run build and test together
./scripts/check.sh
```

## Configuration

Environment variables:
- `BUILD_DIR` - Build directory (default: `build`)
- `JOBS` - Parallel jobs for build (default: auto-detected)

## Output

- Success: Single line with checkmark
- Failure: Full captured output followed by non-zero exit
