# FluidAudioSwift Benchmark Scripts

This directory contains scripts for running comprehensive Metal acceleration benchmarks.

## Available Scripts

### 1. Shell Script (Original)
- **File**: `run-benchmarks.sh`
- **Requirements**: Bash, Python 3, Swift toolchain
- **Usage**: `./scripts/run-benchmarks.sh`

### 2. Python Script with uv (Recommended)
- **File**: `run_benchmarks.py`
- **Requirements**: Python 3.8+, uv, Swift toolchain
- **Dependencies**: Managed by uv

## Using the Python Script with uv

### First-time Setup

1. **Install uv** (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Navigate to the scripts directory and initialize**:
   ```bash
   cd scripts
   uv init --no-readme --no-workspace
   uv add colorama
   ```

### Running Benchmarks

#### Option 1: Using uv run (recommended)
```bash
# From the FluidAudioSwift root directory
cd scripts && uv run run_benchmarks.py
# Note: The script will automatically check if it's in the right directory
```

#### Option 2: Direct execution with dependencies
```bash
# From the FluidAudioSwift root directory
./scripts/run_benchmarks.py
# Note: This requires colorama to be available in your system Python
```

## What the Benchmarks Do

Both scripts perform the same operations:

1. **Build** the FluidAudioSwift package in release mode
2. **Run** Metal acceleration benchmark tests
3. **Parse** and combine JSON results from multiple test methods
4. **Display** a colorized summary of benchmark results
5. **Save** detailed results to a timestamped JSON file

## Output

The benchmarks generate:
- **Console output**: Colorized summary with performance metrics
- **JSON file**: Detailed results saved as `benchmark_results_YYYYMMDD_HHMMSS.json`

### Example Summary Output
```
📊 Benchmark Results Summary:
===============================
✅ Metal Performance Shaders available
🕐 Timestamp: 2024-01-15T10:30:45Z
📈 Total tests run: 12
⚡ Average speedup: 2.45x
🚀 Best speedup: 4.12x
🎉 Excellent Metal acceleration performance!

📋 Test Breakdown:
   • Cosine Distance: 4 tests, 2.1x avg speedup
   • Vector Addition: 4 tests, 2.8x avg speedup
   • Matrix Multiplication: 4 tests, 2.5x avg speedup
```

## Advantages of Python Version

- **Better error handling**: More detailed error messages and graceful failures
- **Integrated dependencies**: No need for separate Python scripts
- **Cross-platform colors**: Uses colorama for consistent output across platforms
- **Type hints**: Better code maintainability and IDE support
- **uv integration**: Fast dependency management and virtual environments

## Requirements

- **Swift**: Xcode command line tools or Swift toolchain
- **Python**: 3.8 or later
- **uv**: Latest version (for Python script)
- **macOS/iOS**: Metal Performance Shaders (for GPU acceleration)

## Troubleshooting

### Common Issues

1. **"No benchmark results found"**
   - Ensure you're running from the FluidAudioSwift root directory
   - Check that Swift tests compile successfully
   - Verify MetalAccelerationBenchmarks tests exist

2. **Import errors in Python**
   - Make sure you're using the virtual environment: `source scripts/.venv/bin/activate`
   - Or use `uv run` which handles dependencies automatically

3. **Permission denied**
   - Make sure the script is executable: `chmod +x scripts/run_benchmarks.py`

4. **Swift build failures**
   - Ensure you have the latest Xcode command line tools
   - Try cleaning: `swift package clean` 