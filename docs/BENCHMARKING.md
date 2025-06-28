# FluidAudioSwift Performance Benchmarking

This document provides comprehensive information about FluidAudioSwift's Metal acceleration benchmarking system, performance optimization, and how to interpret benchmark results.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Benchmark Categories](#benchmark-categories)
- [Running Benchmarks](#running-benchmarks)
- [Understanding Results](#understanding-results)
- [Performance Optimization](#performance-optimization)
- [CI Integration](#ci-integration)
- [Troubleshooting](#troubleshooting)

## Overview

FluidAudioSwift includes a comprehensive benchmarking system that measures the performance impact of Metal Performance Shaders (MPS) acceleration compared to the Accelerate framework. The benchmarking system helps:

- **Quantify performance improvements** from Metal GPU acceleration
- **Identify optimal configurations** for different hardware and workloads
- **Detect performance regressions** in continuous integration
- **Guide optimization decisions** for real-world applications

### Key Performance Benefits

- **3-8x speedup** for batch embedding similarity calculations
- **GPU parallelization** of compute-intensive operations
- **Memory efficiency** through optimized buffer management
- **Automatic fallback** to Accelerate when Metal unavailable

## Quick Start

### Running All Benchmarks

```bash
# Complete benchmark suite (5-10 minutes)
swift test --filter MetalAccelerationBenchmarks

# User-friendly reporting with the convenience script
./scripts/run-benchmarks.sh
```

### Running Specific Benchmark Categories

```bash
# Cosine distance batch size optimization
swift test --filter testCosineDistanceBatchSizeBenchmark

# End-to-end diarization performance
swift test --filter testEndToEndDiarizationBenchmark

# Memory usage analysis
swift test --filter testMemoryUsageBenchmark

# Powerset conversion GPU kernels
swift test --filter testPowersetConversionBatchSizeBenchmark
```

## Benchmark Categories

### 1. Cosine Distance Calculations

Tests Metal MPS matrix operations vs Accelerate vDSP for embedding similarity calculations.

**Test Variations:**
- **Batch sizes**: 8, 16, 32, 64, 128 embeddings
- **Embedding dimensions**: 256, 512, 1024 dimensions
- **Matrix scales**: Various query×candidate combinations

**What it measures:**
- Raw computation speed (milliseconds)
- Memory allocation overhead
- GPU vs CPU throughput
- Optimal batch size identification

### 2. Powerset Conversion Operations

Compares Metal compute kernels vs CPU for speaker activity aggregation.

**Test Variations:**
- **Batch sizes**: 1, 2, 4, 8 audio chunks
- **Frame counts**: 294, 589, 1178, 2356 frames (5s, 10s, 20s, 40s)

**What it measures:**
- GPU kernel dispatch efficiency
- Parallel frame processing speed
- Memory transfer overhead
- Throughput (frames per second)

### 3. End-to-End Diarization

Real-world performance testing with complete diarization pipelines.

**Test Variations:**
- **Audio durations**: 10s, 30s, 60s synthetic audio
- **Metal enabled vs disabled** configurations

**What it measures:**
- Complete pipeline performance
- Real-time processing factor
- Memory usage throughout processing
- Success rates and reliability

### 4. Memory Usage Analysis

Tracks peak memory consumption and efficiency improvements.

**Test Variations:**
- **Small**: 50×100 embeddings (512d)
- **Medium**: 100×200 embeddings (512d)  
- **Large**: 200×300 embeddings (1024d)

**What it measures:**
- Peak memory consumption
- Memory allocation patterns
- GPU memory efficiency
- Memory reduction percentages

### 5. Scalability Testing

Performance characteristics across different problem sizes.

**Test Variations:**
- **Query counts**: 16, 32, 64, 128
- **Candidate counts**: 25, 50, 100, 200
- **Embedding dimensions**: 256, 512, 1024

**What it measures:**
- Performance scaling characteristics
- GPU acceleration thresholds
- Memory bandwidth limitations
- Optimal configuration identification

## Running Benchmarks

### Local Development

For detailed local benchmarking with user-friendly output:

```bash
./scripts/run-benchmarks.sh
```

This script provides:
- ✅ **Colorized terminal output**
- 📊 **Performance summaries and recommendations**
- 💾 **Timestamped JSON results** saved to disk
- 🎯 **Optimization suggestions**

### Programmatic Access

For integration into other tools or automated analysis:

```bash
# Raw JSON output
swift test --filter MetalAccelerationBenchmarks 2>&1 | \
  sed -n '/🔬 BENCHMARK_RESULTS_JSON_START/,/🔬 BENCHMARK_RESULTS_JSON_END/p' | \
  sed '1d;$d' > benchmark_results.json
```

### Continuous Integration

Benchmarks automatically run on every pull request via GitHub Actions. See [CI Integration](#ci-integration) for details.

## Understanding Results

### JSON Output Structure

```json
{
  "timestamp": "2025-06-28T04:37:36Z",
  "metal_available": true,
  "tests": [
    {
      "test_name": "cosine_distance_batch_32",
      "test_type": "cosine_distance",
      "num_queries": 32,
      "num_candidates": 50,
      "embedding_dim": 512,
      "metal_time_ms": 7.94,
      "accelerate_time_ms": 48.40,
      "speedup": 6.09,
      "memory_increase_mb": 0.19,
      "metal_available": true
    }
  ]
}
```

### Key Metrics

#### Speedup Factor
- **> 3.0x**: Excellent Metal acceleration
- **2.0-3.0x**: Good Metal performance 
- **1.2-2.0x**: Moderate improvement
- **< 1.2x**: Limited benefit (GPU overhead)

#### Real-Time Factor
- **< 0.5x**: Faster than real-time (excellent)
- **0.5-1.0x**: Real-time capable (good)
- **> 1.0x**: Slower than real-time (needs optimization)

#### Memory Efficiency
- **Positive %**: Memory reduction vs Accelerate
- **Negative %**: Additional memory overhead
- **GPU memory**: Usually higher initial allocation, better efficiency at scale

### Performance Interpretation

#### When Metal Excels
- **Large batch sizes** (32+ embeddings)
- **High-dimensional embeddings** (512+ dimensions)
- **Repeated operations** (amortized setup cost)
- **Parallel workloads** (multiple audio streams)

#### When Accelerate May Be Better
- **Small operations** (< 16 embeddings)
- **Single computations** (high GPU setup overhead)
- **Memory-constrained environments**
- **Legacy hardware** without Metal support

## Performance Optimization

### Configuration Tuning

#### Optimal Batch Sizes
Based on continuous benchmarking, recommended configurations:

```swift
let config = DiarizerConfig(
    // For most workloads
    metalBatchSize: 32,
    useMetalAcceleration: true,
    
    // For memory-constrained environments
    metalBatchSize: 16,
    
    // For high-throughput applications
    metalBatchSize: 64
)
```

#### Hardware-Specific Optimization

**Apple Silicon (M1/M2/M3):**
- ✅ Use Metal acceleration (3-8x speedup typical)
- ✅ Batch size 32-64 optimal
- ✅ Enable parallel processing for >60s audio

**Intel Macs:**
- ⚠️ Limited Metal acceleration benefits
- ✅ Accelerate framework performs well
- ✅ Focus on CPU-based optimizations

**iOS Devices:**
- ✅ Metal acceleration beneficial on A12+ chips
- ⚠️ Consider memory constraints (use smaller batches)
- ✅ Optimize for thermal management

### Application-Level Optimization

#### For Real-Time Processing
```swift
let realtimeConfig = DiarizerConfig(
    metalBatchSize: 16,           // Lower latency
    useEarlyTermination: true,    // Stop early when possible
    embeddingCacheSize: 50,       // Reduce memory usage
    parallelProcessingThreshold: 30.0  // Shorter parallel threshold
)
```

#### For Batch Processing
```swift
let batchConfig = DiarizerConfig(
    metalBatchSize: 64,           // Maximum throughput
    embeddingCacheSize: 200,      // Larger cache for efficiency
    parallelProcessingThreshold: 10.0,  // Aggressive parallelization
    useMetalAcceleration: true
)
```

#### For Memory-Constrained Environments
```swift
let memoryConfig = DiarizerConfig(
    metalBatchSize: 16,           // Smaller GPU allocations
    embeddingCacheSize: 25,       // Reduced cache size
    fallbackToAccelerate: true,   // Graceful degradation
    useEarlyTermination: true     // Minimize computation
)
```

## CI Integration

### GitHub Actions Workflow

The benchmark system integrates with GitHub Actions to provide automated performance monitoring:

#### Pull Request Comments

Every PR automatically receives a detailed performance report:

```markdown
## 🚀 Metal Acceleration Benchmark Results

### Performance Summary
- **Overall Average Speedup**: 3.2x faster with Metal acceleration
- **Best Speedup Achieved**: 6.1x faster
- **Optimal Batch Size**: 32 embeddings
- **Average Memory Reduction**: 15% lower peak usage

### Detailed Performance Results
| Operation | Configuration | Metal (ms) | Accelerate (ms) | Speedup |
|-----------|---------------|------------|-----------------|---------|
| Cosine Distance (batch_32) | 32×50 (512d) | 7.9 | 48.4 | 6.1x |
| Powerset Conv (batch_4) | 4 batch, 589 frames | 8.1 | 28.4 | 3.5x |
| End-to-End Diarization | 30s audio | 145.2 | 421.8 | 2.9x |

### Recommendations
✅ **Excellent performance improvement** - Metal acceleration is highly beneficial
- Use batch size of **32** for optimal performance
- Metal acceleration is most beneficial for large embedding matrices
```

#### Performance Regression Detection

The CI system automatically detects performance regressions:

- **> 10% slower**: Fails the CI check
- **5-10% slower**: Warning in PR comment
- **Improved performance**: Celebration message

#### Baseline Comparison

Each PR is compared against the main branch baseline to detect:
- Performance improvements or regressions
- Configuration changes impact
- Hardware-specific variations

### Workflow Configuration

The benchmark workflow runs:
- **On every PR** to `main` branch
- **On changes to** Swift source files or workflows
- **With 30-minute timeout** for comprehensive testing
- **On macOS-latest runners** with Apple Silicon

## Troubleshooting

### Common Issues

#### Metal Not Available
```
ℹ️ Metal Performance Shaders not available on this runner
```

**Solutions:**
- Expected on some CI environments
- Framework automatically falls back to Accelerate
- Local testing on Metal-capable hardware recommended

#### Poor Performance Results
```
⚠️ Metal MPS speedup lower than expected (may vary by hardware)
```

**Potential Causes:**
- Small batch sizes (try increasing `metalBatchSize`)
- GPU memory limitations (reduce problem size)
- Thermal throttling (allow cooling between tests)
- Background GPU usage (close other GPU-intensive apps)

#### Memory Issues
```
Failed to allocate Metal buffers
```

**Solutions:**
- Reduce batch size or embedding dimensions
- Close other applications using GPU memory
- Enable `fallbackToAccelerate` for graceful degradation
- Monitor system memory usage during benchmarks

#### Test Timeouts
```
Test timed out after 30 seconds
```

**Solutions:**
- Check for infinite loops in benchmark code
- Reduce test problem sizes for CI environments
- Increase timeout in workflow configuration
- Verify GPU drivers are up to date

### Debugging Performance Issues

#### Enable Debug Logging
```swift
let config = DiarizerConfig(
    debugMode: true,  // Enable detailed logging
    useMetalAcceleration: true
)
```

#### Profile Memory Usage
```bash
# Monitor memory during benchmarks
swift test --filter testMemoryUsageBenchmark & \
top -pid $! -s 1
```

#### Analyze GPU Usage
```bash
# Monitor GPU utilization (macOS)
sudo powermetrics --samplers gpu_power -n 1 --hide-cpu-duty-cycle
```

### Performance Validation

#### Expected Performance Ranges

**Cosine Distance (32×50, 512d):**
- Metal: 5-15ms (Apple Silicon)
- Accelerate: 30-60ms
- Speedup: 3-8x

**End-to-End Diarization (30s audio):**
- Metal: 100-300ms (Apple Silicon)
- Accelerate: 300-800ms  
- Real-time factor: 0.3-1.0x

**Memory Usage:**
- Metal: 2-10MB additional GPU allocation
- Accelerate: 1-5MB CPU allocation
- Net efficiency: 10-30% improvement at scale

#### Reporting Performance Issues

When reporting performance issues, please include:

1. **Hardware specifications** (chip, memory, OS version)
2. **Complete benchmark results** (JSON output)
3. **Configuration used** (DiarizerConfig parameters)
4. **Expected vs actual performance** 
5. **Reproducible test case** (if possible)

---

## Additional Resources

- **Source Code**: [`MetalAccelerationBenchmarks.swift`](../Tests/FluidAudioSwiftTests/MetalAccelerationBenchmarks.swift)
- **CI Workflow**: [`.github/workflows/metal-benchmarks.yml`](../.github/workflows/metal-benchmarks.yml)
- **Benchmark Script**: [`scripts/run-benchmarks.sh`](../scripts/run-benchmarks.sh)
- **Project Documentation**: [`CLAUDE.md`](../CLAUDE.md)

For questions or contributions to the benchmarking system, please open an issue or pull request on GitHub.