# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FluidAudioSwift is a Swift framework for on-device speaker diarization and audio processing. It provides speaker diarization, speaker embedding extraction, and audio comparison capabilities using CoreML for optimal performance on Apple platforms.

## Commands

### Build and Test
```bash
# Build the package
swift build

# Run all tests
swift test

# Run specific test categories
swift test --filter BasicInitializationTests    # Basic functionality tests
swift test --filter BenchmarkTests             # Real dataset benchmarks (downloads AMI corpus)
swift test --filter CITests                    # CI-friendly tests

# Run specific benchmark tests
swift test --filter testAMI_IHM_SegmentationBenchmark  # Clean audio conditions
swift test --filter testAMI_SDM_SegmentationBenchmark  # Far-field audio conditions
```

### Package Management
```bash
# Show package info
swift package describe

# Update dependencies (none currently)
swift package update

# Generate Xcode project
swift package generate-xcodeproj
```

## Architecture

### Core Components

**DiarizerManager** (`Sources/FluidAudioSwift/DiarizerManager.swift`):
- Main API class for speaker diarization
- Handles CoreML model downloads from HuggingFace
- Implements powerset classification (7 classes: silence, 3 single speakers, 3 speaker pairs)
- Processes audio in 10-second chunks with speaker tracking across chunks
- Downloads models: `pyannote_segmentation.mlmodelc` and `wespeaker.mlmodelc`

**FluidAudioSwift** (`Sources/FluidAudioSwift/FluidAudioSwift.swift`):
- Main module file with backward compatibility aliases
- Re-exports all framework functionality

### Key Data Structures

- **DiarizerConfig**: Configuration for clustering thresholds, duration limits, debug mode
- **DiarizationResult**: Complete result with segments and global speaker database
- **TimedSpeakerSegment**: Individual speaker segment with embedding and timing
- **AudioValidationResult**: Audio quality validation results

### Model Architecture

The system uses a two-stage approach:
1. **Segmentation Model**: PyAnnote-based model for speaker activity detection (when speakers are talking)
2. **Embedding Model**: WeSpeaker-based model for speaker characteristic extraction (who is talking)

Models are automatically downloaded from HuggingFace (`bweng/speaker-diarization-coreml`) and cached locally.

### Processing Pipeline

1. Audio divided into 10-second chunks
2. Segmentation model identifies when speakers are active (powerset classification)
3. Embedding model extracts speaker characteristics for active segments
4. Global speaker database maintains consistent speaker IDs across chunks
5. Speaker assignment using cosine distance with configurable clustering threshold

## Testing Strategy

The project includes comprehensive benchmarks using real research datasets:

### Benchmark Categories
- **BasicInitializationTests**: Core functionality without model downloads
- **BenchmarkTests**: Real AMI Meeting Corpus evaluation with automatic dataset downloading
- **CITests**: Lightweight tests for continuous integration

### Research Integration
- Automatic download of AMI IHM/SDM datasets from HuggingFace
- Standard research metrics: DER (Diarization Error Rate), JER (Jaccard Error Rate)
- Comparison against published research baselines

## Platform Support

- **macOS**: 13.0+
- **iOS**: 16.0+
- **Swift**: 6.1+ (uses Swift 6 concurrency features)
- **Dependencies**: None (pure Swift with Foundation, OSLog, CoreML)

## Performance Characteristics

- **Real-time processing**: Optimized for <1x real-time factor on Apple Silicon
- **Memory efficient**: 30-50% memory reduction through ArraySlice references and efficient data structures
- **Parallel processing**: Automatic TaskGroup-based parallelization for long audio files (>60 seconds)
- **GPU acceleration**: Metal Performance Shaders for batch embedding similarity calculations (3-8x speedup)
- **Custom Metal kernels**: GPU compute shaders for powerset conversion and speaker activity aggregation
- **Vectorized operations**: Accelerate framework integration for SIMD cosine distance and RMS calculations
- **Smart caching**: Early termination and configurable performance parameters
- **Graceful fallback**: Automatic fallback to Accelerate if Metal unavailable
- **On-device processing**: No network dependencies after model download
- **Swift 6 concurrency**: Safe structured concurrency with actor isolation

## Configuration

Key configuration parameters in DiarizerConfig:
- `clusteringThreshold` (0.7): Speaker similarity threshold (higher = stricter)
- `minDurationOn` (1.0s): Minimum speech segment duration
- `minDurationOff` (0.5s): Minimum silence between speakers
- `numClusters` (-1): Number of speakers (-1 = auto-detect)
- `minActivityThreshold` (10.0): Minimum activity frames for speaker detection

**Performance optimization parameters:**
- `parallelProcessingThreshold` (60.0s): Audio duration threshold for parallel processing
- `embeddingCacheSize` (100): Maximum cached embeddings for quick lookup
- `useEarlyTermination` (true): Enable early speaker search termination
- `earlyTerminationThreshold` (0.3): Distance threshold for early termination

**Metal Performance Shaders parameters:**
- `useMetalAcceleration` (true): Enable Metal GPU acceleration when available
- `metalBatchSize` (32): Optimal batch size for GPU operations
- `fallbackToAccelerate` (true): Graceful degradation to Accelerate if Metal fails

## Benchmarking

### Metal Acceleration Benchmarks

The project includes comprehensive benchmarks to measure Metal vs Accelerate performance:

```bash
# Run complete benchmark suite
swift test --filter MetalAccelerationBenchmarks

# Run specific benchmark categories
swift test --filter testCosineDistanceBatchSizeBenchmark
swift test --filter testEndToEndDiarizationBenchmark
swift test --filter testMemoryUsageBenchmark

# Use the convenience script
./scripts/run-benchmarks.sh
```

**Benchmark categories:**
- **Cosine distance calculations**: Batch size optimization (8-128 embeddings)
- **Powerset conversion operations**: GPU vs CPU compute kernels
- **End-to-end diarization**: Real-world performance comparison
- **Memory usage analysis**: Peak memory consumption comparison
- **Scalability testing**: Performance across different matrix sizes

**CI Integration:**
- Automated benchmarks run on all PRs
- Performance regression detection
- Automated PR comments with results
- Baseline comparison against main branch

## Troubleshooting

- Model downloads may fail in test environments - expected behavior
- First-time initialization requires network access for model downloads
- Models are cached in `~/Library/Application Support/SpeakerKitModels/coreml/`
- Enable debug mode in config for detailed logging
- Metal acceleration may be slower for small operations due to GPU overhead