# FluidAudioSwift

[![Swift](https://img.shields.io/badge/Swift-5.9+-orange.svg)](https://swift.org)
[![Platform](https://img.shields.io/badge/Platform-macOS%20%7C%20iOS-blue.svg)](https://developer.apple.com)

FluidAudioSwift is a high-performance Swift framework for on-device speaker diarization and audio processing, achieving **state-of-the-art results** competitive with academic research.

## 🎯 Performance

**AMI Benchmark Results** (Single Distant Microphone), a subset of the files:

- **DER: 17.7%** - Competitive with Powerset BCE 2023 (18.5%)
- **JER: 28.0%** - Outperforms EEND 2019 (25.3%) and x-vector clustering (28.7%)
- **RTF: 0.02x** - Real-time processing with 50x speedup

```text
  RTF = Processing Time / Audio Duration

  With RTF = 0.02x:
  - 1 minute of audio takes 0.02 × 60 = 1.2 seconds to process
  - 10 minutes of audio takes 0.02 × 600 = 12 seconds to process

  For real-time speech-to-text:
  - Latency: ~1.2 seconds per minute of audio
  - Throughput: Can process 50x faster than real-time
  - Pipeline impact: Minimal - diarization won't be the bottleneck
```

## Features

- **State-of-the-Art Diarization**: Research-competitive speaker separation with optimal speaker mapping
- **Speaker Embedding Extraction**: Generate speaker embeddings for voice comparison and clustering
- **CoreML Integration**: Native Apple CoreML backend for optimal performance on Apple Silicon and iOS support
- **Metal Performance Shaders**: GPU-accelerated computations with 3-8x speedup for batch operations
- **Real-time Processing**: Support for streaming audio processing with minimal latency
- **Cross-platform**: Full support for macOS 13.0+ and iOS 16.0+
- **Comprehensive CLI**: Professional benchmarking tools with beautiful tabular output
- **Comprehensive Benchmarking**: Built-in performance testing and optimization tools

## Installation

Add FluidAudioSwift to your project using Swift Package Manager:

```swift
dependencies: [
    .package(url: "https://github.com/FluidInference/FluidAudioSwift.git", from: "1.0.0"),
],
```

## Quick Start

```swift
import FluidAudioSwift

// Initialize and process audio
Task {
    let diarizer = DiarizerManager()
    try await diarizer.initialize()

    let audioSamples: [Float] = // your 16kHz audio data
    let result = try await diarizer.performCompleteDiarization(audioSamples, sampleRate: 16000)

    for segment in result.segments {
        print("\(segment.speakerId): \(segment.startTimeSeconds)s - \(segment.endTimeSeconds)s")
    }
}
```

## Configuration

Customize behavior with `DiarizerConfig`:

```swift
let config = DiarizerConfig(
    clusteringThreshold: 0.7,      // Speaker similarity (0.0-1.0, higher = stricter)
    minActivityThreshold: 10.0,    // Minimum activity frames for speaker detection
    minDurationOn: 1.0,           // Minimum speech duration (seconds)
    minDurationOff: 0.5,          // Minimum silence between speakers (seconds)
    numClusters: -1,              // Number of speakers (-1 = auto-detect)
    useMetalAcceleration: true,    // Enable GPU acceleration (recommended)
    metalBatchSize: 32,           // Optimal batch size for GPU operations
    debugMode: false
)
```

## Command Line Interface (CLI)

FluidAudioSwift includes a powerful CLI tool for benchmarking and processing audio files:

```bash
# Build the CLI
swift build

# Run AMI corpus benchmarks
swift run fluidaudio benchmark --dataset ami-sdm
swift run fluidaudio benchmark --dataset ami-ihm --threshold 0.8 --output results.json

# Process individual audio files
swift run fluidaudio process meeting.wav --output results.json
```

### CLI Commands

- **`benchmark`**: Run standardized research benchmarks on AMI Meeting Corpus
- **`process`**: Process individual audio files with speaker diarization
- **`help`**: Show detailed usage information and examples

### Supported Benchmark Datasets

- **AMI-SDM**: Single Distant Microphone (Mix-Headset.wav files) - realistic meeting conditions
- **AMI-IHM**: Individual Headset Microphones (Headset-0.wav files) - clean audio conditions

See [docs/CLI.md](docs/CLI.md) for complete CLI documentation and examples.

## Performance & Benchmarking

FluidAudioSwift includes comprehensive benchmarking tools to measure and optimize performance:

```bash
# Run complete benchmark suite
swift test --filter MetalAccelerationBenchmarks

# Run benchmarks with detailed reporting
./scripts/run-benchmarks.sh

# Research-standard AMI corpus evaluation
swift run fluidaudio benchmark --dataset ami-sdm --output benchmark-results.json
```

### Metal Acceleration

The framework automatically leverages Metal Performance Shaders for GPU acceleration:

- **3-8x speedup** for batch embedding calculations
- **Automatic fallback** to Accelerate framework when Metal unavailable
- **Optimal batch sizes** determined through continuous benchmarking
- **Memory efficient** GPU operations with smart caching

See [docs/BENCHMARKING.md](docs/BENCHMARKING.md) for detailed performance analysis and optimization guidelines.

For technical implementation details, see [docs/METAL_ACCELERATION.md](docs/METAL_ACCELERATION.md).

## API Reference

- **`DiarizerManager`**: Main diarization class
- **`performCompleteDiarization(_:sampleRate:)`**: Process audio and return speaker segments
- **`compareSpeakers(audio1:audio2:)`**: Compare similarity between two audio samples
- **`validateAudio(_:)`**: Validate audio quality and characteristics

## License

Apache 2.0 - see [LICENSE](LICENSE) for details.

## Acknowledgments

This project builds upon the excellent work of the [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) project for speaker diarization algorithms and techniques. We extend our gratitude to the sherpa-onnx contributors for their foundational work in on-device speech processing.


