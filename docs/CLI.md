# FluidAudioSwift CLI Documentation

The FluidAudioSwift Command Line Interface (CLI) provides powerful tools for benchmarking speaker diarization performance and processing audio files from the command line.

## Table of Contents

- [Installation](#installation)
- [Commands Overview](#commands-overview)
- [Benchmark Command](#benchmark-command)
- [Process Command](#process-command)
- [AMI Dataset Setup](#ami-dataset-setup)
- [Output Formats](#output-formats)
- [Performance Metrics](#performance-metrics)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)

## Installation

Build the CLI using Swift Package Manager:

```bash
cd FluidAudioSwift
swift build
```

The CLI will be available as `fluidaudio` in the build output.

## Commands Overview

```bash
swift run fluidaudio <command> [options]
```

### Available Commands

- **`benchmark`**: Run standardized research benchmarks on AMI Meeting Corpus
- **`process`**: Process individual audio files with speaker diarization  
- **`help`**: Show detailed usage information and examples

## Benchmark Command

Run standardized benchmarks on research datasets to evaluate diarization performance.

### Usage

```bash
swift run fluidaudio benchmark [options]
```

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--dataset` | string | `ami-sdm` | Dataset to use (`ami-sdm`, `ami-ihm`) |
| `--threshold` | float | `0.7` | Clustering threshold (0.0-1.0, higher = stricter) |
| `--debug` | flag | `false` | Enable debug mode for detailed logging |
| `--output` | string | `stdout` | Output results to JSON file |

### Supported Datasets

#### AMI-SDM (Single Distant Microphone)
- **Files**: Mix-Headset.wav files
- **Conditions**: Realistic meeting room acoustics, far-field audio
- **Use Case**: Evaluates performance in real-world meeting scenarios
- **Expected DER**: 25-35% (research baseline)

#### AMI-IHM (Individual Headset Microphones)  
- **Files**: Headset-0.wav files
- **Conditions**: Clean close-talking audio
- **Use Case**: Evaluates performance in optimal audio conditions
- **Expected DER**: 18-28% (typically 5-10% lower than SDM)

### Examples

```bash
# Run AMI SDM benchmark with default settings
swift run fluidaudio benchmark

# Run AMI IHM benchmark with custom threshold
swift run fluidaudio benchmark --dataset ami-ihm --threshold 0.8

# Save benchmark results to JSON file
swift run fluidaudio benchmark --dataset ami-sdm --output results.json --debug
```

## Process Command

Process individual audio files with speaker diarization.

### Usage

```bash
swift run fluidaudio process <audio-file> [options]
```

### Supported Audio Formats

- `.wav` (recommended)
- `.m4a`
- `.mp3`

Audio is automatically resampled to 16kHz mono for processing.

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--threshold` | float | `0.7` | Clustering threshold (0.0-1.0) |
| `--debug` | flag | `false` | Enable debug mode |
| `--output` | string | `stdout` | Output results to JSON file |

### Examples

```bash
# Process audio file with default settings
swift run fluidaudio process meeting.wav

# Process with custom threshold and save results
swift run fluidaudio process meeting.wav --threshold 0.6 --output output.json

# Process with debug information
swift run fluidaudio process interview.m4a --debug
```

## AMI Dataset Setup

To run benchmarks on the AMI Meeting Corpus, you need to download the official dataset:

### Download Instructions

1. **Visit**: https://groups.inf.ed.ac.uk/ami/download/
2. **Register** for dataset access (free for research)
3. **Select meetings**: ES2002a, ES2003a, ES2004a, ES2005a, IS1000a, IS1001a, IS1002a, TS3003a, TS3004a
4. **Choose audio streams**:
   - For AMI-SDM: Download **"Headset mix"** files (Mix-Headset.wav)
   - For AMI-IHM: Download **"Individual headsets"** files (Headset-0.wav)

### File Organization

Place downloaded files in the following directory structure:

```
~/FluidAudioSwift_Datasets/
└── ami_official/
    ├── sdm/
    │   ├── ES2002a.Mix-Headset.wav
    │   ├── ES2003a.Mix-Headset.wav
    │   └── ...
    └── ihm/
        ├── ES2002a.Headset-0.wav
        ├── ES2003a.Headset-0.wav
        └── ...
```

### Verification

Run the benchmark command to verify your setup:

```bash
swift run fluidaudio benchmark --dataset ami-sdm
```

If files are missing, the CLI will show specific download instructions.

## Output Formats

### Console Output

Standard console output shows real-time progress and results:

```
🚀 Starting AMI-SDM benchmark evaluation
   Clustering threshold: 0.7
   Debug mode: disabled
✅ Models initialized successfully
📊 Running AMI SDM Benchmark
   🎵 Processing ES2002a.Mix-Headset.wav...
     ✅ DER: 23.4%, JER: 15.2%, RTF: 0.34x

🏆 AMI SDM Benchmark Results:
   Average DER: 25.1%
   Average JER: 16.8%
   Processed Files: 7/9
```

### JSON Output

Use `--output filename.json` to save detailed results:

#### Benchmark Results

```json
{
  "dataset": "AMI-SDM",
  "averageDER": 25.1,
  "averageJER": 16.8,
  "processedFiles": 7,
  "totalFiles": 9,
  "timestamp": "2024-01-15T10:30:00Z",
  "results": [
    {
      "meetingId": "ES2002a",
      "durationSeconds": 1847.2,
      "processingTimeSeconds": 625.8,
      "realTimeFactor": 0.34,
      "der": 23.4,
      "jer": 15.2,
      "speakerCount": 4,
      "segments": [...]
    }
  ]
}
```

#### Processing Results

```json
{
  "audioFile": "meeting.wav",
  "durationSeconds": 120.5,
  "processingTimeSeconds": 45.2,
  "realTimeFactor": 0.38,
  "speakerCount": 3,
  "timestamp": "2024-01-15T10:30:00Z",
  "segments": [
    {
      "speakerId": "Speaker 1",
      "startTimeSeconds": 0.0,
      "endTimeSeconds": 15.3,
      "qualityScore": 0.89,
      "embedding": [0.1, 0.2, ...]
    }
  ],
  "config": {
    "clusteringThreshold": 0.7,
    "minDurationOn": 1.0,
    "debugMode": false
  }
}
```

## Performance Metrics

### Diarization Error Rate (DER)

Primary metric used in speaker diarization research:

```
DER = (Missed Speech + False Alarm + Speaker Error) / Total Speech Time × 100%
```

- **Missed Speech**: Speech segments not detected
- **False Alarm**: Non-speech detected as speech  
- **Speaker Error**: Speech assigned to wrong speaker
- **Lower is better** (0% = perfect)

### Jaccard Error Rate (JER)

Measures overall temporal accuracy:

```
JER = (Total Duration - Overlap Duration) / Union Duration × 100%
```

- **Overlap**: Time where prediction matches ground truth
- **Union**: Total time covered by either prediction or ground truth
- **Lower is better** (0% = perfect)

### Real-Time Factor (RTF)

Processing speed relative to audio duration:

```
RTF = Processing Time / Audio Duration
```

- **RTF < 1.0**: Faster than real-time (good for streaming)
- **RTF = 1.0**: Real-time processing
- **RTF > 1.0**: Slower than real-time

### Research Baselines

#### AMI-SDM (Far-field audio)
- **State-of-the-art (2023)**: 18.5% DER (Powerset BCE)
- **Strong baseline**: 25.3% DER (EEND)
- **Traditional methods**: 28.7% DER (x-vector clustering)

#### AMI-IHM (Close-talking audio)
- **Typically 5-10% lower DER** than SDM
- **Expected range**: 15-25% DER for modern systems

## Examples

### Basic Benchmarking

```bash
# Quick AMI-SDM benchmark
swift run fluidaudio benchmark

# Comprehensive evaluation with output
swift run fluidaudio benchmark --dataset ami-ihm --output ami-ihm-results.json
```

### Audio Processing

```bash
# Process meeting recording
swift run fluidaudio process board-meeting.wav --output meeting-results.json

# Process with stricter speaker separation
swift run fluidaudio process interview.wav --threshold 0.8
```

### Batch Processing Script

```bash
#!/bin/bash
# Process multiple files
for file in audio/*.wav; do
    echo "Processing $file..."
    swift run fluidaudio process "$file" --output "results/$(basename "$file" .wav).json"
done
```

### Performance Tuning

```bash
# Test different thresholds
for threshold in 0.5 0.6 0.7 0.8 0.9; do
    echo "Testing threshold: $threshold"
    swift run fluidaudio benchmark --threshold $threshold --output "results-$threshold.json"
done
```

## Troubleshooting

### Common Issues

#### Models Not Found
```
❌ Failed to initialize models: Model file not found
💡 Make sure you have network access for model downloads
```

**Solution**: Ensure internet connectivity for first-time model download. Models are cached locally after initial download.

#### Audio File Issues
```
❌ Failed to process audio file: Unsupported format
```

**Solution**: Convert audio to WAV format or ensure file is readable:
```bash
ffmpeg -i input.mp4 -ar 16000 -ac 1 output.wav
```

#### Dataset Not Found
```
⚠️ AMI SDM dataset not found
📥 Download instructions: ...
```

**Solution**: Follow the [AMI Dataset Setup](#ami-dataset-setup) instructions.

#### Poor Performance Results

**High DER (>50%)**:
- Check audio quality (noise, overlapping speech)
- Try different clustering thresholds (0.5-0.9)
- Ensure proper ground truth alignment

**Slow Processing (RTF >> 1.0)**:
- Enable Metal acceleration (should be automatic)
- Check system resources and memory usage
- Consider shorter audio segments for testing

### Debug Mode

Enable debug mode for detailed information:

```bash
swift run fluidaudio benchmark --debug
swift run fluidaudio process audio.wav --debug
```

Debug output includes:
- Model loading details
- Audio preprocessing information
- Speaker clustering decisions
- Performance timing breakdowns

### Getting Help

```bash
# Show detailed usage
swift run fluidaudio help

# Check available commands
swift run fluidaudio
```

For additional support, see the main [README.md](../README.md) and [BENCHMARKING.md](BENCHMARKING.md) documentation.