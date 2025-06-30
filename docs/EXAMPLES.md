# FluidAudioSwift CLI Examples

This document provides practical examples and use cases for the FluidAudioSwift CLI tool.

## Table of Contents

- [Basic Usage](#basic-usage)
- [Research Benchmarking](#research-benchmarking)
- [Audio Processing Workflows](#audio-processing-workflows)
- [Performance Optimization](#performance-optimization)
- [Batch Processing](#batch-processing)
- [Result Analysis](#result-analysis)
- [Integration Examples](#integration-examples)

## Basic Usage

### Quick Start

```bash
# Build the CLI
swift build

# Show help
swift run fluidaudio help

# Process a single audio file
swift run fluidaudio process meeting.wav

# Run default benchmark
swift run fluidaudio benchmark
```

### Processing Different Audio Formats

```bash
# WAV files (recommended)
swift run fluidaudio process interview.wav --output results.json

# M4A files
swift run fluidaudio process podcast.m4a --threshold 0.8

# MP3 files  
swift run fluidaudio process conference-call.mp3 --debug
```

## Research Benchmarking

### AMI Corpus Evaluation

```bash
# Standard SDM benchmark (realistic conditions)
swift run fluidaudio benchmark --dataset ami-sdm

# Clean IHM benchmark (optimal conditions)
swift run fluidaudio benchmark --dataset ami-ihm

# Save results for analysis
swift run fluidaudio benchmark --dataset ami-sdm --output sdm-baseline.json
```

### Threshold Optimization Study

```bash
#!/bin/bash
# Test different clustering thresholds
echo "Running threshold optimization study..."

for threshold in 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9; do
    echo "Testing threshold: $threshold"
    
    swift run fluidaudio benchmark \
        --dataset ami-sdm \
        --threshold $threshold \
        --output "threshold-study/sdm-${threshold}.json"
        
    swift run fluidaudio benchmark \
        --dataset ami-ihm \
        --threshold $threshold \
        --output "threshold-study/ihm-${threshold}.json"
done

echo "Threshold study complete. Results in threshold-study/"
```

### Comparative Analysis

```bash
#!/bin/bash
# Compare performance across datasets
mkdir -p benchmark-comparison

# Baseline configurations
swift run fluidaudio benchmark --dataset ami-sdm --output benchmark-comparison/sdm-baseline.json
swift run fluidaudio benchmark --dataset ami-ihm --output benchmark-comparison/ihm-baseline.json

# Optimized configurations
swift run fluidaudio benchmark --dataset ami-sdm --threshold 0.75 --output benchmark-comparison/sdm-optimized.json
swift run fluidaudio benchmark --dataset ami-ihm --threshold 0.65 --output benchmark-comparison/ihm-optimized.json

# Debug mode for detailed analysis
swift run fluidaudio benchmark --dataset ami-sdm --debug --output benchmark-comparison/sdm-debug.json
```

## Audio Processing Workflows

### Meeting Analysis Pipeline

```bash
#!/bin/bash
# Complete meeting analysis workflow

MEETING_FILE="board-meeting-2024-01.wav"
OUTPUT_DIR="meeting-analysis"
mkdir -p "$OUTPUT_DIR"

echo "Analyzing meeting: $MEETING_FILE"

# Standard analysis
swift run fluidaudio process "$MEETING_FILE" \
    --output "$OUTPUT_DIR/standard-analysis.json"

# Conservative speaker separation
swift run fluidaudio process "$MEETING_FILE" \
    --threshold 0.8 \
    --output "$OUTPUT_DIR/conservative-analysis.json"

# Aggressive speaker detection
swift run fluidaudio process "$MEETING_FILE" \
    --threshold 0.6 \
    --output "$OUTPUT_DIR/aggressive-analysis.json"

echo "Meeting analysis complete. Results in $OUTPUT_DIR/"
```

### Interview Processing

```bash
#!/bin/bash
# Interview processing with quality checks

INTERVIEW_FILE="$1"
if [ -z "$INTERVIEW_FILE" ]; then
    echo "Usage: $0 <interview-file>"
    exit 1
fi

BASE_NAME=$(basename "$INTERVIEW_FILE" .wav)
OUTPUT_DIR="interview-results/$BASE_NAME"
mkdir -p "$OUTPUT_DIR"

echo "Processing interview: $INTERVIEW_FILE"

# High-confidence processing (good for interviews)
swift run fluidaudio process "$INTERVIEW_FILE" \
    --threshold 0.75 \
    --output "$OUTPUT_DIR/diarization.json"

# Debug analysis for quality assessment
swift run fluidaudio process "$INTERVIEW_FILE" \
    --threshold 0.75 \
    --debug \
    --output "$OUTPUT_DIR/debug-analysis.json"

echo "Interview processing complete. Results in $OUTPUT_DIR/"
```

## Performance Optimization

### Finding Optimal Settings

```bash
#!/bin/bash
# Performance optimization script

AUDIO_FILE="test-audio.wav"
RESULTS_DIR="optimization-results"
mkdir -p "$RESULTS_DIR"

echo "Running performance optimization for: $AUDIO_FILE"

# Test different threshold values
for threshold in 0.6 0.7 0.8; do
    echo "Testing threshold: $threshold"
    
    # Time the processing
    time swift run fluidaudio process "$AUDIO_FILE" \
        --threshold $threshold \
        --output "$RESULTS_DIR/perf-${threshold}.json" 2>&1 | \
        tee "$RESULTS_DIR/timing-${threshold}.log"
done

echo "Performance optimization complete."
```

### System Performance Test

```bash
#!/bin/bash
# Test system performance with different audio lengths

TEST_DIR="performance-test"
mkdir -p "$TEST_DIR"

echo "Running system performance tests..."

# Short audio (good for quick testing)
swift run fluidaudio process short-sample.wav --output "$TEST_DIR/short-test.json"

# Medium audio (typical use case)  
swift run fluidaudio process medium-sample.wav --output "$TEST_DIR/medium-test.json"

# Long audio (stress test)
swift run fluidaudio process long-sample.wav --output "$TEST_DIR/long-test.json"

echo "System performance test complete."
```

## Batch Processing

### Process Multiple Files

```bash
#!/bin/bash
# Batch process all audio files in a directory

INPUT_DIR="audio-files"
OUTPUT_DIR="diarization-results"
mkdir -p "$OUTPUT_DIR"

echo "Batch processing audio files from: $INPUT_DIR"

# Process all WAV files
for file in "$INPUT_DIR"/*.wav; do
    if [ -f "$file" ]; then
        filename=$(basename "$file" .wav)
        echo "Processing: $filename"
        
        swift run fluidaudio process "$file" \
            --output "$OUTPUT_DIR/${filename}-diarization.json"
    fi
done

# Process other formats
for ext in m4a mp3; do
    for file in "$INPUT_DIR"/*.$ext; do
        if [ -f "$file" ]; then
            filename=$(basename "$file" .$ext)
            echo "Processing: $filename ($ext)"
            
            swift run fluidaudio process "$file" \
                --output "$OUTPUT_DIR/${filename}-diarization.json"
        fi
    done
done

echo "Batch processing complete. Results in: $OUTPUT_DIR"
```

### Parallel Processing

```bash
#!/bin/bash
# Parallel processing with GNU parallel

INPUT_DIR="audio-files"
OUTPUT_DIR="parallel-results"
mkdir -p "$OUTPUT_DIR"

# Function to process a single file
process_file() {
    local file="$1"
    local output_dir="$2"
    local filename=$(basename "$file" .wav)
    
    echo "Processing: $filename"
    swift run fluidaudio process "$file" \
        --output "$output_dir/${filename}-diarization.json"
}

export -f process_file

# Process files in parallel (adjust -j based on your CPU cores)
find "$INPUT_DIR" -name "*.wav" | \
    parallel -j 4 process_file {} "$OUTPUT_DIR"

echo "Parallel processing complete."
```

## Result Analysis

### Extract Key Metrics

```bash
#!/bin/bash
# Extract key metrics from benchmark results

RESULTS_FILE="$1"
if [ -z "$RESULTS_FILE" ]; then
    echo "Usage: $0 <results-file.json>"
    exit 1
fi

echo "Analyzing results from: $RESULTS_FILE"

# Extract DER and JER using jq
if command -v jq &> /dev/null; then
    echo "Average DER: $(jq -r '.averageDER' "$RESULTS_FILE")%"
    echo "Average JER: $(jq -r '.averageJER' "$RESULTS_FILE")%"
    echo "Processed Files: $(jq -r '.processedFiles') / $(jq -r '.totalFiles')"
    echo "Dataset: $(jq -r '.dataset')"
else
    echo "Install jq for JSON parsing: brew install jq"
fi
```

### Compare Results

```bash
#!/bin/bash
# Compare multiple benchmark results

echo "Benchmark Comparison Report"
echo "=========================="

for file in benchmark-results/*.json; do
    if [ -f "$file" ]; then
        filename=$(basename "$file" .json)
        echo "File: $filename"
        
        if command -v jq &> /dev/null; then
            echo "  DER: $(jq -r '.averageDER' "$file")%"
            echo "  JER: $(jq -r '.averageJER' "$file")%"
            echo "  Dataset: $(jq -r '.dataset' "$file")"
            echo "  Files: $(jq -r '.processedFiles')/$(jq -r '.totalFiles')"
        fi
        echo ""
    fi
done
```

### Generate Summary Report

```bash
#!/bin/bash
# Generate comprehensive summary report

RESULTS_DIR="benchmark-results"
REPORT_FILE="benchmark-summary.md"

echo "# Benchmark Summary Report" > "$REPORT_FILE"
echo "Generated: $(date)" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"

echo "## Results Overview" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"
echo "| Dataset | Threshold | DER (%) | JER (%) | Files |" >> "$REPORT_FILE"
echo "|---------|-----------|---------|---------|-------|" >> "$REPORT_FILE"

if command -v jq &> /dev/null; then
    for file in "$RESULTS_DIR"/*.json; do
        if [ -f "$file" ]; then
            dataset=$(jq -r '.dataset' "$file")
            # Extract threshold from filename or config
            threshold="N/A"
            der=$(jq -r '.averageDER' "$file")
            jer=$(jq -r '.averageJER' "$file")
            files="$(jq -r '.processedFiles')/$(jq -r '.totalFiles')"
            
            echo "| $dataset | $threshold | $der | $jer | $files |" >> "$REPORT_FILE"
        fi
    done
fi

echo "" >> "$REPORT_FILE"
echo "## Performance Analysis" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"
echo "Add your analysis here..." >> "$REPORT_FILE"

echo "Summary report generated: $REPORT_FILE"
```

## Integration Examples

### CI/CD Integration

```yaml
# .github/workflows/benchmark.yml
name: Performance Benchmarks

on:
  pull_request:
    branches: [ main ]

jobs:
  benchmark:
    runs-on: macos-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Build CLI
      run: swift build
      
    - name: Run Benchmarks (without dataset)
      run: |
        # Test CLI functionality without requiring full dataset
        swift run fluidaudio help
        
        # Run basic performance tests
        swift test --filter BasicInitializationTests
        swift test --filter MetalAccelerationBenchmarks
        
    - name: Generate Report
      run: |
        echo "# Benchmark Results" > benchmark-report.md
        echo "Generated for PR #${{ github.event.number }}" >> benchmark-report.md
        # Add benchmark results here
```

### Python Integration

```python
#!/usr/bin/env python3
"""
FluidAudioSwift CLI integration example
"""

import subprocess
import json
import sys
from pathlib import Path

def run_diarization(audio_file, threshold=0.7, output_file=None):
    """Run diarization on an audio file"""
    
    cmd = ["swift", "run", "fluidaudio", "process", str(audio_file)]
    
    if threshold != 0.7:
        cmd.extend(["--threshold", str(threshold)])
        
    if output_file:
        cmd.extend(["--output", str(output_file)])
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        if output_file:
            with open(output_file, 'r') as f:
                return json.load(f)
        else:
            # Parse JSON from stdout if available
            return result.stdout
            
    except subprocess.CalledProcessError as e:
        print(f"Error running diarization: {e}")
        print(f"stderr: {e.stderr}")
        return None

def run_benchmark(dataset="ami-sdm", threshold=0.7, output_file=None):
    """Run benchmark evaluation"""
    
    cmd = ["swift", "run", "fluidaudio", "benchmark", "--dataset", dataset]
    
    if threshold != 0.7:
        cmd.extend(["--threshold", str(threshold)])
        
    if output_file:
        cmd.extend(["--output", str(output_file)])
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        if output_file:
            with open(output_file, 'r') as f:
                return json.load(f)
        else:
            return result.stdout
            
    except subprocess.CalledProcessError as e:
        print(f"Error running benchmark: {e}")
        return None

if __name__ == "__main__":
    # Example usage
    audio_file = "example.wav"
    if Path(audio_file).exists():
        result = run_diarization(audio_file, threshold=0.75, output_file="result.json")
        if result:
            print("Diarization successful!")
            print(f"Found {result.get('speakerCount', 'unknown')} speakers")
    else:
        print(f"Audio file not found: {audio_file}")
```

### Makefile Integration

```makefile
# Makefile for FluidAudioSwift CLI workflows

.PHONY: build test benchmark clean help

# Build the CLI
build:
	swift build

# Run basic tests
test: build
	swift test --filter CITests

# Run performance benchmarks
benchmark: build
	swift test --filter MetalAccelerationBenchmarks

# Run AMI benchmarks (requires dataset)
benchmark-ami: build
	@echo "Running AMI SDM benchmark..."
	swift run fluidaudio benchmark --dataset ami-sdm --output ami-sdm-results.json
	@echo "Running AMI IHM benchmark..."
	swift run fluidaudio benchmark --dataset ami-ihm --output ami-ihm-results.json

# Process audio files in batch
process-batch: build
	@echo "Processing audio files..."
	@for file in audio/*.wav; do \
		echo "Processing $$file..."; \
		swift run fluidaudio process "$$file" --output "results/$$(basename $$file .wav).json"; \
	done

# Clean build artifacts
clean:
	swift package clean
	rm -rf .build

# Show help
help:
	@echo "Available targets:"
	@echo "  build         - Build the CLI"
	@echo "  test          - Run basic tests"
	@echo "  benchmark     - Run performance benchmarks"
	@echo "  benchmark-ami - Run AMI corpus benchmarks"
	@echo "  process-batch - Process audio files in batch"
	@echo "  clean         - Clean build artifacts"
	@echo "  help          - Show this help"
```

These examples demonstrate various ways to use the FluidAudioSwift CLI for research, production workflows, and integration with other tools. Adjust the scripts based on your specific needs and environment.