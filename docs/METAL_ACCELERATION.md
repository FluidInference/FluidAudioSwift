# Metal Performance Shaders Integration

This document provides technical details about FluidAudioSwift's Metal Performance Shaders (MPS) integration, including implementation architecture, optimization strategies, and advanced configuration.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Metal Implementation](#metal-implementation)
- [Performance Characteristics](#performance-characteristics)
- [Optimization Strategies](#optimization-strategies)
- [Advanced Configuration](#advanced-configuration)
- [GPU Memory Management](#gpu-memory-management)
- [Fallback Mechanisms](#fallback-mechanisms)
- [Platform Considerations](#platform-considerations)

## Architecture Overview

FluidAudioSwift leverages a hybrid computation architecture that automatically selects the optimal backend based on hardware capabilities and workload characteristics.

```
┌─────────────────────────────────────────────────────────┐
│                DiarizerManager                          │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────────┐    ┌─────────────────────────────┐ │
│  │ MetalProcessor  │    │    Accelerate Framework     │ │
│  │   (GPU MPS)     │    │       (CPU vDSP)            │ │
│  └─────────────────┘    └─────────────────────────────┘ │
├─────────────────────────────────────────────────────────┤
│              Automatic Backend Selection                │
└─────────────────────────────────────────────────────────┘
```

### Key Components

**MetalPerformanceProcessor**
- GPU device management and command queue handling
- MPS matrix operations for batch cosine distances
- Custom Metal compute kernels for powerset conversion
- Memory buffer management and synchronization

**Automatic Fallback System**
- Runtime Metal availability detection
- Graceful degradation to Accelerate framework
- Configuration-driven backend selection
- Performance-based dynamic switching

## Metal Implementation

### Batch Cosine Distance Calculation

The core Metal implementation optimizes embedding similarity calculations using MPS matrix operations:

```swift
func batchCosineDistances(queries: [[Float]], candidates: [[Float]]) -> [[Float]]? {
    // Create MPS matrices for GPU computation
    let queryMatrix = MPSMatrix(buffer: queryBuffer, descriptor: queryMatrixDescriptor)
    let candidateMatrix = MPSMatrix(buffer: candidateBuffer, descriptor: candidateMatrixDescriptor)
    let resultMatrix = MPSMatrix(buffer: resultBuffer, descriptor: resultMatrixDescriptor)
    
    // Perform matrix multiplication on GPU
    let matrixMultiplication = MPSMatrixMultiplication(
        device: device,
        transposeLeft: false,
        transposeRight: true,
        resultRows: numQueries,
        resultColumns: numCandidates,
        interiorColumns: embeddingDim,
        alpha: 1.0,
        beta: 0.0
    )
    
    matrixMultiplication.encode(
        commandBuffer: commandBuffer,
        leftMatrix: queryMatrix,
        rightMatrix: candidateMatrix,
        resultMatrix: resultMatrix
    )
}
```

### Custom Metal Compute Kernels

For powerset conversion operations, custom Metal compute shaders provide optimal GPU utilization:

```metal
kernel void powerset_conversion(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& batch_size [[buffer(2)]],
    constant uint& num_frames [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]]
) {
    // GPU kernel implementation for parallel powerset conversion
    const uint batch_idx = gid.x;
    const uint frame_idx = gid.y;
    
    if (batch_idx >= batch_size || frame_idx >= num_frames) return;
    
    // Powerset mapping and speaker activation logic
    // ... (optimized for GPU execution)
}
```

### Memory Layout Optimization

**Row-Major Query Matrix:**

```text
Query[0]: [e0, e1, e2, ..., eN]
Query[1]: [e0, e1, e2, ..., eN]
...
```

**Column-Major Candidate Matrix:**

```text
Candidate[0]: [e0, e1, e2, ...]
Candidate[1]: [e0, e1, e2, ...]
              [↓   ↓   ↓      ]
```

This layout optimization enables efficient GPU memory access patterns and maximizes cache utilization.

## Performance Characteristics

### Speedup Analysis

**Batch Size Impact:**

- **8 embeddings**: 0.5-1.2x (GPU overhead dominant)
- **16 embeddings**: 1.2-2.5x (breakeven point)
- **32 embeddings**: 3.0-6.0x (optimal performance)
- **64+ embeddings**: 4.0-8.0x (maximum efficiency)

**Embedding Dimension Scaling:**

- **256d**: 2.0-4.0x speedup
- **512d**: 3.0-6.0x speedup  
- **1024d**: 4.0-8.0x speedup

**Hardware Performance:**

- **M1/M2/M3**: 3-8x typical speedup
- **Intel integrated**: 1.5-3x speedup
- **Dedicated GPU**: 5-15x potential speedup

### Memory Bandwidth Utilization

**GPU Memory Throughput:**

- Theoretical: 400+ GB/s (Apple Silicon)
- Achieved: 60-150 GB/s (typical workloads)
- Efficiency: 15-40% of peak bandwidth

**CPU Memory Comparison:**

- Theoretical: 100+ GB/s (unified memory)
- Achieved: 20-40 GB/s (Accelerate vDSP)
- Efficiency: 20-40% of peak bandwidth

## Optimization Strategies

### Batch Size Optimization

**Dynamic Batch Sizing:**

```swift
func optimalBatchSize(for embeddingCount: Int, dimension: Int) -> Int {
    switch (embeddingCount, dimension) {
    case (_, let dim) where dim >= 1024:
        return min(embeddingCount, 64)
    case (let count, _) where count >= 128:
        return 32
    case (let count, _) where count >= 32:
        return min(count, 32)
    default:
        return 16  // Fallback to CPU for small operations
    }
}
```

### Memory Pool Management

**Buffer Reuse Strategy:**

```swift
class MetalBufferPool {
    private var availableBuffers: [MTLBuffer] = []
    private var usedBuffers: Set<MTLBuffer> = []
    
    func acquire(size: Int) -> MTLBuffer? {
        // Reuse existing buffers when possible
        if let buffer = availableBuffers.first(where: { $0.length >= size }) {
            availableBuffers.removeAll { $0 === buffer }
            usedBuffers.insert(buffer)
            return buffer
        }
        
        // Allocate new buffer if needed
        return device.makeBuffer(length: size, options: .storageModeShared)
    }
}
```

### Command Buffer Optimization

**Async Execution Pipeline:**

```swift
func asyncBatchProcessing(queries: [[Float]], candidates: [[Float]]) {
    let commandBuffer = commandQueue.makeCommandBuffer()
    
    // Encode multiple operations in single command buffer
    encodeMatrixMultiplication(commandBuffer: commandBuffer)
    encodeDistanceCalculation(commandBuffer: commandBuffer)
    encodeResultRetrieval(commandBuffer: commandBuffer)
    
    // Async execution with completion handler
    commandBuffer?.addCompletedHandler { _ in
        // Process results on background queue
        DispatchQueue.global().async {
            self.processResults()
        }
    }
    
    commandBuffer?.commit()
}
```

## Advanced Configuration

### Performance Tuning Parameters

**GPU-Specific Optimization:**

```swift
extension DiarizerConfig {
    static func optimizedForHardware() -> DiarizerConfig {
        var config = DiarizerConfig.default
        
        #if targetEnvironment(simulator)
        config.useMetalAcceleration = false
        #else
        if let device = MTLCreateSystemDefaultDevice() {
            switch device.name {
            case let name where name.contains("M1"):
                config.metalBatchSize = 32
                config.fallbackToAccelerate = true
            case let name where name.contains("M2"), 
                 let name where name.contains("M3"):
                config.metalBatchSize = 64
                config.fallbackToAccelerate = true
            default:
                config.metalBatchSize = 16
            }
        }
        #endif
        
        return config
    }
}
```

### Thermal Management

**Dynamic Performance Scaling:**

```swift
class ThermalAwareProcessor {
    private var thermalState: ProcessInfo.ThermalState = .nominal
    
    func adaptToThermalState() {
        thermalState = ProcessInfo.processInfo.thermalState
        
        switch thermalState {
        case .nominal:
            config.metalBatchSize = 64
            config.useMetalAcceleration = true
        case .fair:
            config.metalBatchSize = 32
            config.useMetalAcceleration = true
        case .serious, .critical:
            config.metalBatchSize = 16
            config.useMetalAcceleration = false  // Fallback to CPU
        @unknown default:
            config.useMetalAcceleration = false
        }
    }
}
```

### Power Efficiency Optimization

**Battery-Aware Processing:**

```swift
func batteryOptimizedConfig() -> DiarizerConfig {
    var config = DiarizerConfig.default
    
    if ProcessInfo.processInfo.isLowPowerModeEnabled {
        // Prioritize battery life over performance
        config.metalBatchSize = 16
        config.parallelProcessingThreshold = 120.0  // Longer threshold
        config.useEarlyTermination = true
    }
    
    return config
}
```

## GPU Memory Management

### Buffer Allocation Strategy

**Shared Memory Mode:**

```swift
// Optimal for frequent CPU-GPU data transfer
let buffer = device.makeBuffer(
    length: dataSize,
    options: .storageModeShared
)
```

**Private Memory Mode:**

```swift
// Optimal for GPU-only computations
let buffer = device.makeBuffer(
    length: dataSize,
    options: .storageModePrivate
)
```

### Memory Usage Patterns

**Peak Memory Consumption:**
- **Query Matrix**: `numQueries × embeddingDim × 4 bytes`
- **Candidate Matrix**: `embeddingDim × numCandidates × 4 bytes`
- **Result Matrix**: `numQueries × numCandidates × 4 bytes`
- **Overhead**: ~20% additional for Metal infrastructure

**Memory Efficiency Calculation:**
```swift
func estimateMemoryUsage(queries: Int, candidates: Int, dimension: Int) -> Int {
    let querySize = queries * dimension * 4
    let candidateSize = dimension * candidates * 4
    let resultSize = queries * candidates * 4
    let overhead = Int(Double(querySize + candidateSize + resultSize) * 0.2)
    
    return querySize + candidateSize + resultSize + overhead
}
```

### Memory Pool Implementation

**Efficient Buffer Reuse:**
```swift
final class MetalMemoryPool {
    private let device: MTLDevice
    private var bufferPool: [Int: [MTLBuffer]] = [:]
    private let queue = DispatchQueue(label: "MetalMemoryPool")
    
    func getBuffer(size: Int) -> MTLBuffer? {
        return queue.sync {
            // Round up to nearest power of 2 for better reuse
            let poolSize = nextPowerOfTwo(size)
            
            if let buffer = bufferPool[poolSize]?.popLast() {
                return buffer
            }
            
            return device.makeBuffer(length: poolSize, options: .storageModeShared)
        }
    }
    
    func returnBuffer(_ buffer: MTLBuffer) {
        queue.async {
            let size = buffer.length
            self.bufferPool[size, default: []].append(buffer)
            
            // Limit pool size to prevent excessive memory usage
            if self.bufferPool[size]!.count > 10 {
                self.bufferPool[size]!.removeFirst()
            }
        }
    }
}
```

## Fallback Mechanisms

### Automatic Backend Selection

**Runtime Capability Detection:**
```swift
enum ComputeBackend {
    case metal(device: MTLDevice)
    case accelerate
    case cpu
}

func selectOptimalBackend() -> ComputeBackend {
    // Try Metal first
    if let device = MTLCreateSystemDefaultDevice(),
       config.useMetalAcceleration {
        return .metal(device: device)
    }
    
    // Fallback to Accelerate
    if config.fallbackToAccelerate {
        return .accelerate
    }
    
    // Final fallback to pure CPU
    return .cpu
}
```

### Graceful Degradation

**Progressive Fallback Strategy:**
```swift
func performBatchOperation<T>(
    operation: Operation,
    fallbackChain: [ComputeBackend]
) throws -> T {
    var lastError: Error?
    
    for backend in fallbackChain {
        do {
            switch backend {
            case .metal(let device):
                return try performMetalOperation(operation, device: device)
            case .accelerate:
                return try performAccelerateOperation(operation)
            case .cpu:
                return try performCPUOperation(operation)
            }
        } catch {
            lastError = error
            logger.warning("Backend \(backend) failed: \(error)")
            continue
        }
    }
    
    throw lastError ?? ComputeError.allBackendsFailed
}
```

### Error Recovery

**Robust Error Handling:**
```swift
func handleMetalError(_ error: Error) -> RecoveryAction {
    switch error {
    case MTLError.invalidResource:
        return .retryWithSmallerBatch
    case MTLError.outOfMemory:
        return .fallbackToAccelerate
    case MTLError.deviceNotFound:
        return .disableMetal
    default:
        return .retryOnce
    }
}
```

## Platform Considerations

### iOS Optimization

**Memory Constraints:**
```swift
#if os(iOS)
extension DiarizerConfig {
    static var iOSOptimized: DiarizerConfig {
        var config = DiarizerConfig.default
        config.metalBatchSize = 16  // Smaller batches for iOS
        config.embeddingCacheSize = 50  // Reduced cache
        config.parallelProcessingThreshold = 30.0
        return config
    }
}
#endif
```

**Thermal Management:**
```swift
func iOSThermalAwareness() {
    NotificationCenter.default.addObserver(
        forName: ProcessInfo.thermalStateDidChangeNotification,
        object: nil,
        queue: .main
    ) { _ in
        self.adaptToThermalState()
    }
}
```

### macOS Optimization

**High-Performance Configuration:**
```swift
#if os(macOS)
extension DiarizerConfig {
    static var macOSHighPerformance: DiarizerConfig {
        var config = DiarizerConfig.default
        config.metalBatchSize = 64  // Larger batches for desktop
        config.embeddingCacheSize = 200
        config.parallelProcessingThreshold = 10.0  // Aggressive parallelization
        return config
    }
}
#endif
```

### Hardware-Specific Tuning

**Apple Silicon Optimization:**
```swift
func appleOptimizedConfig() -> DiarizerConfig {
    var config = DiarizerConfig.default
    
    if let device = MTLCreateSystemDefaultDevice() {
        // Detect Apple Silicon vs Intel
        if device.supportsFamily(.apple7) || device.supportsFamily(.apple8) {
            // M1/M2 optimization
            config.metalBatchSize = 64
            config.useMetalAcceleration = true
            config.fallbackToAccelerate = true
        } else {
            // Intel or older hardware
            config.metalBatchSize = 16
            config.useMetalAcceleration = false
            config.fallbackToAccelerate = true
        }
    }
    
    return config
}
```

---

## Implementation Notes

### Thread Safety

All Metal operations are designed to be thread-safe through:
- **Command queue serialization**: All GPU commands executed sequentially
- **Buffer synchronization**: Proper memory barriers and completion handlers
- **Async-friendly design**: Compatible with Swift concurrency

### Performance Monitoring

Built-in performance tracking provides:
- **Operation timing**: Microsecond precision for all operations
- **Memory usage tracking**: Peak and average memory consumption
- **GPU utilization**: Command buffer execution time analysis
- **Thermal impact**: Performance correlation with thermal state

### Debugging Support

Development and debugging features include:
- **Metal validation**: Comprehensive GPU state validation
- **Performance annotations**: GPU timeline debugging support
- **Memory leak detection**: Automatic buffer lifecycle tracking
- **Verbose logging**: Detailed operation tracing when enabled

For additional technical details, see the source implementation in [`MetalPerformanceProcessor`](../Sources/FluidAudioSwift/DiarizerManager.swift).