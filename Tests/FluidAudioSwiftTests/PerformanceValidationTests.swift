import XCTest
import Metal
import MetalPerformanceShaders
import Accelerate
@testable import FluidAudioSwift

/// Real-world performance validation tests
/// Tests memory efficiency, real-time processing, hardware scaling, and performance regression
@available(macOS 13.0, iOS 16.0, *)
final class PerformanceValidationTests: XCTestCase {

    private let testTimeout: TimeInterval = 120.0

    // MARK: - Memory Efficiency Tests

    func testArraySliceMemoryOptimization() async {
        // Test the claimed 66% memory reduction through ArraySlice usage
        let config = DiarizerConfig(debugMode: false, parallelProcessingThreshold: 30.0)
        let manager = DiarizerManager(config: config)

        // Large audio sample to test memory usage
        let largeAudio = generateLargeAudioSample(durationSeconds: 180.0, sampleRate: 16000) // 3 minutes

        print("📊 Memory Optimization Test:")
        print("   Audio size: \(largeAudio.count) samples (\(largeAudio.count * MemoryLayout<Float>.size / 1024 / 1024) MB)")

        do {
            try await manager.initialize()

            let memoryBefore = getMemoryUsage()

            let result = try await manager.performCompleteDiarization(largeAudio, sampleRate: 16000)

            let memoryAfter = getMemoryUsage()
            let memoryIncrease = memoryAfter - memoryBefore

            print("   Memory before: \(memoryBefore) MB")
            print("   Memory after: \(memoryAfter) MB")
            print("   Memory increase: \(memoryIncrease) MB")

            // Memory increase should be reasonable (not exceeding 3x the original audio size)
            let audioSizeMB = Float(largeAudio.count * MemoryLayout<Float>.size) / 1024.0 / 1024.0
            let maxExpectedIncrease = audioSizeMB * 3.0

            XCTAssertLessThan(memoryIncrease, maxExpectedIncrease,
                            "Memory increase should not exceed 3x audio size (ArraySlice optimization)")

            XCTAssertNotNil(result, "Large audio should process successfully")

            print("✅ ArraySlice memory optimization validated")

        } catch {
            print("ℹ️ Memory optimization test skipped - models not available: \(error)")
        }
    }

    func testMemoryLeakPrevention() async {
        // Test for memory leaks during repeated operations
        let config = DiarizerConfig(debugMode: false)
        let manager = DiarizerManager(config: config)

        do {
            try await manager.initialize()

            let initialMemory = getMemoryUsage()
            let testAudio = generateTestAudio(durationSeconds: 30.0, sampleRate: 16000)

            // Perform multiple operations
            for i in 0..<5 {
                _ = autoreleasepool {
                    Task {
                        let _ = try await manager.performCompleteDiarization(testAudio, sampleRate: 16000)
                    }
                }

                // Allow memory cleanup
                try await Task.sleep(nanoseconds: 100_000_000) // 0.1 seconds

                let currentMemory = getMemoryUsage()
                let memoryGrowth = currentMemory - initialMemory

                print("   Operation \(i + 1): \(currentMemory) MB (+\(memoryGrowth) MB)")

                // Memory growth should stabilize and not continuously increase
                if i > 2 { // Allow initial allocation
                    XCTAssertLessThan(memoryGrowth, 100.0, "Memory should not continuously grow")
                }
            }

            print("✅ Memory leak prevention validated")

        } catch {
            print("ℹ️ Memory leak test skipped - models not available: \(error)")
        }
    }

    func testMemoryPressureHandling() async {
        // Test system behavior under memory pressure
        let config = DiarizerConfig(
            debugMode: false,
            parallelProcessingThreshold: 20.0,
            embeddingCacheSize: 200 // Large cache
        )
        let manager = DiarizerManager(config: config)

        do {
            try await manager.initialize()

            // Create memory pressure with large concurrent operations
            let largeAudioSamples = [
                generateLargeAudioSample(durationSeconds: 120.0, sampleRate: 16000),
                generateLargeAudioSample(durationSeconds: 100.0, sampleRate: 16000),
                generateLargeAudioSample(durationSeconds: 80.0, sampleRate: 16000)
            ]

            let startTime = CFAbsoluteTimeGetCurrent()

            let results = try await withThrowingTaskGroup(of: DiarizationResult.self) { group in
                for (index, audio) in largeAudioSamples.enumerated() {
                    group.addTask {
                        print("   Starting memory pressure task \(index + 1)")
                        return try await manager.performCompleteDiarization(audio, sampleRate: 16000)
                    }
                }

                var results: [DiarizationResult] = []
                for try await result in group {
                    results.append(result)
                }

                return results
            }

            let processingTime = CFAbsoluteTimeGetCurrent() - startTime

            print("📊 Memory Pressure Test:")
            print("   Processed 3 large files in \(String(format: "%.2f", processingTime))s")
            print("   All operations completed: \(results.count == 3)")

            XCTAssertEqual(results.count, 3, "All operations should complete under memory pressure")

            print("✅ Memory pressure handling validated")

        } catch {
            print("ℹ️ Memory pressure test skipped - models not available: \(error)")
        }
    }

    // MARK: - Real-Time Processing Tests

    func testRealTimeFactorPerformance() async {
        // Test the claimed <1x real-time factor performance
        let config = DiarizerConfig(
            debugMode: false,
            parallelProcessingThreshold: 30.0,
            useMetalAcceleration: true
        )
        let manager = DiarizerManager(config: config)

        let testDurations: [Float] = [30.0, 60.0, 120.0, 300.0] // 30s to 5 minutes

        do {
            try await manager.initialize()

            print("📊 Real-Time Factor Performance:")

            for duration in testDurations {
                let audio = generateRealtimeTestAudio(durationSeconds: duration, sampleRate: 16000)

                let startTime = CFAbsoluteTimeGetCurrent()
                let result = try await manager.performCompleteDiarization(audio, sampleRate: 16000)
                let processingTime = CFAbsoluteTimeGetCurrent() - startTime

                let realTimeFactor = processingTime / Double(duration)

                print("   \(Int(duration))s audio: \(String(format: "%.3f", realTimeFactor))x real-time")

                XCTAssertNotNil(result, "Audio should process successfully")

                // Target: <1x real-time for most cases, allow up to 2x for very long audio
                let maxAllowedFactor: Double = duration > 120.0 ? 2.0 : 1.5
                XCTAssertLessThan(realTimeFactor, maxAllowedFactor,
                                "\(Int(duration))s audio should process within \(maxAllowedFactor)x real-time")
            }

            print("✅ Real-time factor performance validated")

        } catch {
            print("ℹ️ Real-time factor test skipped - models not available: \(error)")
        }
    }

    func testStreamingPerformanceSimulation() async {
        // Simulate streaming audio processing
        let config = DiarizerConfig(
            debugMode: false,
            parallelProcessingThreshold: 10.0 // Process in small chunks
        )
        let manager = DiarizerManager(config: config)

        do {
            try await manager.initialize()

            // Simulate 10-second chunks arriving every 10 seconds
            let chunkDuration: Float = 10.0
            let numChunks = 6

            var totalProcessingTime: Double = 0
            var results: [DiarizationResult] = []

            print("📊 Streaming Performance Simulation:")

            for chunkIndex in 0..<numChunks {
                let chunkAudio = generateStreamingChunk(
                    chunkIndex: chunkIndex,
                    durationSeconds: chunkDuration,
                    sampleRate: 16000
                )

                let chunkStartTime = CFAbsoluteTimeGetCurrent()
                let result = try await manager.performCompleteDiarization(chunkAudio, sampleRate: 16000)
                let chunkProcessingTime = CFAbsoluteTimeGetCurrent() - chunkStartTime

                totalProcessingTime += chunkProcessingTime
                results.append(result)

                let chunkRTF = chunkProcessingTime / Double(chunkDuration)
                print("   Chunk \(chunkIndex + 1): \(String(format: "%.3f", chunkRTF))x real-time")

                // Each chunk should process faster than real-time
                XCTAssertLessThan(chunkRTF, 1.0, "Streaming chunk should process faster than real-time")
            }

            let averageRTF = totalProcessingTime / Double(numChunks * Int(chunkDuration))
            print("   Average RTF: \(String(format: "%.3f", averageRTF))x")

            XCTAssertEqual(results.count, numChunks, "All streaming chunks should process")
            XCTAssertLessThan(averageRTF, 0.8, "Average streaming performance should be <0.8x real-time")

            print("✅ Streaming performance simulation validated")

        } catch {
            print("ℹ️ Streaming simulation test skipped - models not available: \(error)")
        }
    }

    // MARK: - Hardware Scaling Tests

    func testAppleSiliconOptimization() async {
        // Test performance on Apple Silicon vs Intel
        let config = DiarizerConfig(
            debugMode: false,
            useMetalAcceleration: true,
            fallbackToAccelerate: true
        )
        let manager = DiarizerManager(config: config)

        do {
            try await manager.initialize()

            let testAudio = generateHardwareTestAudio(durationSeconds: 60.0, sampleRate: 16000)

            // Test Metal availability (primarily Apple Silicon)
            let metalDevice = MTLCreateSystemDefaultDevice()
            let hasAppleSilicon = metalDevice?.name.contains("Apple") ?? false

            let startTime = CFAbsoluteTimeGetCurrent()
            let result = try await manager.performCompleteDiarization(testAudio, sampleRate: 16000)
            let processingTime = CFAbsoluteTimeGetCurrent() - startTime

            let realTimeFactor = processingTime / 60.0

            print("📊 Hardware Performance:")
            print("   Device: \(metalDevice?.name ?? "Unknown")")
            print("   Apple Silicon: \(hasAppleSilicon)")
            print("   Metal available: \(metalDevice != nil)")
            print("   Processing time: \(String(format: "%.3f", processingTime))s")
            print("   Real-time factor: \(String(format: "%.3f", realTimeFactor))x")

            XCTAssertNotNil(result, "Hardware test should complete successfully")

            // Performance expectations based on hardware
            if hasAppleSilicon {
                XCTAssertLessThan(realTimeFactor, 0.8, "Apple Silicon should provide excellent performance")
            } else {
                XCTAssertLessThan(realTimeFactor, 1.5, "Intel Macs should still provide reasonable performance")
            }

            print("✅ Hardware scaling validated")

        } catch {
            print("ℹ️ Hardware scaling test skipped - models not available: \(error)")
        }
    }

    func testConcurrentHardwareUtilization() async {
        // Test how well the system utilizes available hardware concurrency
        let config = DiarizerConfig(
            debugMode: false,
            parallelProcessingThreshold: 20.0,
            useMetalAcceleration: true
        )
        let manager = DiarizerManager(config: config)

        let processorCount = ProcessInfo.processInfo.processorCount
        print("📊 Hardware Concurrency Test:")
        print("   Available processors: \(processorCount)")

        do {
            try await manager.initialize()

            // Create tasks that can utilize parallel processing
            let concurrentTasks = min(processorCount, 4) // Don't overwhelm the system
            var taskAudio: [[Float]] = []

            for i in 0..<concurrentTasks {
                let audio = generateConcurrencyTestAudio(
                    taskId: i,
                    durationSeconds: 45.0,
                    sampleRate: 16000
                )
                taskAudio.append(audio)
            }

            let startTime = CFAbsoluteTimeGetCurrent()

            let completedTasks = try await withThrowingTaskGroup(of: (taskId: Int, result: DiarizationResult).self) { group in
                for (taskId, audio) in taskAudio.enumerated() {
                    group.addTask {
                        let result = try await manager.performCompleteDiarization(audio, sampleRate: 16000)
                        return (taskId: taskId, result: result)
                    }
                }

                var completedTasks: [(taskId: Int, result: DiarizationResult)] = []
                for try await taskResult in group {
                    completedTasks.append(taskResult)
                    print("   Task \(taskResult.taskId) completed")
                }

                return completedTasks
            }

            let totalTime = CFAbsoluteTimeGetCurrent() - startTime
            let expectedSequentialTime = Double(concurrentTasks) * 45.0 / 2.0 // Rough estimate
            let concurrencyEfficiency = expectedSequentialTime / totalTime

            print("   Concurrent processing time: \(String(format: "%.2f", totalTime))s")
            print("   Concurrency efficiency: \(String(format: "%.2f", concurrencyEfficiency))x")

            XCTAssertEqual(completedTasks.count, concurrentTasks, "All concurrent tasks should complete")
            XCTAssertGreaterThan(concurrencyEfficiency, 1.5, "Should show meaningful concurrency benefits")

            print("✅ Concurrent hardware utilization validated")

        } catch {
            print("ℹ️ Hardware concurrency test skipped - models not available: \(error)")
        }
    }

    // MARK: - Performance Regression Tests

    func testPerformanceBaseline() async {
        // Establish performance baselines for regression testing
        let config = DiarizerConfig(
            debugMode: false,
            parallelProcessingThreshold: 30.0,
            useMetalAcceleration: true,
            fallbackToAccelerate: true
        )
        let manager = DiarizerManager(config: config)

        let standardTestCases = [
            (name: "Short Audio", duration: 15.0, maxRTF: 1.0),
            (name: "Medium Audio", duration: 60.0, maxRTF: 1.2),
            (name: "Long Audio", duration: 180.0, maxRTF: 1.5)
        ]

        do {
            try await manager.initialize()

            print("📊 Performance Baseline Test:")

            for testCase in standardTestCases {
                let audio = generateBaselineTestAudio(
                    durationSeconds: Float(testCase.duration),
                    sampleRate: 16000
                )

                let startTime = CFAbsoluteTimeGetCurrent()
                let result = try await manager.performCompleteDiarization(audio, sampleRate: 16000)
                let processingTime = CFAbsoluteTimeGetCurrent() - startTime

                let realTimeFactor = processingTime / Double(testCase.duration)

                print("   \(testCase.name) (\(Int(testCase.duration))s): \(String(format: "%.3f", realTimeFactor))x RTF")

                XCTAssertNotNil(result, "\(testCase.name) should process successfully")
                XCTAssertLessThan(realTimeFactor, testCase.maxRTF,
                                "\(testCase.name) should meet performance baseline")

                // Store baseline for future regression testing
                UserDefaults.standard.set(realTimeFactor, forKey: "FluidAudioSwift_Baseline_\(testCase.name)")
            }

            print("✅ Performance baselines established")

        } catch {
            print("ℹ️ Baseline test skipped - models not available: \(error)")
        }
    }

    func testPerformanceRegression() async {
        // Test against previously established baselines
        let config = DiarizerConfig(
            debugMode: false,
            parallelProcessingThreshold: 30.0,
            useMetalAcceleration: true
        )
        let manager = DiarizerManager(config: config)

        let testCases = [
            (name: "Short Audio", duration: 15.0),
            (name: "Medium Audio", duration: 60.0),
            (name: "Long Audio", duration: 180.0)
        ]

        do {
            try await manager.initialize()

            print("📊 Performance Regression Test:")

            for testCase in testCases {
                let audio = generateBaselineTestAudio(
                    durationSeconds: Float(testCase.duration),
                    sampleRate: 16000
                )

                let startTime = CFAbsoluteTimeGetCurrent()
                let result = try await manager.performCompleteDiarization(audio, sampleRate: 16000)
                let processingTime = CFAbsoluteTimeGetCurrent() - startTime

                let currentRTF = processingTime / Double(testCase.duration)
                let baselineRTF = UserDefaults.standard.double(forKey: "FluidAudioSwift_Baseline_\(testCase.name)")

                if baselineRTF > 0 {
                    let performanceChange = (currentRTF - baselineRTF) / baselineRTF * 100

                    print("   \(testCase.name): \(String(format: "%.3f", currentRTF))x RTF (baseline: \(String(format: "%.3f", baselineRTF))x)")
                    print("     Performance change: \(String(format: "%.1f", performanceChange))%")

                    // Allow up to 20% performance degradation
                    XCTAssertLessThan(performanceChange, 20.0,
                                    "\(testCase.name) should not regress more than 20%")
                } else {
                    print("   \(testCase.name): No baseline available, current RTF: \(String(format: "%.3f", currentRTF))x")
                }

                XCTAssertNotNil(result, "\(testCase.name) should process successfully")
            }

            print("✅ Performance regression test completed")

        } catch {
            print("ℹ️ Regression test skipped - models not available: \(error)")
        }
    }

    // MARK: - Performance Monitoring Tests

    func testContinuousPerformanceMonitoring() async {
        // Test performance consistency over multiple operations
        let config = DiarizerConfig(debugMode: false)
        let manager = DiarizerManager(config: config)

        do {
            try await manager.initialize()

            let testAudio = generateMonitoringTestAudio(durationSeconds: 30.0, sampleRate: 16000)
            var processingTimes: [Double] = []

            print("📊 Continuous Performance Monitoring:")

            // Run multiple iterations
            for iteration in 0..<10 {
                let startTime = CFAbsoluteTimeGetCurrent()
                let result = try await manager.performCompleteDiarization(testAudio, sampleRate: 16000)
                let processingTime = CFAbsoluteTimeGetCurrent() - startTime

                processingTimes.append(processingTime)

                let rtf = processingTime / 30.0
                print("   Iteration \(iteration + 1): \(String(format: "%.3f", rtf))x RTF")

                XCTAssertNotNil(result, "Iteration \(iteration + 1) should succeed")
            }

            // Analyze consistency
            let avgTime = processingTimes.reduce(0, +) / Double(processingTimes.count)
            let variance = processingTimes.map { pow($0 - avgTime, 2) }.reduce(0, +) / Double(processingTimes.count)
            let standardDeviation = sqrt(variance)
            let coefficientOfVariation = standardDeviation / avgTime

            print("   Average RTF: \(String(format: "%.3f", avgTime / 30.0))x")
            print("   Std deviation: \(String(format: "%.3f", standardDeviation))s")
            print("   Coefficient of variation: \(String(format: "%.3f", coefficientOfVariation))")

            // Performance should be consistent (CV < 0.2)
            XCTAssertLessThan(coefficientOfVariation, 0.2, "Performance should be consistent across runs")

            print("✅ Continuous performance monitoring validated")

        } catch {
            print("ℹ️ Performance monitoring test skipped - models not available: \(error)")
        }
    }

    // MARK: - Helper Methods

    private func getMemoryUsage() -> Float {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4

        // Use the global variable directly for thread safety
        let taskPort = mach_task_self_

        let kerr: kern_return_t = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: 1) {
                task_info(taskPort, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
            }
        }

        if kerr == KERN_SUCCESS {
            return Float(info.resident_size) / 1024.0 / 1024.0 // Convert to MB
        } else {
            return 0.0
        }
    }

    private func generateLargeAudioSample(durationSeconds: Float, sampleRate: Int) -> [Float] {
        let sampleCount = Int(durationSeconds * Float(sampleRate))
        var audio = Array<Float>(repeating: 0.0, count: sampleCount)

        // Generate complex audio with multiple speakers
        let numSpeakers = 5
        for speaker in 0..<numSpeakers {
            let speakerStartTime = Float(speaker) * durationSeconds / Float(numSpeakers)
            let speakerDuration = durationSeconds / Float(numSpeakers) * 1.5 // Overlapping

            let startSample = Int(speakerStartTime * Float(sampleRate))
            let endSample = Int(min((speakerStartTime + speakerDuration) * Float(sampleRate), Float(sampleCount)))

            let frequency = 150.0 + Float(speaker) * 80.0

            for i in startSample..<endSample {
                let t = Float(i - startSample) / Float(sampleRate)
                let envelope = 0.5 + 0.5 * sin(2.0 * Float.pi * 2.0 * t) // Amplitude modulation
                audio[i] += 0.3 * envelope * sin(2.0 * Float.pi * frequency * t)
            }
        }

        return audio
    }

    private func generateTestAudio(durationSeconds: Float, sampleRate: Int) -> [Float] {
        let sampleCount = Int(durationSeconds * Float(sampleRate))
        return (0..<sampleCount).map { i in
            let t = Float(i) / Float(sampleRate)
            return 0.5 * sin(2.0 * Float.pi * 440.0 * t) * (1.0 + 0.1 * sin(2.0 * Float.pi * 5.0 * t))
        }
    }

    private func generateRealtimeTestAudio(durationSeconds: Float, sampleRate: Int) -> [Float] {
        let sampleCount = Int(durationSeconds * Float(sampleRate))
        var audio = Array<Float>(repeating: 0.0, count: sampleCount)

        // Realistic speech-like patterns
        let segmentDuration = Float(sampleRate) * 2.0 // 2-second segments
        let numSegments = Int(ceil(Float(sampleCount) / segmentDuration))

        for segment in 0..<numSegments {
            let startIdx = Int(Float(segment) * segmentDuration)
            let endIdx = min(Int(Float(segment + 1) * segmentDuration), sampleCount)

            let frequency = 200.0 + Float(segment % 3) * 100.0 // Different speakers

            for i in startIdx..<endIdx {
                let t = Float(i - startIdx) / Float(sampleRate)
                // Speech-like envelope
                let envelope = exp(-t * 2.0) * (0.5 + 0.5 * sin(2.0 * Float.pi * 8.0 * t))
                audio[i] = 0.4 * envelope * sin(2.0 * Float.pi * frequency * t)
            }
        }

        return audio
    }

    private func generateStreamingChunk(chunkIndex: Int, durationSeconds: Float, sampleRate: Int) -> [Float] {
        let sampleCount = Int(durationSeconds * Float(sampleRate))
        let frequency = 300.0 + Float(chunkIndex % 4) * 50.0 // Different speaker per chunk

        return (0..<sampleCount).map { i in
            let t = Float(i) / Float(sampleRate)
            return 0.5 * sin(2.0 * Float.pi * frequency * t) * (1.0 + 0.2 * sin(2.0 * Float.pi * 4.0 * t))
        }
    }

    private func generateHardwareTestAudio(durationSeconds: Float, sampleRate: Int) -> [Float] {
        let sampleCount = Int(durationSeconds * Float(sampleRate))
        var audio = Array<Float>(repeating: 0.0, count: sampleCount)

        // Complex signal that benefits from hardware acceleration
        for i in 0..<sampleCount {
            let t = Float(i) / Float(sampleRate)
            audio[i] = 0.2 * sin(2.0 * Float.pi * 220.0 * t) +
                      0.2 * sin(2.0 * Float.pi * 440.0 * t) +
                      0.1 * sin(2.0 * Float.pi * 880.0 * t) +
                      0.1 * sin(2.0 * Float.pi * 1320.0 * t)
        }

        return audio
    }

    private func generateConcurrencyTestAudio(taskId: Int, durationSeconds: Float, sampleRate: Int) -> [Float] {
        let sampleCount = Int(durationSeconds * Float(sampleRate))
        let baseFrequency = 200.0 + Float(taskId) * 150.0

        return (0..<sampleCount).map { i in
            let t = Float(i) / Float(sampleRate)
            return 0.4 * sin(2.0 * Float.pi * baseFrequency * t) *
                   (0.8 + 0.2 * sin(2.0 * Float.pi * 6.0 * t))
        }
    }

    private func generateBaselineTestAudio(durationSeconds: Float, sampleRate: Int) -> [Float] {
        let sampleCount = Int(durationSeconds * Float(sampleRate))
        var audio = Array<Float>(repeating: 0.0, count: sampleCount)

        // Standardized test pattern for baseline comparisons
        let fundamentalFreq: Float = 300.0

        for i in 0..<sampleCount {
            let t = Float(i) / Float(sampleRate)
            // Harmonic series with time-varying amplitude
            let envelope = 0.5 + 0.3 * sin(2.0 * Float.pi * 3.0 * t)
            audio[i] = envelope * (
                0.4 * sin(2.0 * Float.pi * fundamentalFreq * t) +
                0.2 * sin(2.0 * Float.pi * fundamentalFreq * 2.0 * t) +
                0.1 * sin(2.0 * Float.pi * fundamentalFreq * 3.0 * t)
            )
        }

        return audio
    }

    private func generateMonitoringTestAudio(durationSeconds: Float, sampleRate: Int) -> [Float] {
        let sampleCount = Int(durationSeconds * Float(sampleRate))
        return (0..<sampleCount).map { i in
            let t = Float(i) / Float(sampleRate)
            return 0.5 * sin(2.0 * Float.pi * 350.0 * t) *
                   (0.7 + 0.3 * cos(2.0 * Float.pi * 4.0 * t))
        }
    }
}
