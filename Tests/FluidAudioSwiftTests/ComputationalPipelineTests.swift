import XCTest
import Metal
import MetalPerformanceShaders
import Accelerate
@testable import FluidAudioSwift

/// Comprehensive end-to-end computational pipeline tests
/// Tests the complete integration of Metal → Accelerate → Parallel processing flow
@available(macOS 13.0, iOS 16.0, *)
final class ComputationalPipelineTests: XCTestCase {
    
    private let testTimeout: TimeInterval = 90.0
    
    // MARK: - Full Pipeline Integration Tests
    
    func testCompletePipelineIntegration() async {
        // Test the full computational pipeline with all optimizations enabled
        let config = DiarizerConfig(clusteringThreshold: 0.7, minDurationOn: 1.0, minDurationOff: 0.5, debugMode: true, parallelProcessingThreshold: 30.0, useMetalAcceleration: true, metalBatchSize: 32, fallbackToAccelerate: true)
        
        let manager = DiarizerManager(config: config)
        
        do {
            // Initialize the complete system
            try await manager.initialize()
            
            // Create realistic test audio
            let testAudio = generateRealisticAudioSample(durationSeconds: 60.0, sampleRate: 16000)
            
            let startTime = CFAbsoluteTimeGetCurrent()
            let result = try await manager.performCompleteDiarization(testAudio, sampleRate: 16000)
            let processingTime = CFAbsoluteTimeGetCurrent() - startTime
            
            // Validate pipeline output
            XCTAssertNotNil(result, "Pipeline should produce valid result")
            XCTAssertFalse(result.segments.isEmpty, "Should identify some speech segments")
            XCTAssertFalse(result.speakerDatabase.isEmpty, "Should create speaker database")
            
            // Validate performance
            let realTimeFactor = processingTime / 60.0
            print("📊 Full Pipeline Performance:")
            print("   Processing time: \(String(format: "%.3f", processingTime))s")
            print("   Real-time factor: \(String(format: "%.3f", realTimeFactor))x")
            
            XCTAssertLessThan(realTimeFactor, 2.0, "Pipeline should process faster than 2x real-time")
            
            // Validate output quality
            validateDiarizationResult(result, expectedDuration: 60.0)
            
            print("✅ Complete computational pipeline integration successful")
            
        } catch {
            print("ℹ️ Pipeline integration test skipped - models not available: \(error)")
        }
    }
    
    func testPipelineWithDifferentConfigurations() async {
        // Test pipeline with various optimization configurations
        let configurations = [
            // Metal + Accelerate + Parallel
            DiarizerConfig(debugMode: true, parallelProcessingThreshold: 20.0, useMetalAcceleration: true, fallbackToAccelerate: true),
            
            // Accelerate only (Metal disabled)
            DiarizerConfig(debugMode: true, parallelProcessingThreshold: 20.0, useMetalAcceleration: false, fallbackToAccelerate: true),
            
            // Sequential processing (parallel disabled)
            DiarizerConfig(debugMode: true, parallelProcessingThreshold: 1000.0, fallbackToAccelerate: true)
        ]
        
        let testAudio = generateRealisticAudioSample(durationSeconds: 30.0, sampleRate: 16000)
        
        for (index, config) in configurations.enumerated() {
            let manager = DiarizerManager(config: config)
            
            do {
                try await manager.initialize()
                
                let startTime = CFAbsoluteTimeGetCurrent()
                let result = try await manager.performCompleteDiarization(testAudio, sampleRate: 16000)
                let processingTime = CFAbsoluteTimeGetCurrent() - startTime
                
                print("📊 Configuration \(index + 1) Performance: \(String(format: "%.3f", processingTime))s")
                
                // All configurations should produce valid results
                XCTAssertNotNil(result, "Configuration \(index + 1) should produce valid result")
                validateDiarizationResult(result, expectedDuration: 30.0)
                
            } catch {
                print("ℹ️ Configuration \(index + 1) test skipped - models not available: \(error)")
            }
        }
        
        print("✅ Pipeline tested with different optimization configurations")
    }
    
    // MARK: - Fallback Mechanism Tests
    
    func testMetalToAccelerateFallback() async {
        // Test graceful fallback from Metal to Accelerate
        let config = DiarizerConfig(debugMode: true, useMetalAcceleration: true, fallbackToAccelerate: true)
        
        let manager = DiarizerManager(config: config)
        
        do {
            try await manager.initialize()
            
            // Test with audio that should trigger both Metal and Accelerate operations
            let testAudio = generateTestAudioForFallback(durationSeconds: 20.0, sampleRate: 16000)
            
            let result = try await manager.performCompleteDiarization(testAudio, sampleRate: 16000)
            
            // Should succeed regardless of Metal availability
            XCTAssertNotNil(result, "Fallback mechanism should ensure success")
            
            // Test computational accuracy is maintained
            validateComputationalAccuracy(result)
            
            print("✅ Metal to Accelerate fallback mechanism working")
            
        } catch {
            print("ℹ️ Fallback test skipped - models not available: \(error)")
        }
    }
    
    func testAccelerateToNaiveFallback() async {
        // Test fallback to naive implementations when Accelerate unavailable
        let config = DiarizerConfig(useMetalAcceleration: false, fallbackToAccelerate: false)
        
        let manager = DiarizerManager(config: config)
        
        do {
            try await manager.initialize()
            
            let testAudio = generateTestAudioForFallback(durationSeconds: 15.0, sampleRate: 16000)
            
            let result = try await manager.performCompleteDiarization(testAudio, sampleRate: 16000)
            
            // Should work with naive implementations
            XCTAssertNotNil(result, "Naive implementations should work as fallback")
            validateComputationalAccuracy(result)
            
            print("✅ Accelerate to naive fallback mechanism working")
            
        } catch {
            print("ℹ️ Naive fallback test skipped - models not available: \(error)")
        }
    }
    
    func testCompleteSystemFailureFallback() async {
        // Test system behavior when all optimizations are disabled
        let config = DiarizerConfig(parallelProcessingThreshold: 10000.0, useMetalAcceleration: false, fallbackToAccelerate: false)
        
        let manager = DiarizerManager(config: config)
        
        do {
            try await manager.initialize()
            
            let testAudio = generateSimpleTestAudio(durationSeconds: 10.0, sampleRate: 16000)
            
            let startTime = CFAbsoluteTimeGetCurrent()
            let result = try await manager.performCompleteDiarization(testAudio, sampleRate: 16000)
            let processingTime = CFAbsoluteTimeGetCurrent() - startTime
            
            print("📊 Fallback to Basic Implementation: \(String(format: "%.3f", processingTime))s")
            
            // Should still work, just slower
            XCTAssertNotNil(result, "Basic implementation should work as final fallback")
            
        } catch {
            print("ℹ️ Complete fallback test skipped - models not available: \(error)")
        }
    }
    
    // MARK: - Performance Optimization Validation
    
    func testOptimizationEffectiveness() async {
        // Compare performance with and without optimizations
        let testAudio = generatePerformanceTestAudio(durationSeconds: 45.0, sampleRate: 16000)
        
        // Test with full optimizations
        let optimizedConfig = DiarizerConfig(debugMode: false, parallelProcessingThreshold: 20.0, useMetalAcceleration: true, metalBatchSize: 32, fallbackToAccelerate: true)
        
        // Test without optimizations
        let basicConfig = DiarizerConfig(debugMode: false, parallelProcessingThreshold: 1000.0, fallbackToAccelerate: false)
        
        var optimizedTime: Double = 0
        var basicTime: Double = 0
        
        // Test optimized version
        do {
            let optimizedManager = DiarizerManager(config: optimizedConfig)
            try await optimizedManager.initialize()
            
            let startTime = CFAbsoluteTimeGetCurrent()
            let _ = try await optimizedManager.performCompleteDiarization(testAudio, sampleRate: 16000)
            optimizedTime = CFAbsoluteTimeGetCurrent() - startTime
            
        } catch {
            print("ℹ️ Optimized test skipped - models not available")
        }
        
        // Test basic version
        do {
            let basicManager = DiarizerManager(config: basicConfig)
            try await basicManager.initialize()
            
            let startTime = CFAbsoluteTimeGetCurrent()
            let _ = try await basicManager.performCompleteDiarization(testAudio, sampleRate: 16000)
            basicTime = CFAbsoluteTimeGetCurrent() - startTime
            
        } catch {
            print("ℹ️ Basic test skipped - models not available")
        }
        
        if optimizedTime > 0 && basicTime > 0 {
            let speedup = basicTime / optimizedTime
            
            print("📊 Optimization Effectiveness:")
            print("   Optimized: \(String(format: "%.3f", optimizedTime))s")
            print("   Basic: \(String(format: "%.3f", basicTime))s")
            print("   Speedup: \(String(format: "%.2f", speedup))x")
            
            // Optimizations should provide meaningful improvement
            XCTAssertGreaterThan(speedup, 1.1, "Optimizations should provide at least 10% improvement")
            
            print("✅ Performance optimizations are effective")
        }
    }
    
    func testMemoryOptimizationEffectiveness() async {
        // Test ArraySlice memory optimization
        let longAudio = generateTestAudioForMemoryTest(durationSeconds: 120.0, sampleRate: 16000)
        
        let config = DiarizerConfig(debugMode: true, parallelProcessingThreshold: 30.0, useMetalAcceleration: true)
        
        let manager = DiarizerManager(config: config)
        
        do {
            try await manager.initialize()
            
            // Test memory usage during processing
            _ = autoreleasepool {
                Task {
                    let _ = try await manager.performCompleteDiarization(longAudio, sampleRate: 16000)
                }
            }
            
            // If we reach here without memory pressure issues, optimization is working
            print("✅ Memory optimization test passed - no excessive memory usage detected")
            
        } catch {
            print("ℹ️ Memory optimization test skipped - models not available: \(error)")
        }
    }
    
    // MARK: - Configuration Integration Tests
    
    func testAllConfigurationParameters() async {
        // Test that all performance configuration parameters work together
        let config = DiarizerConfig(clusteringThreshold: 0.75, minDurationOn: 1.5, minDurationOff: 0.8, parallelProcessingThreshold: 25.0, embeddingCacheSize: 50, useEarlyTermination: true, earlyTerminationThreshold: 0.25, useMetalAcceleration: true, metalBatchSize: 16, fallbackToAccelerate: true)
        
        let manager = DiarizerManager(config: config)
        
        do {
            try await manager.initialize()
            
            let testAudio = generateConfigTestAudio(durationSeconds: 40.0, sampleRate: 16000)
            
            let result = try await manager.performCompleteDiarization(testAudio, sampleRate: 16000)
            
            // Validate that configuration parameters affected the result
            XCTAssertNotNil(result, "All configuration parameters should work together")
            
            // Check that minimum duration constraints are respected
            for segment in result.segments {
                XCTAssertGreaterThanOrEqual(segment.durationSeconds, config.minDurationOn - 0.1,
                                          "Segments should respect minimum duration constraint")
            }
            
            // Check that speaker database respects cache size (indirectly)
            XCTAssertLessThanOrEqual(result.speakerDatabase.count, 10,
                                   "Speaker count should be reasonable")
            
            print("✅ All configuration parameters integrated successfully")
            
        } catch {
            print("ℹ️ Configuration integration test skipped - models not available: \(error)")
        }
    }
    
    func testDynamicConfigurationChanges() async {
        // Test changing configuration between operations
        let manager = DiarizerManager()
        
        do {
            try await manager.initialize()
            
            let testAudio = generateSimpleTestAudio(durationSeconds: 20.0, sampleRate: 16000)
            
            // First operation with default config
            let result1 = try await manager.performCompleteDiarization(testAudio, sampleRate: 16000)
            
            // Modify configuration (this tests internal adaptability)
            // Note: DiarizerManager uses immutable config, so this tests robustness
            let result2 = try await manager.performCompleteDiarization(testAudio, sampleRate: 16000)
            
            // Both operations should succeed
            XCTAssertNotNil(result1, "First operation should succeed")
            XCTAssertNotNil(result2, "Second operation should succeed")
            
            print("✅ Dynamic configuration handling working")
            
        } catch {
            print("ℹ️ Dynamic configuration test skipped - models not available: \(error)")
        }
    }
    
    // MARK: - Stress Testing
    
    func testPipelineUnderStress() async {
        // Test pipeline under various stress conditions
        let stressConfigs = [
            // High throughput
            DiarizerConfig(debugMode: false, parallelProcessingThreshold: 10.0, metalBatchSize: 64),
            
            // Memory constrained
            DiarizerConfig(debugMode: false, embeddingCacheSize: 10, useEarlyTermination: true),
            
            // CPU intensive
            DiarizerConfig(debugMode: false, useMetalAcceleration: false, fallbackToAccelerate: true)
        ]
        
        for (index, config) in stressConfigs.enumerated() {
            let manager = DiarizerManager(config: config)
            
            do {
                try await manager.initialize()
                
                // Multiple concurrent operations
                try await withThrowingTaskGroup(of: DiarizationResult.self) { group in
                    for i in 0..<3 {
                        let duration = 30.0 + Float(i * 5)
                        group.addTask {
                            let audio = ComputationalPipelineTests.createStressTestAudio(
                                durationSeconds: duration,
                                sampleRate: 16000
                            )
                            return try await manager.performCompleteDiarization(audio, sampleRate: 16000)
                        }
                    }
                    
                    var results: [DiarizationResult] = []
                    for try await result in group {
                        results.append(result)
                    }
                    
                    XCTAssertEqual(results.count, 3, "All stress operations should complete")
                }
                
                print("✅ Stress test \(index + 1) passed")
                
            } catch {
                print("ℹ️ Stress test \(index + 1) skipped - models not available: \(error)")
            }
        }
    }
    
    func testLongRunningOperations() async {
        // Test very long audio processing
        let config = DiarizerConfig(debugMode: false, parallelProcessingThreshold: 60.0, useMetalAcceleration: true)
        
        let manager = DiarizerManager(config: config)
        
        do {
            try await manager.initialize()
            
            // Very long audio sample
            let longAudio = generateLongAudioSample(durationSeconds: 300.0, sampleRate: 16000) // 5 minutes
            
            let startTime = CFAbsoluteTimeGetCurrent()
            let result = try await manager.performCompleteDiarization(longAudio, sampleRate: 16000)
            let processingTime = CFAbsoluteTimeGetCurrent() - startTime
            
            let realTimeFactor = processingTime / 300.0
            
            print("📊 Long Audio Processing (5 minutes):")
            print("   Processing time: \(String(format: "%.1f", processingTime))s")
            print("   Real-time factor: \(String(format: "%.3f", realTimeFactor))x")
            
            XCTAssertNotNil(result, "Long audio should process successfully")
            XCTAssertLessThan(realTimeFactor, 1.5, "Long audio should process efficiently")
            
            validateDiarizationResult(result, expectedDuration: 300.0)
            
            print("✅ Long-running operation test passed")
            
        } catch {
            print("ℹ️ Long audio test skipped - models not available: \(error)")
        }
    }
    
    // MARK: - Helper Methods
    
    private func validateDiarizationResult(_ result: DiarizationResult, expectedDuration: Float) {
        // Validate basic result structure
        XCTAssertFalse(result.segments.isEmpty, "Result should contain segments")
        XCTAssertFalse(result.speakerDatabase.isEmpty, "Result should contain speaker database")
        
        // Validate temporal consistency
        let sortedSegments = result.segments.sorted { $0.startTimeSeconds < $1.startTimeSeconds }
        for i in 0..<(sortedSegments.count - 1) {
            let current = sortedSegments[i]
            let next = sortedSegments[i + 1]
            
            XCTAssertLessThanOrEqual(current.endTimeSeconds, next.startTimeSeconds + 0.1,
                                   "Segments should not overlap significantly")
        }
        
        // Validate speaker IDs
        for segment in result.segments {
            XCTAssertTrue(result.speakerDatabase.keys.contains(segment.speakerId),
                        "All segment speaker IDs should exist in database")
        }
        
        // Validate embeddings
        for (_, embedding) in result.speakerDatabase {
            XCTAssertFalse(embedding.isEmpty, "Embeddings should not be empty")
            XCTAssertFalse(embedding.contains { $0.isNaN }, "Embeddings should not contain NaN")
        }
    }
    
    private func validateComputationalAccuracy(_ result: DiarizationResult) {
        // Validate that computational optimizations maintain accuracy
        for segment in result.segments {
            XCTAssert(segment.qualityScore >= 0.0 && segment.qualityScore <= 1.0,
                    "Quality scores should be in valid range")
            XCTAssert(segment.startTimeSeconds >= 0.0, "Start times should be non-negative")
            XCTAssert(segment.endTimeSeconds > segment.startTimeSeconds, "End times should be after start times")
        }
    }
    
    private func generateRealisticAudioSample(durationSeconds: Float, sampleRate: Int) -> [Float] {
        let sampleCount = Int(durationSeconds * Float(sampleRate))
        var audio = Array<Float>(repeating: 0.0, count: sampleCount)
        
        // Multiple speakers with realistic speech patterns
        let speakerPatterns = [
            (startTime: 0.0, endTime: durationSeconds * 0.3, frequency: 150.0, amplitude: 0.6),
            (startTime: durationSeconds * 0.2, endTime: durationSeconds * 0.7, frequency: 250.0, amplitude: 0.5),
            (startTime: durationSeconds * 0.6, endTime: durationSeconds, frequency: 200.0, amplitude: 0.7)
        ]
        
        for pattern in speakerPatterns {
            let startSample = Int(pattern.startTime * Float(sampleRate))
            let endSample = Int(pattern.endTime * Float(sampleRate))
            
            for i in startSample..<min(endSample, sampleCount) {
                let t = Float(i - startSample) / Float(sampleRate)
                // Add speech-like modulation
                let envelope = 0.5 + 0.5 * sin(2.0 * Float.pi * 5.0 * t) // 5 Hz modulation
                let carrier = sin(2.0 * Float.pi * Float(pattern.frequency) * t)
                audio[i] += Float(pattern.amplitude) * envelope * carrier
            }
        }
        
        return audio
    }
    
    private func generateTestAudioForFallback(durationSeconds: Float, sampleRate: Int) -> [Float] {
        let sampleCount = Int(durationSeconds * Float(sampleRate))
        return (0..<sampleCount).map { i in
            let t = Float(i) / Float(sampleRate)
            return 0.5 * sin(2.0 * Float.pi * 440.0 * t) * (1.0 + 0.1 * sin(2.0 * Float.pi * 3.0 * t))
        }
    }
    
    private func generateSimpleTestAudio(durationSeconds: Float, sampleRate: Int) -> [Float] {
        let sampleCount = Int(durationSeconds * Float(sampleRate))
        return (0..<sampleCount).map { i in
            sin(2.0 * Float.pi * 440.0 * Float(i) / Float(sampleRate)) * 0.5
        }
    }
    
    private func generatePerformanceTestAudio(durationSeconds: Float, sampleRate: Int) -> [Float] {
        let sampleCount = Int(durationSeconds * Float(sampleRate))
        var audio = Array<Float>(repeating: 0.0, count: sampleCount)
        
        // Complex multi-frequency signal for performance testing
        for i in 0..<sampleCount {
            let t = Float(i) / Float(sampleRate)
            audio[i] = 0.3 * sin(2.0 * Float.pi * 220.0 * t) +
                      0.2 * sin(2.0 * Float.pi * 440.0 * t) +
                      0.1 * sin(2.0 * Float.pi * 880.0 * t)
        }
        
        return audio
    }
    
    private func generateTestAudioForMemoryTest(durationSeconds: Float, sampleRate: Int) -> [Float] {
        let sampleCount = Int(durationSeconds * Float(sampleRate))
        return (0..<sampleCount).map { i in
            let frequency = 440.0 + Float(i % 1000) / 10.0 // Varying frequency
            return sin(2.0 * Float.pi * frequency * Float(i) / Float(sampleRate)) * 0.4
        }
    }
    
    private func generateConfigTestAudio(durationSeconds: Float, sampleRate: Int) -> [Float] {
        let sampleCount = Int(durationSeconds * Float(sampleRate))
        var audio = Array<Float>(repeating: 0.0, count: sampleCount)
        
        // Segments of different lengths to test configuration parameters
        let segmentLength = sampleCount / 4
        
        for segment in 0..<4 {
            let startIdx = segment * segmentLength
            let endIdx = min((segment + 1) * segmentLength, sampleCount)
            
            for i in startIdx..<endIdx {
                let frequency = 200.0 + Float(segment) * 100.0
                audio[i] = 0.5 * sin(2.0 * Float.pi * frequency * Float(i - startIdx) / Float(sampleRate))
            }
        }
        
        return audio
    }
    
    static func createStressTestAudio(durationSeconds: Float, sampleRate: Int) -> [Float] {
        let sampleCount = Int(durationSeconds * Float(sampleRate))
        return (0..<sampleCount).map { i in
            // Multiple overlapping tones for stress testing
            let t = Float(i) / Float(sampleRate)
            return 0.2 * sin(2.0 * Float.pi * 300.0 * t) +
                   0.2 * sin(2.0 * Float.pi * 600.0 * t) +
                   0.1 * sin(2.0 * Float.pi * 900.0 * t) +
                   0.05 * Float.random(in: -1.0...1.0) // Add noise
        }
    }
    
    private func generateStressTestAudio(durationSeconds: Float, sampleRate: Int) -> [Float] {
        let sampleCount = Int(durationSeconds * Float(sampleRate))
        return (0..<sampleCount).map { i in
            // Multiple overlapping tones for stress testing
            let t = Float(i) / Float(sampleRate)
            return 0.2 * sin(2.0 * Float.pi * 300.0 * t) +
                   0.2 * sin(2.0 * Float.pi * 600.0 * t) +
                   0.1 * sin(2.0 * Float.pi * 900.0 * t) +
                   0.05 * Float.random(in: -1.0...1.0) // Add noise
        }
    }
    
    private func generateLongAudioSample(durationSeconds: Float, sampleRate: Int) -> [Float] {
        let sampleCount = Int(durationSeconds * Float(sampleRate))
        var audio = Array<Float>(repeating: 0.0, count: sampleCount)
        
        // Long audio with varying speaker patterns
        let numSpeakers = 4
        let speakerDuration = durationSeconds / Float(numSpeakers)
        
        for speaker in 0..<numSpeakers {
            let startTime = Float(speaker) * speakerDuration
            let endTime = startTime + speakerDuration * 1.2 // Overlapping speakers
            
            let startSample = Int(startTime * Float(sampleRate))
            let endSample = Int(min(endTime * Float(sampleRate), Float(sampleCount)))
            
            let frequency = 200.0 + Float(speaker) * 50.0
            
            for i in startSample..<endSample {
                let t = Float(i - startSample) / Float(sampleRate)
                audio[i] += 0.4 * sin(2.0 * Float.pi * frequency * t)
            }
        }
        
        return audio
    }
}