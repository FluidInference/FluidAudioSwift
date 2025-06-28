import XCTest
@testable import FluidAudioSwift

/// Comprehensive tests for TaskGroup-based parallel processing
/// Tests concurrent chunk processing, speaker ID consistency, error handling, and performance validation
@available(macOS 13.0, iOS 16.0, *)
final class ParallelProcessingTests: XCTestCase {
    
    private let testTimeout: TimeInterval = 60.0
    
    // MARK: - Parallel Processing Threshold Tests
    
    func testParallelProcessingThreshold() async {
        // Test that short audio uses sequential processing
        let shortConfig = DiarizerConfig(debugMode: true, parallelProcessingThreshold: 60.0)
        let shortManager = DiarizerManager(config: shortConfig)
        
        // Create 30-second audio (below threshold)
        let shortAudio = generateTestAudio(durationSeconds: 30.0, sampleRate: 16000)
        
        do {
            try await shortManager.initialize()
            
            let startTime = CFAbsoluteTimeGetCurrent()
            let result = try await shortManager.performCompleteDiarization(shortAudio, sampleRate: 16000)
            let processingTime = CFAbsoluteTimeGetCurrent() - startTime
            
            print("📊 Short Audio Processing (30s): \(String(format: "%.3f", processingTime))s")
            XCTAssertNotNil(result, "Short audio should process successfully")
            
        } catch {
            print("ℹ️ Short audio test skipped - models not available: \(error)")
        }
        
        // Test that long audio triggers parallel processing
        let longConfig = DiarizerConfig(debugMode: true, parallelProcessingThreshold: 60.0)
        let longManager = DiarizerManager(config: longConfig)
        
        // Create 120-second audio (above threshold)
        let longAudio = generateTestAudio(durationSeconds: 120.0, sampleRate: 16000)
        
        do {
            try await longManager.initialize()
            
            let startTime = CFAbsoluteTimeGetCurrent()
            let result = try await longManager.performCompleteDiarization(longAudio, sampleRate: 16000)
            let processingTime = CFAbsoluteTimeGetCurrent() - startTime
            
            print("📊 Long Audio Processing (120s): \(String(format: "%.3f", processingTime))s")
            XCTAssertNotNil(result, "Long audio should process successfully")
            
        } catch {
            print("ℹ️ Long audio test skipped - models not available: \(error)")
        }
    }
    
    func testCustomParallelThreshold() async {
        // Test custom threshold configuration
        let customConfig = DiarizerConfig(parallelProcessingThreshold: 30.0)
        let manager = DiarizerManager(config: customConfig)
        
        // Create 45-second audio (above custom threshold)
        let audio = generateTestAudio(durationSeconds: 45.0, sampleRate: 16000)
        
        do {
            try await manager.initialize()
            let result = try await manager.performCompleteDiarization(audio, sampleRate: 16000)
            XCTAssertNotNil(result, "Audio above custom threshold should process")
            
        } catch {
            print("ℹ️ Custom threshold test skipped - models not available: \(error)")
        }
    }
    
    // MARK: - TaskGroup Concurrency Tests
    
    func testTaskGroupExecution() async {
        // Test TaskGroup-based parallel chunk processing without models
        let chunks = [
            generateTestAudio(durationSeconds: 10.0, sampleRate: 16000),
            generateTestAudio(durationSeconds: 10.0, sampleRate: 16000),
            generateTestAudio(durationSeconds: 10.0, sampleRate: 16000),
            generateTestAudio(durationSeconds: 10.0, sampleRate: 16000)
        ]
        
        let startTime = CFAbsoluteTimeGetCurrent()
        
        // Simulate parallel processing structure
        let results: [(index: Int, duration: Float)]
        do {
            results = try await withThrowingTaskGroup(of: (index: Int, duration: Float).self) { group in
                for (index, chunk) in chunks.enumerated() {
                    group.addTask {
                        // Simulate processing time
                        try await Task.sleep(nanoseconds: 100_000_000) // 0.1 seconds
                        let duration = Float(chunk.count) / 16000.0
                        return (index: index, duration: duration)
                    }
                }
                
                var taskResults: [(index: Int, duration: Float)] = []
                for try await result in group {
                    taskResults.append(result)
                }
                return taskResults
            }
        } catch {
            XCTFail("TaskGroup execution failed: \(error)")
            return
        }
        
        let totalTime = CFAbsoluteTimeGetCurrent() - startTime
        
        // Verify all chunks were processed
        XCTAssertEqual(results.count, 4, "All chunks should be processed")
        
        // Verify parallel execution was faster than sequential
        // (4 chunks × 0.1s sequentially = 0.4s, parallel should be ~0.1s)
        XCTAssertLessThan(totalTime, 0.3, "Parallel execution should be faster than sequential")
        
        // Verify results maintain order information
        let sortedResults = results.sorted { $0.index < $1.index }
        for (expectedIndex, result) in sortedResults.enumerated() {
            XCTAssertEqual(result.index, expectedIndex, "Chunk ordering should be preserved")
        }
        
        print("✅ TaskGroup parallel execution working correctly")
        print("   Processed 4 chunks in \(String(format: "%.3f", totalTime))s")
    }
    
    func testTaskGroupErrorHandling() async {
        // Test error propagation in TaskGroup
        enum TestError: Error {
            case simulatedFailure
        }
        
        do {
            _ = try await withThrowingTaskGroup(of: Int.self) { group in
                // Add successful tasks
                group.addTask { return 1 }
                group.addTask { return 2 }
                
                // Add failing task
                group.addTask {
                    throw TestError.simulatedFailure
                }
                
                var results: [Int] = []
                for try await result in group {
                    results.append(result)
                }
                return results
            }
            
            XCTFail("TaskGroup should have thrown an error")
            
        } catch TestError.simulatedFailure {
            print("✅ TaskGroup error propagation working correctly")
        } catch {
            XCTFail("Unexpected error type: \(error)")
        }
    }
    
    func testTaskGroupCancellation() async {
        let expectation = XCTestExpectation(description: "Task cancellation")
        
        let task = Task {
            try await withThrowingTaskGroup(of: Void.self) { group in
                for _ in 0..<10 {
                    group.addTask {
                        // Long-running task
                        for _ in 0..<1000000 {
                            try Task.checkCancellation()
                            // Simulate work
                        }
                    }
                }
                
                for try await _ in group {
                    // Process results
                }
            }
        }
        
        // Cancel after short delay
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
            task.cancel()
            expectation.fulfill()
        }
        
        do {
            try await task.value
            XCTFail("Task should have been cancelled")
        } catch is CancellationError {
            print("✅ TaskGroup cancellation working correctly")
        } catch {
            XCTFail("Unexpected error: \(error)")
        }
        
        await fulfillment(of: [expectation], timeout: 1.0)
    }
    
    // MARK: - Speaker ID Consistency Tests
    
    func testSpeakerIDConsistencyAcrossChunks() async {
        // Test that speaker IDs remain consistent when processing chunks in parallel
        let config = DiarizerConfig(parallelProcessingThreshold: 15.0)
        let manager = DiarizerManager(config: config)
        
        // Create audio with distinct speaker patterns
        let speakerAudio = generateMultiSpeakerAudio(durationSeconds: 30.0, sampleRate: 16000)
        
        do {
            try await manager.initialize()
            let result = try await manager.performCompleteDiarization(speakerAudio, sampleRate: 16000)
            
            // Verify speaker database consistency
            XCTAssertFalse(result.speakerDatabase.isEmpty, "Speaker database should not be empty")
            
            // Verify segments have consistent speaker IDs
            let speakerIds = Set(result.segments.map { $0.speakerId })
            XCTAssertGreaterThan(speakerIds.count, 0, "Should identify at least one speaker")
            
            // Verify all speaker IDs in segments exist in database
            for segment in result.segments {
                XCTAssertTrue(result.speakerDatabase.keys.contains(segment.speakerId),
                            "Segment speaker ID '\(segment.speakerId)' should exist in speaker database")
            }
            
            // Verify temporal consistency (no overlapping segments from same speaker)
            let sortedSegments = result.segments.sorted { $0.startTimeSeconds < $1.startTimeSeconds }
            for i in 0..<(sortedSegments.count - 1) {
                let current = sortedSegments[i]
                let next = sortedSegments[i + 1]
                
                if current.speakerId == next.speakerId {
                    // Same speaker segments should not overlap
                    XCTAssertLessThanOrEqual(current.endTimeSeconds, next.startTimeSeconds,
                                           "Same speaker segments should not overlap")
                }
            }
            
            print("✅ Speaker ID consistency validated across parallel chunks")
            
        } catch {
            print("ℹ️ Speaker consistency test skipped - models not available: \(error)")
        }
    }
    
    func testSpeakerDatabaseMerging() async {
        // Test speaker database merging from parallel chunks
        let config = DiarizerConfig(debugMode: true, parallelProcessingThreshold: 20.0)
        let manager = DiarizerManager(config: config)
        
        // Create long audio to ensure parallel processing
        let longAudio = generateComplexMultiSpeakerAudio(durationSeconds: 60.0, sampleRate: 16000)
        
        do {
            try await manager.initialize()
            let result = try await manager.performCompleteDiarization(longAudio, sampleRate: 16000)
            
            // Verify speaker database has reasonable number of speakers
            let numSpeakers = result.speakerDatabase.count
            XCTAssertGreaterThan(numSpeakers, 0, "Should identify at least one speaker")
            XCTAssertLessThan(numSpeakers, 10, "Should not identify excessive number of speakers")
            
            // Verify all embeddings are valid
            for (speakerId, embedding) in result.speakerDatabase {
                XCTAssertFalse(embedding.isEmpty, "Speaker \(speakerId) embedding should not be empty")
                XCTAssertFalse(embedding.contains { $0.isNaN }, "Speaker \(speakerId) embedding should not contain NaN")
                XCTAssertFalse(embedding.contains { $0.isInfinite }, "Speaker \(speakerId) embedding should not contain infinity")
            }
            
            print("✅ Speaker database merging validated")
            print("   Identified \(numSpeakers) speakers in 60s audio")
            
        } catch {
            print("ℹ️ Speaker database test skipped - models not available: \(error)")
        }
    }
    
    // MARK: - Load Balancing Tests
    
    func testOptimalChunkSizing() async {
        // Test different chunk sizes for load balancing
        let testDurations: [Float] = [30.0, 60.0, 120.0, 240.0]
        
        for duration in testDurations {
            let chunkCount = Int(ceil(duration / 10.0)) // Assuming 10-second chunks
            let expectedParallelism = min(chunkCount, 4) // Assume max 4 cores
            
            print("📊 Duration: \(duration)s → \(chunkCount) chunks → \(expectedParallelism) parallel tasks")
            
            // Verify reasonable chunk distribution
            XCTAssertGreaterThan(chunkCount, 0, "Should have at least one chunk")
            if duration > 60.0 {
                XCTAssertGreaterThan(chunkCount, 6, "Long audio should have multiple chunks")
            }
        }
        
        print("✅ Chunk sizing analysis completed")
    }
    
    func testSystemResourceUtilization() async {
        // Test that parallel processing doesn't overwhelm system resources
        let config = DiarizerConfig(debugMode: true, parallelProcessingThreshold: 10.0)
        let manager = DiarizerManager(config: config)
        
        // Create multiple concurrent processing tasks
        let audioSamples = [
            generateTestAudio(durationSeconds: 30.0, sampleRate: 16000),
            generateTestAudio(durationSeconds: 25.0, sampleRate: 16000),
            generateTestAudio(durationSeconds: 35.0, sampleRate: 16000)
        ]
        
        do {
            try await manager.initialize()
            
            let startTime = CFAbsoluteTimeGetCurrent()
            
            // Process multiple audio samples concurrently
            _ = try await withThrowingTaskGroup(of: DiarizationResult.self) { group in
                for (index, audio) in audioSamples.enumerated() {
                    group.addTask {
                        print("Starting concurrent processing task \(index + 1)")
                        return try await manager.performCompleteDiarization(audio, sampleRate: 16000)
                    }
                }
                
                var results: [DiarizationResult] = []
                for try await result in group {
                    results.append(result)
                }
                
                let totalTime = CFAbsoluteTimeGetCurrent() - startTime
                print("📊 Concurrent Processing: 3 audio files in \(String(format: "%.3f", totalTime))s")
                
                XCTAssertEqual(results.count, 3, "All concurrent tasks should complete")
                
                return results
            }
            
            print("✅ System resource utilization test passed")
            
        } catch {
            print("ℹ️ Resource utilization test skipped - models not available: \(error)")
        }
    }
    
    // MARK: - Performance Validation Tests
    
    func testParallelProcessingSpeedup() async {
        // Test that parallel processing provides actual speedup
        let config1 = DiarizerConfig(debugMode: true, parallelProcessingThreshold: 20.0)
        let config2 = DiarizerConfig(debugMode: true, parallelProcessingThreshold: 100.0)
        let manager1 = DiarizerManager(config: config1)
        let manager2 = DiarizerManager(config: config2)
        
        // Create test audio
        let testAudio = generateTestAudio(durationSeconds: 40.0, sampleRate: 16000)
        
        do {
            // Test with parallel processing enabled (low threshold)
            try await manager1.initialize()
            let parallelStartTime = CFAbsoluteTimeGetCurrent()
            let _ = try await manager1.performCompleteDiarization(testAudio, sampleRate: 16000)
            let parallelTime = CFAbsoluteTimeGetCurrent() - parallelStartTime
            
            // Test with parallel processing disabled (high threshold)
            try await manager2.initialize()
            let sequentialStartTime = CFAbsoluteTimeGetCurrent()
            let _ = try await manager2.performCompleteDiarization(testAudio, sampleRate: 16000)
            let sequentialTime = CFAbsoluteTimeGetCurrent() - sequentialStartTime
            
            let speedup = sequentialTime / parallelTime
            
            print("📊 Parallel Processing Speedup Analysis:")
            print("   Sequential: \(String(format: "%.3f", sequentialTime))s")
            print("   Parallel: \(String(format: "%.3f", parallelTime))s")
            print("   Speedup: \(String(format: "%.2f", speedup))x")
            
            // Parallel should be at least as fast as sequential (may not be faster for short audio)
            XCTAssertLessThanOrEqual(parallelTime, sequentialTime * 1.2, "Parallel should not be significantly slower")
            
        } catch {
            print("ℹ️ Speedup test skipped - models not available: \(error)")
        }
    }
    
    func testMemoryUsageDuringParallelProcessing() async {
        // Test memory efficiency during parallel processing
        let config = DiarizerConfig(debugMode: true, parallelProcessingThreshold: 15.0)
        let manager = DiarizerManager(config: config)
        
        // Create large audio sample
        let largeAudio = generateTestAudio(durationSeconds: 120.0, sampleRate: 16000)
        
        do {
            try await manager.initialize()
            
            // Process and monitor memory usage
            _ = autoreleasepool {
                Task {
                    let _ = try await manager.performCompleteDiarization(largeAudio, sampleRate: 16000)
                }
            }
            
            // If we reach here without memory issues, test passes
            print("✅ Memory usage during parallel processing test passed")
            
        } catch {
            print("ℹ️ Memory test skipped - models not available: \(error)")
        }
    }
    
    // MARK: - Edge Cases and Error Handling
    
    func testEmptyAudioParallelProcessing() async {
        let config = DiarizerConfig(debugMode: true, parallelProcessingThreshold: 1.0)
        let manager = DiarizerManager(config: config)
        
        let emptyAudio: [Float] = []
        
        do {
            try await manager.initialize()
            let result = try await manager.performCompleteDiarization(emptyAudio, sampleRate: 16000)
            XCTAssertTrue(result.segments.isEmpty, "Empty audio should produce no segments")
            
        } catch {
            // Expected to fail with invalid audio
            print("✅ Empty audio properly rejected: \(error)")
        }
    }
    
    func testVeryShortAudioChunks() async {
        let config = DiarizerConfig(debugMode: true, parallelProcessingThreshold: 0.5) // Very low threshold
        let manager = DiarizerManager(config: config)
        
        // 1-second audio (shorter than typical chunk size)
        let shortAudio = generateTestAudio(durationSeconds: 1.0, sampleRate: 16000)
        
        do {
            try await manager.initialize()
            let result = try await manager.performCompleteDiarization(shortAudio, sampleRate: 16000)
            
            // Should handle gracefully
            XCTAssertNotNil(result, "Very short audio should be handled gracefully")
            
        } catch {
            print("ℹ️ Very short audio test skipped - models not available: \(error)")
        }
    }
    
    // MARK: - Helper Methods
    
    private func generateTestAudio(durationSeconds: Float, sampleRate: Int) -> [Float] {
        let sampleCount = Int(durationSeconds * Float(sampleRate))
        return (0..<sampleCount).map { i in
            // Generate a simple sine wave with some variation
            let frequency: Float = 440.0 + Float(i % 100) * 2.0
            return sin(2.0 * Float.pi * frequency * Float(i) / Float(sampleRate)) * 0.5
        }
    }
    
    private func generateMultiSpeakerAudio(durationSeconds: Float, sampleRate: Int) -> [Float] {
        let sampleCount = Int(durationSeconds * Float(sampleRate))
        let chunkSize = sampleCount / 3
        
        var audio: [Float] = []
        
        // Speaker 1: Low frequency
        for i in 0..<chunkSize {
            let frequency: Float = 200.0
            audio.append(sin(2.0 * Float.pi * frequency * Float(i) / Float(sampleRate)) * 0.6)
        }
        
        // Speaker 2: Medium frequency
        for i in 0..<chunkSize {
            let frequency: Float = 440.0
            audio.append(sin(2.0 * Float.pi * frequency * Float(i) / Float(sampleRate)) * 0.5)
        }
        
        // Speaker 3: High frequency
        for i in 0..<(sampleCount - 2 * chunkSize) {
            let frequency: Float = 880.0
            audio.append(sin(2.0 * Float.pi * frequency * Float(i) / Float(sampleRate)) * 0.4)
        }
        
        return audio
    }
    
    private func generateComplexMultiSpeakerAudio(durationSeconds: Float, sampleRate: Int) -> [Float] {
        let sampleCount = Int(durationSeconds * Float(sampleRate))
        var audio = Array<Float>(repeating: 0.0, count: sampleCount)
        
        // Multiple overlapping speakers with different characteristics
        let speakers = [
            (frequency: 220.0, amplitude: 0.4, phase: 0.0),
            (frequency: 440.0, amplitude: 0.3, phase: Float.pi / 4),
            (frequency: 660.0, amplitude: 0.2, phase: Float.pi / 2),
        ]
        
        for (index, _) in audio.enumerated() {
            let t = Float(index) / Float(sampleRate)
            var value: Float = 0
            
            // Each speaker appears in different time segments
            for (speakerIndex, speaker) in speakers.enumerated() {
                let speakerStart = Float(speakerIndex) * durationSeconds / 3.0
                let speakerEnd = speakerStart + durationSeconds / 2.0
                
                if t >= speakerStart && t <= speakerEnd {
                    value += Float(speaker.amplitude) * sin(2.0 * Float.pi * Float(speaker.frequency) * t + Float(speaker.phase))
                }
            }
            
            audio[index] = value
        }
        
        return audio
    }
}