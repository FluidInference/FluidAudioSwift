import XCTest
import Metal
import MetalPerformanceShaders
import Accelerate
import Foundation
@testable import FluidAudioSwift

/// Comprehensive benchmarks for Metal acceleration vs Accelerate framework
/// Designed for CI integration with structured JSON output for PR comments
@available(macOS 13.0, iOS 16.0, *)
final class MetalAccelerationBenchmarks: XCTestCase {

    private var metalProcessor: MetalPerformanceProcessor!
    private var benchmarkResults: [String: Any] = [:]
    private let testTimeout: TimeInterval = 180.0

    override func setUp() {
        super.setUp()
        metalProcessor = MetalPerformanceProcessor()
        benchmarkResults = [
            "timestamp": ISO8601DateFormatter().string(from: Date()),
            "metal_available": metalProcessor.isAvailable,
            "tests": []
        ]
    }

    override func tearDown() {
        // Output benchmark results as JSON for CI consumption
        if let jsonData = try? JSONSerialization.data(withJSONObject: benchmarkResults, options: [.prettyPrinted]),
           let jsonString = String(data: jsonData, encoding: .utf8) {
            print("\n🔬 BENCHMARK_RESULTS_JSON_START")
            print(jsonString)
            print("🔬 BENCHMARK_RESULTS_JSON_END\n")
        }

        metalProcessor = nil
        super.tearDown()
    }

    // MARK: - Cosine Distance Benchmarks

    func testCosineDistanceBatchSizeBenchmark() {
        let batchSizes = [8, 16, 32, 64, 128]
        let embeddingDim = 512
        let numCandidates = 50

        for batchSize in batchSizes {
            let testResult = benchmarkCosineDistances(
                numQueries: batchSize,
                numCandidates: numCandidates,
                embeddingDim: embeddingDim,
                testName: "cosine_distance_batch_\(batchSize)"
            )
            addBenchmarkResult(testResult)
        }
    }

    func testCosineDistanceEmbeddingDimensionBenchmark() {
        let embeddingDims = [256, 512, 1024]
        let batchSize = 32
        let numCandidates = 50

        for embeddingDim in embeddingDims {
            let testResult = benchmarkCosineDistances(
                numQueries: batchSize,
                numCandidates: numCandidates,
                embeddingDim: embeddingDim,
                testName: "cosine_distance_dim_\(embeddingDim)"
            )
            addBenchmarkResult(testResult)
        }
    }

    func testCosineDistanceScalabilityBenchmark() {
        let scalingFactors = [(16, 25), (32, 50), (64, 100), (128, 200)]
        let embeddingDim = 512

        for (numQueries, numCandidates) in scalingFactors {
            let testResult = benchmarkCosineDistances(
                numQueries: numQueries,
                numCandidates: numCandidates,
                embeddingDim: embeddingDim,
                testName: "cosine_distance_scale_\(numQueries)x\(numCandidates)"
            )
            addBenchmarkResult(testResult)
        }
    }

    // MARK: - Powerset Conversion Benchmarks

    func testPowersetConversionBatchSizeBenchmark() {
        let batchSizes = [1, 2, 4, 8]
        let numFrames = 589 // Typical 10-second chunk

        for batchSize in batchSizes {
            let testResult = benchmarkPowersetConversion(
                batchSize: batchSize,
                numFrames: numFrames,
                testName: "powerset_batch_\(batchSize)"
            )
            addBenchmarkResult(testResult)
        }
    }

    func testPowersetConversionFrameCountBenchmark() {
        let frameCounts = [294, 589, 1178, 2356] // 5s, 10s, 20s, 40s chunks
        let batchSize = 4

        for numFrames in frameCounts {
            let testResult = benchmarkPowersetConversion(
                batchSize: batchSize,
                numFrames: numFrames,
                testName: "powerset_frames_\(numFrames)"
            )
            addBenchmarkResult(testResult)
        }
    }

    // MARK: - End-to-End Diarization Benchmarks

    func testEndToEndDiarizationBenchmark() {
        let audioDurations = [10.0, 30.0, 60.0] // seconds
        let sampleRate = 16000

        for duration in audioDurations {
            let testResult = benchmarkEndToEndDiarization(
                durationSeconds: duration,
                sampleRate: sampleRate,
                testName: "e2e_diarization_\(Int(duration))s"
            )
            if let result = testResult {
                addBenchmarkResult(result)
            }
        }
    }

    // MARK: - Memory Usage Benchmarks

    func testMemoryUsageBenchmark() {
        let testConfigs = [
            (queries: 50, candidates: 100, dim: 512, name: "memory_medium"),
            (queries: 100, candidates: 200, dim: 512, name: "memory_large"),
            (queries: 200, candidates: 300, dim: 1024, name: "memory_xlarge")
        ]

        for config in testConfigs {
            let testResult = benchmarkMemoryUsage(
                numQueries: config.queries,
                numCandidates: config.candidates,
                embeddingDim: config.dim,
                testName: config.name
            )
            addBenchmarkResult(testResult)
        }
    }

    // MARK: - Benchmark Implementation Methods

    private func benchmarkCosineDistances(
        numQueries: Int,
        numCandidates: Int,
        embeddingDim: Int,
        testName: String
    ) -> [String: Any] {

        // Generate test data
        let queries = generateRandomEmbeddings(count: numQueries, dimension: embeddingDim)
        let candidates = generateRandomEmbeddings(count: numCandidates, dimension: embeddingDim)

        var metalTime: Double = 0
        var accelerateTime: Double = 0
        var memoryBefore: Float = 0
        var memoryAfter: Float = 0

        // Benchmark Metal implementation
        if metalProcessor.isAvailable {
            memoryBefore = getMemoryUsage()
            let startTime = CFAbsoluteTimeGetCurrent()

            let _ = metalProcessor.batchCosineDistances(queries: queries, candidates: candidates)

            metalTime = CFAbsoluteTimeGetCurrent() - startTime
            memoryAfter = getMemoryUsage()
        }

        // Benchmark Accelerate implementation
        let accelerateStartTime = CFAbsoluteTimeGetCurrent()

        let _ = accelerateBatchCosineDistances(queries: queries, candidates: candidates)

        accelerateTime = CFAbsoluteTimeGetCurrent() - accelerateStartTime

        let speedup = metalProcessor.isAvailable && metalTime > 0 ? accelerateTime / metalTime : 0

        return [
            "test_name": testName,
            "test_type": "cosine_distance",
            "num_queries": numQueries,
            "num_candidates": numCandidates,
            "embedding_dim": embeddingDim,
            "metal_time_ms": metalTime * 1000,
            "accelerate_time_ms": accelerateTime * 1000,
            "speedup": speedup,
            "memory_increase_mb": memoryAfter - memoryBefore,
            "metal_available": metalProcessor.isAvailable
        ]
    }

    private func benchmarkPowersetConversion(
        batchSize: Int,
        numFrames: Int,
        testName: String
    ) -> [String: Any] {

        // Generate test data
        var segments: [[[Float]]] = []
        for _ in 0..<batchSize {
            var batchSegments: [[Float]] = []
            for _ in 0..<numFrames {
                let frameValues = generateRandomPowersetFrame()
                batchSegments.append(frameValues)
            }
            segments.append(batchSegments)
        }

        var metalTime: Double = 0
        var cpuTime: Double = 0

        // Benchmark Metal implementation
        if metalProcessor.isAvailable {
            let startTime = CFAbsoluteTimeGetCurrent()
            let _ = metalProcessor.performPowersetConversion(segments: segments)
            metalTime = CFAbsoluteTimeGetCurrent() - startTime
        }

        // Benchmark CPU implementation
        let cpuStartTime = CFAbsoluteTimeGetCurrent()
        let _ = cpuPowersetConversion(segments: segments)
        cpuTime = CFAbsoluteTimeGetCurrent() - cpuStartTime

        let speedup = metalProcessor.isAvailable && metalTime > 0 ? cpuTime / metalTime : 0
        let throughput = metalProcessor.isAvailable && metalTime > 0 ?
            Double(batchSize * numFrames) / metalTime : 0

        return [
            "test_name": testName,
            "test_type": "powerset_conversion",
            "batch_size": batchSize,
            "num_frames": numFrames,
            "metal_time_ms": metalTime * 1000,
            "cpu_time_ms": cpuTime * 1000,
            "speedup": speedup,
            "throughput_frames_per_sec": throughput,
            "metal_available": metalProcessor.isAvailable
        ]
    }

    private func benchmarkEndToEndDiarization(
        durationSeconds: Double,
        sampleRate: Int,
        testName: String
    ) -> [String: Any]? {

        let audioSamples = generateSyntheticAudio(
            durationSeconds: durationSeconds,
            sampleRate: sampleRate
        )

        // Test with Metal enabled
        var metalConfig = DiarizerConfig.default
        metalConfig.useMetalAcceleration = true
        metalConfig.debugMode = false

        // Test with Metal disabled (Accelerate only)
        var accelerateConfig = DiarizerConfig.default
        accelerateConfig.useMetalAcceleration = false
        accelerateConfig.debugMode = false

        var metalTime: Double = 0
        var accelerateTime: Double = 0
        var metalSuccess = false
        var accelerateSuccess = false

        // Benchmark with Metal acceleration
        if metalProcessor.isAvailable {
            let metalManager = DiarizerManager(config: metalConfig)

            let expectation = XCTestExpectation(description: "Metal diarization")
            let startTime = CFAbsoluteTimeGetCurrent()

            Task {
                do {
                    try await metalManager.initialize()
                    let _ = try await metalManager.performCompleteDiarization(audioSamples, sampleRate: sampleRate)
                    metalTime = CFAbsoluteTimeGetCurrent() - startTime
                    metalSuccess = true
                } catch {
                    print("Metal diarization failed: \(error)")
                }
                expectation.fulfill()
            }

            wait(for: [expectation], timeout: testTimeout)
        }

        // Benchmark with Accelerate only
        let accelerateManager = DiarizerManager(config: accelerateConfig)

        let accelerateExpectation = XCTestExpectation(description: "Accelerate diarization")
        let accelerateStartTime = CFAbsoluteTimeGetCurrent()

        Task {
            do {
                try await accelerateManager.initialize()
                let _ = try await accelerateManager.performCompleteDiarization(audioSamples, sampleRate: sampleRate)
                accelerateTime = CFAbsoluteTimeGetCurrent() - accelerateStartTime
                accelerateSuccess = true
            } catch {
                print("Accelerate diarization failed: \(error)")
            }
            accelerateExpectation.fulfill()
        }

        wait(for: [accelerateExpectation], timeout: testTimeout)

        guard metalSuccess || accelerateSuccess else {
            print("Both Metal and Accelerate diarization failed")
            return nil
        }

        let speedup = metalSuccess && accelerateSuccess && metalTime > 0 ? accelerateTime / metalTime : 0
        let realTimeFactor = metalSuccess && metalTime > 0 ? metalTime / durationSeconds :
                           (accelerateSuccess ? accelerateTime / durationSeconds : 0)

        return [
            "test_name": testName,
            "test_type": "end_to_end_diarization",
            "audio_duration_seconds": durationSeconds,
            "sample_rate": sampleRate,
            "metal_time_ms": metalTime * 1000,
            "accelerate_time_ms": accelerateTime * 1000,
            "speedup": speedup,
            "real_time_factor": realTimeFactor,
            "metal_success": metalSuccess,
            "accelerate_success": accelerateSuccess,
            "metal_available": metalProcessor.isAvailable
        ]
    }

    private func benchmarkMemoryUsage(
        numQueries: Int,
        numCandidates: Int,
        embeddingDim: Int,
        testName: String
    ) -> [String: Any] {

        let queries = generateRandomEmbeddings(count: numQueries, dimension: embeddingDim)
        let candidates = generateRandomEmbeddings(count: numCandidates, dimension: embeddingDim)

        var metalMemoryBefore: Float = 0
        var metalMemoryPeak: Float = 0

        var accelerateMemoryBefore: Float = 0
        var accelerateMemoryPeak: Float = 0

        // Benchmark Metal memory usage
        if metalProcessor.isAvailable {
            metalMemoryBefore = getMemoryUsage()
            let _ = metalProcessor.batchCosineDistances(queries: queries, candidates: candidates)
            metalMemoryPeak = getMemoryUsage()

            // Allow some time for cleanup
            Thread.sleep(forTimeInterval: 0.1)
            let _ = getMemoryUsage() // metalMemoryAfter - not used in calculation
        }

        // Benchmark Accelerate memory usage
        accelerateMemoryBefore = getMemoryUsage()
        let _ = accelerateBatchCosineDistances(queries: queries, candidates: candidates)
        accelerateMemoryPeak = getMemoryUsage()

        Thread.sleep(forTimeInterval: 0.1)
        let _ = getMemoryUsage() // accelerateMemoryAfter - not used in calculation

        let metalMemoryIncrease = metalMemoryPeak - metalMemoryBefore
        let accelerateMemoryIncrease = accelerateMemoryPeak - accelerateMemoryBefore
        let memoryReduction = accelerateMemoryIncrease > 0 ?
            (accelerateMemoryIncrease - metalMemoryIncrease) / accelerateMemoryIncrease * 100 : 0

        return [
            "test_name": testName,
            "test_type": "memory_usage",
            "num_queries": numQueries,
            "num_candidates": numCandidates,
            "embedding_dim": embeddingDim,
            "metal_memory_increase_mb": metalMemoryIncrease,
            "accelerate_memory_increase_mb": accelerateMemoryIncrease,
            "memory_reduction_percent": memoryReduction,
            "metal_available": metalProcessor.isAvailable
        ]
    }

    // MARK: - Helper Methods

    private func generateRandomEmbeddings(count: Int, dimension: Int) -> [[Float]] {
        var embeddings: [[Float]] = []

        for _ in 0..<count {
            var embedding: [Float] = []
            for _ in 0..<dimension {
                embedding.append(Float.random(in: -1.0...1.0))
            }

            // Normalize
            let magnitude = sqrt(embedding.map { $0 * $0 }.reduce(0, +))
            if magnitude > 0 {
                embedding = embedding.map { $0 / magnitude }
            }

            embeddings.append(embedding)
        }

        return embeddings
    }

    private func generateRandomPowersetFrame() -> [Float] {
        var frame: [Float] = []
        for _ in 0..<7 {
            frame.append(Float.random(in: 0.0...1.0))
        }
        return frame
    }

    private func generateSyntheticAudio(durationSeconds: Double, sampleRate: Int) -> [Float] {
        let numSamples = Int(durationSeconds * Double(sampleRate))
        var samples: [Float] = []

        // Generate synthetic audio with multiple speakers (simple sine waves)
        for i in 0..<numSamples {
            let time = Float(i) / Float(sampleRate)
            let speaker1 = sin(2.0 * Float.pi * 440.0 * time) * 0.3 // 440 Hz
            let speaker2 = sin(2.0 * Float.pi * 880.0 * time) * 0.2 // 880 Hz
            let noise = Float.random(in: -0.1...0.1)

            // Simulate speaker switching every 2 seconds
            let activeTime = fmod(time, 4.0)
            let sample = activeTime < 2.0 ? speaker1 + noise : speaker2 + noise

            samples.append(sample)
        }

        return samples
    }

    private func accelerateBatchCosineDistances(queries: [[Float]], candidates: [[Float]]) -> [[Float]] {
        var results: [[Float]] = []

        for query in queries {
            var queryResults: [Float] = []
            for candidate in candidates {
                let distance = accelerateCosineDistance(query, candidate)
                queryResults.append(distance)
            }
            results.append(queryResults)
        }

        return results
    }

    private func accelerateCosineDistance(_ a: [Float], _ b: [Float]) -> Float {
        guard a.count == b.count, !a.isEmpty else { return Float.infinity }

        let count = a.count
        var dotProduct: Float = 0
        var magnitudeA: Float = 0
        var magnitudeB: Float = 0

        // Use Accelerate for vectorized operations
        vDSP_dotpr(a, 1, b, 1, &dotProduct, vDSP_Length(count))
        vDSP_svesq(a, 1, &magnitudeA, vDSP_Length(count))
        vDSP_svesq(b, 1, &magnitudeB, vDSP_Length(count))

        magnitudeA = sqrt(magnitudeA)
        magnitudeB = sqrt(magnitudeB)

        if magnitudeA > 0 && magnitudeB > 0 {
            return 1 - (dotProduct / (magnitudeA * magnitudeB))
        } else {
            return Float.infinity
        }
    }

    private func cpuPowersetConversion(segments: [[[Float]]]) -> [[[Float]]]? {
        let powerset = [
            [-1, -1, -1], // 0: empty set
            [0, -1, -1],  // 1: {0}
            [1, -1, -1],  // 2: {1}
            [2, -1, -1],  // 3: {2}
            [0, 1, -1],   // 4: {0, 1}
            [0, 2, -1],   // 5: {0, 2}
            [1, 2, -1]    // 6: {1, 2}
        ]

        var results: [[[Float]]] = []

        for batchSegments in segments {
            var batchResults: [[Float]] = []

            for frameValues in batchSegments {
                guard frameValues.count == 7 else { continue }

                // Find max value index
                let maxIndex = frameValues.indices.max(by: { frameValues[$0] < frameValues[$1] }) ?? 0
                let speakers = powerset[maxIndex]

                // Convert to speaker activation
                var speakerActivation: [Float] = [0.0, 0.0, 0.0]
                for speaker in speakers {
                    if speaker >= 0 && speaker < 3 {
                        speakerActivation[speaker] = 1.0
                    }
                }

                batchResults.append(speakerActivation)
            }

            results.append(batchResults)
        }

        return results
    }

    private func getMemoryUsage() -> Float {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size)/4

        // Use the global variable directly for thread safety
        let taskPort = mach_task_self_

        let kerr: kern_return_t = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: 1) {
                task_info(taskPort,
                         task_flavor_t(MACH_TASK_BASIC_INFO),
                         $0,
                         &count)
            }
        }

        if kerr == KERN_SUCCESS {
            return Float(info.resident_size) / 1024.0 / 1024.0 // Convert to MB
        }

        return 0
    }

    private func addBenchmarkResult(_ result: [String: Any]) {
        if var tests = benchmarkResults["tests"] as? [[String: Any]] {
            tests.append(result)
            benchmarkResults["tests"] = tests
        } else {
            benchmarkResults["tests"] = [result]
        }
    }
}
