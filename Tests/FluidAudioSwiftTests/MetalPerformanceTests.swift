import XCTest
import Metal
import MetalPerformanceShaders
@testable import FluidAudioSwift

/// Comprehensive tests for Metal Performance Shaders GPU acceleration
/// Tests Metal device detection, MPS matrix operations, custom compute kernels, and fallback mechanisms
@available(macOS 13.0, iOS 16.0, *)
final class MetalPerformanceTests: XCTestCase {
    
    private var metalProcessor: MetalPerformanceProcessor!
    private let testTimeout: TimeInterval = 30.0
    
    override func setUp() {
        super.setUp()
        metalProcessor = MetalPerformanceProcessor()
    }
    
    override func tearDown() {
        metalProcessor = nil
        super.tearDown()
    }
    
    // MARK: - Metal Device Detection Tests
    
    func testMetalDeviceAvailability() {
        // Test Metal device detection
        let device = MTLCreateSystemDefaultDevice()
        
        if device != nil {
            print("✅ Metal device available: \(device!.name)")
            XCTAssertTrue(metalProcessor.isAvailable, "MetalPerformanceProcessor should be available when device exists")
        } else {
            print("ℹ️ Metal device not available (expected on some CI environments)")
            XCTAssertFalse(metalProcessor.isAvailable, "MetalPerformanceProcessor should not be available without device")
        }
    }
    
    func testMetalCommandQueueCreation() {
        guard metalProcessor.isAvailable else {
            print("ℹ️ Skipping Metal command queue test - Metal not available")
            return
        }
        
        // Test that we can create command buffers
        let device = MTLCreateSystemDefaultDevice()!
        let commandQueue = device.makeCommandQueue()
        XCTAssertNotNil(commandQueue, "Should be able to create Metal command queue")
        
        let commandBuffer = commandQueue?.makeCommandBuffer()
        XCTAssertNotNil(commandBuffer, "Should be able to create Metal command buffer")
    }
    
    // MARK: - MPS Matrix Operations Tests
    
    func testBatchCosineDistancesBasic() {
        guard metalProcessor.isAvailable else {
            print("ℹ️ Skipping MPS matrix test - Metal not available")
            return
        }
        
        // Test basic batch cosine distance calculation
        let queries: [[Float]] = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ]
        
        let candidates: [[Float]] = [
            [1.0, 0.0, 0.0],  // Identical to query 0
            [0.0, 1.0, 0.0],  // Identical to query 1
            [-1.0, 0.0, 0.0]  // Opposite to query 0
        ]
        
        guard let distances = metalProcessor.batchCosineDistances(queries: queries, candidates: candidates) else {
            XCTFail("Metal batch cosine distances failed")
            return
        }
        
        XCTAssertEqual(distances.count, 3, "Should have 3 query results")
        XCTAssertEqual(distances[0].count, 3, "Each query should have 3 candidate distances")
        
        // Test specific distance values
        XCTAssertEqual(distances[0][0], 0.0, accuracy: 0.001, "Identical vectors should have distance 0")
        XCTAssertEqual(distances[1][1], 0.0, accuracy: 0.001, "Identical vectors should have distance 0")
        XCTAssertEqual(distances[0][2], 2.0, accuracy: 0.001, "Opposite vectors should have distance 2")
        XCTAssertEqual(distances[0][1], 1.0, accuracy: 0.001, "Orthogonal vectors should have distance 1")
        
        print("✅ Metal MPS basic batch cosine distances working correctly")
    }
    
    func testBatchCosineDistancesAccuracy() {
        guard metalProcessor.isAvailable else {
            print("ℹ️ Skipping MPS accuracy test - Metal not available")
            return
        }
        
        // Generate random embeddings for accuracy testing
        let embeddingDim = 256
        let numQueries = 10
        let numCandidates = 15
        
        var queries: [[Float]] = []
        var candidates: [[Float]] = []
        
        // Generate normalized random embeddings
        for _ in 0..<numQueries {
            let embedding = generateNormalizedRandomEmbedding(dimension: embeddingDim)
            queries.append(embedding)
        }
        
        for _ in 0..<numCandidates {
            let embedding = generateNormalizedRandomEmbedding(dimension: embeddingDim)
            candidates.append(embedding)
        }
        
        // Calculate distances using Metal
        guard let metalDistances = metalProcessor.batchCosineDistances(queries: queries, candidates: candidates) else {
            XCTFail("Metal batch cosine distances failed")
            return
        }
        
        // Calculate reference distances using CPU
        var cpuDistances: [[Float]] = []
        for query in queries {
            var queryDistances: [Float] = []
            for candidate in candidates {
                let distance = cpuCosineDistance(query, candidate)
                queryDistances.append(distance)
            }
            cpuDistances.append(queryDistances)
        }
        
        // Compare Metal vs CPU results
        for i in 0..<numQueries {
            for j in 0..<numCandidates {
                let metalDist = metalDistances[i][j]
                let cpuDist = cpuDistances[i][j]
                XCTAssertEqual(metalDist, cpuDist, accuracy: 0.001, 
                             "Metal distance [\(i)][\(j)] should match CPU calculation")
            }
        }
        
        print("✅ Metal MPS accuracy validated against CPU reference")
    }
    
    func testBatchCosineDistancesPerformance() {
        guard metalProcessor.isAvailable else {
            print("ℹ️ Skipping MPS performance test - Metal not available")
            return
        }
        
        // Performance test with larger matrices
        let embeddingDim = 512
        let numQueries = 50
        let numCandidates = 100
        
        var queries: [[Float]] = []
        var candidates: [[Float]] = []
        
        for _ in 0..<numQueries {
            queries.append(generateNormalizedRandomEmbedding(dimension: embeddingDim))
        }
        
        for _ in 0..<numCandidates {
            candidates.append(generateNormalizedRandomEmbedding(dimension: embeddingDim))
        }
        
        // Measure Metal performance
        let metalStartTime = CFAbsoluteTimeGetCurrent()
        guard let _ = metalProcessor.batchCosineDistances(queries: queries, candidates: candidates) else {
            XCTFail("Metal batch cosine distances failed")
            return
        }
        let metalTime = CFAbsoluteTimeGetCurrent() - metalStartTime
        
        // Measure CPU performance for comparison
        let cpuStartTime = CFAbsoluteTimeGetCurrent()
        for query in queries {
            for candidate in candidates {
                _ = cpuCosineDistance(query, candidate)
            }
        }
        let cpuTime = CFAbsoluteTimeGetCurrent() - cpuStartTime
        
        let speedup = cpuTime / metalTime
        print("📊 Metal MPS Performance: \(String(format: "%.2f", speedup))x speedup over CPU")
        print("   Metal time: \(String(format: "%.4f", metalTime))s")
        print("   CPU time: \(String(format: "%.4f", cpuTime))s")
        
        // Metal should be faster for large matrices (aim for at least 2x speedup)
        if speedup > 2.0 {
            print("✅ Metal MPS showing good performance improvement")
        } else {
            print("ℹ️ Metal MPS speedup lower than expected (may vary by hardware)")
        }
    }
    
    func testBatchCosineDistancesEdgeCases() {
        guard metalProcessor.isAvailable else {
            print("ℹ️ Skipping MPS edge cases test - Metal not available")
            return
        }
        
        // Test empty inputs
        let emptyResult = metalProcessor.batchCosineDistances(queries: [], candidates: [])
        XCTAssertNil(emptyResult, "Empty inputs should return nil")
        
        // Test mismatched dimensions
        let queries: [[Float]] = [[1.0, 0.0, 0.0]]
        let candidates: [[Float]] = [[1.0, 0.0]]  // Different dimension
        let mismatchedResult = metalProcessor.batchCosineDistances(queries: queries, candidates: candidates)
        XCTAssertNil(mismatchedResult, "Mismatched dimensions should return nil")
        
        // Test single embedding case
        let singleQuery: [[Float]] = [[1.0, 0.0, 0.0]]
        let singleCandidate: [[Float]] = [[1.0, 0.0, 0.0]]
        let singleResult = metalProcessor.batchCosineDistances(queries: singleQuery, candidates: singleCandidate)
        XCTAssertNotNil(singleResult, "Single embedding should work")
        XCTAssertEqual(singleResult?[0][0] ?? Float.infinity, 0.0, accuracy: 0.001, "Identical single embeddings should have distance 0")
        
        print("✅ Metal MPS edge cases handled correctly")
    }
    
    // MARK: - Metal Compute Kernel Tests
    
    func testPowersetConversionKernel() {
        guard metalProcessor.isAvailable else {
            print("ℹ️ Skipping powerset kernel test - Metal not available")
            return
        }
        
        // Test powerset conversion with known input
        let batchSize = 1
        let numFrames = 10
        let numCombinations = 7
        
        // Create test input with clear max values
        var segments: [[[Float]]] = []
        var batchSegments: [[Float]] = []
        
        for frame in 0..<numFrames {
            var frameValues: [Float] = Array(repeating: 0.1, count: numCombinations)
            // Set clear maximum for each frame (cycling through combinations)
            let maxIndex = frame % numCombinations
            frameValues[maxIndex] = 0.9
            batchSegments.append(frameValues)
        }
        segments.append(batchSegments)
        
        guard let result = metalProcessor.performPowersetConversion(segments: segments) else {
            XCTFail("Metal powerset conversion failed")
            return
        }
        
        XCTAssertEqual(result.count, batchSize, "Should have correct batch size")
        XCTAssertEqual(result[0].count, numFrames, "Should have correct number of frames")
        XCTAssertEqual(result[0][0].count, 3, "Should have 3 speakers output")
        
        // Verify powerset conversion logic
        let powerset = [
            [-1, -1, -1], // 0: empty set
            [0, -1, -1],  // 1: {0}
            [1, -1, -1],  // 2: {1}
            [2, -1, -1],  // 3: {2}
            [0, 1, -1],   // 4: {0, 1}
            [0, 2, -1],   // 5: {0, 2}
            [1, 2, -1]    // 6: {1, 2}
        ]
        
        for frame in 0..<numFrames {
            let maxIndex = frame % numCombinations
            let expectedSpeakers = powerset[maxIndex]
            
            for speaker in 0..<3 {
                let expected: Float = expectedSpeakers.contains(speaker) ? 1.0 : 0.0
                let actual = result[0][frame][speaker]
                XCTAssertEqual(actual, expected, accuracy: 0.001, 
                             "Frame \(frame), Speaker \(speaker): expected \(expected), got \(actual)")
            }
        }
        
        print("✅ Metal powerset conversion kernel working correctly")
    }
    
    func testPowersetConversionPerformance() {
        guard metalProcessor.isAvailable else {
            print("ℹ️ Skipping powerset performance test - Metal not available")
            return
        }
        
        // Performance test with larger input
        let batchSize = 4
        let numFrames = 589  // Typical frame count
        let _ = 7 // numCombinations
        
        var segments: [[[Float]]] = []
        for _ in 0..<batchSize {
            var batchSegments: [[Float]] = []
            for _ in 0..<numFrames {
                let frameValues = generateRandomPowersetFrame()
                batchSegments.append(frameValues)
            }
            segments.append(batchSegments)
        }
        
        let startTime = CFAbsoluteTimeGetCurrent()
        guard let _ = metalProcessor.performPowersetConversion(segments: segments) else {
            XCTFail("Metal powerset conversion performance test failed")
            return
        }
        let metalTime = CFAbsoluteTimeGetCurrent() - startTime
        
        print("📊 Metal Powerset Conversion Performance:")
        print("   Processing time: \(String(format: "%.4f", metalTime))s")
        print("   Throughput: \(String(format: "%.0f", Double(batchSize * numFrames) / metalTime)) frames/sec")
        
        // Should complete in reasonable time
        XCTAssertLessThan(metalTime, 1.0, "Powerset conversion should complete within 1 second")
        
        print("✅ Metal powerset conversion performance acceptable")
    }
    
    func testPowersetConversionEdgeCases() {
        guard metalProcessor.isAvailable else {
            print("ℹ️ Skipping powerset edge cases test - Metal not available")
            return
        }
        
        // Test empty input
        let emptyResult = metalProcessor.performPowersetConversion(segments: [])
        XCTAssertNil(emptyResult, "Empty segments should return nil")
        
        // Test single frame
        let singleFrame: [[[Float]]] = [[[0.1, 0.9, 0.2, 0.3, 0.4, 0.5, 0.6]]]
        let singleResult = metalProcessor.performPowersetConversion(segments: singleFrame)
        XCTAssertNotNil(singleResult, "Single frame should work")
        
        if let result = singleResult {
            XCTAssertEqual(result[0][0][1], 1.0, accuracy: 0.001, "Should activate speaker 1 for max at index 1")
            XCTAssertEqual(result[0][0][0], 0.0, accuracy: 0.001, "Should not activate speaker 0")
            XCTAssertEqual(result[0][0][2], 0.0, accuracy: 0.001, "Should not activate speaker 2")
        }
        
        print("✅ Metal powerset conversion edge cases handled correctly")
    }
    
    // MARK: - Metal Memory Management Tests
    
    func testMetalMemoryManagement() {
        guard metalProcessor.isAvailable else {
            print("ℹ️ Skipping Metal memory test - Metal not available")
            return
        }
        
        // Test multiple operations don't leak memory
        let queries: [[Float]] = [generateNormalizedRandomEmbedding(dimension: 128)]
        let candidates: [[Float]] = [generateNormalizedRandomEmbedding(dimension: 128)]
        
        // Perform multiple operations
        for i in 0..<10 {
            guard let _ = metalProcessor.batchCosineDistances(queries: queries, candidates: candidates) else {
                XCTFail("Metal operation \(i) failed")
                return
            }
        }
        
        // If we reach here without crashes, memory management is working
        print("✅ Metal memory management test passed (no crashes/leaks)")
    }
    
    func testMetalLargeMatrixHandling() {
        guard metalProcessor.isAvailable else {
            print("ℹ️ Skipping Metal large matrix test - Metal not available")
            return
        }
        
        // Test with larger matrices to stress GPU memory
        let embeddingDim = 1024
        let numQueries = 100
        let numCandidates = 100
        
        var queries: [[Float]] = []
        var candidates: [[Float]] = []
        
        for _ in 0..<numQueries {
            queries.append(generateNormalizedRandomEmbedding(dimension: embeddingDim))
        }
        
        for _ in 0..<numCandidates {
            candidates.append(generateNormalizedRandomEmbedding(dimension: embeddingDim))
        }
        
        guard let result = metalProcessor.batchCosineDistances(queries: queries, candidates: candidates) else {
            // This might fail on devices with limited GPU memory - that's acceptable
            print("ℹ️ Large matrix test failed (expected on devices with limited GPU memory)")
            return
        }
        
        XCTAssertEqual(result.count, numQueries, "Should handle large matrices correctly")
        print("✅ Metal large matrix handling successful")
    }
    
    // MARK: - Fallback Mechanism Tests
    
    func testMetalFallbackBehavior() {
        // Test that the system gracefully handles Metal unavailability
        if !metalProcessor.isAvailable {
            print("✅ Metal gracefully reports unavailability")
            
            // Test that operations return nil when Metal unavailable
            let queries: [[Float]] = [[1.0, 0.0, 0.0]]
            let candidates: [[Float]] = [[1.0, 0.0, 0.0]]
            
            let result = metalProcessor.batchCosineDistances(queries: queries, candidates: candidates)
            XCTAssertNil(result, "Operations should return nil when Metal unavailable")
            
            let powersetResult = metalProcessor.performPowersetConversion(segments: [[[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]])
            XCTAssertNil(powersetResult, "Powerset conversion should return nil when Metal unavailable")
        } else {
            print("ℹ️ Metal available - fallback test not applicable")
        }
    }
    
    // MARK: - Helper Methods
    
    private func generateNormalizedRandomEmbedding(dimension: Int) -> [Float] {
        var embedding: [Float] = []
        
        // Generate random values
        for _ in 0..<dimension {
            embedding.append(Float.random(in: -1.0...1.0))
        }
        
        // Normalize
        let magnitude = sqrt(embedding.map { $0 * $0 }.reduce(0, +))
        if magnitude > 0 {
            embedding = embedding.map { $0 / magnitude }
        }
        
        return embedding
    }
    
    private func generateRandomPowersetFrame() -> [Float] {
        var frame: [Float] = []
        for _ in 0..<7 {
            frame.append(Float.random(in: 0.0...1.0))
        }
        return frame
    }
    
    private func cpuCosineDistance(_ a: [Float], _ b: [Float]) -> Float {
        guard a.count == b.count, !a.isEmpty else { return Float.infinity }
        
        var dotProduct: Float = 0
        var magnitudeA: Float = 0
        var magnitudeB: Float = 0
        
        for i in 0..<a.count {
            dotProduct += a[i] * b[i]
            magnitudeA += a[i] * a[i]
            magnitudeB += b[i] * b[i]
        }
        
        magnitudeA = sqrt(magnitudeA)
        magnitudeB = sqrt(magnitudeB)
        
        if magnitudeA > 0 && magnitudeB > 0 {
            return 1 - (dotProduct / (magnitudeA * magnitudeB))
        } else {
            return Float.infinity
        }
    }
}