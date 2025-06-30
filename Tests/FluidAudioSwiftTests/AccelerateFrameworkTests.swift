import XCTest
import Accelerate
@testable import FluidAudioSwift

/// Comprehensive tests for Accelerate framework SIMD vectorization
/// Tests vDSP operations, vectorized cosine distance, RMS calculations, and performance validation
final class AccelerateFrameworkTests: XCTestCase, @unchecked Sendable {

    private let testTimeout: TimeInterval = 30.0

    // MARK: - Vectorized Cosine Distance Tests

    func testVectorizedCosineDistanceAccuracy() {
        let manager = DiarizerManager()

        // Test vectors with known geometric relationships
        let testCases: [(a: [Float], b: [Float], expectedDistance: Float, description: String)] = [
            // Identical vectors
            ([1.0, 0.0, 0.0], [1.0, 0.0, 0.0], 0.0, "identical vectors"),
            ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5], 0.0, "identical non-unit vectors"),

            // Orthogonal vectors
            ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], 1.0, "orthogonal unit vectors"),
            ([1.0, 0.0, 0.0], [0.0, 0.0, 1.0], 1.0, "orthogonal unit vectors (different axes)"),

            // Opposite vectors
            ([1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], 2.0, "opposite vectors"),
            ([1.0, 1.0, 1.0], [-1.0, -1.0, -1.0], 2.0, "opposite non-unit vectors"),

            // 45-degree angle (should be sqrt(2))
            ([1.0, 0.0], [1.0, 1.0], 1.0 - (1.0 / sqrt(2.0)), "45-degree angle"),

            // Parallel vectors with different magnitudes
            ([2.0, 0.0, 0.0], [4.0, 0.0, 0.0], 0.0, "parallel vectors different magnitudes"),
        ]

        for testCase in testCases {
            let vectorizedDistance = manager.cosineDistance(testCase.a, testCase.b)
            let referenceDistance = naiveCosineDistance(testCase.a, testCase.b)

            // Test against expected mathematical result
            XCTAssertEqual(vectorizedDistance, testCase.expectedDistance, accuracy: 0.001,
                         "Vectorized distance for \(testCase.description) should match expected value")

            // Test against reference implementation
            XCTAssertEqual(vectorizedDistance, referenceDistance, accuracy: 0.0001,
                         "Vectorized distance for \(testCase.description) should match reference implementation")
        }

        print("✅ Accelerate vectorized cosine distance accuracy validated")
    }

    func testVectorizedCosineDistancePerformance() {
        let manager = DiarizerManager()

        // Test with various embedding dimensions commonly used in speaker recognition
        let dimensions = [128, 256, 512, 1024]

        for dimension in dimensions {
            let embedding1 = generateRandomEmbedding(dimension: dimension)
            let embedding2 = generateRandomEmbedding(dimension: dimension)

            // Measure vectorized performance
            let vectorizedStartTime = CFAbsoluteTimeGetCurrent()
            for _ in 0..<1000 {
                _ = manager.cosineDistance(embedding1, embedding2)
            }
            let vectorizedTime = CFAbsoluteTimeGetCurrent() - vectorizedStartTime

            // Measure naive performance
            let naiveStartTime = CFAbsoluteTimeGetCurrent()
            for _ in 0..<1000 {
                _ = naiveCosineDistance(embedding1, embedding2)
            }
            let naiveTime = CFAbsoluteTimeGetCurrent() - naiveStartTime

            let speedup = naiveTime / vectorizedTime

            print("📊 Accelerate Performance (dim \(dimension)): \(String(format: "%.2f", speedup))x speedup")
            print("   Vectorized: \(String(format: "%.6f", vectorizedTime))s")
            print("   Naive: \(String(format: "%.6f", naiveTime))s")

            // Vectorized should be significantly faster
            XCTAssertGreaterThan(speedup, 1.5, "Vectorized implementation should be at least 1.5x faster for dimension \(dimension)")
        }

        print("✅ Accelerate vectorized cosine distance performance validated")
    }

    func testVectorizedCosineDistanceEdgeCases() {
        let manager = DiarizerManager()

        // Test zero vectors
        let zeroVector = [0.0, 0.0, 0.0] as [Float]
        let normalVector = [1.0, 0.0, 0.0] as [Float]

        let zeroResult = manager.cosineDistance(zeroVector, normalVector)
        XCTAssertEqual(zeroResult, Float.infinity, "Distance with zero vector should be infinity")

        // Test very small vectors
        let smallVector = [1e-10, 1e-10, 1e-10] as [Float]
        let smallResult = manager.cosineDistance(smallVector, normalVector)
        XCTAssert(smallResult.isFinite, "Distance with small vector should be finite")

        // Test mismatched dimensions
        let shortVector = [1.0, 0.0] as [Float]
        let longVector = [1.0, 0.0, 0.0] as [Float]
        let mismatchResult = manager.cosineDistance(shortVector, longVector)
        XCTAssertEqual(mismatchResult, Float.infinity, "Mismatched dimensions should return infinity")

        // Test empty vectors
        let emptyVector: [Float] = []
        let emptyResult = manager.cosineDistance(emptyVector, normalVector)
        XCTAssertEqual(emptyResult, Float.infinity, "Empty vector should return infinity")

        print("✅ Accelerate vectorized cosine distance edge cases handled correctly")
    }

    // MARK: - vDSP Operation Tests

    func testVDSPDotProductAccuracy() {
        let testVectors: [([Float], [Float], Float)] = [
            ([1.0, 2.0, 3.0], [4.0, 5.0, 6.0], 32.0),  // 1*4 + 2*5 + 3*6 = 32
            ([1.0, -1.0, 1.0], [2.0, 2.0, 2.0], 2.0),   // 1*2 + (-1)*2 + 1*2 = 2
            ([0.5, 0.5], [0.5, 0.5], 0.5),               // 0.5*0.5 + 0.5*0.5 = 0.5
        ]

        for (vec1, vec2, expected) in testVectors {
            var result: Float = 0.0

            vec1.withUnsafeBufferPointer { buf1 in
                vec2.withUnsafeBufferPointer { buf2 in
                    vDSP_dotpr(buf1.baseAddress!, 1, buf2.baseAddress!, 1, &result, vDSP_Length(vec1.count))
                }
            }

            XCTAssertEqual(result, expected, accuracy: 0.0001, "vDSP dot product should match expected value")
        }

        print("✅ vDSP dot product accuracy validated")
    }

    func testVDSPMagnitudeCalculation() {
        let testVectors: [([Float], Float)] = [
            ([3.0, 4.0], 5.0),                    // 3-4-5 triangle
            ([1.0, 1.0, 1.0], sqrt(3.0)),         // Unit cube diagonal
            ([2.0, 0.0, 0.0], 2.0),               // Single axis
            ([1.0, -1.0, 1.0, -1.0], 2.0),        // Mixed signs
        ]

        for (vector, expectedMagnitude) in testVectors {
            var magnitudeSquared: Float = 0.0

            vector.withUnsafeBufferPointer { buffer in
                vDSP_dotpr(buffer.baseAddress!, 1, buffer.baseAddress!, 1, &magnitudeSquared, vDSP_Length(vector.count))
            }

            let magnitude = sqrt(magnitudeSquared)
            XCTAssertEqual(magnitude, expectedMagnitude, accuracy: 0.0001, "vDSP magnitude calculation should be accurate")
        }

        print("✅ vDSP magnitude calculation accuracy validated")
    }

    func testVDSPVectorAddition() {
        let vector1: [Float] = [1.0, 2.0, 3.0, 4.0]
        let vector2: [Float] = [0.5, 1.5, 2.5, 3.5]
        let expected: [Float] = [1.5, 3.5, 5.5, 7.5]

        var result = Array<Float>(repeating: 0.0, count: vector1.count)

        vector1.withUnsafeBufferPointer { buf1 in
            vector2.withUnsafeBufferPointer { buf2 in
                result.withUnsafeMutableBufferPointer { bufResult in
                    vDSP_vadd(buf1.baseAddress!, 1, buf2.baseAddress!, 1, bufResult.baseAddress!, 1, vDSP_Length(vector1.count))
                }
            }
        }

        for i in 0..<expected.count {
            XCTAssertEqual(result[i], expected[i], accuracy: 0.0001, "vDSP vector addition should be accurate")
        }

        print("✅ vDSP vector addition accuracy validated")
    }

    // MARK: - RMS and Audio Processing Tests

    func testVectorizedRMSCalculation() {
        // Test RMS calculation using vDSP
        let testAudioSignals: [([Float], Float)] = [
            // DC signal
            (Array(repeating: 1.0, count: 1000), 1.0),

            // Sine wave (RMS = amplitude / sqrt(2))
            (generateSineWave(frequency: 440.0, sampleRate: 16000, duration: 1.0, amplitude: 1.0), 1.0 / sqrt(2.0)),

            // Mixed frequency signal
            (generateComplexSignal(), calculateExpectedRMS()),
        ]

        for (signal, expectedRMS) in testAudioSignals {
            let vectorizedRMS = calculateVectorizedRMS(signal)
            let naiveRMS = calculateNaiveRMS(signal)

            // Test accuracy against expected value
            if expectedRMS > 0 {
                XCTAssertEqual(vectorizedRMS, expectedRMS, accuracy: 0.01, "Vectorized RMS should match expected value")
            }

            // Test accuracy against naive implementation
            XCTAssertEqual(vectorizedRMS, naiveRMS, accuracy: 0.0001, "Vectorized RMS should match naive implementation")
        }

        print("✅ Vectorized RMS calculation accuracy validated")
    }

    func testVectorizedRMSPerformance() {
        let largeAudioSignal = generateSineWave(frequency: 440.0, sampleRate: 16000, duration: 10.0, amplitude: 0.5)

        // Measure vectorized RMS performance
        let vectorizedStartTime = CFAbsoluteTimeGetCurrent()
        for _ in 0..<100 {
            _ = calculateVectorizedRMS(largeAudioSignal)
        }
        let vectorizedTime = CFAbsoluteTimeGetCurrent() - vectorizedStartTime

        // Measure naive RMS performance
        let naiveStartTime = CFAbsoluteTimeGetCurrent()
        for _ in 0..<100 {
            _ = calculateNaiveRMS(largeAudioSignal)
        }
        let naiveTime = CFAbsoluteTimeGetCurrent() - naiveStartTime

        let speedup = naiveTime / vectorizedTime

        print("📊 RMS Calculation Performance: \(String(format: "%.2f", speedup))x speedup")
        print("   Vectorized: \(String(format: "%.6f", vectorizedTime))s")
        print("   Naive: \(String(format: "%.6f", naiveTime))s")

        XCTAssertGreaterThan(speedup, 2.0, "Vectorized RMS should be at least 2x faster")

        print("✅ Vectorized RMS performance validated")
    }

    func testAudioNormalization() {
        // Test vectorized audio normalization
        let unnormalizedAudio: [Float] = [0.1, 0.5, -0.3, 0.8, -0.2, 0.6]
        let targetRMS: Float = 0.5

        let normalizedAudio = normalizeAudioVectorized(unnormalizedAudio, targetRMS: targetRMS)
        let actualRMS = calculateVectorizedRMS(normalizedAudio)

        XCTAssertEqual(actualRMS, targetRMS, accuracy: 0.01, "Normalized audio should have target RMS")
        XCTAssertEqual(normalizedAudio.count, unnormalizedAudio.count, "Normalized audio should have same length")

        print("✅ Vectorized audio normalization working correctly")
    }

    // MARK: - Large Data Performance Tests

    func testLargeVectorOperations() {
        // Test performance with realistic embedding and audio sizes
        let largeDimension = 2048
        let embedding1 = generateRandomEmbedding(dimension: largeDimension)
        let embedding2 = generateRandomEmbedding(dimension: largeDimension)

        let manager = DiarizerManager()

        let startTime = CFAbsoluteTimeGetCurrent()
        for _ in 0..<100 {
            _ = manager.cosineDistance(embedding1, embedding2)
        }
        let processingTime = CFAbsoluteTimeGetCurrent() - startTime

        print("📊 Large Vector Performance (dim \(largeDimension)):")
        print("   100 operations in \(String(format: "%.4f", processingTime))s")
        print("   \(String(format: "%.0f", 100.0 / processingTime)) operations/second")

        // Should handle large vectors efficiently
        XCTAssertLessThan(processingTime, 1.0, "Large vector operations should complete within 1 second")

        print("✅ Large vector operations performance acceptable")
    }

    func testMultipleSimultaneousOperations() {
        // Test concurrent vector operations for thread safety
        let dimension = 512
        let numOperations = 50

        let manager = DiarizerManager()
        let expectation = self.expectation(description: "Concurrent operations")
        expectation.expectedFulfillmentCount = numOperations

        // Local function to avoid capturing self
        @Sendable func generateRandomEmbedding(dimension: Int) -> [Float] {
            return (0..<dimension).map { _ in Float.random(in: -1.0...1.0) }
        }

        DispatchQueue.concurrentPerform(iterations: numOperations) { i in
            let embedding1 = generateRandomEmbedding(dimension: dimension)
            let embedding2 = generateRandomEmbedding(dimension: dimension)

            let distance = manager.cosineDistance(embedding1, embedding2)

            // Verify result is reasonable
            XCTAssert(distance >= 0.0 && distance <= 2.0, "Distance should be in valid range")

            expectation.fulfill()
        }

        wait(for: [expectation], timeout: testTimeout)

        print("✅ Multiple simultaneous vector operations completed successfully")
    }

    // MARK: - Memory Efficiency Tests

    func testVectorOperationMemoryUsage() {
        // Test that vector operations don't create excessive memory pressure
        let dimension = 1024
        let iterations = 1000

        let manager = DiarizerManager()

        autoreleasepool {
            for _ in 0..<iterations {
                let embedding1 = generateRandomEmbedding(dimension: dimension)
                let embedding2 = generateRandomEmbedding(dimension: dimension)
                _ = manager.cosineDistance(embedding1, embedding2)
            }
        }

        // If we reach here without memory issues, the test passes
        print("✅ Vector operations memory usage test passed")
    }

    // MARK: - Helper Methods

    private func generateRandomEmbedding(dimension: Int) -> [Float] {
        return (0..<dimension).map { _ in Float.random(in: -1.0...1.0) }
    }

    private func naiveCosineDistance(_ a: [Float], _ b: [Float]) -> Float {
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

    private func generateSineWave(frequency: Float, sampleRate: Int, duration: Float, amplitude: Float) -> [Float] {
        let sampleCount = Int(Float(sampleRate) * duration)
        return (0..<sampleCount).map { i in
            amplitude * sin(2.0 * Float.pi * frequency * Float(i) / Float(sampleRate))
        }
    }

    private func generateComplexSignal() -> [Float] {
        // Generate a signal with multiple frequency components
        let sampleRate = 16000
        let duration: Float = 1.0
        let sampleCount = Int(Float(sampleRate) * duration)

        return (0..<sampleCount).map { i in
            let t = Float(i) / Float(sampleRate)
            return 0.5 * sin(2.0 * Float.pi * 440.0 * t) +  // 440 Hz
                   0.3 * sin(2.0 * Float.pi * 880.0 * t) +  // 880 Hz
                   0.2 * sin(2.0 * Float.pi * 1320.0 * t)   // 1320 Hz
        }
    }

    private func calculateExpectedRMS() -> Float {
        // For the complex signal: RMS = sqrt((0.5^2 + 0.3^2 + 0.2^2) / 2)
        return sqrt((0.25 + 0.09 + 0.04) / 2.0)
    }

    private func calculateVectorizedRMS(_ signal: [Float]) -> Float {
        var meanSquare: Float = 0.0

        signal.withUnsafeBufferPointer { buffer in
            vDSP_dotpr(buffer.baseAddress!, 1, buffer.baseAddress!, 1, &meanSquare, vDSP_Length(signal.count))
        }

        meanSquare /= Float(signal.count)
        return sqrt(meanSquare)
    }

    private func calculateNaiveRMS(_ signal: [Float]) -> Float {
        let sumOfSquares = signal.reduce(0) { $0 + $1 * $1 }
        let meanSquare = sumOfSquares / Float(signal.count)
        return sqrt(meanSquare)
    }

    private func normalizeAudioVectorized(_ audio: [Float], targetRMS: Float) -> [Float] {
        let currentRMS = calculateVectorizedRMS(audio)
        guard currentRMS > 0 else { return audio }

        var scaleFactor = targetRMS / currentRMS
        var normalizedAudio = Array<Float>(repeating: 0.0, count: audio.count)

        audio.withUnsafeBufferPointer { audioBuffer in
            normalizedAudio.withUnsafeMutableBufferPointer { resultBuffer in
                vDSP_vsmul(audioBuffer.baseAddress!, 1, &scaleFactor, resultBuffer.baseAddress!, 1, vDSP_Length(audio.count))
            }
        }

        return normalizedAudio
    }
}
