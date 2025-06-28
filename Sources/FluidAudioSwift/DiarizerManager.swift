import Foundation
import OSLog
import CoreML
import Accelerate
import Metal
import MetalPerformanceShaders

public struct DiarizerConfig: Sendable {
    public var clusteringThreshold: Float = 0.7 // Similarity threshold for grouping speakers (0.0-1.0, higher = stricter)
    public var minDurationOn: Float = 1.0 // Minimum duration (seconds) for a speaker segment to be considered valid
    public var minDurationOff: Float = 0.5 // Minimum silence duration (seconds) between different speakers
    public var numClusters: Int = -1  // Number of speakers to detect (-1 = auto-detect)
    public var minActivityThreshold: Float = 10.0 // Minimum activity threshold (frames) for speaker to be considered active
    public var debugMode: Bool = false
    public var modelCacheDirectory: URL?
    
    // Performance optimization settings
    public var parallelProcessingThreshold: Double = 60.0 // Seconds - use parallel processing for longer files
    public var embeddingCacheSize: Int = 100 // Maximum cached embeddings for quick lookup
    public var useEarlyTermination: Bool = true // Stop speaker search when confidence is high enough
    public var earlyTerminationThreshold: Float = 0.3 // Distance threshold for early termination
    
    // Metal Performance Shaders settings
    public var useMetalAcceleration: Bool = true // Enable Metal GPU acceleration when available
    public var metalBatchSize: Int = 32 // Optimal batch size for GPU operations
    public var fallbackToAccelerate: Bool = true // Graceful degradation to Accelerate if Metal fails

    public static let `default` = DiarizerConfig()

    public init(
        clusteringThreshold: Float = 0.7,
        minDurationOn: Float = 1.0,
        minDurationOff: Float = 0.5,
        numClusters: Int = -1,
        minActivityThreshold: Float = 10.0,
        debugMode: Bool = false,
        modelCacheDirectory: URL? = nil,
        parallelProcessingThreshold: Double = 60.0,
        embeddingCacheSize: Int = 100,
        useEarlyTermination: Bool = true,
        earlyTerminationThreshold: Float = 0.3,
        useMetalAcceleration: Bool = true,
        metalBatchSize: Int = 32,
        fallbackToAccelerate: Bool = true
    ) {
        self.clusteringThreshold = clusteringThreshold
        self.minDurationOn = minDurationOn
        self.minDurationOff = minDurationOff
        self.numClusters = numClusters
        self.minActivityThreshold = minActivityThreshold
        self.debugMode = debugMode
        self.modelCacheDirectory = modelCacheDirectory
        self.parallelProcessingThreshold = parallelProcessingThreshold
        self.embeddingCacheSize = embeddingCacheSize
        self.useEarlyTermination = useEarlyTermination
        self.earlyTerminationThreshold = earlyTerminationThreshold
        self.useMetalAcceleration = useMetalAcceleration
        self.metalBatchSize = metalBatchSize
        self.fallbackToAccelerate = fallbackToAccelerate
    }
}

/// Complete diarization result with consistent speaker IDs and embeddings
public struct DiarizationResult: Sendable {
    public let segments: [TimedSpeakerSegment]
    public let speakerDatabase: [String: [Float]]  // Speaker ID → representative embedding

    public init(segments: [TimedSpeakerSegment], speakerDatabase: [String: [Float]]) {
        self.segments = segments
        self.speakerDatabase = speakerDatabase
    }
}

/// Speaker segment with embedding and consistent ID across chunks
public struct TimedSpeakerSegment: Sendable, Identifiable {
    public let id = UUID()
    public let speakerId: String              // "Speaker 1", "Speaker 2", etc.
    public let embedding: [Float]             // Voice characteristics
    public let startTimeSeconds: Float       // When segment starts
    public let endTimeSeconds: Float         // When segment ends
    public let qualityScore: Float           // Embedding quality

    public var durationSeconds: Float {
        endTimeSeconds - startTimeSeconds
    }

    public init(speakerId: String, embedding: [Float], startTimeSeconds: Float, endTimeSeconds: Float, qualityScore: Float) {
        self.speakerId = speakerId
        self.embedding = embedding
        self.startTimeSeconds = startTimeSeconds
        self.endTimeSeconds = endTimeSeconds
        self.qualityScore = qualityScore
    }
}

public struct SpeakerEmbedding: Sendable {
    public let embedding: [Float]
    public let qualityScore: Float
    public let durationSeconds: Float

    public init(embedding: [Float], qualityScore: Float, durationSeconds: Float) {
        self.embedding = embedding
        self.qualityScore = qualityScore
        self.durationSeconds = durationSeconds
    }
}

public struct ModelPaths: Sendable {
    public let segmentationPath: String
    public let embeddingPath: String

    public init(segmentationPath: String, embeddingPath: String) {
        self.segmentationPath = segmentationPath
        self.embeddingPath = embeddingPath
    }
}

/// Audio validation result
public struct AudioValidationResult: Sendable {
    public let isValid: Bool
    public let durationSeconds: Float
    public let issues: [String]

    public init(isValid: Bool, durationSeconds: Float, issues: [String] = []) {
        self.isValid = isValid
        self.durationSeconds = durationSeconds
        self.issues = issues
    }
}

// MARK: - Extensions

extension Array {
    func chunked(into size: Int) -> [[Element]] {
        return stride(from: 0, to: count, by: size).map {
            Array(self[$0..<Swift.min($0 + size, count)])
        }
    }
}

// MARK: - Metal Performance Processor

/// Metal Performance Shaders processor for GPU acceleration
@available(macOS 13.0, iOS 16.0, *)
final class MetalPerformanceProcessor: @unchecked Sendable {
    private let device: MTLDevice?
    private let commandQueue: MTLCommandQueue?
    private let logger = Logger(subsystem: "com.fluidinfluence.diarizer", category: "MetalProcessor")
    
    var isAvailable: Bool {
        return device != nil && commandQueue != nil
    }
    
    init() {
        self.device = MTLCreateSystemDefaultDevice()
        self.commandQueue = device?.makeCommandQueue()
        
        if isAvailable {
            logger.info("Metal Performance Shaders initialized successfully")
        } else {
            logger.info("Metal Performance Shaders not available, will use Accelerate fallback")
        }
    }
    
    /// Batch compute cosine distances between embeddings using Metal
    func batchCosineDistances(queries: [[Float]], candidates: [[Float]]) -> [[Float]]? {
        guard isAvailable,
              let device = self.device,
              let commandQueue = self.commandQueue,
              !queries.isEmpty,
              !candidates.isEmpty else {
            return nil
        }
        
        let numQueries = queries.count
        let numCandidates = candidates.count
        let embeddingDim = queries[0].count
        
        // Ensure all embeddings have the same dimension
        guard queries.allSatisfy({ $0.count == embeddingDim }),
              candidates.allSatisfy({ $0.count == embeddingDim }) else {
            logger.error("Inconsistent embedding dimensions")
            return nil
        }
        
        // Create MPS matrices
        let queryMatrixDescriptor = MPSMatrixDescriptor(
            rows: numQueries,
            columns: embeddingDim,
            rowBytes: embeddingDim * MemoryLayout<Float>.size,
            dataType: .float32
        )
        
        let candidateMatrixDescriptor = MPSMatrixDescriptor(
            rows: embeddingDim,
            columns: numCandidates,
            rowBytes: numCandidates * MemoryLayout<Float>.size,
            dataType: .float32
        )
        
        let resultMatrixDescriptor = MPSMatrixDescriptor(
            rows: numQueries,
            columns: numCandidates,
            rowBytes: numCandidates * MemoryLayout<Float>.size,
            dataType: .float32
        )
        
        // Allocate Metal buffers
        let queryBuffer = device.makeBuffer(length: numQueries * embeddingDim * MemoryLayout<Float>.size, options: .storageModeShared)
        let candidateBuffer = device.makeBuffer(length: embeddingDim * numCandidates * MemoryLayout<Float>.size, options: .storageModeShared)
        let resultBuffer = device.makeBuffer(length: numQueries * numCandidates * MemoryLayout<Float>.size, options: .storageModeShared)
        
        guard let queryBuffer = queryBuffer,
              let candidateBuffer = candidateBuffer,
              let resultBuffer = resultBuffer else {
            logger.error("Failed to allocate Metal buffers")
            return nil
        }
        
        // Copy data to Metal buffers
        let queryPtr = queryBuffer.contents().bindMemory(to: Float.self, capacity: numQueries * embeddingDim)
        let candidatePtr = candidateBuffer.contents().bindMemory(to: Float.self, capacity: embeddingDim * numCandidates)
        
        // Copy queries (row-major)
        for (i, query) in queries.enumerated() {
            for (j, value) in query.enumerated() {
                queryPtr[i * embeddingDim + j] = value
            }
        }
        
        // Copy candidates (column-major for matrix multiplication)
        for (j, candidate) in candidates.enumerated() {
            for (i, value) in candidate.enumerated() {
                candidatePtr[i * numCandidates + j] = value
            }
        }
        
        // Create MPS matrices
        let queryMatrix = MPSMatrix(buffer: queryBuffer, descriptor: queryMatrixDescriptor)
        let candidateMatrix = MPSMatrix(buffer: candidateBuffer, descriptor: candidateMatrixDescriptor)
        let resultMatrix = MPSMatrix(buffer: resultBuffer, descriptor: resultMatrixDescriptor)
        
        // Perform matrix multiplication (dot products)
        let matrixMultiplication = MPSMatrixMultiplication(
            device: device,
            transposeLeft: false,
            transposeRight: false,
            resultRows: numQueries,
            resultColumns: numCandidates,
            interiorColumns: embeddingDim,
            alpha: 1.0,
            beta: 0.0
        )
        
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            logger.error("Failed to create Metal command buffer")
            return nil
        }
        
        matrixMultiplication.encode(
            commandBuffer: commandBuffer,
            leftMatrix: queryMatrix,
            rightMatrix: candidateMatrix,
            resultMatrix: resultMatrix
        )
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        // Extract results and convert to cosine distances
        let resultPtr = resultBuffer.contents().bindMemory(to: Float.self, capacity: numQueries * numCandidates)
        var distances: [[Float]] = Array(repeating: Array(repeating: 0.0, count: numCandidates), count: numQueries)
        
        // Calculate magnitudes for normalization
        var queryMagnitudes: [Float] = []
        var candidateMagnitudes: [Float] = []
        
        for query in queries {
            let magnitude = sqrt(query.map { $0 * $0 }.reduce(0, +))
            queryMagnitudes.append(magnitude)
        }
        
        for candidate in candidates {
            let magnitude = sqrt(candidate.map { $0 * $0 }.reduce(0, +))
            candidateMagnitudes.append(magnitude)
        }
        
        // Convert dot products to cosine distances
        for i in 0..<numQueries {
            for j in 0..<numCandidates {
                let dotProduct = resultPtr[i * numCandidates + j]
                let magnitude1 = queryMagnitudes[i]
                let magnitude2 = candidateMagnitudes[j]
                
                if magnitude1 > 0 && magnitude2 > 0 {
                    let similarity = dotProduct / (magnitude1 * magnitude2)
                    distances[i][j] = 1 - similarity
                } else {
                    distances[i][j] = Float.infinity
                }
            }
        }
        
        return distances
    }
    
    /// Accelerated powerset conversion using Metal compute shader
    func performPowersetConversion(segments: [[[Float]]]) -> [[[Float]]]? {
        guard isAvailable,
              let device = self.device,
              let commandQueue = self.commandQueue,
              !segments.isEmpty else {
            return nil
        }
        
        let batchSize = segments.count
        let numFrames = segments[0].count
        let numCombinations = segments[0][0].count
        let numSpeakers = 3
        
        // Metal shader source for powerset conversion
        let shaderSource = """
        #include <metal_stdlib>
        using namespace metal;
        
        kernel void powerset_conversion(
            device const float* segments [[buffer(0)]],
            device float* binarized [[buffer(1)]],
            constant uint& num_frames [[buffer(2)]],
            constant uint& num_combinations [[buffer(3)]],
            uint2 index [[thread_position_in_grid]]
        ) {
            const int powerset[7][3] = {
                {-1, -1, -1}, // 0: empty set
                {0, -1, -1},  // 1: {0}
                {1, -1, -1},  // 2: {1}
                {2, -1, -1},  // 3: {2}
                {0, 1, -1},   // 4: {0, 1}
                {0, 2, -1},   // 5: {0, 2}
                {1, 2, -1}    // 6: {1, 2}
            };
            
            uint b = index.x; // batch
            uint f = index.y; // frame
            
            if (b >= 1 || f >= num_frames) return;
            
            // Find max value index in this frame
            float max_val = -1.0;
            uint best_idx = 0;
            
            for (uint c = 0; c < num_combinations; c++) {
                float val = segments[b * num_frames * num_combinations + f * num_combinations + c];
                if (val > max_val) {
                    max_val = val;
                    best_idx = c;
                }
            }
            
            // Clear output for this frame
            for (uint s = 0; s < 3; s++) {
                binarized[b * num_frames * 3 + f * 3 + s] = 0.0;
            }
            
            // Set active speakers based on powerset
            for (uint i = 0; i < 3; i++) {
                int speaker = powerset[best_idx][i];
                if (speaker >= 0) {
                    binarized[b * num_frames * 3 + f * 3 + speaker] = 1.0;
                }
            }
        }
        """
        
        // Create Metal library and function
        guard let library = try? device.makeLibrary(source: shaderSource, options: nil),
              let function = library.makeFunction(name: "powerset_conversion") else {
            logger.error("Failed to create Metal compute function")
            return nil
        }
        
        guard let computePipelineState = try? device.makeComputePipelineState(function: function) else {
            logger.error("Failed to create Metal compute pipeline state")
            return nil
        }
        
        // Allocate Metal buffers
        let inputSize = batchSize * numFrames * numCombinations * MemoryLayout<Float>.size
        let outputSize = batchSize * numFrames * numSpeakers * MemoryLayout<Float>.size
        
        guard let inputBuffer = device.makeBuffer(length: inputSize, options: .storageModeShared),
              let outputBuffer = device.makeBuffer(length: outputSize, options: .storageModeShared) else {
            logger.error("Failed to allocate Metal buffers for powerset conversion")
            return nil
        }
        
        // Copy input data
        let inputPtr = inputBuffer.contents().bindMemory(to: Float.self, capacity: batchSize * numFrames * numCombinations)
        for b in 0..<batchSize {
            for f in 0..<numFrames {
                for c in 0..<numCombinations {
                    inputPtr[b * numFrames * numCombinations + f * numCombinations + c] = segments[b][f][c]
                }
            }
        }
        
        // Execute compute shader
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
            logger.error("Failed to create Metal command buffer or encoder")
            return nil
        }
        
        computeEncoder.setComputePipelineState(computePipelineState)
        computeEncoder.setBuffer(inputBuffer, offset: 0, index: 0)
        computeEncoder.setBuffer(outputBuffer, offset: 0, index: 1)
        
        var numFramesConstant = UInt32(numFrames)
        var numCombinationsConstant = UInt32(numCombinations)
        computeEncoder.setBytes(&numFramesConstant, length: MemoryLayout<UInt32>.size, index: 2)
        computeEncoder.setBytes(&numCombinationsConstant, length: MemoryLayout<UInt32>.size, index: 3)
        
        let threadGroupSize = MTLSize(width: 1, height: min(numFrames, computePipelineState.maxTotalThreadsPerThreadgroup), depth: 1)
        let threadGroups = MTLSize(width: batchSize, height: (numFrames + threadGroupSize.height - 1) / threadGroupSize.height, depth: 1)
        
        computeEncoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
        computeEncoder.endEncoding()
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        // Extract results
        let outputPtr = outputBuffer.contents().bindMemory(to: Float.self, capacity: batchSize * numFrames * numSpeakers)
        var result: [[[Float]]] = Array(repeating: Array(repeating: Array(repeating: 0.0, count: numSpeakers), count: numFrames), count: batchSize)
        
        for b in 0..<batchSize {
            for f in 0..<numFrames {
                for s in 0..<numSpeakers {
                    result[b][f][s] = outputPtr[b * numFrames * numSpeakers + f * numSpeakers + s]
                }
            }
        }
        
        return result
    }
}

// MARK: - Error Types

public enum DiarizerError: Error, LocalizedError {
    case notInitialized
    case modelDownloadFailed
    case embeddingExtractionFailed
    case invalidAudioData
    case processingFailed(String)

    public var errorDescription: String? {
        switch self {
        case .notInitialized:
            return "Diarization system not initialized. Call initialize() first."
        case .modelDownloadFailed:
            return "Failed to download required models."
        case .embeddingExtractionFailed:
            return "Failed to extract speaker embedding from audio."
        case .invalidAudioData:
            return "Invalid audio data provided."
        case .processingFailed(let message):
            return "Processing failed: \(message)"
        }
    }
}

private struct Segment: Hashable {
    let start: Double
    let end: Double
}

private struct SlidingWindow {
    var start: Double
    var duration: Double
    var step: Double

    func time(forFrame index: Int) -> Double {
        return start + Double(index) * step
    }

    func segment(forFrame index: Int) -> Segment {
        let s = time(forFrame: index)
        return Segment(start: s, end: s + duration)
    }
}

private struct SlidingWindowFeature {
    var data: [[[Float]]] // (1, 589, 3)
    var slidingWindow: SlidingWindow
}

// MARK: - Diarizer Implementation

/// Speaker diarization manager
@available(macOS 13.0, iOS 16.0, *)
public final class DiarizerManager: @unchecked Sendable {

    private let logger = Logger(subsystem: "com.fluidinfluence.diarizer", category: "Diarizer")
    private let config: DiarizerConfig

    // ML models
    private var segmentationModel: MLModel?
    private var embeddingModel: MLModel?
    
    // Metal performance processor
    private lazy var metalProcessor: MetalPerformanceProcessor? = {
        guard config.useMetalAcceleration else { return nil }
        return MetalPerformanceProcessor()
    }()

    public init(config: DiarizerConfig = .default) {
        self.config = config
    }

    public var isAvailable: Bool {
        return segmentationModel != nil && embeddingModel != nil
    }

    public func initialize() async throws {
        logger.info("Initializing diarization system")

        try await cleanupBrokenModels()

        let modelPaths = try await downloadModels()

        let segmentationURL = URL(fileURLWithPath: modelPaths.segmentationPath)
        let embeddingURL = URL(fileURLWithPath: modelPaths.embeddingPath)

        self.segmentationModel = try MLModel(contentsOf: segmentationURL)
        self.embeddingModel = try MLModel(contentsOf: embeddingURL)

        logger.info("Diarization system initialized successfully")
    }

    private func cleanupBrokenModels() async throws {
        let modelsDirectory = getModelsDirectory()
        let segmentationModelPath = modelsDirectory.appendingPathComponent("pyannote_segmentation.mlmodelc")
        let embeddingModelPath = modelsDirectory.appendingPathComponent("wespeaker.mlmodelc")

        if FileManager.default.fileExists(atPath: segmentationModelPath.path) &&
           !isModelCompiled(at: segmentationModelPath) {
            logger.info("Removing broken segmentation model")
            try FileManager.default.removeItem(at: segmentationModelPath)
        }

        if FileManager.default.fileExists(atPath: embeddingModelPath.path) &&
           !isModelCompiled(at: embeddingModelPath) {
            logger.info("Removing broken embedding model")
            try FileManager.default.removeItem(at: embeddingModelPath)
        }
    }

    private func getSegments(audioChunk: [Float], chunkSize: Int = 160_000) throws -> [[[Float]]] {
        guard let segmentationModel = self.segmentationModel else {
            throw DiarizerError.notInitialized
        }

        let audioArray = try MLMultiArray(shape: [1, 1, NSNumber(value: chunkSize)], dataType: .float32)
        for i in 0..<min(audioChunk.count, chunkSize) {
            audioArray[i] = NSNumber(value: audioChunk[i])
        }

        let input = try MLDictionaryFeatureProvider(dictionary: ["audio": audioArray])

        let output = try segmentationModel.prediction(from: input)

        guard let segmentOutput = output.featureValue(for: "segments")?.multiArrayValue else {
            throw DiarizerError.processingFailed("Missing segments output from segmentation model")
        }

        let frames = segmentOutput.shape[1].intValue
        let combinations = segmentOutput.shape[2].intValue

        var segments = Array(repeating: Array(repeating: Array(repeating: 0.0 as Float, count: combinations), count: frames), count: 1)

        for f in 0..<frames {
            for c in 0..<combinations {
                let index = f * combinations + c
                segments[0][f][c] = segmentOutput[index].floatValue
            }
        }

        return powersetConversion(segments)
    }

    private func powersetConversion(_ segments: [[[Float]]]) -> [[[Float]]] {
        // Try Metal acceleration first
        if let metalProcessor = self.metalProcessor,
           metalProcessor.isAvailable,
           let metalResult = metalProcessor.performPowersetConversion(segments: segments) {
            if config.debugMode {
                logger.debug("Used Metal for powerset conversion")
            }
            return metalResult
        }
        
        // Fallback to CPU implementation
        return powersetConversionCPU(segments)
    }
    
    private func powersetConversionCPU(_ segments: [[[Float]]]) -> [[[Float]]] {
        let powerset: [[Int]] = [
            [], // 0
            [0], // 1
            [1], // 2
            [2], // 3
            [0, 1], // 4
            [0, 2], // 5
            [1, 2], // 6
        ]

        let batchSize = segments.count
        let numFrames = segments[0].count
        let numSpeakers = 3

        // Pre-allocate with more efficient ContiguousArray for better cache performance
        var binarized: [[[Float]]] = []
        binarized.reserveCapacity(batchSize)
        
        for _ in 0..<batchSize {
            var batchFrames: [[Float]] = []
            batchFrames.reserveCapacity(numFrames)
            
            for _ in 0..<numFrames {
                batchFrames.append(Array(repeating: 0.0 as Float, count: numSpeakers))
            }
            binarized.append(batchFrames)
        }

        for b in 0..<batchSize {
            for f in 0..<numFrames {
                let frame = segments[b][f]

                // Find index of max value in this frame
                guard let bestIdx = frame.indices.max(by: { frame[$0] < frame[$1] }) else {
                    continue
                }

                // Mark the corresponding speakers as active
                for speaker in powerset[bestIdx] {
                    binarized[b][f][speaker] = 1.0
                }
            }
        }

        return binarized
    }

    private func createSlidingWindowFeature(binarizedSegments: [[[Float]]], chunkOffset: Double = 0.0) -> SlidingWindowFeature {
        let slidingWindow = SlidingWindow(
            start: chunkOffset,
            duration: 0.0619375,
            step: 0.016875
        )

        return SlidingWindowFeature(
            data: binarizedSegments,
            slidingWindow: slidingWindow
        )
    }

    private func getEmbedding(
        audioChunk: [Float],
        binarizedSegments: [[[Float]]],
        slidingWindowFeature: SlidingWindowFeature,
        embeddingModel: MLModel,
        sampleRate: Int = 16000
    ) throws -> [[Float]] {
        let chunkSize = 10 * sampleRate
        let audioTensor = audioChunk
        let numFrames = slidingWindowFeature.data[0].count
        let numSpeakers = slidingWindowFeature.data[0][0].count

        // Pre-allocate and compute clean_frames efficiently
        var cleanFrames = ContiguousArray<Float>()
        cleanFrames.reserveCapacity(numFrames)
        
        let segmentData = slidingWindowFeature.data[0]
        for f in 0..<numFrames {
            let frame = segmentData[f]
            let speakerSum = frame.reduce(0, +)
            cleanFrames.append((speakerSum < 2.0) ? 1.0 : 0.0)
        }

        // Pre-allocate cleanSegmentData more efficiently
        var cleanSegmentData: [[[Float]]] = []
        cleanSegmentData.reserveCapacity(1)
        
        var batchData: [[Float]] = []
        batchData.reserveCapacity(numFrames)
        
        for f in 0..<numFrames {
            var frameData = ContiguousArray<Float>()
            frameData.reserveCapacity(numSpeakers)
            
            let cleanMask = cleanFrames[f]
            for s in 0..<numSpeakers {
                frameData.append(segmentData[f][s] * cleanMask)
            }
            batchData.append(Array(frameData))
        }
        cleanSegmentData.append(batchData)


        // Use efficient ArraySlice references instead of duplicating audio data
        let audioSlice = ArraySlice(audioTensor)

        // Transpose mask shape to (numSpeakers, 589)
        var cleanMasks: [[Float]] = Array(repeating: Array(repeating: 0.0, count: numFrames), count: numSpeakers)

        for s in 0..<numSpeakers {
            for f in 0..<numFrames {
                cleanMasks[s][f] = cleanSegmentData[0][f][s]
            }
        }

        // Prepare MLMultiArray inputs
        guard let waveformArray = try? MLMultiArray(shape: [numSpeakers, chunkSize] as [NSNumber], dataType: .float32),
              let maskArray = try? MLMultiArray(shape: [numSpeakers, numFrames] as [NSNumber], dataType: .float32) else {
            throw DiarizerError.processingFailed("Failed to allocate MLMultiArray for embeddings")
        }

        // Optimize MLMultiArray population using safe bulk operations
        audioSlice.withUnsafeBufferPointer { audioBuffer in
            for s in 0..<numSpeakers {
                let speakerOffset = s * chunkSize
                for i in 0..<min(chunkSize, audioBuffer.count) {
                    waveformArray[speakerOffset + i] = NSNumber(value: audioBuffer[i])
                }
            }
        }

        // Bulk populate mask array efficiently
        for s in 0..<numSpeakers {
            let speakerMaskOffset = s * numFrames
            let speakerMask = cleanMasks[s]
            speakerMask.withUnsafeBufferPointer { maskBuffer in
                for f in 0..<numFrames {
                    maskArray[speakerMaskOffset + f] = NSNumber(value: maskBuffer[f])
                }
            }
        }

        // Run model
        let inputs: [String: Any] = [
            "waveform": waveformArray,
            "mask": maskArray,
        ]

        guard let output = try? embeddingModel.prediction(from: MLDictionaryFeatureProvider(dictionary: inputs)),
              let multiArray = output.featureValue(for: "embedding")?.multiArrayValue else {
            throw DiarizerError.processingFailed("Embedding model prediction failed")
        }

        return convertToSendableArray(multiArray)
    }

    private func convertToSendableArray(_ multiArray: MLMultiArray) -> [[Float]] {
        let shape = multiArray.shape.map { $0.intValue }
        let numRows = shape[0]
        let numCols = shape[1]
        let strides = multiArray.strides.map { $0.intValue }

        var result: [[Float]] = Array(repeating: Array(repeating: 0.0, count: numCols), count: numRows)

        for i in 0..<numRows {
            for j in 0..<numCols {
                let index = i * strides[0] + j * strides[1]
                result[i][j] = multiArray[index].floatValue
            }
        }

        return result
    }

    private func getAnnotation(
        annotation: inout [Segment: Int],
        binarizedSegments: [[[Float]]],
        slidingWindow: SlidingWindow
    ) {
        let segmentation = binarizedSegments[0] // shape: [589][3]
        let numFrames = segmentation.count

        // Step 1: argmax to get dominant speaker per frame
        var frameSpeakers: [Int] = []
        for frame in segmentation {
            if let maxIdx = frame.indices.max(by: { frame[$0] < frame[$1] }) {
                frameSpeakers.append(maxIdx)
            } else {
                frameSpeakers.append(0) // fallback
            }
        }

        // Step 2: group contiguous same-speaker segments
        var currentSpeaker = frameSpeakers[0]
        var startFrame = 0

        for i in 1..<numFrames {
            if frameSpeakers[i] != currentSpeaker {
                let startTime = slidingWindow.time(forFrame: startFrame)
                let endTime = slidingWindow.time(forFrame: i)

                let segment = Segment(start: startTime, end: endTime)
                annotation[segment] = currentSpeaker // Use raw speaker index
                currentSpeaker = frameSpeakers[i]
                startFrame = i
            }
        }

        // Final segment
        let finalStart = slidingWindow.time(forFrame: startFrame)
        let finalEnd = slidingWindow.segment(forFrame: numFrames - 1).end
        let finalSegment = Segment(start: finalStart, end: finalEnd)
        annotation[finalSegment] = currentSpeaker // Use raw speaker index
    }

    // MARK: - Model Management

    /// Download required models for diarization
    public func downloadModels() async throws -> ModelPaths {
        logger.info("Downloading diarization models from Hugging Face")

        let modelsDirectory = getModelsDirectory()

        let segmentationModelPath = modelsDirectory.appendingPathComponent("pyannote_segmentation.mlmodelc").path
        let embeddingModelPath = modelsDirectory.appendingPathComponent("wespeaker.mlmodelc").path

        // Force redownload - remove existing models first
        try? FileManager.default.removeItem(at: URL(fileURLWithPath: segmentationModelPath))
        try? FileManager.default.removeItem(at: URL(fileURLWithPath: embeddingModelPath))
        logger.info("Removed existing models to force fresh download")

        // Download segmentation model bundle from Hugging Face
        try await downloadMLModelCBundle(
            repoPath: "bweng/speaker-diarization-coreml",
            modelName: "pyannote_segmentation.mlmodelc",
            outputPath: URL(fileURLWithPath: segmentationModelPath)
        )
        logger.info("Downloaded segmentation model bundle from Hugging Face")

        // Download embedding model bundle from Hugging Face
        try await downloadMLModelCBundle(
            repoPath: "bweng/speaker-diarization-coreml",
            modelName: "wespeaker.mlmodelc",
            outputPath: URL(fileURLWithPath: embeddingModelPath)
        )
        logger.info("Downloaded embedding model bundle from Hugging Face")

        logger.info("Successfully downloaded and compiled diarization models from Hugging Face")
        return ModelPaths(segmentationPath: segmentationModelPath, embeddingPath: embeddingModelPath)
    }

    /// Check if a model is properly compiled
    private func isModelCompiled(at url: URL) -> Bool {
        let coreMLDataPath = url.appendingPathComponent("coremldata.bin")
        return FileManager.default.fileExists(atPath: coreMLDataPath.path)
    }

    /// Download a complete .mlmodelc bundle from Hugging Face
    private func downloadMLModelCBundle(repoPath: String, modelName: String, outputPath: URL) async throws {
        logger.info("Downloading \(modelName) bundle from Hugging Face")

        // Create output directory
        try FileManager.default.createDirectory(at: outputPath, withIntermediateDirectories: true)

        // Files typically found in a .mlmodelc bundle
        let bundleFiles = [
            "model.mil",
            "coremldata.bin",
            "metadata.json"
        ]

        // Weight files that are referenced by model.mil
        let weightFiles = [
            "weights/weight.bin"
        ]

        // Download each file in the bundle
        for fileName in bundleFiles {
            let fileURL = URL(string: "https://huggingface.co/\(repoPath)/resolve/main/\(modelName)/\(fileName)")!

            do {
                let (tempFile, response) = try await URLSession.shared.download(from: fileURL)

                // Check if download was successful
                if let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode == 200 {
                    let destinationPath = outputPath.appendingPathComponent(fileName)

                    // Remove existing file if it exists
                    try? FileManager.default.removeItem(at: destinationPath)

                    // Move downloaded file to destination
                    try FileManager.default.moveItem(at: tempFile, to: destinationPath)
                    logger.info("Downloaded \(fileName) for \(modelName)")
                } else {
                    logger.warning("Failed to download \(fileName) for \(modelName) - file may not exist")
                    // Create empty file if it doesn't exist (some files are optional)
                    if fileName == "metadata.json" {
                        let destinationPath = outputPath.appendingPathComponent(fileName)
                        try "{}".write(to: destinationPath, atomically: true, encoding: .utf8)
                    }
                }
            } catch {
                logger.warning("Error downloading \(fileName): \(error.localizedDescription)")
                // For critical files, create minimal versions
                if fileName == "coremldata.bin" {
                    let destinationPath = outputPath.appendingPathComponent(fileName)
                    try Data().write(to: destinationPath)
                } else if fileName == "metadata.json" {
                    let destinationPath = outputPath.appendingPathComponent(fileName)
                    try "{}".write(to: destinationPath, atomically: true, encoding: .utf8)
                }
            }
        }

        // Download weight files
        for weightFile in weightFiles {
            let fileURL = URL(string: "https://huggingface.co/\(repoPath)/resolve/main/\(modelName)/\(weightFile)")!

            do {
                let (tempFile, response) = try await URLSession.shared.download(from: fileURL)

                // Check if download was successful
                if let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode == 200 {
                    let destinationPath = outputPath.appendingPathComponent(weightFile)

                    // Create weights directory if it doesn't exist
                    let weightsDir = destinationPath.deletingLastPathComponent()
                    try FileManager.default.createDirectory(at: weightsDir, withIntermediateDirectories: true)

                    // Remove existing file if it exists
                    try? FileManager.default.removeItem(at: destinationPath)

                    // Move downloaded file to destination
                    try FileManager.default.moveItem(at: tempFile, to: destinationPath)
                    logger.info("Downloaded \(weightFile) for \(modelName)")
                } else {
                    logger.warning("Failed to download \(weightFile) for \(modelName)")
                    throw DiarizerError.modelDownloadFailed
                }
            } catch {
                logger.error("Critical error downloading \(weightFile): \(error.localizedDescription)")
                throw DiarizerError.modelDownloadFailed
            }
        }

        // Also try to download analytics directory if it exists
        let analyticsURL = URL(string: "https://huggingface.co/\(repoPath)/resolve/main/\(modelName)/analytics/coremldata.bin")!
        do {
            let (tempFile, response) = try await URLSession.shared.download(from: analyticsURL)
            if let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode == 200 {
                let analyticsDir = outputPath.appendingPathComponent("analytics")
                try FileManager.default.createDirectory(at: analyticsDir, withIntermediateDirectories: true)
                let destinationPath = analyticsDir.appendingPathComponent("coremldata.bin")
                try? FileManager.default.removeItem(at: destinationPath)
                try FileManager.default.moveItem(at: tempFile, to: destinationPath)
                logger.info("Downloaded analytics/coremldata.bin for \(modelName)")
            }
        } catch {
            logger.info("Analytics directory not found or not needed for \(modelName)")
        }

        logger.info("Completed downloading \(modelName) bundle")
    }

    /// Compile a model
    private func compileModel(at sourceURL: URL, outputPath: URL) async throws -> URL {
        logger.info("Compiling model from \(sourceURL.lastPathComponent)")

        // Remove existing compiled model if it exists
        if FileManager.default.fileExists(atPath: outputPath.path) {
            try FileManager.default.removeItem(at: outputPath)
        }

        // Compile the model
        let compiledModelURL = try await MLModel.compileModel(at: sourceURL)

        // Move to the desired location
        try FileManager.default.moveItem(at: compiledModelURL, to: outputPath)

        // Clean up the source file
        try? FileManager.default.removeItem(at: sourceURL)

        logger.info("Successfully compiled model to \(outputPath.lastPathComponent)")
        return outputPath
    }

    private func getModelsDirectory() -> URL {
        let directory: URL

        if let customDirectory = config.modelCacheDirectory {
            directory = customDirectory.appendingPathComponent("coreml", isDirectory: true)
        } else {
            let appSupport = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
            directory = appSupport.appendingPathComponent("SpeakerKitModels/coreml", isDirectory: true)
        }

        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return directory.standardizedFileURL
    }

    // MARK: - Audio Analysis

    /// Compare similarity between two audio samples using efficient diarization
    public func compareSpeakers(audio1: [Float], audio2: [Float]) async throws -> Float {
        // Use the efficient method to get embeddings
        let result1 = try await performCompleteDiarization(audio1)
        let result2 = try await performCompleteDiarization(audio2)

        // Get the most representative embedding from each audio
        guard let segment1 = result1.segments.max(by: { $0.qualityScore < $1.qualityScore }),
              let segment2 = result2.segments.max(by: { $0.qualityScore < $1.qualityScore }) else {
            throw DiarizerError.embeddingExtractionFailed
        }

        let distance = cosineDistance(segment1.embedding, segment2.embedding)
        return max(0, (1.0 - distance) * 100) // Convert to similarity percentage
    }

    /// Validate if an embedding is valid
    public func validateEmbedding(_ embedding: [Float]) -> Bool {
        guard !embedding.isEmpty else { return false }

        // Check for NaN or infinite values
        guard embedding.allSatisfy({ $0.isFinite }) else { return false }

        // Check magnitude
        let magnitude = sqrt(embedding.map { $0 * $0 }.reduce(0, +))
        guard magnitude > 0.1 else { return false }

        return true
    }

    /// Validate audio quality and characteristics
    public func validateAudio(_ samples: [Float]) -> AudioValidationResult {
        let duration = Float(samples.count) / 16000.0
        var issues: [String] = []

        if duration < 1.0 {
            issues.append("Audio too short (minimum 1 second)")
        }

        if samples.isEmpty {
            issues.append("No audio data")
        }

        // Check for silence
        let rmsEnergy = calculateRMSEnergy(samples)
        if rmsEnergy < 0.01 {
            issues.append("Audio too quiet or silent")
        }

        return AudioValidationResult(
            isValid: issues.isEmpty,
            durationSeconds: duration,
            issues: issues
        )
    }

    // MARK: - Utility Functions

    /// Batch assign speakers using Metal acceleration when available
    private func batchAssignSpeakers(embeddings: [[Float]], speakerDB: inout [String: [Float]]) -> [String] {
        guard embeddings.count > 1,
              !speakerDB.isEmpty,
              let metalProcessor = self.metalProcessor,
              metalProcessor.isAvailable else {
            // Fallback to individual assignment
            return embeddings.map { assignSpeaker(embedding: $0, speakerDB: &speakerDB) }
        }
        
        let candidateEmbeddings = Array(speakerDB.values)
        let candidateIds = Array(speakerDB.keys)
        
        // Use Metal for batch distance computation
        if let distanceMatrix = metalProcessor.batchCosineDistances(queries: embeddings, candidates: candidateEmbeddings) {
            var assignments: [String] = []
            
            for (embeddingIndex, embedding) in embeddings.enumerated() {
                let distances = distanceMatrix[embeddingIndex]
                let minDistanceIndex = distances.indices.min(by: { distances[$0] < distances[$1] }) ?? 0
                let minDistance = distances[minDistanceIndex]
                let bestSpeakerId = candidateIds[minDistanceIndex]
                
                if minDistance > config.clusteringThreshold {
                    // New speaker
                    let newSpeakerId = "Speaker \(speakerDB.count + 1)"
                    speakerDB[newSpeakerId] = embedding
                    assignments.append(newSpeakerId)
                    logger.info("Metal: Created new speaker: \(newSpeakerId)")
                } else {
                    // Existing speaker - update embedding
                    updateSpeakerEmbedding(bestSpeakerId, embedding, speakerDB: &speakerDB)
                    assignments.append(bestSpeakerId)
                    if config.debugMode {
                        logger.debug("Metal: Matched existing speaker: \(bestSpeakerId)")
                    }
                }
            }
            
            return assignments
        }
        
        // Fallback to Accelerate if Metal fails
        logger.info("Metal batch processing failed, falling back to individual assignment")
        return embeddings.map { assignSpeaker(embedding: $0, speakerDB: &speakerDB) }
    }

    /// Calculate cosine distance between two embeddings using vectorized operations
    public func cosineDistance(_ a: [Float], _ b: [Float]) -> Float {
        guard a.count == b.count, !a.isEmpty else {
            logger.error("Invalid embeddings for distance calculation")
            return Float.infinity
        }

        // Use Accelerate framework for vectorized operations
        return a.withUnsafeBufferPointer { aBuffer in
            b.withUnsafeBufferPointer { bBuffer in
                let count = vDSP_Length(a.count)
                
                // Calculate dot product using vDSP
                var dotProduct: Float = 0
                vDSP_dotpr(aBuffer.baseAddress!, 1, bBuffer.baseAddress!, 1, &dotProduct, count)
                
                // Calculate squared magnitudes using vDSP
                var magnitudeSquaredA: Float = 0
                var magnitudeSquaredB: Float = 0
                vDSP_svesq(aBuffer.baseAddress!, 1, &magnitudeSquaredA, count)
                vDSP_svesq(bBuffer.baseAddress!, 1, &magnitudeSquaredB, count)
                
                let magnitudeA = sqrt(magnitudeSquaredA)
                let magnitudeB = sqrt(magnitudeSquaredB)
                
                guard magnitudeA > 0 && magnitudeB > 0 else {
                    logger.info("Zero magnitude embedding detected")
                    return Float.infinity
                }
                
                let similarity = dotProduct / (magnitudeA * magnitudeB)
                return 1 - similarity
            }
        }
    }

    private func calculateRMSEnergy(_ samples: [Float]) -> Float {
        guard !samples.isEmpty else { return 0 }
        
        // Use Accelerate framework for efficient RMS calculation
        return samples.withUnsafeBufferPointer { buffer in
            var sum: Float = 0
            let count = vDSP_Length(samples.count)
            vDSP_svesq(buffer.baseAddress!, 1, &sum, count)
            return sqrt(sum / Float(samples.count))
        }
    }

    private func calculateEmbeddingQuality(_ embedding: [Float]) -> Float {
        // Use Accelerate framework for efficient magnitude calculation
        let magnitude = embedding.withUnsafeBufferPointer { buffer in
            var sum: Float = 0
            let count = vDSP_Length(embedding.count)
            vDSP_svesq(buffer.baseAddress!, 1, &sum, count)
            return sqrt(sum)
        }
        // Simple quality score based on magnitude
        return min(1.0, magnitude / 10.0)
    }

    /// Select the embedding for the most active speaker based on speaker activity
    private func selectMostActiveSpeaker(
        embeddings: [[Float]],
        binarizedSegments: [[[Float]]]
    ) -> (embedding: [Float], activity: Float) {
        guard !embeddings.isEmpty, !binarizedSegments.isEmpty else {
            return ([], 0.0)
        }

        let numSpeakers = min(embeddings.count, binarizedSegments[0][0].count)
        var speakerActivities: [Float] = []

        // Calculate total activity for each speaker
        for speakerIndex in 0..<numSpeakers {
            var totalActivity: Float = 0.0
            let numFrames = binarizedSegments[0].count

            for frameIndex in 0..<numFrames {
                totalActivity += binarizedSegments[0][frameIndex][speakerIndex]
            }

            speakerActivities.append(totalActivity)
        }

        // Find the most active speaker
        guard let maxActivityIndex = speakerActivities.indices.max(by: { speakerActivities[$0] < speakerActivities[$1] }) else {
            return (embeddings[0], 0.0)
        }

        let maxActivity = speakerActivities[maxActivityIndex]
        let normalizedActivity = maxActivity / Float(binarizedSegments[0].count)

        return (embeddings[maxActivityIndex], normalizedActivity)
    }

    // MARK: - Cleanup

    // MARK: - Combined Efficient Diarization

    /// Perform complete diarization with consistent speaker IDs across chunks
    /// This is more efficient than calling performSegmentation + extractEmbedding separately
    public func performCompleteDiarization(_ samples: [Float], sampleRate: Int = 16000) async throws -> DiarizationResult {
        guard segmentationModel != nil, embeddingModel != nil else {
            throw DiarizerError.notInitialized
        }

        logger.info("Starting complete diarization for \(samples.count) samples")

        let totalDuration = Double(samples.count) / Double(sampleRate)
        
        // For long audio files, use parallel processing with post-hoc speaker alignment
        if totalDuration > config.parallelProcessingThreshold {
            return try await performParallelDiarization(samples, sampleRate: sampleRate)
        }
        
        // For shorter files, use sequential processing for better speaker consistency
        return try await performSequentialDiarization(samples, sampleRate: sampleRate)
    }
    
    /// Sequential processing for optimal speaker consistency (shorter files)
    private func performSequentialDiarization(_ samples: [Float], sampleRate: Int = 16000) async throws -> DiarizationResult {
        let chunkSize = sampleRate * 10 // 10 seconds
        var allSegments: [TimedSpeakerSegment] = []
        var speakerDB: [String: [Float]] = [:]  // Global speaker database

        // Process in 10-second chunks sequentially
        for chunkStart in stride(from: 0, to: samples.count, by: chunkSize) {
            let chunkEnd = min(chunkStart + chunkSize, samples.count)
            let chunk = Array(samples[chunkStart..<chunkEnd])
            let chunkOffset = Double(chunkStart) / Double(sampleRate)

            let chunkSegments = try await processChunkWithSpeakerTracking(
                chunk,
                chunkOffset: chunkOffset,
                speakerDB: &speakerDB,
                sampleRate: sampleRate
            )
            allSegments.append(contentsOf: chunkSegments)
        }

        logger.info("Sequential diarization finished: \(allSegments.count) segments, \(speakerDB.count) speakers")
        return DiarizationResult(segments: allSegments, speakerDatabase: speakerDB)
    }
    
    /// Parallel processing for long audio files with post-processing speaker alignment
    private func performParallelDiarization(_ samples: [Float], sampleRate: Int = 16000) async throws -> DiarizationResult {
        let chunkSize = sampleRate * 10 // 10 seconds
        let totalChunks = (samples.count + chunkSize - 1) / chunkSize
        
        logger.info("Using parallel processing for \(totalChunks) chunks")
        
        // Process chunks in parallel using TaskGroup
        let chunkResults = try await withThrowingTaskGroup(of: (offset: Double, segments: [TimedSpeakerSegment]).self) { group in
            var results: [(offset: Double, segments: [TimedSpeakerSegment])] = []
            
            for chunkIndex in 0..<totalChunks {
                let chunkStart = chunkIndex * chunkSize
                let chunkEnd = min(chunkStart + chunkSize, samples.count)
                let chunkOffset = Double(chunkStart) / Double(sampleRate)
                let chunk = Array(samples[chunkStart..<chunkEnd])
                
                group.addTask { [self] in
                    // Process each chunk independently
                    var localSpeakerDB: [String: [Float]] = [:]
                    let segments = try await self.processChunkWithSpeakerTracking(
                        chunk,
                        chunkOffset: chunkOffset,
                        speakerDB: &localSpeakerDB,
                        sampleRate: sampleRate
                    )
                    return (offset: chunkOffset, segments: segments)
                }
            }
            
            // Collect results in order
            for try await result in group {
                results.append(result)
            }
            
            return results.sorted { $0.offset < $1.offset }
        }
        
        // Align speakers across chunks using global clustering
        let (alignedSegments, globalSpeakerDB) = alignSpeakersAcrossChunks(chunkResults.flatMap { $0.segments })
        
        logger.info("Parallel diarization finished: \(alignedSegments.count) segments, \(globalSpeakerDB.count) speakers")
        return DiarizationResult(segments: alignedSegments, speakerDatabase: globalSpeakerDB)
    }
    
    /// Align speakers across parallel-processed chunks using embedding similarity with Metal acceleration
    private func alignSpeakersAcrossChunks(_ segments: [TimedSpeakerSegment]) -> ([TimedSpeakerSegment], [String: [Float]]) {
        var globalSpeakerDB: [String: [Float]] = [:]
        var alignedSegments: [TimedSpeakerSegment] = []
        
        // Group segments into batches for Metal processing
        let batchSize = config.metalBatchSize
        let segmentBatches = segments.chunked(into: batchSize)
        
        for batch in segmentBatches {
            let embeddings = batch.map { $0.embedding }
            
            // Use batch assignment when we have multiple speakers in the database
            let speakerIds: [String]
            if globalSpeakerDB.count > 1 && embeddings.count > 1 {
                speakerIds = batchAssignSpeakers(embeddings: embeddings, speakerDB: &globalSpeakerDB)
            } else {
                // Fall back to individual assignment for small batches or empty database
                speakerIds = embeddings.map { assignSpeakerGlobally(embedding: $0, speakerDB: &globalSpeakerDB) }
            }
            
            // Create aligned segments with assigned speaker IDs
            for (index, segment) in batch.enumerated() {
                let alignedSegment = TimedSpeakerSegment(
                    speakerId: speakerIds[index],
                    embedding: segment.embedding,
                    startTimeSeconds: segment.startTimeSeconds,
                    endTimeSeconds: segment.endTimeSeconds,
                    qualityScore: segment.qualityScore
                )
                alignedSegments.append(alignedSegment)
            }
        }
        
        return (alignedSegments, globalSpeakerDB)
    }
    
    /// Assign speaker ID to global database (similar to existing method but standalone)
    private func assignSpeakerGlobally(embedding: [Float], speakerDB: inout [String: [Float]]) -> String {
        if speakerDB.isEmpty {
            let speakerId = "Speaker 1"
            speakerDB[speakerId] = embedding
            return speakerId
        }

        var minDistance: Float = Float.greatestFiniteMagnitude
        var identifiedSpeaker: String? = nil

        for (speakerId, refEmbedding) in speakerDB {
            let distance = cosineDistance(embedding, refEmbedding)
            if distance < minDistance {
                minDistance = distance
                identifiedSpeaker = speakerId
                
                // Early termination if we find a very close match
                if config.useEarlyTermination && distance < config.earlyTerminationThreshold {
                    break
                }
            }
        }

        if let bestSpeaker = identifiedSpeaker {
            if minDistance > config.clusteringThreshold {
                // New speaker
                let newSpeakerId = "Speaker \(speakerDB.count + 1)"
                speakerDB[newSpeakerId] = embedding
                return newSpeakerId
            } else {
                // Existing speaker - update embedding
                updateSpeakerEmbedding(bestSpeaker, embedding, speakerDB: &speakerDB)
                return bestSpeaker
            }
        }

        return "Unknown"
    }

    /// Process a single chunk with speaker tracking across chunks
    private func processChunkWithSpeakerTracking(
        _ chunk: [Float],
        chunkOffset: Double,
        speakerDB: inout [String: [Float]],
        sampleRate: Int = 16000
    ) async throws -> [TimedSpeakerSegment] {
        let chunkSize = sampleRate * 10 // 10 seconds
        var paddedChunk = chunk
        if chunk.count < chunkSize {
            paddedChunk += Array(repeating: 0.0, count: chunkSize - chunk.count)
        }

        // Step 1: Get segmentation (when speakers are active)
        let binarizedSegments = try getSegments(audioChunk: paddedChunk)
        let slidingFeature = createSlidingWindowFeature(binarizedSegments: binarizedSegments, chunkOffset: chunkOffset)

        // Step 2: Get embeddings using same segmentation results
        guard let embeddingModel = self.embeddingModel else {
            throw DiarizerError.notInitialized
        }

        let embeddings = try getEmbedding(
            audioChunk: paddedChunk,
            binarizedSegments: binarizedSegments,
            slidingWindowFeature: slidingFeature,
            embeddingModel: embeddingModel,
            sampleRate: sampleRate
        )

        // Step 3: Calculate speaker activities
        let speakerActivities = calculateSpeakerActivities(binarizedSegments)

        // Step 4: Assign consistent speaker IDs using global database
        var speakerLabels: [String] = []
        for (speakerIndex, activity) in speakerActivities.enumerated() {
            if activity > config.minActivityThreshold { // Use configurable activity threshold
                let embedding = embeddings[speakerIndex]
                if validateEmbedding(embedding) {
                    let speakerId = assignSpeaker(embedding: embedding, speakerDB: &speakerDB)
                    speakerLabels.append(speakerId)
                } else {
                    speakerLabels.append("")  // Invalid embedding
                }
            } else {
                speakerLabels.append("")  // No activity
            }
        }

        // Step 5: Create temporal segments with consistent speaker IDs
        return createTimedSegments(
            binarizedSegments: binarizedSegments,
            slidingWindow: slidingFeature.slidingWindow,
            embeddings: embeddings,
            speakerLabels: speakerLabels,
            speakerActivities: speakerActivities
        )
    }

    /// Calculate total activity for each speaker across all frames
    private func calculateSpeakerActivities(_ binarizedSegments: [[[Float]]]) -> [Float] {
        let numSpeakers = binarizedSegments[0][0].count
        let numFrames = binarizedSegments[0].count
        var activities: [Float] = Array(repeating: 0.0, count: numSpeakers)

        for speakerIndex in 0..<numSpeakers {
            for frameIndex in 0..<numFrames {
                activities[speakerIndex] += binarizedSegments[0][frameIndex][speakerIndex]
            }
        }

        return activities
    }

    /// Assign speaker ID using global database (like main.swift)
    private func assignSpeaker(embedding: [Float], speakerDB: inout [String: [Float]]) -> String {
        if speakerDB.isEmpty {
            let speakerId = "Speaker 1"
            speakerDB[speakerId] = embedding
            logger.info("Created new speaker: \(speakerId)")
            return speakerId
        }

        var minDistance: Float = Float.greatestFiniteMagnitude
        var identifiedSpeaker: String? = nil

        for (speakerId, refEmbedding) in speakerDB {
            let distance = cosineDistance(embedding, refEmbedding)
            if distance < minDistance {
                minDistance = distance
                identifiedSpeaker = speakerId
                
                // Early termination if we find a very close match
                if config.useEarlyTermination && distance < config.earlyTerminationThreshold {
                    break
                }
            }
        }

        if let bestSpeaker = identifiedSpeaker {
            if minDistance > config.clusteringThreshold {
                // New speaker
                let newSpeakerId = "Speaker \(speakerDB.count + 1)"
                speakerDB[newSpeakerId] = embedding
                logger.info("Created new speaker: \(newSpeakerId) (distance: \(String(format: "%.3f", minDistance)))")
                return newSpeakerId
            } else {
                // Existing speaker - update embedding (exponential moving average)
                updateSpeakerEmbedding(bestSpeaker, embedding, speakerDB: &speakerDB)
                logger.debug("Matched existing speaker: \(bestSpeaker) (distance: \(String(format: "%.3f", minDistance)))")
                return bestSpeaker
            }
        }

        return "Unknown"
    }

    /// Update speaker embedding with exponential moving average
    private func updateSpeakerEmbedding(_ speakerId: String, _ newEmbedding: [Float], speakerDB: inout [String: [Float]], alpha: Float = 0.9) {
        guard var oldEmbedding = speakerDB[speakerId] else { return }

        for i in 0..<oldEmbedding.count {
            oldEmbedding[i] = alpha * oldEmbedding[i] + (1 - alpha) * newEmbedding[i]
        }
        speakerDB[speakerId] = oldEmbedding
    }

    /// Create timed segments with speaker IDs
    private func createTimedSegments(
        binarizedSegments: [[[Float]]],
        slidingWindow: SlidingWindow,
        embeddings: [[Float]],
        speakerLabels: [String],
        speakerActivities: [Float]
    ) -> [TimedSpeakerSegment] {
        let segmentation = binarizedSegments[0]
        let numFrames = segmentation.count
        var segments: [TimedSpeakerSegment] = []

        // Find dominant speaker per frame
        var frameSpeakers: [Int] = []
        for frame in segmentation {
            if let maxIdx = frame.indices.max(by: { frame[$0] < frame[$1] }) {
                frameSpeakers.append(maxIdx)
            } else {
                frameSpeakers.append(0)
            }
        }

        // Group contiguous same-speaker segments
        var currentSpeaker = frameSpeakers[0]
        var startFrame = 0

        for i in 1..<numFrames {
            if frameSpeakers[i] != currentSpeaker {
                if let segment = createSegmentIfValid(
                    speakerIndex: currentSpeaker,
                    startFrame: startFrame,
                    endFrame: i,
                    slidingWindow: slidingWindow,
                    embeddings: embeddings,
                    speakerLabels: speakerLabels,
                    speakerActivities: speakerActivities
                ) {
                    segments.append(segment)
                }
                currentSpeaker = frameSpeakers[i]
                startFrame = i
            }
        }

        // Final segment
        if let segment = createSegmentIfValid(
            speakerIndex: currentSpeaker,
            startFrame: startFrame,
            endFrame: numFrames,
            slidingWindow: slidingWindow,
            embeddings: embeddings,
            speakerLabels: speakerLabels,
            speakerActivities: speakerActivities
        ) {
            segments.append(segment)
        }

        return segments
    }

    /// Create a segment if the speaker is valid
    private func createSegmentIfValid(
        speakerIndex: Int,
        startFrame: Int,
        endFrame: Int,
        slidingWindow: SlidingWindow,
        embeddings: [[Float]],
        speakerLabels: [String],
        speakerActivities: [Float]
    ) -> TimedSpeakerSegment? {
        guard speakerIndex < speakerLabels.count,
              !speakerLabels[speakerIndex].isEmpty,
              speakerIndex < embeddings.count else {
            return nil
        }

        let startTime = slidingWindow.time(forFrame: startFrame)
        let endTime = slidingWindow.time(forFrame: endFrame)
        let embedding = embeddings[speakerIndex]
        let activity = speakerActivities[speakerIndex]
        let quality = calculateEmbeddingQuality(embedding) * (activity / Float(endFrame - startFrame))

        return TimedSpeakerSegment(
            speakerId: speakerLabels[speakerIndex],
            embedding: embedding,
            startTimeSeconds: Float(startTime),
            endTimeSeconds: Float(endTime),
            qualityScore: quality
        )
    }

    /// Clean up resources
    public func cleanup() async {
        segmentationModel = nil
        embeddingModel = nil
        logger.info("Diarization resources cleaned up")
    }
}

