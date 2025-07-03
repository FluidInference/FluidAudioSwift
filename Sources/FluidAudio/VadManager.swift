import Foundation
import AVFoundation
import SoundAnalysis
import OSLog

/// Configuration for Voice Activity Detection
public struct VadConfig: Sendable {
    public var enableVAD: Bool = true  // Enable Voice Activity Detection
    public var vadThreshold: Float = 0.6  // SoundAnalysis VAD confidence threshold
    public var energyVADThreshold: Float = 0.01  // Energy-based VAD threshold (fallback)
    public var debugMode: Bool = false  // Enable debug logging

    public static let `default` = VadConfig()

    public init(
        enableVAD: Bool = true,
        vadThreshold: Float = 0.6,
        energyVADThreshold: Float = 0.01,
        debugMode: Bool = false
    ) {
        self.enableVAD = enableVAD
        self.vadThreshold = vadThreshold
        self.energyVADThreshold = energyVADThreshold
        self.debugMode = debugMode
    }
}

/// Voice Activity Detection Manager
///
/// This class provides Voice Activity Detection using Apple's SoundAnalysis framework
/// with fallback to energy-based VAD for better compatibility.
///
/// Features:
/// - SoundAnalysis-based VAD for high accuracy
/// - Energy-based VAD fallback
/// - Configurable thresholds
/// - Debug logging support
///
/// Example usage:
/// ```swift
/// let vadManager = VadManager(config: VadConfig(vadThreshold: 0.7))
/// let filteredAudio = vadManager.detectVoiceActivity(in: audioSamples)
/// let hasSpeech = vadManager.isSpeechDetected(in: audioSamples)
/// ```
@available(macOS 13.0, iOS 16.0, *)
public final class VadManager: NSObject, SNResultsObserving, @unchecked Sendable {

    private let logger = Logger(subsystem: "com.fluidinfluence.vad", category: "VadManager")
    private let config: VadConfig

    // Voice Activity Detection
    private let soundAnalysisRequest: SNClassifySoundRequest?
    private var vadResults: [Bool] = []

    public init(config: VadConfig = .default) {
        self.config = config

        // Initialize SoundAnalysis VAD
        do {
            self.soundAnalysisRequest = try SNClassifySoundRequest(classifierIdentifier: .version1)
        } catch {
            self.soundAnalysisRequest = nil
        }

        super.init()
    }

    // MARK: - Public API

    /// Detect voice activity in audio samples using SoundAnalysis or energy-based VAD
    public func detectVoiceActivity(in samples: [Float], windowSize: Int = 1600, threshold: Float? = nil) -> [Float] {
        guard config.enableVAD else { return samples }

        let vadThreshold = threshold ?? config.energyVADThreshold

        // Use SoundAnalysis VAD if available, otherwise fall back to energy-based VAD
        if soundAnalysisRequest != nil {
            return detectVoiceActivityWithSoundAnalysis(in: samples)
        } else {
            return detectVoiceActivityWithEnergy(in: samples, windowSize: windowSize, threshold: vadThreshold)
        }
    }

    /// Simple VAD detector - returns true if speech is detected
    public func isSpeechDetected(in samples: [Float]) -> Bool {
        guard config.enableVAD else { return true }

        // Use SoundAnalysis if available for better accuracy
        if soundAnalysisRequest != nil {
            return isSpeechDetectedWithSoundAnalysis(in: samples)
        } else {
            // Fall back to energy-based detection
            let energy = calculateRMSEnergy(samples)
            return energy > config.energyVADThreshold
        }
    }

    /// Convenience method to get VAD-filtered audio samples
    public func getVADFilteredAudio(from samples: [Float]) -> [Float] {
        return detectVoiceActivity(in: samples, threshold: config.energyVADThreshold)
    }

    /// Check if SoundAnalysis VAD is available
    public var isSoundAnalysisAvailable: Bool {
        return soundAnalysisRequest != nil
    }

    // MARK: - Private Implementation

    private func detectVoiceActivityWithSoundAnalysis(in samples: [Float]) -> [Float] {
        guard let request = soundAnalysisRequest else { return samples }

        vadResults.removeAll()

                guard let audioBuffer = createAudioBuffer(from: samples) else {
            logger.error("Failed to create audio buffer for SoundAnalysis VAD")
            return samples
        }

        do {
            let analyzer = SNAudioStreamAnalyzer(format: audioBuffer.format)
            try analyzer.add(request, withObserver: self)
            analyzer.analyze(audioBuffer, atAudioFramePosition: 0)
        } catch {
            logger.error("SoundAnalysis processing failed: \(error.localizedDescription)")
            return samples
        }

        return applyVADResults(to: samples)
    }

    private func detectVoiceActivityWithEnergy(in samples: [Float], windowSize: Int = 1600, threshold: Float = 0.01) -> [Float] {
        var segments: [Float] = []
        var current: [Float] = []
        var silenceCount = 0
        let maxSilenceFrames = 3

        for i in stride(from: 0, to: samples.count, by: windowSize) {
            let end = min(i + windowSize, samples.count)
            let window = Array(samples[i..<end])
            let energy = calculateRMSEnergy(window)

            if energy > threshold {
                silenceCount = 0
                if current.isEmpty {
                    let contextStart = max(0, i - windowSize)
                    current.append(contentsOf: samples[contextStart..<end])
                } else {
                    current.append(contentsOf: window)
                }
            } else {
                silenceCount += 1
                if !current.isEmpty && silenceCount <= maxSilenceFrames {
                    current.append(contentsOf: window)
                } else if !current.isEmpty {
                    if current.count > windowSize {
                        segments.append(contentsOf: current)
                    }
                    current.removeAll()
                }
            }
        }

        if !current.isEmpty && current.count > windowSize {
            segments.append(contentsOf: current)
        }

        return segments.isEmpty ? samples : segments
    }

    private func createAudioBuffer(from samples: [Float]) -> AVAudioPCMBuffer? {
        let format = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 16000, channels: 1, interleaved: false)
        guard let audioFormat = format else { return nil }

        let buffer = AVAudioPCMBuffer(pcmFormat: audioFormat, frameCapacity: AVAudioFrameCount(samples.count))
        guard let audioBuffer = buffer else { return nil }

        audioBuffer.frameLength = audioBuffer.frameCapacity

        if let channelData = audioBuffer.floatChannelData?[0] {
            for i in 0..<samples.count {
                channelData[i] = samples[i]
            }
        }

        return audioBuffer
    }

    private func applyVADResults(to samples: [Float]) -> [Float] {
        guard !vadResults.isEmpty else { return samples }

        var filteredSamples: [Float] = []
        let samplesPerVADResult = samples.count / vadResults.count
        var currentSegment: [Float] = []

        for (index, isSpeech) in vadResults.enumerated() {
            let startIndex = index * samplesPerVADResult
            let endIndex = min((index + 1) * samplesPerVADResult, samples.count)
            let segmentSamples = Array(samples[startIndex..<endIndex])

            if isSpeech {
                currentSegment.append(contentsOf: segmentSamples)
            } else if !currentSegment.isEmpty {
                let paddingSamples = min(samplesPerVADResult, segmentSamples.count)
                currentSegment.append(contentsOf: segmentSamples.prefix(paddingSamples))

                if currentSegment.count > samplesPerVADResult {
                    filteredSamples.append(contentsOf: currentSegment)
                }
                currentSegment.removeAll()
            }
        }

        if !currentSegment.isEmpty && currentSegment.count > samplesPerVADResult {
            filteredSamples.append(contentsOf: currentSegment)
        }

        return filteredSamples.isEmpty ? samples : filteredSamples
    }

    private func isSpeechDetectedWithSoundAnalysis(in samples: [Float]) -> Bool {
        guard let request = soundAnalysisRequest else { return false }

        vadResults.removeAll()

                guard let audioBuffer = createAudioBuffer(from: samples) else {
            logger.error("Failed to create audio buffer for SoundAnalysis VAD")
            return false
        }

        do {
            let analyzer = SNAudioStreamAnalyzer(format: audioBuffer.format)
            try analyzer.add(request, withObserver: self)
            analyzer.analyze(audioBuffer, atAudioFramePosition: 0)
        } catch {
            logger.error("SoundAnalysis processing failed: \(error.localizedDescription)")
            return false
        }

        // Return true if any part was classified as speech
        return vadResults.contains(true)
    }

    private func calculateRMSEnergy(_ samples: [Float]) -> Float {
        guard !samples.isEmpty else { return 0 }
        let squaredSum = samples.reduce(0) { $0 + $1 * $1 }
        return sqrt(squaredSum / Float(samples.count))
    }

    // MARK: - SNResultsObserving

    public func request(_ request: SNRequest, didProduce result: SNResult) {
        guard let classificationResult = result as? SNClassificationResult else { return }

        // Log all classifications for debugging
        if config.debugMode {
            let allClassifications = classificationResult.classifications.map { "\($0.identifier): \($0.confidence)" }.joined(separator: ", ")
            logger.debug("SoundAnalysis classifications: \(allClassifications)")
        }

        let speechConfidence = classificationResult.classifications.first { classification in
            classification.identifier == "Speech" ||
            classification.identifier.contains("speech") ||
            classification.identifier.contains("voice")
        }?.confidence ?? 0.0

        let isSpeech = speechConfidence > Double(config.vadThreshold)
        if config.debugMode {
            logger.debug("Speech confidence: \(speechConfidence), isSpeech: \(isSpeech)")
        }
        vadResults.append(isSpeech)
    }

    public func request(_ request: SNRequest, didFailWithError error: Error) {
        logger.error("SoundAnalysis request failed: \(error.localizedDescription)")
    }

    public func requestDidComplete(_ request: SNRequest) {
        // Optional: Handle completion
    }

    // MARK: - Cleanup

    /// Clean up VAD resources
    public func cleanup() {
        vadResults.removeAll()
        logger.info("VAD resources cleaned up")
    }
}
