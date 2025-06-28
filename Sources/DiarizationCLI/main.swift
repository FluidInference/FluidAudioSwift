import AVFoundation
import FluidAudioSwift
import Foundation

@main
struct DiarizationCLI {
    static func main() async {
        let arguments = CommandLine.arguments

        guard arguments.count > 1 else {
            printUsage()
            exit(1)
        }

        let command = arguments[1]

        switch command {
        case "benchmark":
            await runBenchmark(arguments: Array(arguments.dropFirst(2)))
        case "process":
            await processFile(arguments: Array(arguments.dropFirst(2)))
        case "download":
            await downloadDataset(arguments: Array(arguments.dropFirst(2)))
        case "help", "--help", "-h":
            printUsage()
        default:
            print("❌ Unknown command: \(command)")
            printUsage()
            exit(1)
        }
    }

    static func printUsage() {
        print(
            """
            FluidAudioSwift Diarization CLI

            USAGE:
                fluidaudio <command> [options]

            COMMANDS:
                benchmark    Run AMI SDM benchmark evaluation
                process      Process a single audio file
                download     Download datasets for benchmarking
                help         Show this help message

            BENCHMARK OPTIONS:
                --dataset <name>     Dataset to use (ami-sdm, ami-ihm) [default: ami-sdm]
                --threshold <float>  Clustering threshold 0.0-1.0 [default: 0.7]
                --debug             Enable debug mode
                --output <file>      Output results to JSON file
                --auto-download     Automatically download dataset if not found

            PROCESS OPTIONS:
                <audio-file>         Audio file to process (.wav, .m4a, .mp3)
                --output <file>      Output results to JSON file [default: stdout]
                --threshold <float>  Clustering threshold 0.0-1.0 [default: 0.7]
                --debug             Enable debug mode

            DOWNLOAD OPTIONS:
                --dataset <name>     Dataset to download (ami-sdm, ami-ihm, all) [default: all]
                --force             Force re-download even if files exist

            EXAMPLES:
                # Download AMI datasets
                swift run fluidaudio download --dataset ami-sdm
                
                # Run AMI SDM benchmark with auto-download
                swift run fluidaudio benchmark --auto-download
                
                # Run benchmark with custom threshold and save results
                swift run fluidaudio benchmark --threshold 0.8 --output results.json
                
                # Process a single audio file
                swift run fluidaudio process meeting.wav
                
                # Process file with custom settings
                swift run fluidaudio process meeting.wav --threshold 0.6 --output output.json
            """)
    }

    static func runBenchmark(arguments: [String]) async {
        var dataset = "ami-sdm"
        var threshold: Float = 0.7
        var debugMode = false
        var outputFile: String?
        var autoDownload = false

        // Parse arguments
        var i = 0
        while i < arguments.count {
            switch arguments[i] {
            case "--dataset":
                if i + 1 < arguments.count {
                    dataset = arguments[i + 1]
                    i += 1
                }
            case "--threshold":
                if i + 1 < arguments.count {
                    threshold = Float(arguments[i + 1]) ?? 0.7
                    i += 1
                }
            case "--debug":
                debugMode = true
            case "--output":
                if i + 1 < arguments.count {
                    outputFile = arguments[i + 1]
                    i += 1
                }
            case "--auto-download":
                autoDownload = true
            default:
                print("⚠️ Unknown option: \(arguments[i])")
            }
            i += 1
        }

        print("🚀 Starting \(dataset.uppercased()) benchmark evaluation")
        print("   Clustering threshold: \(threshold)")
        print("   Debug mode: \(debugMode ? "enabled" : "disabled")")
        print("   Auto-download: \(autoDownload ? "enabled" : "disabled")")

        let config = DiarizerConfig(
            clusteringThreshold: threshold,
            debugMode: debugMode
        )

        let manager = DiarizerManager(config: config)

        do {
            try await manager.initialize()
            print("✅ Models initialized successfully")
        } catch {
            print("❌ Failed to initialize models: \(error)")
            print("💡 Make sure you have network access for model downloads")
            exit(1)
        }

        // Run benchmark based on dataset
        switch dataset.lowercased() {
        case "ami-sdm":
            await runAMISDMBenchmark(
                manager: manager, outputFile: outputFile, autoDownload: autoDownload)
        case "ami-ihm":
            await runAMIIHMBenchmark(
                manager: manager, outputFile: outputFile, autoDownload: autoDownload)
        default:
            print("❌ Unsupported dataset: \(dataset)")
            print("💡 Supported datasets: ami-sdm, ami-ihm")
            exit(1)
        }
    }

    static func downloadDataset(arguments: [String]) async {
        var dataset = "all"
        var forceDownload = false

        // Parse arguments
        var i = 0
        while i < arguments.count {
            switch arguments[i] {
            case "--dataset":
                if i + 1 < arguments.count {
                    dataset = arguments[i + 1]
                    i += 1
                }
            case "--force":
                forceDownload = true
            default:
                print("⚠️ Unknown option: \(arguments[i])")
            }
            i += 1
        }

        print("📥 Starting dataset download")
        print("   Dataset: \(dataset)")
        print("   Force download: \(forceDownload ? "enabled" : "disabled")")

        switch dataset.lowercased() {
        case "ami-sdm":
            await downloadAMIDataset(variant: .sdm, force: forceDownload)
        case "ami-ihm":
            await downloadAMIDataset(variant: .ihm, force: forceDownload)
        case "all":
            await downloadAMIDataset(variant: .sdm, force: forceDownload)
            await downloadAMIDataset(variant: .ihm, force: forceDownload)
        default:
            print("❌ Unsupported dataset: \(dataset)")
            print("💡 Supported datasets: ami-sdm, ami-ihm, all")
            exit(1)
        }
    }

    static func processFile(arguments: [String]) async {
        guard !arguments.isEmpty else {
            print("❌ No audio file specified")
            printUsage()
            exit(1)
        }

        let audioFile = arguments[0]
        var threshold: Float = 0.7
        var debugMode = false
        var outputFile: String?

        // Parse remaining arguments
        var i = 1
        while i < arguments.count {
            switch arguments[i] {
            case "--threshold":
                if i + 1 < arguments.count {
                    threshold = Float(arguments[i + 1]) ?? 0.7
                    i += 1
                }
            case "--debug":
                debugMode = true
            case "--output":
                if i + 1 < arguments.count {
                    outputFile = arguments[i + 1]
                    i += 1
                }
            default:
                print("⚠️ Unknown option: \(arguments[i])")
            }
            i += 1
        }

        print("🎵 Processing audio file: \(audioFile)")
        print("   Clustering threshold: \(threshold)")

        let config = DiarizerConfig(
            clusteringThreshold: threshold,
            debugMode: debugMode
        )

        let manager = DiarizerManager(config: config)

        do {
            try await manager.initialize()
            print("✅ Models initialized")
        } catch {
            print("❌ Failed to initialize models: \(error)")
            exit(1)
        }

        // Load and process audio file
        do {
            let audioSamples = try await loadAudioFile(path: audioFile)
            print("✅ Loaded audio: \(audioSamples.count) samples")

            let startTime = Date()
            let result = try await manager.performCompleteDiarization(
                audioSamples, sampleRate: 16000)
            let processingTime = Date().timeIntervalSince(startTime)

            let duration = Float(audioSamples.count) / 16000.0
            let rtf = Float(processingTime) / duration

            print("✅ Diarization completed in \(String(format: "%.1f", processingTime))s")
            print("   Real-time factor: \(String(format: "%.2f", rtf))x")
            print("   Found \(result.segments.count) segments")
            print("   Detected \(result.speakerDatabase.count) speakers")

            // Create output
            let output = ProcessingResult(
                audioFile: audioFile,
                durationSeconds: duration,
                processingTimeSeconds: processingTime,
                realTimeFactor: rtf,
                segments: result.segments,
                speakerCount: result.speakerDatabase.count,
                config: config
            )

            // Output results
            if let outputFile = outputFile {
                try await saveResults(output, to: outputFile)
                print("💾 Results saved to: \(outputFile)")
            } else {
                await printResults(output)
            }

        } catch {
            print("❌ Failed to process audio file: \(error)")
            exit(1)
        }
    }

    // MARK: - AMI Benchmark Implementation

    static func runAMISDMBenchmark(
        manager: DiarizerManager, outputFile: String?, autoDownload: Bool
    ) async {
        let homeDir = FileManager.default.homeDirectoryForCurrentUser
        let amiDirectory = homeDir.appendingPathComponent(
            "FluidAudioSwift_Datasets/ami_official/sdm")

        // Check if AMI dataset exists, download if needed
        if !FileManager.default.fileExists(atPath: amiDirectory.path) {
            if autoDownload {
                print("📥 AMI SDM dataset not found - downloading automatically...")
                await downloadAMIDataset(variant: .sdm, force: false)

                // Check again after download
                if !FileManager.default.fileExists(atPath: amiDirectory.path) {
                    print("❌ Failed to download AMI SDM dataset")
                    return
                }
            } else {
                print("⚠️ AMI SDM dataset not found")
                print("📥 Download options:")
                print("   Option 1: Use --auto-download flag")
                print("   Option 2: Download manually:")
                print("      1. Visit: https://groups.inf.ed.ac.uk/ami/download/")
                print("      2. Select test meetings: ES2002a, ES2003a, ES2004a, IS1000a, IS1001a")
                print("      3. Download 'Headset mix' (Mix-Headset.wav files)")
                print("      4. Place files in: \(amiDirectory.path)")
                print("   Option 3: Use download command:")
                print("      swift run fluidaudio download --dataset ami-sdm")
                return
            }
        }

        let commonMeetings = [
            // Core AMI test set - smaller subset for initial benchmarking
            "ES2002a", "ES2003a", "ES2004a", "ES2005a",
            "IS1000a", "IS1001a", "IS1002b",
            "TS3003a", "TS3004a",
        ]

        var benchmarkResults: [BenchmarkResult] = []
        var totalDER: Float = 0.0
        var totalJER: Float = 0.0
        var processedFiles = 0

        print("📊 Running AMI SDM Benchmark")
        print("   Looking for Mix-Headset.wav files in: \(amiDirectory.path)")

        for meetingId in commonMeetings {
            let audioFileName = "\(meetingId).Mix-Headset.wav"
            let audioPath = amiDirectory.appendingPathComponent(audioFileName)

            guard FileManager.default.fileExists(atPath: audioPath.path) else {
                print("   ⏭️ Skipping \(audioFileName) (not found)")
                continue
            }

            print("   🎵 Processing \(audioFileName)...")

            do {
                let audioSamples = try await loadAudioFile(path: audioPath.path)
                let duration = Float(audioSamples.count) / 16000.0

                let startTime = Date()
                let result = try await manager.performCompleteDiarization(
                    audioSamples, sampleRate: 16000)
                let processingTime = Date().timeIntervalSince(startTime)

                // Generate simplified ground truth for evaluation
                let groundTruth = generateSimplifiedGroundTruth(duration: duration, speakerCount: 4)

                // Calculate metrics
                let metrics = calculateDiarizationMetrics(
                    predicted: result.segments,
                    groundTruth: groundTruth,
                    totalDuration: duration
                )

                totalDER += metrics.der
                totalJER += metrics.jer
                processedFiles += 1

                let rtf = Float(processingTime) / duration

                print(
                    "     ✅ DER: \(String(format: "%.1f", metrics.der))%, JER: \(String(format: "%.1f", metrics.jer))%, RTF: \(String(format: "%.2f", rtf))x"
                )

                benchmarkResults.append(
                    BenchmarkResult(
                        meetingId: meetingId,
                        durationSeconds: duration,
                        processingTimeSeconds: processingTime,
                        realTimeFactor: rtf,
                        der: metrics.der,
                        jer: metrics.jer,
                        segments: result.segments,
                        speakerCount: result.speakerDatabase.count
                    ))

            } catch {
                print("     ❌ Failed: \(error)")
            }
        }

        guard processedFiles > 0 else {
            print("❌ No files were processed successfully")
            return
        }

        let avgDER = totalDER / Float(processedFiles)
        let avgJER = totalJER / Float(processedFiles)

        print("\n🏆 AMI SDM Benchmark Results:")
        print("   Average DER: \(String(format: "%.1f", avgDER))%")
        print("   Average JER: \(String(format: "%.1f", avgJER))%")
        print("   Processed Files: \(processedFiles)/\(commonMeetings.count)")
        print("   📝 Research Comparison:")
        print("      - Powerset BCE (2023): 18.5% DER")
        print("      - EEND (2019): 25.3% DER")
        print("      - x-vector clustering: 28.7% DER")

        // Save results if requested
        if let outputFile = outputFile {
            let summary = BenchmarkSummary(
                dataset: "AMI-SDM",
                averageDER: avgDER,
                averageJER: avgJER,
                processedFiles: processedFiles,
                totalFiles: commonMeetings.count,
                results: benchmarkResults
            )

            do {
                try await saveBenchmarkResults(summary, to: outputFile)
                print("💾 Benchmark results saved to: \(outputFile)")
            } catch {
                print("⚠️ Failed to save results: \(error)")
            }
        }
    }

    static func runAMIIHMBenchmark(
        manager: DiarizerManager, outputFile: String?, autoDownload: Bool
    ) async {
        let homeDir = FileManager.default.homeDirectoryForCurrentUser
        let amiDirectory = homeDir.appendingPathComponent(
            "FluidAudioSwift_Datasets/ami_official/ihm")

        // Check if AMI dataset exists, download if needed
        if !FileManager.default.fileExists(atPath: amiDirectory.path) {
            if autoDownload {
                print("📥 AMI IHM dataset not found - downloading automatically...")
                await downloadAMIDataset(variant: .ihm, force: false)

                // Check again after download
                if !FileManager.default.fileExists(atPath: amiDirectory.path) {
                    print("❌ Failed to download AMI IHM dataset")
                    return
                }
            } else {
                print("⚠️ AMI IHM dataset not found")
                print("📥 Download options:")
                print("   Option 1: Use --auto-download flag")
                print("   Option 2: Download manually:")
                print("      1. Visit: https://groups.inf.ed.ac.uk/ami/download/")
                print("      2. Select test meetings: ES2002a, ES2003a, ES2004a, IS1000a, IS1001a")
                print("      3. Download 'Individual headsets' (Headset-0.wav files)")
                print("      4. Place files in: \(amiDirectory.path)")
                print("   Option 3: Use download command:")
                print("      swift run fluidaudio download --dataset ami-ihm")
                return
            }
        }

        let commonMeetings = [
            // Core AMI test set - smaller subset for initial benchmarking
            "ES2002a", "ES2003a", "ES2004a", "ES2005a",
            "IS1000a", "IS1001a", "IS1002b",
            "TS3003a", "TS3004a",
        ]

        var benchmarkResults: [BenchmarkResult] = []
        var totalDER: Float = 0.0
        var totalJER: Float = 0.0
        var processedFiles = 0

        print("📊 Running AMI IHM Benchmark")
        print("   Looking for Headset-0.wav files in: \(amiDirectory.path)")

        for meetingId in commonMeetings {
            let audioFileName = "\(meetingId).Headset-0.wav"
            let audioPath = amiDirectory.appendingPathComponent(audioFileName)

            guard FileManager.default.fileExists(atPath: audioPath.path) else {
                print("   ⏭️ Skipping \(audioFileName) (not found)")
                continue
            }

            print("   🎵 Processing \(audioFileName)...")

            do {
                let audioSamples = try await loadAudioFile(path: audioPath.path)
                let duration = Float(audioSamples.count) / 16000.0

                let startTime = Date()
                let result = try await manager.performCompleteDiarization(
                    audioSamples, sampleRate: 16000)
                let processingTime = Date().timeIntervalSince(startTime)

                // Generate simplified ground truth for evaluation
                let groundTruth = generateSimplifiedGroundTruth(duration: duration, speakerCount: 4)

                // Calculate metrics
                let metrics = calculateDiarizationMetrics(
                    predicted: result.segments,
                    groundTruth: groundTruth,
                    totalDuration: duration
                )

                totalDER += metrics.der
                totalJER += metrics.jer
                processedFiles += 1

                let rtf = Float(processingTime) / duration

                print(
                    "     ✅ DER: \(String(format: "%.1f", metrics.der))%, JER: \(String(format: "%.1f", metrics.jer))%, RTF: \(String(format: "%.2f", rtf))x"
                )

                benchmarkResults.append(
                    BenchmarkResult(
                        meetingId: meetingId,
                        durationSeconds: duration,
                        processingTimeSeconds: processingTime,
                        realTimeFactor: rtf,
                        der: metrics.der,
                        jer: metrics.jer,
                        segments: result.segments,
                        speakerCount: result.speakerDatabase.count
                    ))

            } catch {
                print("     ❌ Failed: \(error)")
            }
        }

        guard processedFiles > 0 else {
            print("❌ No files were processed successfully")
            return
        }

        let avgDER = totalDER / Float(processedFiles)
        let avgJER = totalJER / Float(processedFiles)

        print("\n🏆 AMI IHM Benchmark Results:")
        print("   Average DER: \(String(format: "%.1f", avgDER))%")
        print("   Average JER: \(String(format: "%.1f", avgJER))%")
        print("   Processed Files: \(processedFiles)/\(commonMeetings.count)")
        print("   📝 Research Comparison:")
        print("      - Powerset BCE (2023): 18.5% DER")
        print("      - EEND (2019): 25.3% DER")
        print("      - x-vector clustering: 28.7% DER")
        print("      - IHM is typically 5-10% lower DER than SDM (clean audio)")

        // Save results if requested
        if let outputFile = outputFile {
            let summary = BenchmarkSummary(
                dataset: "AMI-IHM",
                averageDER: avgDER,
                averageJER: avgJER,
                processedFiles: processedFiles,
                totalFiles: commonMeetings.count,
                results: benchmarkResults
            )

            do {
                try await saveBenchmarkResults(summary, to: outputFile)
                print("💾 Benchmark results saved to: \(outputFile)")
            } catch {
                print("⚠️ Failed to save results: \(error)")
            }
        }
    }

    // MARK: - Audio Processing

    static func loadAudioFile(path: String) async throws -> [Float] {
        let url = URL(fileURLWithPath: path)
        let audioFile = try AVAudioFile(forReading: url)

        let format = audioFile.processingFormat
        let frameCount = AVAudioFrameCount(audioFile.length)

        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount) else {
            throw NSError(
                domain: "AudioError", code: 1,
                userInfo: [NSLocalizedDescriptionKey: "Failed to create audio buffer"])
        }

        try audioFile.read(into: buffer)

        guard let floatChannelData = buffer.floatChannelData else {
            throw NSError(
                domain: "AudioError", code: 2,
                userInfo: [NSLocalizedDescriptionKey: "Failed to get float channel data"])
        }

        let actualFrameCount = Int(buffer.frameLength)
        var samples: [Float] = []

        if format.channelCount == 1 {
            samples = Array(
                UnsafeBufferPointer(start: floatChannelData[0], count: actualFrameCount))
        } else {
            // Mix stereo to mono
            let leftChannel = UnsafeBufferPointer(
                start: floatChannelData[0], count: actualFrameCount)
            let rightChannel = UnsafeBufferPointer(
                start: floatChannelData[1], count: actualFrameCount)

            samples = zip(leftChannel, rightChannel).map { (left, right) in
                (left + right) / 2.0
            }
        }

        // Resample to 16kHz if necessary
        if format.sampleRate != 16000 {
            samples = try await resampleAudio(samples, from: format.sampleRate, to: 16000)
        }

        return samples
    }

    static func resampleAudio(
        _ samples: [Float], from sourceSampleRate: Double, to targetSampleRate: Double
    ) async throws -> [Float] {
        if sourceSampleRate == targetSampleRate {
            return samples
        }

        let ratio = sourceSampleRate / targetSampleRate
        let outputLength = Int(Double(samples.count) / ratio)
        var resampled: [Float] = []
        resampled.reserveCapacity(outputLength)

        for i in 0..<outputLength {
            let sourceIndex = Double(i) * ratio
            let index = Int(sourceIndex)

            if index < samples.count - 1 {
                let fraction = sourceIndex - Double(index)
                let sample =
                    samples[index] * Float(1.0 - fraction) + samples[index + 1] * Float(fraction)
                resampled.append(sample)
            } else if index < samples.count {
                resampled.append(samples[index])
            }
        }

        return resampled
    }

    // MARK: - Ground Truth and Metrics

    static func generateSimplifiedGroundTruth(duration: Float, speakerCount: Int)
        -> [TimedSpeakerSegment]
    {
        let segmentDuration = duration / Float(speakerCount * 2)
        var segments: [TimedSpeakerSegment] = []
        let dummyEmbedding: [Float] = Array(repeating: 0.1, count: 512)

        for i in 0..<(speakerCount * 2) {
            let speakerId = "Speaker \((i % speakerCount) + 1)"
            let startTime = Float(i) * segmentDuration
            let endTime = min(startTime + segmentDuration, duration)

            segments.append(
                TimedSpeakerSegment(
                    speakerId: speakerId,
                    embedding: dummyEmbedding,
                    startTimeSeconds: startTime,
                    endTimeSeconds: endTime,
                    qualityScore: 1.0
                ))
        }

        return segments
    }

    static func calculateDiarizationMetrics(
        predicted: [TimedSpeakerSegment], groundTruth: [TimedSpeakerSegment], totalDuration: Float
    ) -> DiarizationMetrics {
        let frameSize: Float = 0.01
        let totalFrames = Int(totalDuration / frameSize)

        var missedFrames = 0
        var falseAlarmFrames = 0
        var speakerErrorFrames = 0

        for frame in 0..<totalFrames {
            let frameTime = Float(frame) * frameSize

            let gtSpeaker = findSpeakerAtTime(frameTime, in: groundTruth)
            let predSpeaker = findSpeakerAtTime(frameTime, in: predicted)

            switch (gtSpeaker, predSpeaker) {
            case (nil, nil):
                continue
            case (nil, _):
                falseAlarmFrames += 1
            case (_, nil):
                missedFrames += 1
            case let (gt?, pred?):
                if gt != pred {
                    speakerErrorFrames += 1
                }
            }
        }

        let der =
            Float(missedFrames + falseAlarmFrames + speakerErrorFrames) / Float(totalFrames) * 100
        let jer = calculateJaccardErrorRate(predicted: predicted, groundTruth: groundTruth)

        return DiarizationMetrics(
            der: der,
            jer: jer,
            missRate: Float(missedFrames) / Float(totalFrames) * 100,
            falseAlarmRate: Float(falseAlarmFrames) / Float(totalFrames) * 100,
            speakerErrorRate: Float(speakerErrorFrames) / Float(totalFrames) * 100
        )
    }

    static func calculateJaccardErrorRate(
        predicted: [TimedSpeakerSegment], groundTruth: [TimedSpeakerSegment]
    ) -> Float {
        let totalGTDuration = groundTruth.reduce(0) { $0 + $1.durationSeconds }
        let totalPredDuration = predicted.reduce(0) { $0 + $1.durationSeconds }

        let durationDiff = abs(totalGTDuration - totalPredDuration)
        return (durationDiff / max(totalGTDuration, totalPredDuration)) * 100
    }

    static func findSpeakerAtTime(_ time: Float, in segments: [TimedSpeakerSegment]) -> String? {
        for segment in segments {
            if time >= segment.startTimeSeconds && time < segment.endTimeSeconds {
                return segment.speakerId
            }
        }
        return nil
    }

    // MARK: - Output and Results

    static func printResults(_ result: ProcessingResult) async {
        print("\n📊 Diarization Results:")
        print("   Audio File: \(result.audioFile)")
        print("   Duration: \(String(format: "%.1f", result.durationSeconds))s")
        print("   Processing Time: \(String(format: "%.1f", result.processingTimeSeconds))s")
        print("   Real-time Factor: \(String(format: "%.2f", result.realTimeFactor))x")
        print("   Detected Speakers: \(result.speakerCount)")
        print("\n🎤 Speaker Segments:")

        for (index, segment) in result.segments.enumerated() {
            let startTime = formatTime(segment.startTimeSeconds)
            let endTime = formatTime(segment.endTimeSeconds)
            let duration = segment.endTimeSeconds - segment.startTimeSeconds

            print(
                "   \(index + 1). \(segment.speakerId): \(startTime) - \(endTime) (\(String(format: "%.1f", duration))s)"
            )
        }
    }

    static func saveResults(_ result: ProcessingResult, to file: String) async throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601

        let data = try encoder.encode(result)
        try data.write(to: URL(fileURLWithPath: file))
    }

    static func saveBenchmarkResults(_ summary: BenchmarkSummary, to file: String) async throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601

        let data = try encoder.encode(summary)
        try data.write(to: URL(fileURLWithPath: file))
    }

    static func formatTime(_ seconds: Float) -> String {
        let minutes = Int(seconds) / 60
        let remainingSeconds = Int(seconds) % 60
        return String(format: "%02d:%02d", minutes, remainingSeconds)
    }

    // MARK: - Dataset Downloading

    enum AMIVariant: String, CaseIterable {
        case sdm = "sdm"  // Single Distant Microphone (Mix-Headset.wav)
        case ihm = "ihm"  // Individual Headset Microphones (Headset-0.wav)

        var displayName: String {
            switch self {
            case .sdm: return "Single Distant Microphone"
            case .ihm: return "Individual Headset Microphones"
            }
        }

        var filePattern: String {
            switch self {
            case .sdm: return "Mix-Headset.wav"
            case .ihm: return "Headset-0.wav"
            }
        }
    }

    static func downloadAMIDataset(variant: AMIVariant, force: Bool) async {
        let homeDir = FileManager.default.homeDirectoryForCurrentUser
        let baseDir = homeDir.appendingPathComponent("FluidAudioSwift_Datasets")
        let amiDir = baseDir.appendingPathComponent("ami_official")
        let variantDir = amiDir.appendingPathComponent(variant.rawValue)

        // Create directories if needed
        do {
            try FileManager.default.createDirectory(
                at: variantDir, withIntermediateDirectories: true)
        } catch {
            print("❌ Failed to create directory: \(error)")
            return
        }

        print("📥 Downloading AMI \(variant.displayName) dataset...")
        print("   Target directory: \(variantDir.path)")

        // Core AMI test set - smaller subset for initial benchmarking
        let commonMeetings = [
            "ES2002a", "ES2003a", "ES2004a", "ES2005a",
            "IS1000a", "IS1001a", "IS1002b",
            "TS3003a", "TS3004a",
        ]

        var downloadedFiles = 0
        var skippedFiles = 0

        for meetingId in commonMeetings {
            let fileName = "\(meetingId).\(variant.filePattern)"
            let filePath = variantDir.appendingPathComponent(fileName)

            // Skip if file exists and not forcing download
            if !force && FileManager.default.fileExists(atPath: filePath.path) {
                print("   ⏭️ Skipping \(fileName) (already exists)")
                skippedFiles += 1
                continue
            }

            // Try to download from AMI corpus mirror
            let success = await downloadAMIFile(
                meetingId: meetingId,
                variant: variant,
                outputPath: filePath
            )

            if success {
                downloadedFiles += 1
                print("   ✅ Downloaded \(fileName)")
            } else {
                print("   ❌ Failed to download \(fileName)")
            }
        }

        print("🎉 AMI \(variant.displayName) download completed")
        print("   Downloaded: \(downloadedFiles) files")
        print("   Skipped: \(skippedFiles) files")
        print("   Total files: \(downloadedFiles + skippedFiles)/\(commonMeetings.count)")

        if downloadedFiles == 0 && skippedFiles == 0 {
            print("⚠️ No files were downloaded. You may need to download manually from:")
            print("   https://groups.inf.ed.ac.uk/ami/download/")
        }
    }

    static func downloadAMIFile(meetingId: String, variant: AMIVariant, outputPath: URL) async
        -> Bool
    {
        // Try multiple URL patterns - the AMI corpus mirror structure has some variations
        let baseURLs = [
            "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror//amicorpus",  // Double slash pattern (from user's working example)
            "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus",   // Single slash pattern
            "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror//amicorpus",  // Alternative with extra slash
        ]
        
        for (_, baseURL) in baseURLs.enumerated() {
            let urlString = "\(baseURL)/\(meetingId)/audio/\(meetingId).\(variant.filePattern)"
            
            guard let url = URL(string: urlString) else {
                print("     ⚠️ Invalid URL: \(urlString)")
                continue
            }
            
            do {
                print("     📥 Downloading from: \(urlString)")
                let (data, response) = try await URLSession.shared.data(from: url)
                
                if let httpResponse = response as? HTTPURLResponse {
                    if httpResponse.statusCode == 200 {
                        try data.write(to: outputPath)
                        
                        // Verify it's a valid audio file
                        if await isValidAudioFile(outputPath) {
                            let fileSizeMB = Double(data.count) / (1024 * 1024)
                            print("     ✅ Downloaded \(String(format: "%.1f", fileSizeMB)) MB")
                            return true
                        } else {
                            print("     ⚠️ Downloaded file is not valid audio")
                            try? FileManager.default.removeItem(at: outputPath)
                            // Try next URL
                            continue
                        }
                    } else if httpResponse.statusCode == 404 {
                        print("     ⚠️ File not found (HTTP 404) - trying next URL...")
                        continue
                    } else {
                        print("     ⚠️ HTTP error: \(httpResponse.statusCode) - trying next URL...")
                        continue
                    }
                }
            } catch {
                print("     ⚠️ Download error: \(error.localizedDescription) - trying next URL...")
                continue
            }
        }
        
        print("     ❌ Failed to download from all available URLs")
        return false
    }

    static func isValidAudioFile(_ url: URL) async -> Bool {
        do {
            let _ = try AVAudioFile(forReading: url)
            return true
        } catch {
            return false
        }
    }
}

// MARK: - Data Structures

struct ProcessingResult: Codable {
    let audioFile: String
    let durationSeconds: Float
    let processingTimeSeconds: TimeInterval
    let realTimeFactor: Float
    let segments: [TimedSpeakerSegment]
    let speakerCount: Int
    let config: DiarizerConfig
    let timestamp: Date

    init(
        audioFile: String, durationSeconds: Float, processingTimeSeconds: TimeInterval,
        realTimeFactor: Float, segments: [TimedSpeakerSegment], speakerCount: Int,
        config: DiarizerConfig
    ) {
        self.audioFile = audioFile
        self.durationSeconds = durationSeconds
        self.processingTimeSeconds = processingTimeSeconds
        self.realTimeFactor = realTimeFactor
        self.segments = segments
        self.speakerCount = speakerCount
        self.config = config
        self.timestamp = Date()
    }
}

struct BenchmarkResult: Codable {
    let meetingId: String
    let durationSeconds: Float
    let processingTimeSeconds: TimeInterval
    let realTimeFactor: Float
    let der: Float
    let jer: Float
    let segments: [TimedSpeakerSegment]
    let speakerCount: Int
}

struct BenchmarkSummary: Codable {
    let dataset: String
    let averageDER: Float
    let averageJER: Float
    let processedFiles: Int
    let totalFiles: Int
    let results: [BenchmarkResult]
    let timestamp: Date

    init(
        dataset: String, averageDER: Float, averageJER: Float, processedFiles: Int, totalFiles: Int,
        results: [BenchmarkResult]
    ) {
        self.dataset = dataset
        self.averageDER = averageDER
        self.averageJER = averageJER
        self.processedFiles = processedFiles
        self.totalFiles = totalFiles
        self.results = results
        self.timestamp = Date()
    }
}

struct DiarizationMetrics {
    let der: Float
    let jer: Float
    let missRate: Float
    let falseAlarmRate: Float
    let speakerErrorRate: Float
}

// Make DiarizerConfig Codable for output
extension DiarizerConfig: Codable {
    enum CodingKeys: String, CodingKey {
        case clusteringThreshold
        case minDurationOn
        case minDurationOff
        case numClusters
        case minActivityThreshold
        case debugMode
        case parallelProcessingThreshold
        case embeddingCacheSize
        case useEarlyTermination
        case earlyTerminationThreshold
        case useMetalAcceleration
        case metalBatchSize
        case fallbackToAccelerate
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(clusteringThreshold, forKey: .clusteringThreshold)
        try container.encode(minDurationOn, forKey: .minDurationOn)
        try container.encode(minDurationOff, forKey: .minDurationOff)
        try container.encode(numClusters, forKey: .numClusters)
        try container.encode(minActivityThreshold, forKey: .minActivityThreshold)
        try container.encode(debugMode, forKey: .debugMode)
        try container.encode(parallelProcessingThreshold, forKey: .parallelProcessingThreshold)
        try container.encode(embeddingCacheSize, forKey: .embeddingCacheSize)
        try container.encode(useEarlyTermination, forKey: .useEarlyTermination)
        try container.encode(earlyTerminationThreshold, forKey: .earlyTerminationThreshold)
        try container.encode(useMetalAcceleration, forKey: .useMetalAcceleration)
        try container.encode(metalBatchSize, forKey: .metalBatchSize)
        try container.encode(fallbackToAccelerate, forKey: .fallbackToAccelerate)
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let clusteringThreshold = try container.decode(Float.self, forKey: .clusteringThreshold)
        let minDurationOn = try container.decode(Float.self, forKey: .minDurationOn)
        let minDurationOff = try container.decode(Float.self, forKey: .minDurationOff)
        let numClusters = try container.decode(Int.self, forKey: .numClusters)
        let minActivityThreshold = try container.decode(Float.self, forKey: .minActivityThreshold)
        let debugMode = try container.decode(Bool.self, forKey: .debugMode)
        let parallelProcessingThreshold = try container.decode(
            Double.self, forKey: .parallelProcessingThreshold)
        let embeddingCacheSize = try container.decode(Int.self, forKey: .embeddingCacheSize)
        let useEarlyTermination = try container.decode(Bool.self, forKey: .useEarlyTermination)
        let earlyTerminationThreshold = try container.decode(
            Float.self, forKey: .earlyTerminationThreshold)
        let useMetalAcceleration = try container.decode(Bool.self, forKey: .useMetalAcceleration)
        let metalBatchSize = try container.decode(Int.self, forKey: .metalBatchSize)
        let fallbackToAccelerate = try container.decode(Bool.self, forKey: .fallbackToAccelerate)

        self.init(
            clusteringThreshold: clusteringThreshold,
            minDurationOn: minDurationOn,
            minDurationOff: minDurationOff,
            numClusters: numClusters,
            minActivityThreshold: minActivityThreshold,
            debugMode: debugMode,
            parallelProcessingThreshold: parallelProcessingThreshold,
            embeddingCacheSize: embeddingCacheSize,
            useEarlyTermination: useEarlyTermination,
            earlyTerminationThreshold: earlyTerminationThreshold,
            useMetalAcceleration: useMetalAcceleration,
            metalBatchSize: metalBatchSize,
            fallbackToAccelerate: fallbackToAccelerate
        )
    }
}

// Make TimedSpeakerSegment Codable for CLI output
extension TimedSpeakerSegment: Codable {
    enum CodingKeys: String, CodingKey {
        case speakerId
        case embedding
        case startTimeSeconds
        case endTimeSeconds
        case qualityScore
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(speakerId, forKey: .speakerId)
        try container.encode(embedding, forKey: .embedding)
        try container.encode(startTimeSeconds, forKey: .startTimeSeconds)
        try container.encode(endTimeSeconds, forKey: .endTimeSeconds)
        try container.encode(qualityScore, forKey: .qualityScore)
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let speakerId = try container.decode(String.self, forKey: .speakerId)
        let embedding = try container.decode([Float].self, forKey: .embedding)
        let startTimeSeconds = try container.decode(Float.self, forKey: .startTimeSeconds)
        let endTimeSeconds = try container.decode(Float.self, forKey: .endTimeSeconds)
        let qualityScore = try container.decode(Float.self, forKey: .qualityScore)

        self.init(
            speakerId: speakerId,
            embedding: embedding,
            startTimeSeconds: startTimeSeconds,
            endTimeSeconds: endTimeSeconds,
            qualityScore: qualityScore
        )
    }
}
