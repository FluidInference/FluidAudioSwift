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
                benchmark    Run AMI SDM benchmark evaluation with real annotations
                process      Process a single audio file
                download     Download datasets for benchmarking
                help         Show this help message

            BENCHMARK OPTIONS:
                --dataset <name>        Dataset to use (ami-sdm, ami-ihm) [default: ami-sdm]
                --threshold <float>     Clustering threshold 0.0-1.0 [default: 0.7]
                --min-duration-on <float>   Minimum speaker segment duration in seconds [default: 1.0]
                --min-duration-off <float>  Minimum silence between speakers in seconds [default: 0.5]
                --min-activity <float>      Minimum activity threshold in frames [default: 10.0]
                --single-file <name>    Test only one specific meeting file (e.g., ES2004a)
                --debug                 Enable debug mode
                --output <file>         Output results to JSON file
                --auto-download         Automatically download dataset if not found

            NOTE: Benchmark now uses real AMI manual annotations from Tests/ami_public_1.6.2/
                  If annotations are not found, falls back to simplified placeholder.

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
        var minDurationOn: Float = 1.0
        var minDurationOff: Float = 0.5
        var minActivityThreshold: Float = 10.0
        var singleFile: String?
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
            case "--min-duration-on":
                if i + 1 < arguments.count {
                    minDurationOn = Float(arguments[i + 1]) ?? 1.0
                    i += 1
                }
            case "--min-duration-off":
                if i + 1 < arguments.count {
                    minDurationOff = Float(arguments[i + 1]) ?? 0.5
                    i += 1
                }
            case "--min-activity":
                if i + 1 < arguments.count {
                    minActivityThreshold = Float(arguments[i + 1]) ?? 10.0
                    i += 1
                }
            case "--single-file":
                if i + 1 < arguments.count {
                    singleFile = arguments[i + 1]
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
        print("   Min duration on: \(minDurationOn)s")
        print("   Min duration off: \(minDurationOff)s")
        print("   Min activity threshold: \(minActivityThreshold)")
        print("   Debug mode: \(debugMode ? "enabled" : "disabled")")
        print("   Auto-download: \(autoDownload ? "enabled" : "disabled")")

        let config = DiarizerConfig(
            clusteringThreshold: threshold,
            minDurationOn: minDurationOn,
            minDurationOff: minDurationOff,
            minActivityThreshold: minActivityThreshold,
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
            await runAMIBenchmark(
                manager: manager, outputFile: outputFile, autoDownload: autoDownload, singleFile: singleFile, variant: .sdm)
        case "ami-ihm":
            await runAMIBenchmark(
                manager: manager, outputFile: outputFile, autoDownload: autoDownload, singleFile: singleFile, variant: .ihm)
        default:
            print("❌ Unsupported dataset: \(dataset)")
            print("💡 Supported datasets: ami-sdm, ami-ihm")
            exit(1)
        }
    }

    static func runAMIBenchmark(
        manager: DiarizerManager, outputFile: String?, autoDownload: Bool, singleFile: String?, variant: AMIVariant
    ) async {
        let homeDir = FileManager.default.homeDirectoryForCurrentUser
        let amiDirectory = homeDir.appendingPathComponent(
            "FluidAudioSwift_Datasets/ami_official/\(variant.rawValue)")

        // Check if AMI dataset exists, download if needed
        if !FileManager.default.fileExists(atPath: amiDirectory.path) {
            if autoDownload {
                print("📥 AMI \(variant.displayName) dataset not found - downloading automatically...")
                await downloadAMIDataset(variant: variant, force: false)

                // Check again after download
                if !FileManager.default.fileExists(atPath: amiDirectory.path) {
                    print("❌ Failed to download AMI \(variant.displayName) dataset")
                    return
                }
            } else {
                print("⚠️ AMI \(variant.displayName) dataset not found")
                print("📥 Download options:")
                print("   Option 1: Use --auto-download flag")
                print("   Option 2: Download manually:")
                print("      1. Visit: https://groups.inf.ed.ac.uk/ami/download/")
                print("      2. Select test meetings: ES2002a, ES2003a, ES2004a, IS1000a, IS1001a")
                print("      3. Download '\(variant.filePattern)' files")
                print("      4. Place files in: \(amiDirectory.path)")
                print("   Option 3: Use download command:")
                print("      swift run fluidaudio download --dataset ami-\(variant.rawValue)")
                return
            }
        }

        let commonMeetings: [String]
        if let singleFile = singleFile {
            commonMeetings = [singleFile]
            print("📋 Testing single file: \(singleFile)")
        } else {
            commonMeetings = [
                "ES2002a", "ES2003a", "ES2004a", "ES2005a",
                "IS1000a", "IS1001a", "IS1002b",
                "TS3003a", "TS3004a",
            ]
        }

        var benchmarkResults: [BenchmarkResult] = []
        var totalDER: Float = 0.0
        var totalJER: Float = 0.0
        var processedFiles = 0

        print("📊 Running AMI \(variant.displayName) Benchmark")
        print("   Looking for \(variant.filePattern) files in: \(amiDirectory.path)")

        for meetingId in commonMeetings {
            let audioFileName = "\(meetingId).\(variant.filePattern)"
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

                // Load ground truth from AMI annotations if available, else fallback
                let groundTruth = await Self.loadAMIGroundTruth(for: meetingId, duration: duration)

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

        printBenchmarkResults(benchmarkResults, avgDER: avgDER, avgJER: avgJER, dataset: "AMI-\(variant.displayName)")

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

            do {
                try await saveBenchmarkResults(summary, to: outputFile)
                print("💾 Benchmark results saved to: \(outputFile)")
            } catch {
                print("⚠️ Failed to save results: \(error)")
            }
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

// (Retain all data structures, AMIAnnotationParser, findOptimalSpeakerMapping, printBenchmarkResults, etc. as in your current file)
