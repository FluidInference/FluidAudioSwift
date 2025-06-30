# FluidAudioSwift Documentation

Welcome to the FluidAudioSwift documentation! This directory contains comprehensive guides and technical documentation for the FluidAudioSwift framework.

## Documentation Overview

### 📚 User Guides

- **[Getting Started](../README.md)** - Quick start guide and basic usage examples
- **[CLI Documentation](CLI.md)** - Complete command-line interface guide for benchmarking and audio processing
- **[Performance & Benchmarking](BENCHMARKING.md)** - Complete guide to benchmarking system and performance optimization
- **[Examples & Use Cases](EXAMPLES.md)** - Practical examples and integration scripts

### 🔧 Technical Documentation

- **[Metal Acceleration](METAL_ACCELERATION.md)** - Deep dive into Metal Performance Shaders integration and GPU optimization
- **[Project Documentation](../CLAUDE.md)** - Development guidelines and project structure

## Quick Navigation

### For Users
- Want to **get started quickly**? → [README.md](../README.md#quick-start)
- Need to **run benchmarks**? → [CLI.md](CLI.md#benchmark-command)
- Want to **process audio files**? → [CLI.md](CLI.md#process-command)
- Need to **optimize performance**? → [BENCHMARKING.md](BENCHMARKING.md#performance-optimization)
- Looking for **practical examples**? → [EXAMPLES.md](EXAMPLES.md)
- Looking for **configuration options**? → [README.md](../README.md#configuration)

### For Developers
- Understanding **Metal implementation**? → [METAL_ACCELERATION.md](METAL_ACCELERATION.md#metal-implementation)
- Contributing **performance improvements**? → [BENCHMARKING.md](BENCHMARKING.md#ci-integration)
- Working on **platform optimization**? → [METAL_ACCELERATION.md](METAL_ACCELERATION.md#platform-considerations)

### For Researchers
- Need **AMI corpus evaluation**? → [CLI.md](CLI.md#ami-dataset-setup)
- Want **research-standard metrics**? → [CLI.md](CLI.md#performance-metrics)
- Looking for **batch evaluation scripts**? → [EXAMPLES.md](EXAMPLES.md#research-benchmarking)

### For DevOps/CI
- Setting up **automated benchmarks**? → [BENCHMARKING.md](BENCHMARKING.md#ci-integration)
- Need **CLI integration**? → [EXAMPLES.md](EXAMPLES.md#integration-examples)
- Monitoring **performance regressions**? → [BENCHMARKING.md](BENCHMARKING.md#understanding-results)
- Troubleshooting **CI issues**? → [BENCHMARKING.md](BENCHMARKING.md#troubleshooting)

## Key Features Covered

### 🚀 Performance Optimization
- **Metal GPU acceleration** with 3-8x speedup
- **Automatic fallback** to Accelerate framework
- **Batch size optimization** for different workloads
- **Memory efficiency** improvements

### 📊 Benchmarking System
- **Comprehensive test suite** covering all major operations
- **Research-standard evaluation** on AMI Meeting Corpus
- **Command-line interface** for easy benchmarking
- **CI integration** with automated PR comments
- **Performance regression detection**
- **Hardware-specific optimization guidance**

### 🔧 Advanced Configuration
- **Thermal management** for sustained performance
- **Battery-aware processing** for mobile devices
- **Platform-specific optimizations** for iOS/macOS
- **Dynamic backend selection**

## Document Index

| Document | Purpose | Audience | Length |
|----------|---------|----------|---------|
| [CLI.md](CLI.md) | Command-line interface usage | Users, Researchers | ~500+ lines |
| [EXAMPLES.md](EXAMPLES.md) | Practical examples and scripts | All users | ~400+ lines |
| [BENCHMARKING.md](BENCHMARKING.md) | Performance testing and optimization | All users | ~500+ lines |
| [METAL_ACCELERATION.md](METAL_ACCELERATION.md) | Technical Metal implementation details | Developers | ~555 lines |
| [README.md](../README.md) | Quick start and basic usage | All users | ~100 lines |
| [CLAUDE.md](../CLAUDE.md) | Development guidelines | Contributors | ~175 lines |

## Contributing to Documentation

We welcome contributions to improve our documentation! When contributing:

1. **Check existing docs** to avoid duplication
2. **Follow markdown best practices** for consistency
3. **Include code examples** where helpful
4. **Test all links** and references
5. **Update this index** when adding new documents

### Documentation Standards

- Use **clear, concise language**
- Include **practical examples** and code snippets
- Provide **cross-references** between related sections
- Add **table of contents** for longer documents
- Include **troubleshooting sections** for complex topics

## Support

- **Issues**: Report documentation issues on [GitHub Issues](https://github.com/FluidInference/FluidAudioSwift/issues)
- **Discussions**: Join conversations on [GitHub Discussions](https://github.com/FluidInference/FluidAudioSwift/discussions)
- **Contributions**: Submit improvements via [Pull Requests](https://github.com/FluidInference/FluidAudioSwift/pulls)

---

*Last updated: {{ date }}*