#!/bin/bash

# FluidAudioSwift Metal Acceleration Benchmarks Runner
# This script runs comprehensive benchmarks and generates a report

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 FluidAudioSwift Metal Acceleration Benchmarks${NC}"
echo "=================================================="

# Check if we're in the right directory
if [ ! -f "Package.swift" ]; then
    echo -e "${RED}Error: Please run this script from the FluidAudioSwift root directory${NC}"
    exit 1
fi

# Build the package
echo -e "${YELLOW}📦 Building package...${NC}"
swift build --configuration release

echo -e "${YELLOW}🔬 Running Metal acceleration benchmarks...${NC}"
echo "This may take several minutes..."

# Run benchmarks and capture output
BENCHMARK_OUTPUT=$(swift test --filter MetalAccelerationBenchmarks --configuration release 2>&1)

# Extract JSON results from multiple test methods and combine them
JSON_BLOCKS=$(echo "$BENCHMARK_OUTPUT" | awk '/🔬 BENCHMARK_RESULTS_JSON_START/,/🔬 BENCHMARK_RESULTS_JSON_END/ {if ($0 !~ /🔬/) print}')

if [ -z "$JSON_BLOCKS" ]; then
    echo -e "${RED}❌ No benchmark results found. Check the test output above for errors.${NC}"
    echo "$BENCHMARK_OUTPUT"
    exit 1
fi

# Use our Python script to combine the JSON blocks
JSON_RESULTS=$(echo "$JSON_BLOCKS" | python3 scripts/combine_benchmark_json.py --pretty)

if [ $? -ne 0 ] || [ -z "$JSON_RESULTS" ]; then
    echo -e "${RED}❌ Failed to parse benchmark results. Raw output:${NC}"
    echo "$JSON_BLOCKS"
    exit 1
fi

echo -e "${GREEN}✅ Benchmarks completed successfully!${NC}"

# Save results to file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_FILE="benchmark_results_${TIMESTAMP}.json"
echo "$JSON_RESULTS" > "$RESULTS_FILE"

echo -e "${BLUE}📊 Benchmark Results Summary:${NC}"
echo "==============================="

# Parse and display summary using Python
python3 << EOF
import json
import sys

try:
    results = json.loads('''$JSON_RESULTS''')
    
    metal_available = results.get('metal_available', False)
    tests = results.get('tests', [])
    
    if not metal_available:
        print("ℹ️  Metal Performance Shaders not available on this device")
        print("   Benchmarks show Accelerate-only performance")
    else:
        print("✅ Metal Performance Shaders available")
    
    print(f"🕐 Timestamp: {results.get('timestamp', 'N/A')}")
    print(f"📈 Total tests run: {len(tests)}")
    
    if tests:
        # Calculate overall statistics
        speedups = [t.get('speedup', 0) for t in tests if t.get('speedup', 0) > 0]
        
        if speedups:
            avg_speedup = sum(speedups) / len(speedups)
            max_speedup = max(speedups)
            
            print(f"⚡ Average speedup: {avg_speedup:.2f}x")
            print(f"🚀 Best speedup: {max_speedup:.2f}x")
            
            if avg_speedup >= 2.0:
                print("🎉 Excellent Metal acceleration performance!")
            elif avg_speedup >= 1.5:
                print("✅ Good Metal acceleration benefits")
            elif avg_speedup >= 1.0:
                print("⚠️  Moderate Metal acceleration benefits")
            else:
                print("⚠️  Metal overhead detected (expected for small operations)")
        
        # Show test breakdown
        test_types = {}
        for test in tests:
            test_type = test.get('test_type', 'unknown')
            if test_type not in test_types:
                test_types[test_type] = []
            test_types[test_type].append(test)
        
        print("\n📋 Test Breakdown:")
        for test_type, type_tests in test_types.items():
            count = len(type_tests)
            avg_speedup = sum(t.get('speedup', 0) for t in type_tests) / count if count > 0 else 0
            print(f"   • {test_type.replace('_', ' ').title()}: {count} tests, {avg_speedup:.2f}x avg speedup")

except json.JSONDecodeError as e:
    print(f"Error parsing benchmark results: {e}")
    sys.exit(1)
except Exception as e:
    print(f"Error processing results: {e}")
    sys.exit(1)
EOF

echo
echo -e "${BLUE}📁 Full results saved to: ${RESULTS_FILE}${NC}"
echo -e "${YELLOW}💡 Tip: Use 'jq' to explore the JSON results in detail:${NC}"
echo "   cat $RESULTS_FILE | jq '.tests[] | select(.test_type == \"cosine_distance\")'"

echo
echo -e "${GREEN}🎯 Benchmark run complete!${NC}"