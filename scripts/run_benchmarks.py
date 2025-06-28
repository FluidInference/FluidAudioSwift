#!/usr/bin/env python3
"""
FluidAudioSwift Metal Acceleration Benchmarks Runner

This script runs comprehensive benchmarks and generates a report.
Converted from bash script to Python with uv support.
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

try:
    from colorama import Fore, Style, init
    init(autoreset=True)
except ImportError:
    # Fallback if colorama is not available
    class Fore:
        RED = GREEN = YELLOW = BLUE = ""
    
    class Style:
        RESET_ALL = ""


def print_colored(message: str, color: str = "") -> None:
    """Print a colored message."""
    print(f"{color}{message}{Style.RESET_ALL}")


def run_command(cmd: List[str], capture_output: bool = True, check: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command and return the result."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=capture_output,
            text=True,
            check=check
        )
        return result
    except subprocess.CalledProcessError as e:
        print_colored(f"❌ Command failed: {' '.join(cmd)}", Fore.RED)
        print_colored(f"Error: {e}", Fore.RED)
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
        raise


def parse_json_blocks(raw_content: str) -> List[Dict[str, Any]]:
    """Parse multiple JSON blocks from raw content, handling split boundaries."""
    # Split on the boundary between JSON objects
    blocks = raw_content.strip().split('\n}\n{')
    
    if not blocks:
        return []
    
    parsed_blocks = []
    for i, block in enumerate(blocks):
        try:
            # Add back the braces that were removed by splitting
            if i == 0:
                # First block - add closing brace if needed
                if not block.strip().endswith('}'):
                    block += '\n}'
            elif i == len(blocks) - 1:
                # Last block - add opening brace if needed
                if not block.strip().startswith('{'):
                    block = '{\n' + block
            else:
                # Middle blocks - add both braces
                if not block.strip().startswith('{'):
                    block = '{\n' + block
                if not block.strip().endswith('}'):
                    block += '\n}'
            
            parsed = json.loads(block)
            parsed_blocks.append(parsed)
        except json.JSONDecodeError as e:
            print_colored(f"Warning: Failed to parse JSON block {i}: {e}", Fore.YELLOW)
            continue
    
    return parsed_blocks


def combine_benchmark_results(parsed_blocks: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Combine multiple benchmark result blocks into a single JSON structure."""
    if not parsed_blocks:
        return None
    
    # Use the first block as the base structure
    combined_result = {
        "timestamp": parsed_blocks[0].get("timestamp", ""),
        "metal_available": parsed_blocks[0].get("metal_available", False),
        "tests": []
    }
    
    # Collect all tests from all blocks
    for block in parsed_blocks:
        if "tests" in block and isinstance(block["tests"], list):
            combined_result["tests"].extend(block["tests"])
    
    return combined_result


def extract_json_results(benchmark_output: str) -> str:
    """Extract JSON results from benchmark output."""
    lines = benchmark_output.split('\n')
    json_lines = []
    in_json_block = False
    
    for line in lines:
        if '🔬 BENCHMARK_RESULTS_JSON_START' in line:
            in_json_block = True
            continue
        elif '🔬 BENCHMARK_RESULTS_JSON_END' in line:
            in_json_block = False
            continue
        elif in_json_block and '🔬' not in line:
            json_lines.append(line)
    
    return '\n'.join(json_lines)


def display_summary(results: Dict[str, Any]) -> None:
    """Display a formatted summary of benchmark results."""
    print_colored("📊 Benchmark Results Summary:", Fore.BLUE)
    print_colored("===============================", Fore.BLUE)
    
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


def main() -> int:
    """Main entry point for the benchmark runner."""
    print_colored("🚀 FluidAudioSwift Metal Acceleration Benchmarks", Fore.BLUE)
    print_colored("==================================================", Fore.BLUE)
    
    # Find the project root directory
    current_dir = Path.cwd()
    project_root = None
    
    # Check current directory first
    if (current_dir / "Package.swift").exists():
        project_root = current_dir
    # Check parent directory (in case we're in scripts/)
    elif (current_dir.parent / "Package.swift").exists():
        project_root = current_dir.parent
        os.chdir(project_root)
        print_colored(f"📁 Changed directory to project root: {project_root}", Fore.YELLOW)
    else:
        print_colored("Error: Could not find Package.swift. Please run from FluidAudioSwift root or scripts directory", Fore.RED)
        return 1
    
    try:
        # Build the package
        print_colored("📦 Building package...", Fore.YELLOW)
        run_command(["swift", "build", "--configuration", "release"], capture_output=False)
        
        print_colored("🔬 Running Metal acceleration benchmarks...", Fore.YELLOW)
        print("This may take several minutes...")
        
        # Run benchmarks and capture output
        result = run_command([
            "swift", "test", 
            "--filter", "MetalAccelerationBenchmarks", 
            "--configuration", "release"
        ])
        
        benchmark_output = result.stdout + result.stderr
        
        # Extract JSON results
        json_blocks_raw = extract_json_results(benchmark_output)
        
        if not json_blocks_raw.strip():
            print_colored("❌ No benchmark results found. Check the test output above for errors.", Fore.RED)
            print(benchmark_output)
            return 1
        
        # Parse and combine JSON results
        parsed_blocks = parse_json_blocks(json_blocks_raw)
        
        if not parsed_blocks:
            print_colored("❌ Failed to parse benchmark results. Raw output:", Fore.RED)
            print(json_blocks_raw)
            return 1
        
        combined_results = combine_benchmark_results(parsed_blocks)
        
        if combined_results is None:
            print_colored("❌ Failed to combine benchmark results.", Fore.RED)
            return 1
        
        print_colored("✅ Benchmarks completed successfully!", Fore.GREEN)
        
        # Save results to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"benchmark_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(combined_results, f, indent=2)
        
        # Display summary
        display_summary(combined_results)
        
        print()
        print_colored(f"📁 Full results saved to: {results_file}", Fore.BLUE)
        print_colored("💡 Tip: Use 'jq' to explore the JSON results in detail:", Fore.YELLOW)
        print(f"   cat {results_file} | jq '.tests[] | select(.test_type == \"cosine_distance\")'")
        
        print()
        print_colored("🎯 Benchmark run complete!", Fore.GREEN)
        
        return 0
        
    except subprocess.CalledProcessError:
        return 1
    except KeyboardInterrupt:
        print_colored("\n❌ Benchmark run interrupted by user", Fore.RED)
        return 1
    except Exception as e:
        print_colored(f"❌ Unexpected error: {e}", Fore.RED)
        return 1


if __name__ == "__main__":
    sys.exit(main()) 