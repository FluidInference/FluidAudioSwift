#!/usr/bin/env python3
"""
Combine multiple JSON benchmark results from MetalAccelerationBenchmarks into a single valid JSON structure.

The Swift test suite outputs multiple JSON blocks separated by markers, one for each test method.
This script combines them into a single coherent JSON result.
"""

import json
import sys
import argparse
from typing import List, Dict, Any, Optional


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
            print(f"Warning: Failed to parse JSON block {i}: {e}", file=sys.stderr)
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


def main():
    parser = argparse.ArgumentParser(description="Combine benchmark JSON results")
    parser.add_argument("input_file", nargs="?", help="Input file with raw JSON blocks (default: stdin)")
    parser.add_argument("-o", "--output", help="Output file (default: stdout)")
    parser.add_argument("--pretty", action="store_true", help="Pretty print JSON output")
    
    args = parser.parse_args()
    
    # Read input
    if args.input_file:
        try:
            with open(args.input_file, 'r') as f:
                raw_content = f.read()
        except FileNotFoundError:
            print(f"Error: Input file '{args.input_file}' not found", file=sys.stderr)
            sys.exit(1)
        except IOError as e:
            print(f"Error reading input file: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        raw_content = sys.stdin.read()
    
    if not raw_content.strip():
        print("Error: No input data provided", file=sys.stderr)
        sys.exit(1)
    
    # Parse and combine
    parsed_blocks = parse_json_blocks(raw_content)
    
    if not parsed_blocks:
        print("Error: No valid JSON blocks found in input", file=sys.stderr)
        sys.exit(1)
    
    combined_result = combine_benchmark_results(parsed_blocks)
    
    if combined_result is None:
        print("Error: Failed to combine benchmark results", file=sys.stderr)
        sys.exit(1)
    
    # Output result
    json_options = {"indent": 2} if args.pretty else {}
    json_output = json.dumps(combined_result, **json_options)
    
    if args.output:
        try:
            with open(args.output, 'w') as f:
                f.write(json_output)
        except IOError as e:
            print(f"Error writing output file: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        print(json_output)
    
    # At this point combined_result is guaranteed to not be None
    num_tests = len(combined_result['tests']) if combined_result else 0
    print(f"Successfully combined {len(parsed_blocks)} JSON blocks into {num_tests} tests", file=sys.stderr)


if __name__ == "__main__":
    main() 