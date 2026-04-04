#!/usr/bin/env python3
"""
Script to analyze JSON instruction traces and compute the percentage of instructions 
that have at least one operand with ".reuse" in the operand string.
"""

import json
import sys
import argparse
from pathlib import Path


def analyze_reuse_instructions(json_file_path):
    """
    Analyze the JSON file to compute the percentage of instructions with reuse operands.
    
    Args:
        json_file_path (str): Path to the JSON file containing instruction traces
        
    Returns:
        dict: Analysis results containing counts and percentage
    """
    
    try:
        # Load the JSON file
        with open(json_file_path, 'r') as f:
            data = json.load(f)
        
        total_instructions = 0
        instructions_with_reuse = 0
        
        # Check if the JSON has the expected structure
        if 'kernels' not in data:
            raise ValueError("JSON file does not contain 'kernels' field")
        
        # Iterate through all kernels
        for kernel in data['kernels']:
            if 'instructions' not in kernel:
                continue
                
            # Iterate through all instructions in the kernel
            for instruction in kernel['instructions']:
                total_instructions += 1
                
                # Check if this instruction has operands
                if 'operands' not in instruction:
                    continue
                
                # Check if any operand contains ".reuse"
                has_reuse_operand = False
                for operand in instruction['operands']:
                    if 'operand_string' in operand:
                        operand_string = operand['operand_string']
                        if '.reuse' in operand_string:
                            has_reuse_operand = True
                            break
                
                if has_reuse_operand:
                    instructions_with_reuse += 1
        
        # Calculate percentage
        if total_instructions > 0:
            percentage = (instructions_with_reuse / total_instructions) * 100
        else:
            percentage = 0.0
        
        return {
            'total_instructions': total_instructions,
            'instructions_with_reuse': instructions_with_reuse,
            'percentage': percentage
        }
        
    except FileNotFoundError:
        raise FileNotFoundError(f"JSON file not found: {json_file_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON file: {e}")
    except Exception as e:
        raise Exception(f"Error analyzing file: {e}")


def main():
    """Main function to handle command line arguments and run the analysis."""
    
    parser = argparse.ArgumentParser(
        description="Analyze instruction traces to compute percentage of instructions with reuse operands"
    )
    parser.add_argument(
        'json_file', 
        help='Path to the JSON file containing instruction traces'
    )
    parser.add_argument(
        '-v', '--verbose', 
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    # Check if file exists
    json_path = Path(args.json_file)
    if not json_path.exists():
        print(f"Error: File '{args.json_file}' does not exist.")
        sys.exit(1)
    
    try:
        # Run the analysis
        print(f"Analyzing instruction traces in: {args.json_file}")
        if args.verbose:
            print("Loading and parsing JSON file...")
        
        results = analyze_reuse_instructions(args.json_file)
        
        # Display results
        print("\n" + "="*60)
        print("INSTRUCTION REUSE ANALYSIS RESULTS")
        print("="*60)
        print(f"Total instructions analyzed: {results['total_instructions']:,}")
        print(f"Instructions with .reuse operands: {results['instructions_with_reuse']:,}")
        print(f"Percentage with .reuse operands: {results['percentage']:.2f}%")
        print("="*60)
        
        if args.verbose:
            print(f"\nInstructions without .reuse operands: {results['total_instructions'] - results['instructions_with_reuse']:,}")
            if results['total_instructions'] > 0:
                non_reuse_percentage = 100 - results['percentage']
                print(f"Percentage without .reuse operands: {non_reuse_percentage:.2f}%")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
