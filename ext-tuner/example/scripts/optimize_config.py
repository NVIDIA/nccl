#!/usr/bin/env python3
"""
NCCL Tuner Configuration Optimizer

Reads a CSV file containing performance data across different tuning parameters
and generates optimal NCCL tuner configurations based on the best performing
combinations.

By default, creates growing size ranges that interpolate between the actual data sizes
for each unique dimension (node count, rank count combination). This ensures that
different cluster configurations get their own optimized size boundaries, as
performance characteristics often vary significantly between topologies.

Each dimension gets its own set of ranges starting from 0 and extending to the maximum
size for that dimension, with boundaries at midpoints between consecutive data sizes.

CSV Input Format:
collective,size_bytes,algorithm,protocol,channels,nodes,ranks,pipeOps,regBuff,bandwidth_gbps,latency_us

Output Format (NCCL Tuner Config):
collective_type,min_bytes,max_bytes,algorithm,protocol,channels,nNodes,nRanks,numPipeOps,regBuff

Usage Examples:
  # Auto-create dimension-specific interpolated ranges (default)
  python3 optimize_config.py data.csv

  # Use custom size ranges (applied to all topologies)
  python3 optimize_config.py data.csv --size-ranges "0-1024,1025-65536,65537-1048576"

  # Use hardcoded default ranges (applied to all topologies)
  python3 optimize_config.py data.csv --no-auto-ranges
"""

import csv
import argparse
import sys
import os
from collections import defaultdict
from typing import Dict, List, Tuple, Any

class PerformanceData:
    def __init__(self, row: Dict[str, str]):
        self.collective = row['collective']
        self.size_bytes = int(row['size_bytes'])
        self.algorithm = row['algorithm']
        self.protocol = row['protocol']
        self.channels = int(row['channels']) if row['channels'] != '-1' else -1
        self.nodes = int(row['nodes']) if row['nodes'] != '-1' else -1
        self.ranks = int(row['ranks']) if row['ranks'] != '-1' else -1
        self.pipeOps = int(row['pipeOps']) if row['pipeOps'] != '-1' else -1
        self.regBuff = int(row['regBuff']) if row['regBuff'] != '-1' else -1

        # Performance metrics
        self.bandwidth_gbps = float(row.get('bandwidth_gbps', 0))  # Higher is better
        self.latency_us = float(row.get('latency_us', 0))  # Lower is better

    def get_config_key(self) -> Tuple:
        """Generate a key for grouping similar configurations"""
        return (self.collective, self.nodes, self.ranks, self.pipeOps, self.regBuff)

    def get_size_range_key(self, topology_size_ranges: Dict[Tuple[int, int], List[Tuple[int, int]]]) -> Tuple[int, int]:
        """Find which size range this data point belongs to for its dimension"""
        topology_key = (self.nodes, self.ranks)

        # Get size ranges for this dimension, or fall back to default
        if topology_key in topology_size_ranges:
            size_ranges = topology_size_ranges[topology_key]
        elif (-1, -1) in topology_size_ranges:
            size_ranges = topology_size_ranges[(-1, -1)]
        else:
            # Fallback to first available dimension ranges
            size_ranges = next(iter(topology_size_ranges.values()))

        for min_size, max_size in size_ranges:
            if min_size <= self.size_bytes <= max_size:
                return (min_size, max_size)
        # If no range found, create a single-point range
        return (self.size_bytes, self.size_bytes)

class ConfigOptimizer:
    def __init__(self, optimization_metric: str = 'latency_us'):
        self.optimization_metric = optimization_metric
        # Default size ranges - will be overridden by auto-detection
        self.size_ranges = [
            (0, 1024),
            (1025, 64*1024),
            (64*1024+1, 1024*1024),
            (1024*1024+1, 16*1024*1024),
            (16*1024*1024+1, 4*1024*1024*1024-1)
        ]
        self.auto_size_ranges = True

    def set_size_ranges(self, ranges: List[Tuple[int, int]]):
        """Set custom size ranges for optimization"""
        self.size_ranges = ranges
        self.auto_size_ranges = False

    def auto_determine_size_ranges(self, data: List[PerformanceData]) -> Dict[Tuple[int, int], List[Tuple[int, int]]]:
        """Create growing size ranges for each unique (nodes, ranks) dimension"""
        if not data:
            return {(-1, -1): self.size_ranges}

        # Group data by dimension (nodes, ranks)
        topology_data = defaultdict(list)
        for item in data:
            topology_key = (item.nodes, item.ranks)
            topology_data[topology_key].append(item)

        topology_ranges = {}

        for topology_key, items in topology_data.items():
            nodes, ranks = topology_key

            # Extract unique sizes for this dimension and sort them
            unique_sizes = sorted(set(item.size_bytes for item in items))

            if len(unique_sizes) <= 1:
                # Only one size, create a single range from 0 to that size
                size = unique_sizes[0] if unique_sizes else 0
                ranges = [(0, size)]
            else:
                # Create growing ranges that interpolate between data points
                ranges = []

                for i, size in enumerate(unique_sizes):
                    if i == 0:
                        # First range: 0 to midpoint between first and second size
                        if len(unique_sizes) > 1:
                            next_size = unique_sizes[i + 1]
                            max_size = (size + next_size) // 2
                        else:
                            max_size = size
                        min_size = 0
                    elif i == len(unique_sizes) - 1:
                        # Last range: previous max + 1 to current size (and beyond)
                        min_size = ranges[-1][1] + 1
                        max_size = size
                    else:
                        # Intermediate ranges: previous max + 1 to midpoint with next size
                        min_size = ranges[-1][1] + 1
                        next_size = unique_sizes[i + 1]
                        max_size = (size + next_size) // 2

                    ranges.append((min_size, max_size))

            topology_ranges[topology_key] = ranges

            print(f"Dimension {nodes} nodes, {ranks} ranks: {len(ranges)} size ranges from {len(unique_sizes)} unique sizes:")
            for i, (min_size, max_size) in enumerate(ranges):
                # Count data points that fall in this range for this dimension
                count = sum(1 for item in items if min_size <= item.size_bytes <= max_size)
                actual_sizes = sorted(set(item.size_bytes for item in items if min_size <= item.size_bytes <= max_size))
                if actual_sizes:
                    size_list = ', '.join(f"{s:,}" for s in actual_sizes[:3])
                    if len(actual_sizes) > 3:
                        size_list += f", ... (+{len(actual_sizes)-3} more)"
                    print(f"  Range {i+1}: {min_size:,} - {max_size:,} bytes ({count} data points, sizes: {size_list})")

        return topology_ranges

    def load_data(self, csv_file: str) -> List[PerformanceData]:
        """Load performance data from CSV file"""
        data = []
        try:
            with open(csv_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        data.append(PerformanceData(row))
                    except (ValueError, KeyError) as e:
                        print(f"Warning: Skipping invalid row: {row} - {e}")
        except FileNotFoundError:
            print(f"Error: File {csv_file} not found")
            sys.exit(1)
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")
            sys.exit(1)

        print(f"Loaded {len(data)} performance data points")

        # Auto-determine size ranges if enabled
        if self.auto_size_ranges and data:
            self.topology_size_ranges = self.auto_determine_size_ranges(data)
        else:
            # Use default ranges for all topologies
            self.topology_size_ranges = {(-1, -1): self.size_ranges}

        return data

    def is_better(self, new_data: PerformanceData, current_best: PerformanceData) -> bool:
        """Determine if new_data is better than current_best"""
        if self.optimization_metric == 'bandwidth_gbps':
            return new_data.bandwidth_gbps > current_best.bandwidth_gbps
        elif self.optimization_metric == 'latency_us':
            return new_data.latency_us < current_best.latency_us
        else:
            # Default to latency
            return new_data.latency_us < current_best.latency_us

    def optimize_configurations(self, data: List[PerformanceData]) -> List[str]:
        """Find optimal configurations and return as NCCL config strings"""
        # Group data by configuration key and size range
        grouped_data = defaultdict(lambda: defaultdict(list))

        for item in data:
            config_key = item.get_config_key()
            size_range = item.get_size_range_key(self.topology_size_ranges)
            grouped_data[config_key][size_range].append(item)

        # Store optimal configurations before combining ranges
        optimal_configs = []

        for config_key, size_ranges_dict in grouped_data.items():
            collective, nodes, ranks, pipeOps, regBuff = config_key

            for (min_size, max_size), items in size_ranges_dict.items():
                if not items:
                    continue

                # Find the best performing configuration for this size range
                best_item = items[0]
                for item in items[1:]:
                    if self.is_better(item, best_item):
                        best_item = item

                # Store the optimal configuration with its range
                optimal_configs.append({
                    'collective': collective,
                    'min_size': min_size,
                    'max_size': max_size,
                    'algorithm': best_item.algorithm,
                    'protocol': best_item.protocol,
                    'channels': best_item.channels,
                    'nodes': best_item.nodes,
                    'ranks': best_item.ranks,
                    'pipeOps': best_item.pipeOps,
                    'regBuff': best_item.regBuff,
                    'metric_value': getattr(best_item, self.optimization_metric)
                })

        # Combine sequential ranges with identical tunings
        combined_configs = self.combine_sequential_ranges(optimal_configs)

        # Generate config strings
        configs = []
        for config in combined_configs:
            config_str = f"{config['collective']},{config['min_size']},{config['max_size']},{config['algorithm']},{config['protocol']},{config['channels']},{config['nodes']},{config['ranks']},{config['pipeOps']},{config['regBuff']}"
            configs.append(config_str)

            print(f"Optimal for {config['collective']} [{config['min_size']}-{config['max_size']}] nodes={config['nodes']} ranks={config['ranks']}: "
                  f"{config['algorithm']}/{config['protocol']} channels={config['channels']} "
                  f"({self.optimization_metric}={config['metric_value']:.3f})")

        return configs

    def combine_sequential_ranges(self, configs: List[Dict]) -> List[Dict]:
        """Combine sequential ranges that have identical tuning parameters"""
        if not configs:
            return configs

        # Group by collective and topology (nodes, ranks)
        topology_groups = defaultdict(list)
        for config in configs:
            topology_key = (config['collective'], config['nodes'], config['ranks'],
                          config['pipeOps'], config['regBuff'])
            topology_groups[topology_key].append(config)

        combined_configs = []

        for topology_key, topology_configs in topology_groups.items():
            # Sort by min_size to ensure proper ordering
            topology_configs.sort(key=lambda x: x['min_size'])

            # Group by tuning parameters (algorithm, protocol, channels)
            tuning_groups = defaultdict(list)
            for config in topology_configs:
                tuning_key = (config['algorithm'], config['protocol'], config['channels'])
                tuning_groups[tuning_key].append(config)

            # For each tuning group, combine sequential ranges
            for tuning_key, tuning_configs in tuning_groups.items():
                if not tuning_configs:
                    continue

                # Sort by min_size
                tuning_configs.sort(key=lambda x: x['min_size'])

                # Combine sequential ranges
                current_config = tuning_configs[0].copy()

                for next_config in tuning_configs[1:]:
                    # Check if ranges are adjacent or overlapping
                    if current_config['max_size'] + 1 >= next_config['min_size']:
                        # Extend the current range
                        current_config['max_size'] = max(current_config['max_size'], next_config['max_size'])
                        # Update metric value to the better one
                        if self.optimization_metric == 'bandwidth_gbps':
                            if next_config['metric_value'] > current_config['metric_value']:
                                current_config['metric_value'] = next_config['metric_value']
                        else:  # latency_us or default
                            if next_config['metric_value'] < current_config['metric_value']:
                                current_config['metric_value'] = next_config['metric_value']
                    else:
                        # Gap between ranges, save current and start new one
                        combined_configs.append(current_config)
                        current_config = next_config.copy()

                # Add the last configuration
                combined_configs.append(current_config)

        # Sort final configs by collective, nodes, ranks, then min_size
        combined_configs.sort(key=lambda x: (x['collective'], x['nodes'], x['ranks'], x['min_size']))

        original_count = len(configs)
        combined_count = len(combined_configs)
        if combined_count < original_count:
            print(f"Combined {original_count} ranges into {combined_count} ranges "
                  f"(reduced by {original_count - combined_count})")

        return combined_configs

    def append_to_config_file(self, configs: List[str], config_file: str, add_header: bool = True):
        """Append optimized configurations to NCCL tuner config file"""
        try:
            # Create directory if it doesn't exist
            config_dir = os.path.dirname(config_file)
            if config_dir and not os.path.exists(config_dir):
                os.makedirs(config_dir)
                print(f"Created directory: {config_dir}")

            # Check if file exists and has content
            file_exists = os.path.exists(config_file)
            add_separator = False

            if file_exists:
                with open(config_file, 'r') as f:
                    content = f.read().strip()
                    add_separator = len(content) > 0
                print(f"Appending to existing file: {config_file}")
            else:
                print(f"Creating new file: {config_file}")

            with open(config_file, 'a') as f:
                if add_separator:
                    f.write("\n\n")

                if add_header:
                    f.write(f"# Optimized configurations generated by optimize_config.py\n")
                    f.write(f"# Optimization metric: {self.optimization_metric}\n")
                    f.write(f"# Format: collective_type,min_bytes,max_bytes,algorithm,protocol,channels,nNodes,nRanks,numPipeOps,regBuff\n")

                for config in configs:
                    f.write(f"{config}\n")

            if file_exists:
                print(f"Appended {len(configs)} optimized configurations to {config_file}")
            else:
                print(f"Created {config_file} with {len(configs)} optimized configurations")

        except PermissionError:
            print(f"Error: Permission denied writing to {config_file}")
            print("Try running with appropriate permissions or choose a different output location")
            sys.exit(1)
        except OSError as e:
            print(f"Error: Cannot create/write to {config_file}: {e}")
            print("Check that the path is valid and you have write permissions")
            sys.exit(1)
        except Exception as e:
            print(f"Unexpected error writing to {config_file}: {e}")
            sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Optimize NCCL tuner configurations from performance data")
    parser.add_argument("csv_file", help="Input CSV file with performance data")
    parser.add_argument("-o", "--output", default="nccl_tuner.conf",
                       help="Output NCCL tuner config file (default: nccl_tuner.conf)")
    parser.add_argument("-m", "--metric", choices=['bandwidth_gbps', 'latency_us'],
                       default='latency_us', help="Optimization metric (default: latency_us)")
    parser.add_argument("--no-header", action="store_true",
                       help="Don't add header comments to output file")
    parser.add_argument("--dry-run", action="store_true",
                       help="Print configurations without writing to file")
    parser.add_argument("--no-auto-ranges", action="store_true",
                       help="Disable automatic size range determination (use default ranges)")
    parser.add_argument("--size-ranges", type=str,
                       help="Custom size ranges as comma-separated pairs: 'min1-max1,min2-max2,...'")

    args = parser.parse_args()

    optimizer = ConfigOptimizer(args.metric)

    # Handle size range configuration
    if args.size_ranges:
        # Parse custom size ranges
        try:
            ranges = []
            for range_str in args.size_ranges.split(','):
                min_size, max_size = map(int, range_str.split('-'))
                ranges.append((min_size, max_size))
            optimizer.set_size_ranges(ranges)
            print(f"Using custom size ranges: {ranges}")
        except ValueError:
            print("Error: Invalid size ranges format. Use 'min1-max1,min2-max2,...'")
            sys.exit(1)
    elif args.no_auto_ranges:
        # Disable auto-ranging
        optimizer.auto_size_ranges = False
        print("Using default hardcoded size ranges")
    else:
        # Auto-ranging is enabled by default - creates one bucket per unique size
        optimizer.auto_size_ranges = True
        print("Auto-ranging enabled: will create one bucket per unique size in data")

    # Load and optimize data
    data = optimizer.load_data(args.csv_file)
    if not data:
        print("No valid data found in CSV file")
        sys.exit(1)

    configs = optimizer.optimize_configurations(data)

    if args.dry_run:
        print("\nGenerated configurations:")
        for config in configs:
            print(config)
    else:
        optimizer.append_to_config_file(configs, args.output, not args.no_header)

if __name__ == "__main__":
    main()
