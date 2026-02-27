# NCCL Tuner Configuration Scripts

This directory contains scripts for optimizing NCCL tuner configurations based on performance data.

## optimize_config.py

A Python script that reads performance data from CSV files and generates optimal NCCL tuner configurations.

### Usage

```bash
python scripts/optimize_config.py [options] <input_csv_file>
```

### Options

- `-o, --output FILE`: Output NCCL tuner config file (default: `nccl_tuner.conf`)
- `-m, --metric METRIC`: Optimization metric (`cost_metric`, `bandwidth_gbps`, `latency_us`)
- `--no-header`: Don't add header comments to output file
- `--dry-run`: Print configurations without writing to file

### CSV Input Format

The input CSV file should have the following columns:

```csv
collective,size_bytes,algorithm,protocol,channels,nodes,ranks,pipeOps,regBuff,cost_metric,bandwidth_gbps,latency_us
```

**Required columns:**
- `collective`: NCCL collective type (`allreduce`, `broadcast`, `reduce`, etc.)
- `size_bytes`: Message size in bytes
- `algorithm`: NCCL algorithm (`tree`, `ring`, `nvls`, etc.)
- `protocol`: NCCL protocol (`simple`, `ll`, `ll128`)
- `channels`: Number of channels (or `-1` for default)
- `nodes`: Number of nodes (or `-1` for any)
- `ranks`: Number of ranks (or `-1` for any)
- `pipeOps`: Number of pipeline operations (or `-1` for any)
- `regBuff`: Registered buffer flag (`0`, `1`, or `-1` for any)

**Optional metrics (must have at least one present):**
- `bandwidth_gbps`: Bandwidth in GB/s (higher is better)
- `latency_us`: Latency in microseconds (lower is better)

### Examples

**Basic usage with cost optimization:**
```bash
python scripts/optimize_config.py sample_performance_data.csv
```

**Optimize for bandwidth and write to custom file:**
```bash
python scripts/optimize_config.py -m bandwidth_gbps -o my_tuner.conf performance_data.csv
```

**Preview configurations without writing:**
```bash
python scripts/optimize_config.py --dry-run performance_data.csv
```

### How It Works

1. **Data Loading**: Reads CSV performance data and validates format
2. **Grouping**: Groups data by collective type, topology (nodes/ranks), and other parameters
3. **Size Ranges**: Automatically bins data into size ranges for optimization
4. **Optimization**: Finds the best performing configuration for each group/size combination
5. **Output**: Generates NCCL tuner config format and appends to specified file

### Default Size Ranges

The script uses these default size ranges (in bytes):
- Small: 0 - 1,024
- Medium: 1,025 - 65,536
- Large: 65,537 - 1,048,576
- XLarge: 1,048,577 - 16,777,216
- XXLarge: 16,777,217 - 4,294,967,295

### Sample Data

See `sample_performance_data.csv` for an example of the expected input format.

### Integration with NCCL

The generated configuration file can be used directly with the NCCL tuner plugin:

```bash
export NCCL_TUNER_CONFIG_FILE=/path/to/optimized_config.conf
export NCCL_TUNER_PLUGIN=/path/to/libnccl-tuner.so
mpirun -np 8 your_nccl_application
```

### Performance Data Collection

To collect performance data for optimization, you can:

1. **Use NCCL benchmarks** with different algorithm/protocol combinations
2. **Profile your applications** with various tuner settings
3. **Run systematic sweeps** across parameter combinations
4. **Use NCCL debug output** to collect timing information

The key is to have comprehensive data covering:
- Different message sizes (small to large)
- Various topologies (single node, multi-node)
- All relevant algorithm/protocol combinations
- Different channel counts and pipeline configurations
