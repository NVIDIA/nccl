# NCCL Inspector Performance Summary Exporter

This tool processes NCCL Inspector log files and generates comprehensive performance analysis reports including visualizations and statistical summaries.
One can build similar exporters to integrate with various observability systems like Elastic, Prometheus or other Custom Metric systems.

## Features

- **Performance Analysis**: Generates statistical summaries for collective operations
- **Communication Type Classification**: Automatically categorizes communication patterns
- **Visualizations**: Creates scatter plots, histograms, and box plots for performance metrics
- **Data Export**: Converts logs to Parquet format for efficient processing
- **Multi-format Log Support**: Processes `.log`, `.log.gz`, `.jsonl`, and `.jsonl.gz` files
- **Parallel Processing**: Utilizes multi-core processing for faster analysis

## Requirements

- Python 3.7+
- Access to NCCL Inspector log files

## Installation

### Clone the Repository

```bash
git clone https://github.com/NVIDIA/nccl.git
cd nccl/ext-profiler/inspector/exporter/example
```

Install the required dependencies using the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

## Usage

The script processes NCCL Inspector log files from a specified directory.

**Note:** To generate NCCL Inspector log files, you need to run your NCCL application with the inspector plugin enabled. The log files will be output to a directory specified by the `NCCL_INSPECTOR_DUMP_DIR` environment variable. For detailed setup instructions and environment variable configuration, see the [Inspector README](../../../README.md).

### Basic Usage

```bash
python perf_summary_exporter.py --input_dir /path/to/nccl/inspector/logs
```

This mode processes all log files in the specified directory and its subdirectories recursively.

### Command Line Arguments

- `--input_dir <path>`: **Required**. Directory containing NCCL Inspector log files (searches recursively in subdirectories)
- `--output_dir <name>`: **Optional**. Custom output directory name (default: `<input_directory_name>-analysis`)

## Output

The tool generates:

1. **Parquet Files**: One per log file containing processed log data (stored in `parquet_files/` subdirectory)
2. **Summary Directory**: Contains comprehensive analysis results
3. **Visualizations**: Scatter plots, histograms, and box plots for each message size
4. **CSV Files**: Detailed summaries for each message size and collective type
5. **Log File**: Processing log with detailed information

## Example Output Structure

```
<output_dir_name>/
├── output.log
├── parquet_files/
│   ├── <filename1>.parquet
│   ├── <filename2>.parquet
│   └── ...
└── summary/
    ├── scatter_plot_<comm_type>_<coll_type>.png
    ├── combined_scatter_plot_<comm_type>_<coll_type>.png
    └── msg_size_<human_readable_size>/
        ├── histograms/
        │   └── histogram_<comm_type>_<coll_type>_<size>.png
        ├── boxplots/
        │   └── boxplot_<comm_type>_<coll_type>_<size>.png
        └── summary_<comm_type>_<coll_type>_<size>.csv
```

## Supported Communicator Types

- `single-rank`
- `nvlink-only`
- `hca-only`
- `mixed`

## Supported Collective Types

- `AllReduce`
- `AllGather`
- `ReduceScatter`
- `Broadcast`

## Log File Formats

### Supported Formats

- `.log` - Plain text JSON lines
- `.log.gz` - Compressed JSON lines
- `.jsonl` - JSON lines format
- `.jsonl.gz` - Compressed JSON lines

### Expected JSON Structure

```json
{
  "header": {
    "id": "0x9e7a479f95a66c",
    "rank": 31,
    "n_ranks": 32,
    "nnodes": 4
  },
  "metadata": {
    "inspector_output_format_version": "v4.0",
    "git_rev": "75e61acda-dirty",
    "rec_mechanism": "nccl_profiler_interface",
    "dump_timestamp_us": 1749490229087081,
    "hostname": "example-hostname",
    "pid": 468528
  },
  "coll_perf": {
    "coll": "ReduceScatter",
    "coll_sn": 129,
    "coll_msg_size_bytes": 65536,
    "coll_exec_time_us": 110,
    "coll_timing_source": "kernel_gpu",
    "coll_algobw_gbs": 19.065018,
    "coll_busbw_gbs": 18.469236
  }
}
```

## Troubleshooting

### Common Issues

1. **No log files found**: Ensure the log directory path is correct and contains valid log files
2. **Missing dependencies**: Ensure all requirements are installed in your virtual environment
3. **Mixed file formats**: The tool will exit if it detects mixed `.log`, `.log.gz`, `.jsonl`, and `.jsonl.gz` files in the same directory. This is typically indicative of corrupt input directories caused by multiple overlapping NCCL Inspector runs with different output format options. Clean the directory and re-run with consistent settings.

### Log Files

The tool creates detailed logs in the output directory. Check `output.log` for processing information and any error messages.

## Support

Please refer to the github issues page at https://github.com/NVIDIA/nccl/issues. Your question may already have been asked by another user. If not, feel free to create a new issue and refer to the "inspector plugin" in the title.
