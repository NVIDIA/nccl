# NCCL Inspector Performance Summary Exporter

This tool processes NCCL Inspector log files and generates comprehensive performance analysis reports including visualizations and statistical summaries. It supports both local analysis and data upload to Kibana for centralized monitoring.

## Features

- **Performance Analysis**: Generates statistical summaries for collective operations
- **Communication Type Classification**: Automatically categorizes communication patterns
- **Visualizations**: Creates scatter plots, histograms, and box plots for performance metrics
- **Data Export**: Converts logs to Parquet format for efficient processing
- **Kibana Integration**: Optional upload to Kibana for centralized monitoring
- **Multi-format Log Support**: Processes `.log`, `.log.gz`, `.jsonl`, and `.jsonl.gz` files
- **Parallel Processing**: Utilizes multi-core processing for faster analysis

## Requirements

- Python 3.7+
- Access to NCCL Inspector log files
- Optional: Kibana access for data upload

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd nccl/ext-profiler/inspector/exporter/elastic
```

### 2. Create Python Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate
```

### 3. Install Dependencies

#### 3.1 Install Standard Dependencies

Install the required dependencies using the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

#### 3.2 Install PySLURM

Install pyslurm from [pyslurm-prebuilds](https://gitlab-master.nvidia.com/fact/pyslurm-prebuilds)

```bash
git clone ssh://git@gitlab-master.nvidia.com:12051/fact/pyslurm-prebuilds.git
pip3 install ./pyslurm-prebuilds/pyslurm-23.2.2-cp310-cp310-linux_x86_64.whl
```

#### 3.3 Install nvdataflow

Install nvdataflow from NVIDIA artifactory:

```bash
python3 -m pip install --index-url=https://sc-hw-artf.nvidia.com/artifactory/api/pypi/hwinf-gpuwa-pypi/simple nvdataflow
```

#### 3.4 Verify Installation

Verify core dependencies:

```bash
python -c "import pandas, tqdm, duckdb, matplotlib, pyslurm, nvdataflow; print('All dependencies installed successfully')"
```

## Usage

The script supports two main modes of operation:

### Mode 1: Using Job ID (Recommended)

```bash
python perf_summary_exporter.py --jobid <job_id> [--upload]
```

This mode automatically constructs the log file path using the standard NCCL Inspector directory structure.

### Mode 2: Using Custom Directory

```bash
python perf_summary_exporter.py --job_insp_rootdir /path/to/logs
```

This mode allows you to specify a custom directory containing log files.

### Command Line Arguments

- `--jobid <job_id>`: SLURM job ID for automatic path construction
- `--job_insp_rootdir <path>`: Custom root directory containing log files
- `--upload`: Optional flag to upload results to Kibana (requires `--jobid`)
- `--config <path>`: Path to configuration file (default: `perf_summary_exporter_config.ini` in script directory)

**Note**: Either `--jobid` or `--job_insp_rootdir` must be provided. The `--upload` option can only be used with `--jobid` as it requires valid SLURM job metadata.

## Configuration

The tool supports configuration through an INI file to customize default paths and Kibana settings. If no config file is specified, the tool will look for `perf_summary_exporter_config.ini` in the same directory as the script.

### Sample Configuration File

Create a file named `perf_summary_exporter_config.ini` with the following content:

```ini
[paths]
# Default root directory for SLURM job logs when only --jobid is provided
# The {jobid} placeholder will be replaced with the actual job ID
default_log_root = /path/to/nccl/logs/user-jobs

[kibana]
# Kibana project name for detailed data upload
kibana_project_detailed = your-kibana-project-detailed
# Kibana project name for summary data upload
kibana_project_summary = your-kibana-project-summary
```

### Configuration Options

#### [paths] Section
- **default_log_root**: Base directory for SLURM job logs. When using `--jobid`, the tool constructs the path as `{default_log_root}/{jobid}`.

#### [kibana] Section
- **kibana_project_detailed**: Project name for uploading detailed performance data to Kibana
- **kibana_project_summary**: Project name for uploading summary statistics to Kibana

### Using Custom Configuration

```bash
# Use a custom config file
python perf_summary_exporter.py --jobid 12345 --config /path/to/custom_config.ini

# Use default config file (perf_summary_exporter_config.ini in script directory)
python perf_summary_exporter.py --jobid 12345
```

## Examples

### Basic Analysis

```bash
# Analyze logs for job ID 12345
python perf_summary_exporter.py --jobid 12345

# Analyze logs in custom directory
python perf_summary_exporter.py --job_insp_rootdir /path/to/nccl/logs
```

### Analysis with Kibana Upload

```bash
# Analyze and upload to Kibana (only works with --jobid)
python perf_summary_exporter.py --jobid 12345 --upload

# Note: This will fail - upload requires jobid
# python perf_summary_exporter.py --job_insp_rootdir /path/to/logs --upload
```

## Output Structure

The script creates an organized output directory structure:

```
<jobid>-insp/
├── output.log                          # Execution log
├── parquet_files/                      # Converted log data
│   ├── logfile1.parquet
│   └── logfile2.parquet
└── summary/                           # Analysis results
    ├── scatter_plot_<comm_type>_<coll_type>.png
    ├── combined_scatter_plot_<comm_type>_<coll_type>.png
    └── msg_size_<size>_<human_readable>/
        ├── histograms/
        │   └── histogram_<comm_type>_<coll_type>_<size>.png
        ├── boxplots/
        │   └── boxplot_<comm_type>_<coll_type>_<size>.png
        └── summary_<comm_type>_<coll_type>_<size>.csv
```

## Sample Visualizations

The following examples show the types of visualizations generated by the tool, using data from a 256-process distributed training job (Job ID: 3667016):

### Combined Scatter Plots (Overview)

Combined scatter plots provide a high-level view of performance across different message sizes:

![HCA AllGather Combined Scatter Plot](sample_images/combined_scatter_plot_hca-only_AllGather.png)
*Combined scatter plot showing AllGather performance using HCA communication across multiple message sizes*

![HCA ReduceScatter Combined Scatter Plot](sample_images/combined_scatter_plot_hca-only_ReduceScatter.png)
*Combined scatter plot showing ReduceScatter performance using HCA communication across multiple message sizes*

### Individual Scatter Plots (Detailed Analysis)

Individual scatter plots focus on specific communication types and collective operations:

![NVLink AllGather Scatter Plot](sample_images/scatter_plot_nvlink-only_AllGather.png)
*Detailed scatter plot showing AllGather performance using NVLink communication*

![NVLink ReduceScatter Scatter Plot](sample_images/scatter_plot_nvlink-only_ReduceScatter.png)
*Detailed scatter plot showing ReduceScatter performance using NVLink communication*

### What These Visualizations Show

- **X-axis**: Operation sequence number (time progression)
- **Y-axis**: Bandwidth in GB/s
- **Color coding**: Different message sizes
- **Patterns**: Performance consistency, bottlenecks, and scaling behavior
- **Communication types**: HCA (network), NVLink (GPU interconnect), and mixed patterns

These visualizations help identify:
- Performance trends and patterns over time
- Communication bottlenecks and outliers
- Impact of message size on bandwidth
- Efficiency of different communication types
- Performance scaling characteristics

## Communication Types

The tool automatically classifies communication patterns:

- **single-rank**: Single process operations
- **nvlink-only**: Single-node, multi-GPU operations
- **hca-only**: Multi-node, single GPU per node
- **mixed**: Multi-node, multi-GPU operations

## Collective Types

Supported NCCL collective operations:

- **AllReduce**: Reduction across all ranks
- **AllGather**: Gather from all ranks to all ranks
- **ReduceScatter**: Reduce and scatter results
- **Broadcast**: One-to-many communication

## Visualization Types

The tool generates several types of visualizations:

### 1. Scatter Plots
- **Basic Scatter Plot**: Shows bandwidth vs operation sequence number
- **Combined Scatter Plot**: Multi-panel view for different message sizes

### 2. Statistical Plots
- **Histograms**: Distribution of bandwidth values
- **Box Plots**: Statistical summary with quartiles and outliers

## Performance Metrics

The tool analyzes various performance metrics:

- **Collective Bus Bandwidth (GB/s)**: Effective bandwidth utilization
- **Message Size**: Data volume per operation
- **Sequence Numbers**: Operation ordering
- **Execution Statistics**: Min, max, mean, and distribution analysis
- **Timing Information**: Start/end timestamps and operation duration
- **Time Range Analysis**: Shows when operations occurred and how long they lasted

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
    "hostname": "cw-dfw-h100-003-131-026",
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

### Generated Timing Fields

The tool automatically generates additional timing fields during analysis:

#### 1. Aggregated Summary Fields (in scatter plots, Kibana uploads)
These fields are computed from grouped collective operations using DuckDB:

```json
{
  "id": "rank_0_node_worker-1",
  "coll_sn": 129,
  "coll_msg_size_bytes": 65536,
  "mean_coll_busbw_gbs": 18.469236,
  "coll_start_timestamp_us": 1749490229087081,
  "coll_end_timestamp_us": 1749490229087191,
  "coll_duration_us": 110
}
```

#### 2. Human-Readable Fields (in CSV output files only)
Additional formatting is applied for CSV exports in message-size specific directories:

```csv
coll_start_datetime,coll_end_datetime,coll_duration_human
"2024-01-15 14:30:25.087","2024-01-15 14:30:25.087","110.0μs"
```

**Field Generation Logic:**
- `coll_start_timestamp_us`: MIN(dump_timestamp_us) for grouped operations
- `coll_end_timestamp_us`: MAX(dump_timestamp_us) for grouped operations
- `coll_duration_us`: Calculated as (MAX - MIN) dump_timestamp_us
- `coll_start_datetime`: Human-readable timestamp (CSV files only)
- `coll_end_datetime`: Human-readable timestamp (CSV files only)
- `coll_duration_human`: Human-readable duration (CSV files only)

## Troubleshooting

### Common Issues

1. **No log files found**
   - Verify the path exists and contains log files
   - Check file permissions
   - Ensure files have the correct extensions

2. **Import errors**
   - Verify virtual environment is activated
   - Reinstall dependencies: `pip install -r requirements.txt`

3. **Memory issues with large datasets**
   - The tool uses parallel processing and may consume significant memory
   - Reduce the number of worker processes if needed

4. **Kibana upload failures**
   - Verify network connectivity
   - Check nvdataflow configuration
   - Ensure proper authentication

### Performance Optimization

- Use SSD storage for faster I/O operations
- Ensure sufficient RAM (4-8GB recommended for large datasets)
- Use parallel processing (automatically configured)
- Access to dedicated CPU node is important for running the script

### Allocating a Dedicated CPU Node

For optimal performance, especially with large datasets, allocate a dedicated CPU node using SLURM:

```bash
# Allocate a dedicated CPU node
$ salloc -N1 --account=sw_aidot --time=04:00:00 -p cpu_interactive
salloc: Pending job allocation 3459042
salloc: job 3459042 queued and waiting for resources
salloc: job 3459042 has been allocated resources
salloc: Granted job allocation 3459042
salloc: Waiting for resource configuration
salloc: Nodes cw-dfw-cpu1-003-017-005 are ready for job

# SSH to the allocated node
$ ssh cw-dfw-cpu1-003-017-005

# Verify CPU resources (example output)
$ lscpu
Architecture:            x86_64
CPU(s):                  96
Vendor ID:               GenuineIntel
Model name:              Intel(R) Xeon(R) Gold 6442Y
Thread(s) per core:      2
Core(s) per socket:      24
Socket(s):               2
```

**SLURM Parameters Explained:**
- `-N1`: Request 1 node
- `--account=sw_aidot`: Specify account for billing
- `--time=04:00:00`: Request 4 hours of runtime
- `-p cpu_interactive`: Use CPU interactive partition


## Environment Variables

The tool respects standard Python and system environment variables:

- `PYTHONPATH`: Python module search path
- `OMP_NUM_THREADS`: OpenMP thread count for numerical operations

### Libraries Affected by OMP_NUM_THREADS:

- **pandas** (via NumPy backend): Mathematical operations, array computations, and data aggregations
- **pyarrow**: Parquet file reading/writing operations and columnar data processing
- **duckdb**: SQL query execution and analytical operations on large datasets

**Recommendation**: For optimal performance on multi-core systems, consider setting:
```bash
export OMP_NUM_THREADS=<number_of_physical_cores>
# Example for a 24-core system:
export OMP_NUM_THREADS=24
```

## Dependencies Details

| Package | Version | Purpose |
|---------|---------|---------|
| pandas | >=1.3.0 | Data manipulation and analysis |
| tqdm | >=4.60.0 | Progress bars |
| duckdb | >=0.8.0 | SQL analytics engine |
| matplotlib | >=3.3.0 | Plotting and visualization |
| pyslurm | >=20.11.0 | SLURM job information |
| nvdataflow | >=1.0.0 | Kibana data upload |
| pyarrow | >=5.0.0 | Parquet file support |

## Logging

The script automatically creates detailed execution logs for debugging and monitoring:

- **Log Location**: `<jobid>-insp/output.log`
- **Log Level**: INFO (includes warnings and errors)
- **Log Content**:
  - File processing progress
  - Performance analysis results
  - Error messages and stack traces
  - Kibana upload status
  - Execution timing information

**Example log output:**
```
2024-01-15 10:30:15,123 - INFO - Processing files: 100%|██████████| 8/8 [00:02<00:00,  3.45file/s]
2024-01-15 10:30:17,456 - INFO - Generating summary for jobid: 12345, single-rank and AllReduce and upload=True
2024-01-15 10:30:17,890 - INFO - Example time range - ID: rank_0_node_worker-1, Coll_SN: 1, Start: 2024-01-15 14:30:25.123, End: 2024-01-15 14:30:25.156, Duration: 33.2ms
2024-01-15 10:30:18,789 - INFO - Scatter plot saved to /path/to/output/summary/scatter_plot_single-rank_AllReduce.png

```
