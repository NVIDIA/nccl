from pathlib import Path
import argparse
import glob
import gzip
import sys
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import json
from tqdm.auto import tqdm
import duckdb
import math
import matplotlib.pyplot as plt
import matplotlib.dates
from matplotlib.gridspec import GridSpec
import os
import logging
import contextlib
from datetime import datetime
import numpy as np

def setup_logging(output_dir):
    log_file = output_dir / "output.log"
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


@contextlib.contextmanager
def smart_open(filename, mode="r"):
    if filename.endswith(".gz"):
        opener = gzip.open
    else:
        opener = open

    with opener(filename, mode) as f:
        yield f


def get_log_files_and_output_dir():
    parser = argparse.ArgumentParser(description="Process log files in a directory.")
    parser.add_argument(
        "--input_dir",
        type=str,
        help="The directory containing NCCL Inspector log files to process.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Custom output directory name (default: auto-generated from input directory)."
    )
    args = parser.parse_args()

    if args.input_dir:
        # Use the provided input directory
        root_dir = Path(args.input_dir)
        if not root_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {root_dir}")

    logfiles = list(glob.iglob(str(Path(root_dir) / "**" / "*.log"), recursive=True))
    gzlogfiles = list(
        glob.iglob(str(Path(root_dir) / "**" / "*.log.gz"), recursive=True)
    )
    jsonlfiles = list(
        glob.iglob(str(Path(root_dir) / "**" / "*.jsonl"), recursive=True)
    )
    gzjsonlfiles = list(
        glob.iglob(str(Path(root_dir) / "**" / "*.jsonl.gz"), recursive=True)
    )
    if (
            sum((1 for x in [logfiles, gzlogfiles, jsonlfiles, gzjsonlfiles] if len(x) > 0))
            > 1
    ):
        ### TODO: we could probably generate some logic to pick the "right" file to load, but for now, bail
        logging.critical("Appear to have mixed .log/.log.gz/.jsonl/.jsonl.gz; bailing!")
        sys.exit(1)

    files = logfiles + gzlogfiles + jsonlfiles + gzjsonlfiles

    if not files:
        print("No inspector logs found")
        sys.exit(1)

    # Generate output directory name from input directory
    if args.output_dir:
        output_dir_name = args.output_dir
    else:
        output_dir_name = f"{root_dir.name}-analysis"

    return files, output_dir_name

def bytes_to_human_readable(size_bytes):
    """
    Convert bytes to human-readable format using decimal (SI) units.

    Uses powers of 1000 (decimal/SI standard):
    - 1 KB = 1,000 bytes
    - 1 MB = 1,000,000 bytes
    - 1 GB = 1,000,000,000 bytes

    Not binary units (powers of 1024):
    - Does NOT use KiB, MiB, GiB (1024-based)

    Args:
        size_bytes: Number of bytes to convert

    Returns:
        Human-readable string (e.g., "1.50MB", "2.34GB")
    """
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.log10(int(size_bytes)) / 3)
    s = round(size_bytes * math.pow(10, -3 * i), 2)
    return f"{s:.2f}{size_name[i]}"

def timestamp_to_datetime(timestamp_us):
    """Convert microsecond timestamp to datetime string"""
    return datetime.fromtimestamp(timestamp_us / 1000000).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

def microseconds_to_human_readable(microseconds):
    """Convert microseconds to human readable format"""
    if microseconds < 1000:
        return f"{microseconds:.1f}μs"
    elif microseconds < 1000000:
        return f"{microseconds/1000:.1f}ms"
    else:
        return f"{microseconds/1000000:.1f}s"

def get_comm_type(row) -> str:
    if row["n_ranks"] == 1:
        return "single-rank"
    elif row["nnodes"] == 1:
        return "nvlink-only"
    elif row["n_ranks"] == row["nnodes"]:
        return "hca-only"
    else:
        return "mixed"

def parse_file(filepath: Path, output_dir):
    filename = Path(filepath).stem
    parquet_file = output_dir / f"{filename}.parquet"

    # Check if parquet file exists and is newer than source file
    if parquet_file.exists():
        source_mtime = Path(filepath).stat().st_mtime
        parquet_mtime = parquet_file.stat().st_mtime
        if parquet_mtime >= source_mtime:
            logging.info(f"Parquet file {parquet_file} is up to date. Skipping...")
            return
        else:
            logging.info(f"Source file {filepath} is newer than parquet. Regenerating...")

    # Check if file is empty or too small
    file_size = Path(filepath).stat().st_size
    if file_size == 0:
        logging.warning(f"Skipping empty file: {filepath}")
        return

    recs = []
    try:
        with smart_open(filepath, "r") as infile:
            for lineno, line in enumerate(infile):
                try:
                    json_recs = json.loads(line)
                except json.JSONDecodeError:
                    logging.error(f"Failed to parse line {filepath}:{lineno}")
                    continue

                # Validate that required fields exist
                if not all(key in json_recs for key in ["header", "metadata", "coll_perf"]):
                    logging.error(f"Missing required fields in {filepath}:{lineno}")
                    continue

                header = json_recs["header"]
                metadata = json_recs["metadata"]
                comm_type = get_comm_type(header)
                coll_perf = json_recs["coll_perf"]
                recs.append(
                    dict(
                        **header,
                        comm_type=comm_type,
                        **coll_perf,
                        **metadata,
                    )
                )
    except Exception as e:
        logging.error(f"Error reading file {filepath}: {e}")
        return

    # Skip files with no valid records
    if not recs:
        logging.warning(f"No valid records found in file: {filepath}. Skipping...")
        return

    df = pd.DataFrame(recs)
    df.to_parquet(parquet_file)
    logging.info(f"Created parquet file {parquet_file} with {len(recs)} records")

def create_per_node_parquet_files(files, output_dir):
    output_dir = Path(output_dir) / "parquet_files"
    output_dir.mkdir(parents=True, exist_ok=True)
    max_workers = min(64, len(files), os.cpu_count() or 1)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        list(
            tqdm(
                executor.map(parse_file, files, [output_dir] * len(files)),
                total=len(files),
                desc="Processing files",
                unit="file",
            )
        )
    return output_dir

def generate_scatter_plot(df, comm_type, coll_type, output_file):
    plt.figure(figsize=(10, 6), dpi=100)
    distinct_msg_sizes = df["coll_msg_size_bytes"].unique()

    for msg_size in distinct_msg_sizes:
        df_msg_size = df[df["coll_msg_size_bytes"] == msg_size]
        mean_busbw = df_msg_size["mean_coll_busbw_gbs"].mean()
        plt.scatter(
            df_msg_size["coll_sn"],
            df_msg_size["mean_coll_busbw_gbs"],
            label=f"MsgSize: {bytes_to_human_readable(msg_size)} (Mean: {mean_busbw:.2f} GB/s)",
            alpha=0.5,
        )

    plt.xlabel("Operation Sequence Number")
    plt.ylabel("Mean Collective Bus BW (GB/s)")
    plt.title(f"Comm Type: {comm_type}, Coll Type: {coll_type}")
    plt.legend(title="Message Size", loc="upper right")
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    logging.info(f"Scatter plot saved to {output_file}")

def generate_combined_scatter_plot(df, comm_type, coll_type, output_file, max_cols=3):
    distinct_msg_sizes = df["coll_msg_size_bytes"].unique()
    num_plots = len(distinct_msg_sizes)

    # Compute number of rows and columns
    num_cols = min(max_cols, num_plots)  # Limit max columns
    num_rows = (num_plots + num_cols - 1) // num_cols  # Calculate rows dynamically

    # Create figure with GridSpec
    fig = plt.figure(figsize=(5 * num_cols, 5 * num_rows), dpi=100)
    gs = GridSpec(num_rows, num_cols, figure=fig)

    for i, msg_size in enumerate(distinct_msg_sizes):
        row, col = divmod(i, num_cols)  # Determine row & column index
        ax = fig.add_subplot(gs[row, col])  # Create subplot at position

        df_msg_size = df[df["coll_msg_size_bytes"] == msg_size]
        mean_busbw = df_msg_size["mean_coll_busbw_gbs"].mean()
        ax.scatter(
            df_msg_size["coll_sn"],
            df_msg_size["mean_coll_busbw_gbs"],
            label=f"MsgSize: {bytes_to_human_readable(msg_size)} (Mean: {mean_busbw:.2f} GB/s)",
            alpha=0.5,
        )
        ax.set_xlabel("Op Seq No")
        ax.set_ylabel("Mean Collective Bus BW (GB/s)")
        ax.set_title(f"Message Size: {bytes_to_human_readable(msg_size)}({msg_size})")
        ax.legend(loc="upper right")

    fig.suptitle(f"Comm Type: {comm_type}, Coll Type: {coll_type}", ha="center", y=0.98)

    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    logging.info(f"Combined scatter plot saved to {output_file}")

def generate_histogram(df, comm_type, coll_type, output_file, message_size):
    plt.figure(figsize=(10, 6), dpi=100)
    data_range = df["mean_coll_busbw_gbs"].max() - df["mean_coll_busbw_gbs"].min()
    num_bins = min(50, int(data_range) + 1)
    plt.hist(
        df["mean_coll_busbw_gbs"],
        bins=num_bins,
        alpha=0.7,
        color="b",
        edgecolor="black",
        linewidth=1.2,
    )
    plt.xlabel("Mean Collective Bus BW (GB/s)")
    plt.ylabel("Frequency")
    plt.title(
        f"Comm Type: {comm_type}, Coll Type: {coll_type} Mean Collective Bus BW Histogram\nMsg Size: {message_size}"
    )
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0f}"))
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.2f} GB/s"))
    plt.gca().xaxis.get_offset_text().set_visible(False)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    logging.info(f"Histogram saved to {output_file}")

def generate_boxplot(df, comm_type, coll_type, output_file, message_size):
    plt.figure(figsize=(10, 6))
    boxprops = dict(linestyle="-", linewidth=2, color="blue")
    flierprops = dict(marker="o", color="red", alpha=0.5)
    medianprops = dict(linestyle="-", linewidth=2.5, color="orange")
    whiskerprops = dict(linestyle="--", linewidth=2, color="green")
    capprops = dict(linestyle="-", linewidth=2, color="black")

    plt.boxplot(
        df["mean_coll_busbw_gbs"],
        vert=False,
        patch_artist=True,
        boxprops=boxprops,
        flierprops=flierprops,
        medianprops=medianprops,
        whiskerprops=whiskerprops,
        capprops=capprops,
    )

    plt.xlabel("Mean Coll Bus BW (GB/s)")
    plt.title(
        f"Box Plot of Coll Bus BW (CommType: {comm_type} - Coll Type: {coll_type} - Msg Size: {message_size})"
    )

    # Adding labels for min, max, and median
    stats = df["mean_coll_busbw_gbs"].describe(percentiles=[0.5])
    plt.annotate(
        f"Min: {stats['min']:.2f}",
        xy=(stats["min"], 1),
        xytext=(stats["min"], 1.1),
        arrowprops=dict(facecolor="black", shrink=0.05),
    )
    plt.annotate(
        f"Median: {stats['50%']:.2f}",
        xy=(stats["50%"], 1),
        xytext=(stats["50%"], 1.1),
        arrowprops=dict(facecolor="black", shrink=0.05),
    )
    plt.annotate(
        f"Max: {stats['max']:.2f}",
        xy=(stats["max"], 1),
        xytext=(stats["max"], 1.1),
        arrowprops=dict(facecolor="black", shrink=0.05),
    )

    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    logging.info(f"Box plot saved to {output_file}")


def summarize_data_per_comm_coll_type(output_root, comm_type, coll_type, output_dir_name):
    """Summarize parquet data per communication and collective type using DuckDB"""
    logging.info(f"Summarizing data per comm/coll type for {output_dir_name}, {comm_type} and {coll_type}")

    # Check if there are any parquet files
    parquet_dir = output_root / "parquet_files"
    parquet_files = list(parquet_dir.glob("*.parquet"))
    if not parquet_files:
        logging.warning(f"No parquet files found for {comm_type} and {coll_type}")
        return None

    # Clean up invalid/empty parquet files by moving them to a separate directory
    invalid_dir = parquet_dir / "invalid"
    invalid_dir.mkdir(exist_ok=True)

    invalid_count = 0
    for pf in parquet_files:
        try:
            # Check file size first
            if pf.stat().st_size == 0:
                logging.warning(f"Moving zero-byte parquet file {pf} to invalid directory")
                pf.rename(invalid_dir / pf.name)
                invalid_count += 1
                continue

            # Use pyarrow to check parquet metadata without reading data
            import pyarrow.parquet as pq
            parquet_file = pq.ParquetFile(pf)
            if parquet_file.metadata.num_rows == 0:
                logging.warning(f"Moving empty parquet file {pf} (0 rows) to invalid directory")
                pf.rename(invalid_dir / pf.name)
                invalid_count += 1
        except Exception as e:
            logging.warning(f"Moving invalid parquet file {pf} to invalid directory: {e}")
            pf.rename(invalid_dir / pf.name)
            invalid_count += 1

    # Check if any valid files remain
    remaining_files = list(parquet_dir.glob("*.parquet"))
    if not remaining_files:
        logging.warning(f"No valid parquet files found for {comm_type} and {coll_type} (moved {invalid_count} invalid files)")
        return None

    logging.info(f"Found {len(remaining_files)} valid parquet files (moved {invalid_count} invalid files)")

    try:
        duckdb.execute(
            f"CREATE OR REPLACE VIEW logs AS SELECT * FROM read_parquet('{parquet_dir}/*.parquet')"
        )
        df = duckdb.execute(f"""
            SELECT
                id,
                coll_sn,
                coll_msg_size_bytes,
                AVG(coll_busbw_gbs) as mean_coll_busbw_gbs,
                COUNT(*) as log_count,
                ARRAY_DISTINCT(LIST(n_ranks)) as n_ranks,
                ARRAY_DISTINCT(LIST(nnodes)) as nnodes,
                MIN(dump_timestamp_us) as coll_start_timestamp_us,
                MAX(dump_timestamp_us) as coll_end_timestamp_us,
                (MAX(dump_timestamp_us) - MIN(dump_timestamp_us)) as coll_duration_us
            FROM logs
            WHERE coll = '{coll_type}' and comm_type = '{comm_type}'
            GROUP BY id, coll_sn, coll_msg_size_bytes
            ORDER BY coll_sn
        """).df()
    except Exception as e:
        logging.error(f"Error executing DuckDB query for {comm_type} and {coll_type}: {e}")
        return None

    if df.empty:
        logging.info(f"No data for {comm_type} and {coll_type}")
        return None

    # Add human-readable formatting
    df["human_readable_coll_msg_size_bytes"] = df["coll_msg_size_bytes"].apply(
        bytes_to_human_readable
    )

    # Log example of time range data for first few rows
    if len(df) > 0:
        sample_row = df.iloc[0]
        start_time = timestamp_to_datetime(sample_row['coll_start_timestamp_us'])
        end_time = timestamp_to_datetime(sample_row['coll_end_timestamp_us'])
        duration = microseconds_to_human_readable(sample_row['coll_duration_us'])
        logging.info(f"Example time range - ID: {sample_row['id']}, Coll_SN: {sample_row['coll_sn']}, "
                     f"Start: {start_time}, End: {end_time}, Duration: {duration}")

    return df


def generate_visualizations(df, output_root, comm_type, coll_type):
    """Generate all visualizations and save CSV files for the processed data"""
    logging.info(f"Generating visualizations for {comm_type} and {coll_type}")

    summary_dir = output_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    # Scatter Plot for all message sizes
    output_file = summary_dir / f"scatter_plot_{comm_type}_{coll_type}.png"
    generate_scatter_plot(df, comm_type, coll_type, output_file)

    # Combined Scatter Plot for all message sizes
    output_file = summary_dir / f"combined_scatter_plot_{comm_type}_{coll_type}.png"
    generate_combined_scatter_plot(df, comm_type, coll_type, output_file)

    distinct_msg_sizes = df["coll_msg_size_bytes"].unique()
    for msg_size in distinct_msg_sizes:
        hr_msg_size = bytes_to_human_readable(msg_size)
        msg_size_dir = summary_dir / f"msg_size_{msg_size}_{hr_msg_size}"
        msg_size_hist_dir = msg_size_dir / "histograms"
        msg_size_boxplot_dir = msg_size_dir / "boxplots"
        msg_size_dir.mkdir(parents=True, exist_ok=True)
        msg_size_hist_dir.mkdir(parents=True, exist_ok=True)
        msg_size_boxplot_dir.mkdir(parents=True, exist_ok=True)

        df_msg_size = df[df["coll_msg_size_bytes"] == msg_size]

        # Add human-readable time formatting
        df_msg_size = df_msg_size.copy()
        df_msg_size["coll_start_datetime"] = df_msg_size["coll_start_timestamp_us"].apply(timestamp_to_datetime)
        df_msg_size["coll_end_datetime"] = df_msg_size["coll_end_timestamp_us"].apply(timestamp_to_datetime)
        df_msg_size["coll_duration_human"] = df_msg_size["coll_duration_us"].apply(microseconds_to_human_readable)

        # Histogram
        output_file = (
            msg_size_hist_dir / f"histogram_{comm_type}_{coll_type}_{msg_size}.png"
        )
        generate_histogram(
            df_msg_size,
            comm_type,
            coll_type,
            output_file,
            bytes_to_human_readable(msg_size),
        )

        # Box Plot
        output_file = (
            msg_size_boxplot_dir / f"boxplot_{comm_type}_{coll_type}_{msg_size}.png"
        )
        generate_boxplot(
            df_msg_size,
            comm_type,
            coll_type,
            output_file,
            bytes_to_human_readable(msg_size),
        )

        output_file = msg_size_dir / f"summary_{comm_type}_{coll_type}_{msg_size}.csv"
        df_msg_size.to_csv(output_file, index=False)
        logging.info(
            f"Summary for {comm_type}, {coll_type}, and msg_size {msg_size} written to {output_file}"
        )


def generate_summary(output_root, comm_type, coll_type, output_dir_name):
    """Generate summary by summarizing data per comm/coll type and creating visualizations"""
    logging.info(f"Generating summary for {output_dir_name}, {comm_type} and {coll_type}")

    # Step 1: Summarize data per communication and collective type
    df = summarize_data_per_comm_coll_type(output_root, comm_type, coll_type, output_dir_name)

    # Step 2: Generate visualizations if data exists
    if df is not None:
        generate_visualizations(df, output_root, comm_type, coll_type)
    else:
        logging.warning(f"No data found for {comm_type} and {coll_type} - skipping visualization generation")


def generate_summary_wrapper(args):
    return generate_summary(*args)


if __name__ == "__main__":
    files, output_dir_name = get_log_files_and_output_dir()
    print(f"Number of log files found: {len(files)}")
    print(f"Output directory: {output_dir_name}")
    output_dir = Path(output_dir_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir)
    create_per_node_parquet_files(files, output_dir)
    comm_types = ["single-rank", "nvlink-only", "hca-only", "mixed"]
    coll_types = ["AllReduce", "AllGather", "ReduceScatter", "Broadcast"]
    summary_args = [
        (output_dir, comm_type, coll_type, output_dir_name)
        for comm_type in comm_types
        for coll_type in coll_types
    ]
    max_workers = min(64, len(summary_args), os.cpu_count() or 1)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        list(
            tqdm(
                executor.map(generate_summary_wrapper, summary_args),
                total=len(summary_args),
                desc="Generating summaries",
            )
        )
        print("Done!")
