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
import pyslurm
from nvdataflow import post
from datetime import datetime
import configparser


def load_config(config_file=None):
    """Load configuration from ini file"""
    config = configparser.ConfigParser()

    # Set defaults
    config.add_section('paths')
    config.set('paths', 'default_log_root', '/home/svc-nccl-inspector/nccl-inspector/logs/user-jobs')

    config.add_section('kibana')
    config.set('kibana', 'kibana_project_detailed', 'aidot-fact-nccl-inspector')
    config.set('kibana', 'kibana_project_summary', 'aidot-fact-nccl-inspector-summary')

    # Load config file if provided
    if config_file:
        if not Path(config_file).exists():
            raise FileNotFoundError(f"Configuration file not found: {config_file}")
        config.read(config_file)
    else:
        # Try to load default config file in same directory as script
        default_config = Path(__file__).parent / "perf_summary_exporter_config.ini"
        if default_config.exists():
            config.read(str(default_config))

    return config


# Global config instance
config = None

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


def get_log_files_and_jobid():
    parser = argparse.ArgumentParser(description="Process log files in a directory.")
    parser.add_argument(
        "--job_insp_rootdir",
        type=str,
        help="The root directory to search for log files.",
    )
    parser.add_argument(
        "--jobid", type=str, help="The job ID to construct the log file path."
    )
    parser.add_argument(
        "--upload", action="store_true", help="Flag to indicate if results should be uploaded."
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file (default: perf_summary_exporter_config.ini in script directory)"
    )
    args = parser.parse_args()

    # Initialize global config
    global config
    config = load_config(args.config)

    # Validate that --upload can only be used with --jobid
    if args.upload and not args.jobid:
        raise ValueError("The --upload option requires --jobid to be specified. Cannot upload data without a valid SLURM job ID.")

    root_dir = None
    if args.job_insp_rootdir:
        root_dir = args.job_insp_rootdir
        jobid = "unknown-jobid"
    if args.jobid:
        if root_dir is None:
            default_log_root = config.get('paths', 'default_log_root')
            root_dir = f"{default_log_root}/{args.jobid}"
        jobid = args.jobid
    if root_dir is None:
        raise ValueError("Either --job_insp_rootdir or --jobid must be provided.")

    if not Path(root_dir).exists():
        print("No inspector logs found")
        sys.exit(1)

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

    return files, jobid, args.upload


def get_comm_type(row) -> str:
    if row["n_ranks"] == 1:
        return "single-rank"
    elif row["nnodes"] == 1:
        return "nvlink-only"
    elif row["n_ranks"] == row["nnodes"]:
        return "hca-only"
    else:
        return "mixed"


def parse_file(filepath: Path, output_dir, regenerate=False):
    filename = Path(filepath).stem
    parquet_file = output_dir / f"{filename}.parquet"

    if parquet_file.exists() and not regenerate:
        logging.info(f"Parquet file {parquet_file} already exists. Skipping...")
        return

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


def create_per_node_parquet_files(files, jobid, output_root):
    output_dir = Path(output_root) / "parquet_files"
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


def bytes_to_human_readable(size_bytes):
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.log10(int(size_bytes)) / 3)
    s = round(size_bytes * math.pow(10, -3 * i), 2)
    return f"{s:.2f}{size_name[i]}"


def microseconds_to_human_readable(microseconds):
    """Convert microseconds to human readable format"""
    if microseconds < 1000:
        return f"{microseconds:.1f}μs"
    elif microseconds < 1000000:
        return f"{microseconds/1000:.1f}ms"
    else:
        return f"{microseconds/1000000:.1f}s"


def timestamp_to_datetime(timestamp_us):
    """Convert microsecond timestamp to datetime string"""
    return datetime.fromtimestamp(timestamp_us / 1000000).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]


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

def get_job_details(jobid):
    jobids = [int(jobid)]
    job_filter = pyslurm.db.JobFilter(ids=jobids)
    jobs = pyslurm.db.Jobs.load(job_filter)
    job_data = {}
    for job_id, job_info in jobs.items():
        job_dict = {}
        available_attrs = dir(job_info)
        for attr in [
            "name",
            "cluster",
            "state",
            "partition",
            "user_name",
            "start_time",
            "end_time",
            "submit_time",
            "time_limit",
            "num_tasks",
            "num_cpus",
            "memory_requested",
            "nodes",
            "node_list",
        ]:
            if attr in available_attrs:
                job_dict[attr] = getattr(job_info, attr, "N/A")
        job_data[job_id] = job_dict
    return job_data

def upload_to_kibana(df, comm_type, coll_type, jobid):
    logging.info(f"Uploading data to Kibana for jobid {jobid}: {comm_type} and {coll_type}")
    job_data = get_job_details(jobid)
    job_data = job_data[int(jobid)]

    if not job_data:
        logging.error(f"No job data found for job ID: {jobid}")
        return
    cluster_name = job_data['cluster']
    user_name = job_data['user_name']

    bulk_data = []
    for _, row in df.iterrows():
        if isinstance(row['n_ranks'], (list, tuple)) and row['n_ranks']:
            n_ranks = row['n_ranks'][0]
        else:
            n_ranks = 'unknown'
        msg_size_human_readable = row['human_readable_coll_msg_size_bytes']
        tag = f"{msg_size_human_readable}_r_{n_ranks}"
        jobid_tag = f"{jobid}-{tag}"
        coll_sn = row['coll_sn']
        msg_size = row['coll_msg_size_bytes']

        data = {
            "_id": f"{cluster_name}-{jobid}-{tag}-{coll_sn}-{msg_size}",
            "s_jobid": f"{jobid}",
            "s_cluster": f"{cluster_name}",
            "s_user": user_name,
            "s_tag": tag,
            "s_comm_type": comm_type,
            "s_coll_type": coll_type,
            "ts_current_time": int(datetime.now().timestamp() * 1000),
            "d_mean_coll_busbw": row['mean_coll_busbw_gbs'],
            "d_coll_sn": coll_sn,
            "d_msg_size": msg_size,
            "s_msg_size_human_readable": msg_size_human_readable,
            "s_jobid_tag": jobid_tag,
            "d_coll_start_timestamp_us": row['coll_start_timestamp_us'],
            "d_coll_end_timestamp_us": row['coll_end_timestamp_us'],
            "d_coll_duration_us": row['coll_duration_us'],
        }
        bulk_data.append(data)

    # Post all data to Kibana at once
    kibana_project = config.get('kibana', 'kibana_project_detailed')
    post(data=bulk_data, project=kibana_project)
    logging.info(f"All data for job {jobid} uploaded successfully")

def upload_summary_to_kibana(df, comm_type, coll_type, jobid):
    logging.info(f"Uploading summary data to Kibana for jobid {jobid} : {comm_type} and {coll_type}")
    job_data = get_job_details(jobid)
    job_data = job_data[int(jobid)]

    if not job_data:
        logging.error(f"No job data found for job ID: {jobid}")
        return
    cluster_name = job_data['cluster']
    user_name = job_data['user_name']

    # Group by both coll_msg_size_bytes and comm_type and compute summary statistics
    summary_df = df.groupby("coll_msg_size_bytes").agg(
        avg_busbw=("mean_coll_busbw_gbs", "mean"),
        max_busbw=("mean_coll_busbw_gbs", "max"),
        min_busbw=("mean_coll_busbw_gbs", "min"),
        op_count=("coll_sn", "count"),
        n_ranks=("n_ranks", "first")
    ).reset_index()

    bulk_data = []
    for _, row in summary_df.iterrows():
        n_ranks = row['n_ranks'][0] if row['n_ranks'] else 'unknown'
        msg_size = row['coll_msg_size_bytes']
        msg_size_human_readable = bytes_to_human_readable(msg_size)
        tag = f"{msg_size_human_readable}_r_{n_ranks}"

        data = {
            "_id": f"{cluster_name}-{jobid}-{comm_type}-{msg_size}",
            "s_jobid": f"{jobid}",
            "s_cluster": f"{cluster_name}",
            "s_user": user_name,
            "s_tag": tag,
            "s_comm_type": comm_type,
            "s_coll_type": coll_type,
            "s_msg_size_human_readable": msg_size_human_readable,
            "d_msg_size": msg_size,
            "d_avg_busbw": row['avg_busbw'],
            "d_max_busbw": row['max_busbw'],
            "d_min_busbw": row['min_busbw'],
            "d_op_count": row['op_count'],
            "ts_current_time": int(datetime.now().timestamp() * 1000),
        }
        bulk_data.append(data)
        logging.info(f"Appended record for comm_type: {comm_type}, coll_type: {coll_type}, msg_size_human_readable: {msg_size_human_readable}. Total records appended: {len(bulk_data)}")

    # Post all summary data to Kibana at once
    kibana_project_summary = config.get('kibana', 'kibana_project_summary')
    post(data=bulk_data, project=kibana_project_summary)
    logging.info(f"Summary data for job {jobid} uploaded successfully")

def generate_summary(output_root, comm_type, coll_type, upload, jobid):
    logging.info(f"Generating summary for jobid: {jobid}, {comm_type} and {coll_type} and upload={upload}")
    summary_dir = output_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    # Check if there are any parquet files
    parquet_dir = output_root / "parquet_files"
    parquet_files = list(parquet_dir.glob("*.parquet"))
    if not parquet_files:
        logging.warning(f"No parquet files found for {comm_type} and {coll_type}")
        return

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
        return

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
        return

    if df.empty:
        logging.info(f"No data for {comm_type} and {coll_type}")
    else:
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
        if upload:
            upload_to_kibana(df, comm_type, coll_type, jobid)
            upload_summary_to_kibana(df, comm_type, coll_type, jobid)
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


def generate_summary_wrapper(args):
    return generate_summary(*args)


if __name__ == "__main__":
    files, jobid, upload = get_log_files_and_jobid()
    print(f"Number of log files found: {len(files)}")
    print(f"Job ID: {jobid}")
    output_dir = Path(f"{jobid}-insp")
    print(f"Output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir)
    create_per_node_parquet_files(files, jobid, output_dir)
    comm_types = ["single-rank", "nvlink-only", "hca-only", "mixed"]
    coll_types = ["AllReduce", "AllGather", "ReduceScatter", "Broadcast"]
    summary_args = [
        (output_dir, comm_type, coll_type, upload, jobid)
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
