# NCCL Inspector Job Performance — Grafana Dashboard

This guide applies to the dashboard JSON **[`nccl-inspector-job-performance-template.json`](nccl-inspector-job-performance-template.json)**

**Built-in help:** After import, open **Dashboard settings → General → Description** for a summary of expected metrics and labels. Each template variable also has an inline **description** visible in **Dashboard settings → Variables**.

---

## Overview

This is a Grafana dashboard template for visualizing NCCL collective and P2P performance metrics collected by the [NCCL Inspector plugin](../README.md). It displays bus bandwidth (GB/s) and execution time (µs) broken down by collective type, message size, rank count, and node topology (NVLink vs network).

### Dashboard panels

| Row | Metrics shown |
|-----|---------------|
| NCCL Inspector - P2P [Recv] | P2P Recv bus bandwidth and exec time (NVLink-only and multi-node) |
| NCCL Inspector - P2P [Send] | P2P Send bus bandwidth and exec time (NVLink-only and multi-node) |
| NCCL Inspector - ReduceScatter | ReduceScatter bus bandwidth and exec time |
| NCCL Inspector - AllReduce | AllReduce bus bandwidth and exec time |
| NCCL Inspector - AllGather | AllGather bus bandwidth and exec time |

Each row splits panels into **NVLink-only** (`n_nodes="1"`) and **network** (`n_nodes!="1"`) views.

---

## Prerequisites

### 1. NCCL Inspector running and exporting Prometheus metrics

The NCCL Inspector plugin must be running with Prometheus output enabled:

```bash
export NCCL_INSPECTOR_PROM_DUMP=1
export NCCL_INSPECTOR_DUMP_DIR=/var/lib/node_exporter/nccl_inspector/
export NCCL_INSPECTOR_DUMP_THREAD_INTERVAL_MICROSECONDS=30000000  # 30s minimum
```

Refer to the [NCCL Inspector README](../README.md) for full setup instructions.

### 2. Prometheus scraping NCCL Inspector metrics

A Prometheus instance (or Mimir/Thanos/VictoriaMetrics or any compatible remote write target) must be scraping the NCCL Inspector output directory, exposing the following metrics:

**Collective metrics** (AllReduce, AllGather, ReduceScatter, etc.):

| Metric | Description |
|--------|-------------|
| `nccl_bus_bandwidth_gbs` | Bus bandwidth in GB/s |
| `nccl_collective_exec_time_microseconds` | Execution time in µs |

**P2P metrics** (Send/Recv):

| Metric | Description |
|--------|-------------|
| `nccl_p2p_bus_bandwidth_gbs` | P2P bus bandwidth in GB/s |
| `nccl_p2p_exec_time_microseconds` | P2P execution time in µs |

**Required labels on all NCCL metrics:**

| Label | Description |
|-------|-------------|
| `cluster` | Cluster identifier (must match your `$cluster` variable) |
| `slurm_job_id` | Job identifier (used to filter panels by job) |
| `n_nodes` | Number of nodes (`"1"` = NVLink-only, `>1` = multi-node) |
| `nranks` | Total number of ranks |
| `message_size` | Bucketed message size range string (e.g. `4-5GB`) |

**Additional labels on collective metrics:**

| Label | Description |
|-------|-------------|
| `collective` | Collective name (e.g. `AllReduce`, `AllGather`, `ReduceScatter`) |
| `algo_proto` | Algorithm/protocol (e.g. `Ring_ll`) |

**Additional labels on P2P metrics:**

| Label | Description |
|-------|-------------|
| `p2p_operation` | `Send` or `Recv` |

### 3. Grafana

- Grafana with permission to import dashboards and manage datasources.
- The Prometheus instance above must be configured as a Grafana datasource.

### 4. A metric that exposes job ids and hostnames (optional but recommended)

The `jobid` and `device` dashboard variables are populated from any Prometheus metric in your environment that carries job ids and hostnames. This is not specific to any exporter — use whatever metric already exposes this in your stack (e.g. a GPU metrics exporter, a SLURM exporter, a custom metric). Replace the placeholders `your_job_metric`, `your_job_label`, `your_cluster_label`, and `your_hostname_label` in the variable queries with the actual metric and label names from your environment.

If no such metric exists, you can set the `jobid` variable to a **Custom** or **Text box** type and enter job ids manually.

### 5. MySQL datasource for cluster/region metadata (optional)

The `cluster` and `telemetry_region` variables are driven by SQL queries against a placeholder table `slurm_clusters`. This is **not** part of NCCL — it is a stand-in for whatever metadata store you use to map cluster names to telemetry regions.

- If you have a MySQL-compatible DB with cluster metadata, configure it as a Grafana datasource and update the SQL (see [Updating metadata SQL](#updating-metadata-sql-slurm_clusters) below).
- If you have no DB, replace these variables with **Custom** variables (static list of clusters) or **Query** variables backed by `label_values` from Prometheus.

### 6. Multi-region Prometheus datasources (optional)

The `prom_datasource` variable uses a regex filter on `${telemetry_region}` to select the correct Prometheus/Mimir datasource for the chosen cluster. This assumes your Grafana datasource **names** include a region substring.

- If you operate a single Prometheus instance, set `prom_datasource` to a **Custom** or **Datasource** variable with no regex filter and remove the `telemetry_region` dependency.
- If you use multiple regions, ensure datasource names in Grafana contain a region identifier that matches what `telemetry_region` returns.

### 7. Loki (optional)

A `loki_datasource` variable is included for completeness. All panels in this dashboard are Prometheus-only. Safe to leave unset or remove.

---

## Import the dashboard

1. In Grafana, open **Dashboards** → **New** → **Import** (or **+** → **Import**).
2. Either:
   - **Upload JSON file** and select `nccl-inspector-job-performance-template.json`, or
   - Paste the file contents into **Import via panel json**.
3. Choose a **folder** (optional) and click **Import**.

If Grafana reports missing datasources, add or map them in **Connections** → **Data sources** first, then re-import or fix the datasource dropdowns on the affected panels.

---

## Configure variables

Set variables **in order** from top to bottom — later ones depend on earlier ones.

| Variable | Type | Purpose |
|----------|------|---------|
| **MySQL (metadata)** (`mysql_metadata`) | Datasource | MySQL-compatible datasource holding cluster/region rows. Not part of NCCL. Replace or remove if not using a DB. |
| **Cluster** (`cluster`) | Query (SQL) | Returns cluster identifiers. Values must match the cluster label on your NCCL Prometheus metrics. Replace `slurm_clusters` SQL with your schema. |
| **Telemetry region** (`telemetry_region`) | Query (SQL) | Returns a region string used to select the correct Prometheus datasource by name regex. Replace SQL with your schema, or remove if single-region. |
| **prom_datasource** | Datasource | Prometheus/Mimir instance for the selected region. Regex filters by `${telemetry_region}`. Adjust regex if datasource names differ. |
| **loki_datasource** | Datasource | Optional Loki instance. Panels are Prometheus-only; safe to leave unset. |
| **jobid** | Query (Prometheus) | Job ids from `your_job_metric`. Replace placeholder metric and label names with your environment's equivalents. Can also be set as a Custom/Text box variable. |
| **device** | Query (Prometheus) | Hostnames of GPU nodes for the selected job. Replace `your_hostname_label` with your environment's hostname label. Optional. |
| **csp_instance_id** | Query (Prometheus) | Per-node instance/cloud ids. Replace `your_node_info_metric` with your environment's metric, or remove if not needed. |

After changing **cluster** or **region**, refresh the dashboard so `prom_datasource` and `jobid` update.

---

## Updating metadata SQL (`slurm_clusters`)

The template uses a placeholder table `slurm_clusters` which is **not** defined by NCCL. To adapt it:

1. Open **Dashboard settings** (gear) → **Variables**.
2. Select **`cluster`** → update the SQL to select the cluster identifier column from your schema (must match the cluster label on your NCCL Prometheus metrics). Save.
3. Select **`telemetry_region`** → update the SQL to return the region string matching your Grafana Prometheus/Loki datasource names. Save.
4. Refresh the dashboard and verify `prom_datasource` resolves correctly.

If you do not use MySQL at all, change these variables to **Custom** (static values) or **Query** variables using Prometheus `label_values`, and remove the `telemetry_region` regex dependency from `prom_datasource` if operating a single region.

---

## Assumptions

The following assumptions are baked into the panel queries. If your setup differs, edit the panel PromQL directly.

| Assumption | Detail |
|------------|--------|
| NVLink-only panels filter on `n_nodes="1"` | Multi-node panels filter on `n_nodes!="1"` |
| Job filtering uses `slurm_job_id="$jobid"` | The `slurm_job_id` label must exist on all NCCL metrics |
| Cluster filtering uses `cluster="$cluster"` | The `cluster` label must exist on all NCCL metrics |
| Metrics are averaged across ranks | Queries use `avg by(message_size, nranks, ...)` |
| Time range defaults to last 24 hours | Narrow to the job window for cleaner curves |
| `prom_datasource` is selected per telemetry region | Requires datasource names to include a region substring, or simplify to a single datasource |

---

## Using the dashboard

1. Select a **time range** that covers the job's run window (default: last 24 hours).
2. Set **cluster** → **telemetry_region** → **prom_datasource** in order.
3. Set **jobid** to a specific job (or leave as **All** to see all jobs in the time range).
4. Expand the rows for the collective types of interest and read **BusBw** and **Exec Time** panels.

---

## Troubleshooting

| Symptom | Likely cause |
|---------|-------------|
| No data in any panel | `prom_datasource` not set, or NCCL metrics not scraped by Prometheus |
| No data in P2P panels | NCCL Inspector P2P tracking not enabled (`NCCL_INSPECTOR_ENABLE_P2P=1` required) |
| `jobid` variable is empty | `your_job_metric` / `your_job_label` placeholders not replaced, or no matching metric in Prometheus |
| NVLink-only panels empty but NET panels have data | All jobs ran multi-node; expected behavior |
| `prom_datasource` does not resolve | `telemetry_region` value does not match any Grafana datasource name; adjust the regex |
| Metrics present but wrong values | `cluster` label value does not match between SQL metadata and Prometheus labels |

---

## Reuse and provisioning

- **UID** `nccl-inspector-job-performance-template` is fixed for template use; change it only if it collides with an existing dashboard in your Grafana instance.
- For **GitOps / provisioning**, store the JSON in your repo and register it in Grafana's [dashboard provisioning](https://grafana.com/docs/grafana/latest/administration/provisioning/#dashboards) config so it deploys automatically with your stack.
