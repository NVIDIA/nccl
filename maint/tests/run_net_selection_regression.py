#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Host-only regression of actual source; topology dependencies are test doubles."""
import argparse
import hashlib
import json
import re
import shlex
import subprocess
import time
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract(path):
    source = path.read_text()
    start = source.index("ncclResult_t ncclTopoGetNetDev(")
    body = source.index("{", start)
    depth = 0
    for i in range(body, len(source)):
        depth += (source[i] == "{") - (source[i] == "}")
        if depth == 0:
            return source[start:i + 1] + "\n"
    raise ValueError("Unterminated ncclTopoGetNetDev")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cxx", default="c++")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    fixture = Path(__file__).with_name("net_selection_fixture.cc")
    summary = {"scope": "HOST_SOURCE_REGRESSION_WITH_TOPOLOGY_TEST_DOUBLES", "gpu_tested": False,
               "fixture_sha256": sha(fixture), "arms": {}, "status": "FAIL"}
    for name, source in [("baseline", args.baseline), ("candidate", args.candidate)]:
        arm = args.out / name
        arm.mkdir()
        (arm / "net_selection_under_test.inc").write_text(extract(source))
        command = shlex.split(args.cxx) + ["-std=c++17", "-Wall", "-Wextra", "-Werror", "-O2",
                                         "-I", str(arm.resolve()), str(fixture.resolve()), "-o", str((arm / "fixture").resolve())]
        record = {"source_sha256": sha(source), "compile_argv": command, "started_at": time.time()}
        try:
            build = subprocess.run(command, capture_output=True, text=True, timeout=120)
            (arm / "build.stdout").write_text(build.stdout)
            (arm / "build.stderr").write_text(build.stderr)
            record["compile_exit_code"] = build.returncode
            if build.returncode == 0:
                run = subprocess.run([str((arm / "fixture").resolve())], capture_output=True, text=True, timeout=15)
                (arm / "run.stdout").write_text(run.stdout)
                (arm / "run.stderr").write_text(run.stderr)
                record["exit_code"] = run.returncode
                match = re.search(r"CASES=(\d+) FAILURES=(\d+)", run.stdout)
                record["cases"] = int(match[1]) if match else None
                record["failures"] = int(match[2]) if match else None
                record["trigger_reproduced"] = "level2_no_read_cross_nic: FAIL result=0 proxy=1" in run.stdout
        except (OSError, subprocess.TimeoutExpired) as exc:
            record["error"] = str(exc)
        record["finished_at"] = time.time()
        summary["arms"][name] = record
    base, candidate = summary["arms"]["baseline"], summary["arms"]["candidate"]
    if (base.get("compile_exit_code") == candidate.get("compile_exit_code") == 0
            and base.get("exit_code") == 1 and base.get("trigger_reproduced")
            and candidate.get("exit_code") == 0 and candidate.get("cases") == 18 and candidate.get("failures") == 0):
        summary["status"] = "HOST_REGRESSION_PASS"
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] == "HOST_REGRESSION_PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
