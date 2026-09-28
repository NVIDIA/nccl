# SPDX-FileCopyrightText: Copyright (c) 2026 Jake Wang
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information

import contextlib
import csv
import io
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from optimize_config import ConfigOptimizer


class OptimizationMetricTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.path = Path(self.temp_dir.name) / "measurements.csv"
        self.fields = ["collective", "size_bytes", "algorithm", "protocol", "channels",
                       "nodes", "ranks", "pipeOps", "regBuff"]

    def load(self, rows, metric="latency_us", include_metric=True):
        fields = self.fields + ([metric] if include_metric else [])
        with self.path.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(fields)
            for algorithm, value in rows:
                row = ["allreduce", 1024, algorithm, "simple", 4, 1, 8, 1, 0]
                writer.writerow(row + ([value] if include_metric else []))
        optimizer = ConfigOptimizer(metric)
        with contextlib.redirect_stdout(io.StringIO()):
            data = optimizer.load_data(str(self.path))
        return optimizer, data

    def test_nonfinite_and_negative_measurements_are_skipped(self):
        for metric in ["latency_us", "bandwidth_gbps"]:
            for invalid in ["nan", "NaN", "inf", "-inf", "-1", "", "invalid"]:
                with self.subTest(metric=metric, invalid=invalid):
                    _, data = self.load([("tree", invalid), ("ring", "5")], metric)
                    self.assertEqual([item.algorithm for item in data], ["ring"])

    def test_selected_metric_must_be_present(self):
        for metric in ["latency_us", "bandwidth_gbps"]:
            with self.subTest(metric=metric):
                _, data = self.load([("tree", "5")], metric, include_metric=False)
                self.assertEqual(data, [])

    def test_zero_measurement_is_valid(self):
        for metric in ["latency_us", "bandwidth_gbps"]:
            with self.subTest(metric=metric):
                _, data = self.load([("tree", "0")], metric)
                self.assertEqual(len(data), 1)

    def test_valid_measurements_choose_the_best_configuration(self):
        for metric, expected in [("latency_us", "tree"), ("bandwidth_gbps", "ring")]:
            with self.subTest(metric=metric):
                optimizer, data = self.load([("nvls", "nan"), ("tree", "3"), ("ring", "5")], metric)
                with contextlib.redirect_stdout(io.StringIO()):
                    configs = optimizer.optimize_configurations(data)
                self.assertEqual(len(configs), 1)
                self.assertEqual(configs[0].split(",")[3], expected)

    def test_no_valid_measurements_do_not_modify_output(self):
        self.load([("tree", "nan")])
        output = self.path.with_suffix(".conf")
        output.write_text("existing configuration\n")
        script = Path(__file__).with_name("optimize_config.py")
        result = subprocess.run([sys.executable, str(script), str(self.path), "-o", str(output)],
                                capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("No valid data found", result.stdout)
        self.assertEqual(output.read_text(), "existing configuration\n")


if __name__ == "__main__":
    unittest.main()
