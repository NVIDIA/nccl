# SPDX-License-Identifier: Apache-2.0
"""Compatibility checks for upstream and Proxy Inspector JSON streams."""
import sys
import unittest
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inspector_json_reader import DumpContext


class DumpContextTests(unittest.TestCase):
    def marker(self, version):
        return {
            "header": {"id": "comm", "rank": 0},
            "metadata": {"inspector_output_format_version": version,
                         "dump_timestamp_us": 100},
            "dump_stats": {"coll_records": 1, "p2p_records": 1},
        }

    def test_upstream_v43_preserves_record_context_and_user_tags(self):
        marker = self.marker("v4.3")
        context = DumpContext()
        context.restore(marker)
        for kind, tag in (("coll_perf", "coll_user_tag"), ("p2p_perf", "p2p_user_tag")):
            record = {"header": marker["header"],
                      "metadata": {**marker["metadata"], "dump_timestamp_us": 101},
                      kind: {tag: "18446744073709551615"}}
            self.assertEqual(context.restore(record), record)
        context.finish()

    def test_marker_streams_restore_context(self):
        for version in ("v4.3", "v4.4"):
            with self.subTest(version=version):
                marker = self.marker(version)
                context = DumpContext()
                context.restore(marker)
                for kind in ("coll_perf", "p2p_perf"):
                    self.assertEqual(context.restore({kind: {}}),
                                     {"header": marker["header"],
                                      "metadata": marker["metadata"], kind: {}})
                context.finish()

    def test_upstream_v43_still_detects_truncation_and_extra_records(self):
        marker = self.marker("v4.3")
        context = DumpContext()
        context.restore(marker)
        record = {"header": marker["header"], "metadata": marker["metadata"],
                  "coll_perf": {}}
        context.restore(record)
        with self.assertRaisesRegex(ValueError, "Truncated"):
            context.finish()
        with self.assertRaisesRegex(ValueError, "beyond"):
            context.restore(record)

    def test_v44_rejects_per_record_context(self):
        marker = self.marker("v4.4")
        context = DumpContext()
        context.restore(marker)
        with self.assertRaisesRegex(ValueError, "Unexpected fields"):
            context.restore({"header": marker["header"], "metadata": marker["metadata"],
                             "coll_perf": {}})


if __name__ == "__main__":
    unittest.main()
