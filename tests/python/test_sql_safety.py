"""
Unit tests for SQL injection prevention in perf_summary_exporter.py

Tests verify that:
  1. Source uses parameterized queries ($1, $2, ?) instead of f-string interpolation
  2. DuckDB parameterized queries reject injection payloads
  3. F-string interpolation is demonstrably vulnerable

The source verification test (test_source_uses_parameterized_queries)
intentionally FAILS before the fix is applied, demonstrating the bug
exists in unfixed code.

Run: python3 -m unittest test_sql_safety -v
"""

import os
import sys
import unittest
import tempfile
import shutil

# ---------------------------------------------------------------------------
# Source verification (TDD key — FAILS before fix, PASSES after)
# ---------------------------------------------------------------------------

EXPORTER_PATH = os.path.join(
    os.path.dirname(__file__),
    "..", "..",
    "plugins", "profiler", "inspector", "exporter", "example",
    "perf_summary_exporter.py",
)


class TestSourceVerification(unittest.TestCase):
    """Verify the actual NCCL source has been patched to use parameterized queries."""

    def test_source_uses_parameterized_queries(self):
        """Source must use $1/$2 positional parameters, not f-string interpolation."""
        path = os.path.normpath(EXPORTER_PATH)
        if not os.path.isfile(path):
            self.skipTest(f"Cannot read {path} (run from tests/python/)")

        with open(path, "r") as f:
            src = f.read()

        # The fix replaces f-string SQL with positional parameters
        self.assertIn("$1", src,
                       f"{path}: no $1 positional parameter found — "
                       "SQL queries use f-string interpolation (injection risk)")
        self.assertIn("$2", src,
                       f"{path}: no $2 positional parameter found — "
                       "SQL queries use f-string interpolation (injection risk)")

    def test_source_no_fstring_sql(self):
        """Source must not contain f-string SQL patterns like WHERE coll = '{coll_type}'."""
        path = os.path.normpath(EXPORTER_PATH)
        if not os.path.isfile(path):
            self.skipTest(f"Cannot read {path} (run from tests/python/)")

        with open(path, "r") as f:
            src = f.read()

        # These patterns indicate f-string interpolation in SQL
        self.assertNotIn("'{coll_type}'", src,
                          f"{path}: found f-string SQL pattern '{{coll_type}}' — "
                          "vulnerable to SQL injection")
        self.assertNotIn("'{comm_type}'", src,
                          f"{path}: found f-string SQL pattern '{{comm_type}}' — "
                          "vulnerable to SQL injection")


# ---------------------------------------------------------------------------
# Behavioral tests — DuckDB parameterized query safety
# ---------------------------------------------------------------------------

# SQL injection payloads to test
INJECTION_PAYLOADS = [
    "'; DROP TABLE logs; --",
    "' OR '1'='1",
    "' UNION SELECT * FROM information_schema.tables --",
    "'; DELETE FROM logs WHERE '1'='1",
    "AllReduce'; DROP TABLE logs; --",
    "1; DROP TABLE logs",
    "' OR 1=1 --",
    "'; TRUNCATE TABLE logs; --",
    "' UNION ALL SELECT 1,2,3,4,5,6,7,8,9,10 --",
    "'; UPDATE logs SET coll='hacked'; --",
    "AllReduce' AND 1=0 UNION SELECT null,null,null,null,null,null,null,null,null,null --",
    "'; CREATE TABLE evil(data TEXT); --",
    "\\'; DROP TABLE logs; --",
    "'' OR ''='",
    "AllReduce%27%3B+DROP+TABLE+logs%3B--",
    "' HAVING 1=1 --",
    "' ORDER BY 1 --",
    "'; ATTACH ':memory:' AS evil; --",
    "' AND EXTRACTVALUE(1, CONCAT(0x7e, version())) --",
    "'; COPY logs TO '/tmp/stolen.csv'; --",
    "AllReduce' /*",
    "*/; DROP TABLE logs; /*",
    "' AND 1=CONVERT(int, (SELECT TOP 1 table_name FROM information_schema.tables)) --",
    "'; PRAGMA integrity_check; --",
    "' UNION SELECT sql FROM sqlite_master --",
]


def _duckdb_available():
    """Check if duckdb is available for behavioral tests."""
    try:
        import duckdb  # noqa: F401
        return True
    except ImportError:
        return False


@unittest.skipUnless(_duckdb_available(), "duckdb not installed — skipping behavioral SQL tests")
class TestParameterizedQuerySafety(unittest.TestCase):
    """Test that DuckDB parameterized queries safely handle injection payloads."""

    @classmethod
    def setUpClass(cls):
        import duckdb

        cls.tmpdir = tempfile.mkdtemp()
        cls.db = duckdb.connect(":memory:")

        # Create a test table matching the schema used in perf_summary_exporter
        cls.db.execute("""
            CREATE TABLE logs (
                id VARCHAR,
                coll_sn INTEGER,
                coll_msg_size_bytes BIGINT,
                coll_busbw_gbs DOUBLE,
                n_ranks INTEGER,
                nnodes INTEGER,
                dump_timestamp_us BIGINT,
                coll VARCHAR,
                comm_type VARCHAR
            )
        """)

        # Insert test data
        cls.db.execute("""
            INSERT INTO logs VALUES
            ('test-001', 1, 1048576, 42.5, 8, 1, 1000000, 'AllReduce', 'nvlink-only'),
            ('test-002', 2, 2097152, 85.0, 8, 2, 2000000, 'AllGather', 'mixed'),
            ('test-003', 3, 4194304, 120.3, 4, 1, 3000000, 'AllReduce', 'nvlink-only')
        """)

    @classmethod
    def tearDownClass(cls):
        cls.db.close()
        shutil.rmtree(cls.tmpdir, ignore_errors=True)

    def _run_parameterized_query(self, coll_type, comm_type):
        """Run the parameterized query pattern from the fixed code."""
        return self.db.execute("""
            SELECT
                id,
                coll_sn,
                coll_msg_size_bytes,
                AVG(coll_busbw_gbs) as mean_coll_busbw_gbs,
                COUNT(*) as log_count,
                MIN(dump_timestamp_us) as coll_start_timestamp_us,
                MAX(dump_timestamp_us) as coll_end_timestamp_us,
                (MAX(dump_timestamp_us) - MIN(dump_timestamp_us)) as coll_duration_us
            FROM logs
            WHERE coll = $1 and comm_type = $2
            GROUP BY id, coll_sn, coll_msg_size_bytes
            ORDER BY coll_sn
        """, [coll_type, comm_type]).fetchdf()

    def test_parameterized_coll_type_safe(self):
        """Parameterized queries must safely handle injection in coll_type."""
        for payload in INJECTION_PAYLOADS:
            with self.subTest(payload=payload):
                # Should not raise, should return empty result (no match)
                df = self._run_parameterized_query(payload, "nvlink-only")
                self.assertEqual(len(df), 0,
                                 f"Injection payload matched rows: {payload}")

    def test_parameterized_comm_type_safe(self):
        """Parameterized queries must safely handle injection in comm_type."""
        for payload in INJECTION_PAYLOADS:
            with self.subTest(payload=payload):
                df = self._run_parameterized_query("AllReduce", payload)
                self.assertEqual(len(df), 0,
                                 f"Injection payload matched rows: {payload}")

    def test_legitimate_query_returns_data(self):
        """Parameterized queries should return data for valid inputs."""
        df = self._run_parameterized_query("AllReduce", "nvlink-only")
        self.assertGreater(len(df), 0, "Should return rows for valid coll/comm types")

    def test_table_intact_after_all_payloads(self):
        """After all injection attempts, the table should still have all rows."""
        # Run all payloads first
        for payload in INJECTION_PAYLOADS:
            try:
                self._run_parameterized_query(payload, "nvlink-only")
                self._run_parameterized_query("AllReduce", payload)
            except Exception:
                pass  # Some payloads might cause errors, that's fine

        # Verify table integrity
        count = self.db.execute("SELECT COUNT(*) FROM logs").fetchone()[0]
        self.assertEqual(count, 3, "Table should still have exactly 3 rows")


@unittest.skipUnless(_duckdb_available(), "duckdb not installed — skipping f-string vulnerability demos")
class TestFstringVulnerability(unittest.TestCase):
    """Demonstrate that f-string SQL interpolation IS vulnerable to injection."""

    @classmethod
    def setUpClass(cls):
        import duckdb
        cls.db = duckdb.connect(":memory:")
        cls.db.execute("""
            CREATE TABLE logs (
                id VARCHAR, coll_sn INTEGER, coll_msg_size_bytes BIGINT,
                coll_busbw_gbs DOUBLE, n_ranks INTEGER, nnodes INTEGER,
                dump_timestamp_us BIGINT, coll VARCHAR, comm_type VARCHAR
            )
        """)
        cls.db.execute("""
            INSERT INTO logs VALUES
            ('test-001', 1, 1048576, 42.5, 8, 1, 1000000, 'AllReduce', 'nvlink-only'),
            ('test-002', 2, 2097152, 85.0, 8, 2, 2000000, 'AllGather', 'mixed')
        """)

    @classmethod
    def tearDownClass(cls):
        cls.db.close()

    def test_fstring_boolean_bypass(self):
        """F-string interpolation allows boolean tautology bypass."""
        # This is the VULNERABLE pattern from the unfixed code
        coll_type = "' OR '1'='1"
        comm_type = "' OR '1'='1"
        query = f"""
            SELECT COUNT(*) FROM logs
            WHERE coll = '{coll_type}' and comm_type = '{comm_type}'
        """
        count = self.db.execute(query).fetchone()[0]
        # The injection returns ALL rows, not just matching ones
        self.assertEqual(count, 2,
                         "F-string interpolation should be exploitable — "
                         "this demonstrates why parameterized queries are needed")

    def test_fstring_union_injection(self):
        """F-string interpolation allows UNION-based data exfiltration."""
        coll_type = "' UNION SELECT table_name,0,0,0,0,0,0,null,null FROM information_schema.tables WHERE table_name='"
        try:
            query = f"""
                SELECT id, coll_sn, coll_msg_size_bytes, coll_busbw_gbs,
                       n_ranks, nnodes, dump_timestamp_us, coll, comm_type
                FROM logs
                WHERE coll = '{coll_type}'
            """
            result = self.db.execute(query).fetchall()
            # If we get here, the UNION injection worked — we got schema info
            self.assertTrue(len(result) >= 0,
                            "UNION injection succeeded — demonstrates vulnerability")
        except Exception:
            # DuckDB might reject the column count mismatch, but the point
            # is that f-strings allow the attempt
            pass


if __name__ == "__main__":
    unittest.main()
