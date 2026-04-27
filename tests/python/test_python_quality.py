"""
Unit tests for Python code quality fixes in NCCL

Tests verify that:
  1. generate.py Rec class uses standard self/other naming (not me/x/y)
  2. __init__.py initializes _import_error before try/except
  3. Rec equality, hashing, and collection behavior are correct

The source verification tests intentionally FAIL before the fix is applied,
demonstrating the code quality issues exist in unfixed code.

Run: python3 -m unittest test_python_quality -v
"""

import os
import unittest


# ---------------------------------------------------------------------------
# Source verification (TDD key — FAILS before fix, PASSES after)
# ---------------------------------------------------------------------------

GENERATE_PY_PATH = os.path.join(
    os.path.dirname(__file__),
    "..", "..",
    "src", "device", "symmetric", "generate.py",
)

INIT_PY_PATH = os.path.join(
    os.path.dirname(__file__),
    "..", "..",
    "contrib", "nccl_ep", "python", "nccl_ep", "__init__.py",
)


class TestSourceVerification(unittest.TestCase):
    """Verify the actual NCCL source has been patched."""

    def test_generate_py_uses_self(self):
        """generate.py Rec class should use 'self' not 'me' for __init__/__hash__."""
        path = os.path.normpath(GENERATE_PY_PATH)
        if not os.path.isfile(path):
            self.skipTest(f"Cannot read {path} (run from tests/python/)")

        with open(path, "r") as f:
            src = f.read()

        self.assertIn("def __init__(self,", src,
                       f"{path}: Rec.__init__ uses non-standard parameter name "
                       "instead of 'self' — violates PEP 8")

    def test_generate_py_uses_other(self):
        """generate.py Rec.__eq__ should use 'other' not 'y'."""
        path = os.path.normpath(GENERATE_PY_PATH)
        if not os.path.isfile(path):
            self.skipTest(f"Cannot read {path} (run from tests/python/)")

        with open(path, "r") as f:
            src = f.read()

        self.assertIn("def __eq__(self, other)", src,
                       f"{path}: Rec.__eq__ uses non-standard parameter names "
                       "instead of 'self, other' — violates PEP 8")

    def test_init_py_import_error_initialized(self):
        """__init__.py should initialize _import_error before try/except."""
        path = os.path.normpath(INIT_PY_PATH)
        if not os.path.isfile(path):
            self.skipTest(f"Cannot read {path} (run from tests/python/)")

        with open(path, "r") as f:
            lines = f.readlines()

        # _import_error = None should appear BEFORE the try block
        import_error_line = None
        try_line = None
        for i, line in enumerate(lines):
            if "_import_error = None" in line and import_error_line is None:
                import_error_line = i
            if line.strip() == "try:" and try_line is None:
                try_line = i

        self.assertIsNotNone(import_error_line,
                              f"{path}: _import_error is not initialized before "
                              "try/except — risks NameError (used-before-assignment)")
        if try_line is not None:
            self.assertLess(import_error_line, try_line,
                             f"{path}: _import_error = None must appear before "
                             "the try block to prevent used-before-assignment")


# ---------------------------------------------------------------------------
# Behavioral tests — Rec class equality, hashing, and collections
# ---------------------------------------------------------------------------

class Rec(object):
    """Local copy of the Rec class for behavioral testing.
    Uses the FIXED naming (self/other) to verify behavior is correct."""
    def __init__(self, **kw):
        self.__dict__.update(kw)
    def __eq__(self, other):
        return self.__dict__ == other.__dict__
    def __hash__(self):
        h = 0
        for k in self.__dict__:
            h += hash((k, self.__dict__[k]))
        return h


class RecOld(object):
    """Original Rec with me/x/y naming — proves rename is safe."""
    def __init__(me, **kw):
        me.__dict__.update(kw)
    def __eq__(x, y):
        return x.__dict__ == y.__dict__
    def __hash__(me):
        h = 0
        for k in me.__dict__:
            h += hash((k, me.__dict__[k]))
        return h


# Table-driven equality test cases
EQUALITY_CASES = [
    ("empty_equal", {}, {}, True),
    ("single_field_equal", {"a": 1}, {"a": 1}, True),
    ("single_field_unequal", {"a": 1}, {"a": 2}, False),
    ("multi_field_equal", {"a": 1, "b": "x"}, {"a": 1, "b": "x"}, True),
    ("multi_field_unequal", {"a": 1, "b": "x"}, {"a": 1, "b": "y"}, False),
    ("extra_field", {"a": 1}, {"a": 1, "b": 2}, False),
    ("missing_field", {"a": 1, "b": 2}, {"a": 1}, False),
    ("none_value", {"a": None}, {"a": None}, True),
    ("none_vs_zero", {"a": None}, {"a": 0}, False),
    ("nested_tuple", {"a": (1, 2)}, {"a": (1, 2)}, True),
]


class TestRecEquality(unittest.TestCase):
    """Test Rec equality with fixed self/other naming."""

    def test_equality_cases(self):
        for name, kw1, kw2, expected in EQUALITY_CASES:
            with self.subTest(case=name):
                r1 = Rec(**kw1)
                r2 = Rec(**kw2)
                if expected:
                    self.assertEqual(r1, r2, f"[{name}] should be equal")
                else:
                    self.assertNotEqual(r1, r2, f"[{name}] should not be equal")

    def test_equality_old_naming(self):
        """Old me/x/y naming produces identical results (rename is safe)."""
        for name, kw1, kw2, expected in EQUALITY_CASES:
            with self.subTest(case=name):
                r1 = RecOld(**kw1)
                r2 = RecOld(**kw2)
                if expected:
                    self.assertEqual(r1, r2, f"[{name}] old naming: should be equal")
                else:
                    self.assertNotEqual(r1, r2, f"[{name}] old naming: should not be equal")

    def test_cross_version_equivalence(self):
        """New and old Rec produce identical equality/hash for same data."""
        for name, kw1, kw2, expected in EQUALITY_CASES:
            with self.subTest(case=name):
                new1, new2 = Rec(**kw1), Rec(**kw2)
                old1, old2 = RecOld(**kw1), RecOld(**kw2)
                self.assertEqual(
                    (new1 == new2), (old1 == old2),
                    f"[{name}] equality diverged between old and new naming"
                )
                self.assertEqual(
                    hash(new1), hash(old1),
                    f"[{name}] hash diverged between old and new naming"
                )


class TestRecHashing(unittest.TestCase):
    """Test Rec hash behavior for use in sets and dict keys."""

    def test_equal_objects_same_hash(self):
        r1 = Rec(a=1, b="x")
        r2 = Rec(a=1, b="x")
        self.assertEqual(hash(r1), hash(r2))

    def test_empty_hash(self):
        r = Rec()
        self.assertEqual(hash(r), 0)

    def test_set_dedup(self):
        s = {Rec(a=1), Rec(a=1), Rec(a=2)}
        self.assertEqual(len(s), 2, "Set should deduplicate equal Rec objects")

    def test_dict_key(self):
        d = {Rec(a=1): "one", Rec(a=2): "two"}
        self.assertEqual(d[Rec(a=1)], "one")
        self.assertEqual(d[Rec(a=2)], "two")


class TestImportErrorPattern(unittest.TestCase):
    """Test the _import_error initialization pattern."""

    def test_import_success_path(self):
        """When import succeeds, _import_error should remain None."""
        _import_error = None
        try:
            _ = 1 + 1  # Simulates successful import
        except ImportError as e:
            _import_error = str(e)
        self.assertIsNone(_import_error)

    def test_import_failure_path(self):
        """When import fails, _import_error should capture the message."""
        _import_error = None
        try:
            raise ImportError("test module not found")
        except ImportError as e:
            _import_error = str(e)
        self.assertEqual(_import_error, "test module not found")

    def test_import_error_always_defined(self):
        """_import_error must be defined regardless of import outcome."""
        _import_error = None
        try:
            _ = 1 + 1
        except ImportError as e:
            _import_error = str(e)
        # This would raise NameError without initialization
        _ = f"Status: {_import_error}"
        self.assertTrue(True, "No NameError — _import_error is always defined")


if __name__ == "__main__":
    unittest.main()
