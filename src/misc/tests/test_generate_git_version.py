# SPDX-FileCopyrightText: Copyright (c) 2026 Jake Wang
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information

import importlib.util
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "generate_git_version.py"


class GitVersionTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.git(self.source, "init", "-b", "source-branch")
        self.script = self.source / "src" / "misc" / SCRIPT.name
        self.script.parent.mkdir(parents=True)
        shutil.copyfile(SCRIPT, self.script)
        self.git(self.source, "add", ".")
        self.git(self.source, "-c", "user.name=Test", "-c", "user.email=test@example.com", "-c", "commit.gpgsign=false",
                 "commit", "-m", "Initial source")
        self.expected_hash = self.git(self.source, "describe", "--dirty", "--always", "--exclude", "*")
        self.output = self.root / "output" / "nccl_git_version.h"

    def git(self, directory, *args):
        return subprocess.check_output(["git", "-C", str(directory), *args],
                                       stderr=subprocess.DEVNULL, text=True).strip()

    def generate(self, directory):
        subprocess.run([sys.executable, str(self.script), str(self.output)],
                       cwd=directory, check=True, stdout=subprocess.DEVNULL)
        return self.output.read_text()

    def check_source_version(self, content):
        self.assertIn('#define NCCL_GIT_BRANCH "source-branch"', content)
        self.assertIn(f'#define NCCL_GIT_COMMIT_HASH "{self.expected_hash}"', content)

    def test_build_outside_source_tree(self):
        self.check_source_version(self.generate(self.root))

    def test_build_inside_another_repository(self):
        other = self.root / "other"
        other.mkdir()
        self.git(other, "init", "-b", "other-branch")
        self.git(other, "-c", "user.name=Test", "-c", "user.email=test@example.com", "-c", "commit.gpgsign=false",
                 "commit", "--allow-empty", "-m", "Unrelated repository")
        self.check_source_version(self.generate(other))

    def test_unchanged_header_is_not_rewritten(self):
        self.generate(self.root)
        before = self.output.stat().st_mtime_ns
        self.generate(self.root)
        self.assertEqual(before, self.output.stat().st_mtime_ns)

    def test_unavailable_git_keeps_fallback(self):
        spec = importlib.util.spec_from_file_location("generate_git_version", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with mock.patch.object(module.subprocess, "check_output", side_effect=FileNotFoundError):
            self.assertEqual(module.run_git(["git", "describe"]), "unknown")


if __name__ == "__main__":
    unittest.main()
