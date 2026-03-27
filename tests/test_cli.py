"""
tests/test_cli.py
-----------------
Unit tests for autoresearch/cli.py.

Covers:
  - sys.path is NOT mutated by importing cli (fix for sys.path hack removal)
  - _step_prepare calls autoresearch.prepare.prepare() directly (not subprocess)
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, call
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TestCliSysPath(unittest.TestCase):

    def test_no_sys_path_insert_in_cli_source(self):
        """cli.py must not contain a sys.path.insert() call.

        Checking the source file directly is immune to test-session pollution
        (multiple test files all insert the project root, so counting
        occurrences in a shared sys.path is unreliable).
        """
        cli_path = Path(__file__).resolve().parent.parent / "autoresearch" / "cli.py"
        source = cli_path.read_text(encoding="utf-8")
        self.assertNotIn(
            "sys.path.insert(",
            source,
            "cli.py must not call sys.path.insert() — the package is installed "
            "via pip install -e . so this is unnecessary and hides missing deps.",
        )


class TestStepPrepare(unittest.TestCase):

    def _make_args(self, tmp_dir: Path):
        args = MagicMock()
        args.corpus = str(tmp_dir / "corpus.txt")
        args.data_dir = str(tmp_dir / "data")
        args.max_pairs = 10
        args.model = None
        return args

    def test_step_prepare_calls_prepare_directly(self):
        """_step_prepare must call autoresearch.prepare.prepare(), NOT subprocess.run."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            corpus = tmp / "corpus.txt"
            corpus.write_text("Some test corpus content " * 20, encoding="utf-8")
            args = MagicMock()
            args.corpus = "corpus.txt"
            args.data_dir = "data/"
            args.max_pairs = 5
            args.model = None

            # Patch both prepare and subprocess.run to verify which is called
            with patch("autoresearch.cli.ROOT", tmp), \
                 patch("autoresearch.prepare.prepare") as mock_prepare, \
                 patch("subprocess.run") as mock_subprocess:

                # Create the corpus at the right path
                (tmp / "corpus.txt").write_text("corpus content " * 20, encoding="utf-8")

                from autoresearch.cli import _step_prepare
                result = _step_prepare(args)

                # prepare() must have been called
                mock_prepare.assert_called_once()
                # subprocess.run must NOT have been called by _step_prepare
                mock_subprocess.assert_not_called()

        self.assertTrue(result)

    def test_step_prepare_returns_false_if_corpus_missing(self):
        """Missing corpus file must skip gracefully without raising."""
        args = MagicMock()
        args.corpus = "definitely_nonexistent_corpus.txt"
        args.data_dir = "data/"
        args.max_pairs = 5
        args.model = None

        from autoresearch.cli import _step_prepare
        result = _step_prepare(args)
        self.assertFalse(result)

    def test_step_prepare_returns_false_on_prepare_error(self):
        """If prepare() raises, _step_prepare must return False (not propagate)."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "corpus.txt").write_text("content " * 50, encoding="utf-8")
            args = MagicMock()
            args.corpus = "corpus.txt"
            args.data_dir = "data/"
            args.max_pairs = 5
            args.model = None

            with patch("autoresearch.cli.ROOT", tmp), \
                 patch("autoresearch.prepare.prepare",
                       side_effect=RuntimeError("prepare failed")):
                from autoresearch.cli import _step_prepare
                result = _step_prepare(args)

        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main(verbosity=2)
