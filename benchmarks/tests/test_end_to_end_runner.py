"""Check command diagnostics and separation of online and trial timings."""

import contextlib
import csv
import io
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from benchmarks.compose_protocol_cost.run import _run
from benchmarks.compose_protocol_cost.run_end_to_end import TIMING_FIELDS, _write_summary


class EndToEndRunnerTests(unittest.TestCase):
    def test_command_output_reaches_console_and_log(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "build.log"
            console = io.StringIO()
            with contextlib.redirect_stdout(console):
                _run([sys.executable, "-c", "print('build progress')"], log=log)
            self.assertIn("build progress", console.getvalue())
            self.assertEqual(log.read_text(), "build progress\n")

    def test_failed_command_preserves_diagnostics(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "failure.log"
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(subprocess.CalledProcessError) as error:
                    _run([sys.executable, "-c", "print('failed build'); raise SystemExit(7)"], log=log)
            self.assertEqual(error.exception.returncode, 7)
            self.assertIn("failed build", log.read_text())

    def test_summary_keeps_lifecycle_time_out_of_online_latency(self):
        rows = []
        for online in (10.0, 12.0):
            row = dict.fromkeys(TIMING_FIELDS, 0.0)
            row.update(scenario_id="P1", claim_sequence="full", proofs=1,
                       online_total=online, trial_wall=online + 20.0)
            rows.append(row)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.csv"
            _write_summary(path, rows)
            with path.open() as handle:
                result = next(csv.DictReader(handle))
            self.assertEqual(float(result["online_total_mean"]), 11.0)
            self.assertEqual(float(result["trial_wall_mean"]), 31.0)
            self.assertEqual(float(result["online_total_variance"]), 2.0)


if __name__ == "__main__":
    unittest.main()
