import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from causal_chamber.summarize_support import load_results, summarize_support


class SupportSummaryTests(unittest.TestCase):
    def test_screen_counts_are_per_target_not_per_seed_or_budget(self):
        result = {
            "features": ["red", "green"],
            "source_screens": {
                "a.csv": {"invariant_support": []},
                "b.csv": {"invariant_support": [0, 1]},
                "c.csv": {"invariant_support": [0]},
            },
            "results": [{"method": "pseudo_lasso", "target": target, "subset_seed": seed, "target_budget": budget}
                        for target in ("a.csv", "b.csv", "c.csv") for seed in (0, 1, 2) for budget in (10, 30)],
        }
        summary = summarize_support(result)
        self.assertEqual(summary["num_targets"], 3)
        self.assertEqual(summary["empty_targets"], ["a.csv"])
        self.assertEqual(summary["mean_size"], 1)
        self.assertEqual(summary["nonempty_mean_size"], 1.5)
        self.assertEqual(summary["size_counts"], {0: 1, 1: 1, 2: 1})
        self.assertEqual(summary["feature_counts"], {"red": 2, "green": 1})
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            legacy = root / "results.json"
            legacy.write_text(json.dumps(result))
            self.assertEqual(load_results(root), result)
            legacy.unlink()
            method_dir = root / "pseudo_lasso"
            method_dir.mkdir()
            path = method_dir / "results.json"
            path.write_text(json.dumps(result))
            for source in (root, method_dir, path):
                self.assertEqual(load_results(source), result)
            run = subprocess.run([sys.executable, "-m", "causal_chamber.summarize_support", str(root), "--details"],
                                 capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertIn("Empty A (target-only lasso): 1/3 (33.3%)", run.stdout)
            self.assertIn("b.csv: |A|=2, A={red, green}", run.stdout)
            centered_dir = root / 'pseudo_lasso_centered'
            centered_dir.mkdir()
            centered = {**result,'results':[{**row,'method':'pseudo_lasso_centered'} for row in result['results']]}
            (centered_dir/'results.json').write_text(json.dumps(centered))
            self.assertEqual(load_results(root,'pseudo_lasso_centered'),centered)
            self.assertEqual(summarize_support(centered)['num_targets'],3)
            self.assertEqual(load_results(root),result)

    def test_missing_screens_and_all_empty(self):
        with self.assertRaisesRegex(ValueError, "No saved"):
            summarize_support({"source_screens": {}})
        result = {"source_screens": {"a": {"invariant_support": [], "invariant_features": []}}}
        summary = summarize_support(result)
        self.assertIsNone(summary["nonempty_mean_size"])
        self.assertEqual(summary["std_size"], 0)
        result["results"] = [{"method": "pseudo_lasso", "target": "missing"}]
        with self.assertRaisesRegex(ValueError, "Missing source screens"):
            summarize_support(result)
