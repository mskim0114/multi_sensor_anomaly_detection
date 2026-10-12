"""New trial scenarios (combustion, vapor_release) follow exactly the anomaly-scenario rules."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("trial_runner", ROOT / "jetson_deploy/scripts/12_run_trial.py")
runner = importlib.util.module_from_spec(spec)
sys.modules["trial_runner"] = runner
spec.loader.exec_module(runner)

NEW = ("combustion", "vapor_release")


class ScenarioTests(unittest.TestCase):
    def test_existing_identifiers_are_unchanged_and_new_ones_appended(self):
        self.assertEqual(runner.SCENARIOS[:4], ("normal", "overload", "thermal_abnormal", "dust"))
        self.assertEqual(runner.SCENARIOS[4:], NEW)

    def test_new_scenarios_need_an_intervention_severity(self):
        for scenario in NEW:
            for severity, accepted in ((0, False), (1, True), (2, True), (3, True), (4, False)):
                with self.subTest(scenario=scenario, severity=severity):
                    if accepted:
                        runner.validate_severity(scenario, severity)
                    else:
                        with self.assertRaises(ValueError):
                            runner.validate_severity(scenario, severity)

    def test_new_scenarios_get_the_same_canonical_phase_plan_as_dust(self):
        reference = runner.build_plan("dust", 90, 180, 90, None)
        for scenario in NEW:
            with self.subTest(scenario=scenario):
                self.assertEqual(runner.build_plan(scenario, 90, 180, 90, None), reference)

    def test_trial_directories_and_test_mode_isolation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for scenario in NEW:
                path, trial_id = runner.allocate_trial_dir(root, scenario, False)
                self.assertEqual((path.parent.name, trial_id), (scenario, "trial_001"))
                bench, _ = runner.allocate_trial_dir(root, scenario, True)
                self.assertEqual(bench.parent.name, "_smoke")
                self.assertTrue(bench.name.startswith(f"{scenario}_"))

    def test_safety_boundary_names_the_new_intervention_sources(self):
        parser_help = runner.__doc__ or ""
        self.assertIn("no ignition or vapour source", parser_help, "safety boundary states it controls nothing")


if __name__ == "__main__":
    unittest.main()
