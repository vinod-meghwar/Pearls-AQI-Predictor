import unittest

from scripts.pipeline_runner import get_pipeline_steps


class PipelineRunnerTests(unittest.TestCase):
    def test_hourly_pipeline_steps(self):
        self.assertEqual(
            get_pipeline_steps("hourly"),
            ["data_extraction.py", "feature_engineering.py"],
        )

    def test_daily_pipeline_steps(self):
        self.assertEqual(
            get_pipeline_steps("daily"),
            ["model_train.py", "promote_model.py"],
        )


if __name__ == "__main__":
    unittest.main()
