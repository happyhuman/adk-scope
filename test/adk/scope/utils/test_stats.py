import unittest
from google.adk.scope.utils import stats


class TestStats(unittest.TestCase):
    def test_precision(self):
        self.assertEqual(stats.calculate_precision(10, 20), 0.5)
        self.assertEqual(stats.calculate_precision(0, 20), 0.0)
        self.assertEqual(stats.calculate_precision(10, 0), 1.0)  # Edge case

    def test_recall(self):
        self.assertEqual(stats.calculate_recall(10, 20), 0.5)
        self.assertEqual(stats.calculate_recall(0, 20), 0.0)
        self.assertEqual(stats.calculate_recall(10, 0), 1.0)  # Edge case

    def test_f1(self):
        self.assertAlmostEqual(stats.calculate_f1(0.5, 0.5), 0.5)
        self.assertAlmostEqual(stats.calculate_f1(1.0, 1.0), 1.0)
        self.assertAlmostEqual(stats.calculate_f1(0.0, 1.0), 0.0)
        self.assertAlmostEqual(stats.calculate_f1(0.0, 0.0), 0.0)
        self.assertAlmostEqual(stats.calculate_f1(0.75, 0.5), 0.6)  # 2*(0.375)/1.25 = 0.75/1.25 = 0.6


if __name__ == "__main__":
    unittest.main()
