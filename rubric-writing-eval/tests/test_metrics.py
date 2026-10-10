import unittest

from src.metrics import agreement_summary, quadratic_weighted_kappa


class MetricTests(unittest.TestCase):
    def test_perfect_qwk(self):
        self.assertAlmostEqual(quadratic_weighted_kappa([1, 3, 6, 9], [1, 3, 6, 9]), 1.0)

    def test_agreement_summary(self):
        result = agreement_summary([5, 6, 7], [5, 5, 8])
        self.assertAlmostEqual(result["exact_agreement"], 1 / 3)
        self.assertEqual(result["adjacent_agreement"], 1.0)
        self.assertAlmostEqual(result["mae"], 2 / 3)


if __name__ == "__main__":
    unittest.main()
