import unittest
from main import mae, rmse

class MetricTests(unittest.TestCase):
    def test_mae(self): self.assertAlmostEqual(mae([1, 3], [2, 1]), 1.5)
    def test_rmse(self): self.assertAlmostEqual(rmse([1, 3], [2, 1]), (2.5) ** 0.5)
    def test_empty_input_rejected(self):
        with self.assertRaises(ValueError): rmse([], [])

if __name__ == "__main__": unittest.main()
