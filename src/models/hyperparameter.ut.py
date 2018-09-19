import unittest
from hyperparameter import Hyperparameter


class HyperparameterTest(unittest.TestCase):
    """Tests for hyperparameter.py."""

    def setUp(self):
        # default_weights = {'rgb': 1.0, 'warped_optical_flow': 1.5}
        # self.hyperparameter = Hyperparameter(default_weights)
        pass

    def tearDown(self):
        pass

    def test_true(self):
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
