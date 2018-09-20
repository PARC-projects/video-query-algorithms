import unittest
from hyperparameter import Hyperparameter


class HyperparameterTest(unittest.TestCase):
    """Tests for hyperparameter.py."""

    def setUp(self):
        default_weights = {'rgb': 1.0, 'warped_optical_flow': 1.5}
        self.hyperparameter = Hyperparameter(default_weights)
        pass

    def tearDown(self):
        pass

    def test_true(self):
        self.assertTrue(True)

    def test_add_buffer_threshold_set_to_valid_value(self):
        # Arrange
        buffer = 1
        thresh = 1.00006
        self.eps_threshold = thresh

        # Act
        self.hyperparameter.add_buffer(buffer)

        # Assert
        self.assertEqual(self.eps_threshold, buffer - thresh)


if __name__ == '__main__':
    unittest.main()
