import unittest
import numpy as np

from autoia import Interval, autoia


class TestReferenceRegression(unittest.TestCase):
    def test_coupled_semantic_binary64_regression(self):
        def model(p):
            p1, p2 = p
            A = np.array([
                [2.0 + p1, 0.2*p2],
                [0.2*p2, 1.5-p1],
            ])
            b = np.array([1.0+0.1*p2, 0.5+p1])
            u = np.linalg.solve(A, b)
            return np.array([u[0], u[0]*u[1], u[1]**2], dtype=object)

        R = autoia(model, [Interval(-0.2, 0.2), Interval(-0.5, 0.5)], order=5)

        expected_lower_hex = [
            '0x1.d0b2d6a243c0bp-2',
            '0x1.4ffc71c8f7182p-4',
            '0x1.4b4a115bfb672p-6',
        ]
        expected_upper_hex = [
            '0x1.269c4cb6d38efp-1',
            '0x1.0cff0398c477cp-2',
            '0x1.51032e1fd920dp-2',
        ]
        expected_lower = np.array([float.fromhex(x) for x in expected_lower_hex])
        expected_upper = np.array([float.fromhex(x) for x in expected_upper_hex])

        self.assertTrue(np.array_equal(R.lower, expected_lower))
        self.assertTrue(np.array_equal(R.upper, expected_upper))


if __name__ == "__main__":
    unittest.main()
