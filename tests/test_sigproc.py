import unittest
import numpy as np
import pandas as pd

from src.sigproc import _cast_ndarray
from src.sigproc import calc_section_average


class TestCastNdarray(unittest.TestCase):
    """ ----------------------------------------------------------------------
    testclass _cast_ndarray
    ---------------------------------------------------------------------- """
    def test_cast_ndaray_in_int(self):
        value = int(1)
        expected = np.array(1)
        actual = _cast_ndarray(value)
        self.assertEqual(expected, actual)


class TestCalcSectionAverage(unittest.TestCase):
    """ ----------------------------------------------------------------------
    testclass calc_section_average
    ---------------------------------------------------------------------- """

    def test_calc_section_average_list(self):
        values = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        window = 3
        expected = np.array([2, 5, 8])
        actual = calc_section_average(values, window)
        self.assertEqual(expected, actual)

    def test_calc_section_average_ndarray(self):
        values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
        window = 3
        expected = np.array([2, 5, 8])
        actual = calc_section_average(values, window)
        self.assertEqual(expected, actual)


if __name__ == "__main__":
    unittest.main()
