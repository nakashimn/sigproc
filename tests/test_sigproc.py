import os
import sys
import unittest
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+"/src")
from sigproc import _cast_ndarray
from sigproc import calc_section_average
from sigproc import calc_delay
from sigproc import calc_corrcoef

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
        expected = [2.0, 5.0, 8.0]
        actual = list(calc_section_average(values, window))
        self.assertEqual(expected, actual)

    def test_calc_section_average_ndarray(self):
        values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
        window = 3
        expected = [2.0, 5.0, 8.0]
        actual = list(calc_section_average(values, window))
        self.assertEqual(expected, actual)


class TestCalcCorrcoef(unittest.TestCase):
    """ ----------------------------------------------------------------------
    testclass calc_section_average
    ---------------------------------------------------------------------- """

    def test_calc_corrcoef_dataframe_0(self):
        x = np.arange(0, 1, 0.01)
        header0 = ["{}".format(i) for i in range(100)]
        header1 = ["{}".format(i+50) for i in range(100)]
        df_values_sin0 = pd.DataFrame(np.sin(2*np.pi*x), index=header0)
        df_values_sin1 = pd.DataFrame(np.sin(2*np.pi*x), index=header1)
        expected = 1.0
        actual = calc_corrcoef(df_values_sin0, df_values_sin1, delay=0)
        self.assertAlmostEqual(expected, actual)

    def test_calc_corrcoef_dataframe_1(self):
        x = np.arange(0, 1, 0.01)
        header0 = ["{}".format(i) for i in range(100)]
        header1 = ["{}".format(i+50) for i in range(100)]
        df_values_sin0 = pd.DataFrame(np.sin(2*np.pi*x), index=header0)
        df_values_sin1 = pd.DataFrame(np.sin(2*np.pi*x+np.pi), index=header1)
        expected = -1.0
        actual = calc_corrcoef(df_values_sin0, df_values_sin1, delay=0)
        self.assertAlmostEqual(expected, actual)

    def test_calc_corrcoef_dataframe_2(self):
        x = np.arange(0, 1, 0.01)
        header0 = ["{}".format(i) for i in range(100)]
        header1 = ["{}".format(i+50) for i in range(100)]
        df_values_sin0 = pd.DataFrame(np.sin(2*np.pi*x), index=header0).T
        df_values_sin1 = pd.DataFrame(np.sin(2*np.pi*x), index=header1)
        expected = 1.0
        actual = calc_corrcoef(df_values_sin0, df_values_sin1, delay=0)
        self.assertAlmostEqual(expected, actual)

    def test_calc_corrcoef_dataframe_3(self):
        x = np.arange(0, 1, 0.01)
        header0 = ["{}".format(i) for i in range(100)]
        header1 = ["{}".format(i+50) for i in range(100)]
        df_values_sin0 = pd.DataFrame(np.sin(2*np.pi*x), index=header0).T
        df_values_sin1 = pd.DataFrame(np.sin(2*np.pi*x+np.pi), index=header1)
        expected = -1.0
        actual = calc_corrcoef(df_values_sin0, df_values_sin1, delay=0)
        self.assertAlmostEqual(expected, actual)

    def test_calc_corrcoef_ndarray_0(self):
        x = np.arange(0, 1, 0.01)
        ar_values_sin0 = np.sin(2*np.pi*x)
        ar_values_sin1 = np.sin(2*np.pi*x)
        expected = 1.0
        actual = calc_corrcoef(ar_values_sin0, ar_values_sin1, delay=0)
        self.assertAlmostEqual(expected, actual)

    def test_calc_corrcoef_ndarray_1(self):
        x = np.arange(0, 1, 0.01)
        ar_values_sin0 = np.sin(2*np.pi*x)
        ar_values_sin1 = np.sin(2*np.pi*x+np.pi)
        expected = -1.0
        actual = calc_corrcoef(ar_values_sin0, ar_values_sin1, delay=0)
        self.assertAlmostEqual(expected, actual)


if __name__ == "__main__":
    unittest.main()
