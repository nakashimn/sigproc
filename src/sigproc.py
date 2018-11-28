# -*- coding: utf-8 -*-

## 	@package sigproc.py 信号処理用パッケージ
#   @brief 信号処理用に作成した関数を簡易的にまとめています。
#   @author nakashimn
#   @date 2018.11.27
#   @version 0.1

import pandas
import numpy as np

""" ----------------------------------------------------------------------------
## 型変換(ndarray型)
# @param input 入力信号
# @return output_nd 出力信号(ndarray型)
---------------------------------------------------------------------------- """
def _cast_ndarray(input):
    try:
        if type(input) == int:
            output_nd = np.array(input)
            print("input : data type is int.")
        elif type(input) == float:
            output_nd = np.array(input)
            print("input : data type is float.")
        elif type(input) == str:
            output_nd = np.array(input)
            print("input : data type is str.")
        elif type(input) == list:
            output_nd = np.array(input)
        elif type(input) == np.ndarray:
            output_nd = input.copy()
        elif type(input) == pandas.core.frame.DataFrame:
            output_nd = input.values.copy()
        elif type(input) == pandas.core.series.Series:
            output_nd = input.values.copy()
        else:
            output_nd = np.array(input)
            print("input : data type is wrong.")
        return output_nd
    except Exception as e:
        print("Error:sigproc._cast_ndarray : {}".format(e.args[0]))
        return np.nan

""" ----------------------------------------------------------------------------
## 型変換(DataFrame型)
# @param input 入力信号
# @return output_df 出力信号(DataFrame型)
---------------------------------------------------------------------------- """
def _cast_dataframe(input):
    try:
        if type(input) == int:
            output_df = pandas.DataFrame([input])
            print("input : data type is int.")
        elif type(input) == float:
            output_df = pandas.DataFrame([input])
            print("input : data type is float.")
        elif type(input) == str:
            output_df = pandas.DataFrame([input])
            print("input : data type is str.")
        elif type(input) == list:
            output_df = pandas.DataFrame(input)
        elif type(input) == np.ndarray:
            output_df = pandas.DataFrame(input.copy())
        elif type(input) == pandas.core.frame.DataFrame:
            output_df = input.copy()
        elif type(input) == pandas.core.series.Series:
            output_df = input.copy()
        else:
            output_df = pandas.DataFrame(input)
            print("input : data type is wrong.")
        return output_df
    except Exception as e:
        print("Error:sigproc._cast_dataframe : {}".format(e.args[0]))
        return np.nan

""" ----------------------------------------------------------------------------
## 区間平均
# @param input 入力信号
# @param window 窓区間幅
# @param ofset =0 オフセット
# @return section_average 区間平均(ndarray型)
---------------------------------------------------------------------------- """
def calc_section_average(input, window, ofset=0):
    try:
        _signal = _cast_ndarray(input)
        _signal = np.delete(_signal,range(ofset))
        _signal = np.append(_signal,[np.nan]*(window-len(_signal)%window))
        _signal = _signal.reshape(int(len(_signal)/window),window)
        section_average = np.nanmean(_signal,axis=1)
        return section_average
    except Exception as e:
        print("Error:sigproc.calc_section_average : {}".format(e.args[0]))
        return np.nan

""" ----------------------------------------------------------------------------
## 無効値補間
# @param input 入力信号
# @param invalid_values 無効値
# @return interpolated 無効値補間後信号(ndarray型)
---------------------------------------------------------------------------- """
def interpolate_invalid(input, invalid_values=[]):
    try:
        _signal = _cast_dataframe(input)
        if type(invalid_values) == int or type(invalid_values) == float:
            invalid_values = [invalid_values]
        for invalid_value in invalid_values:
            _signal[_signal==invalid_value] = np.nan
        interpolated = _signal.interpolate(limit_direction="backward").dropna().T.values[0]
        return interpolated
    except Exception as e:
        print("Error:sigproc.interpolate_invalid : {}".format(e.args[0]))
        return np.nan

""" ----------------------------------------------------------------------------
## 相関関数算出
# @param data_meas 測定信号
# @param data_ref 参照信号
# @param invalid_values 無効値
# @return corrfunc 相関関数
---------------------------------------------------------------------------- """
def calc_corrfunc(data_meas, data_ref, invalid_values=[]):
    try:
        _data_meas = interpolate_invalid(data_meas, invalid_values)
        _data_ref = interpolate_invalid(data_ref, invalid_values)
        corrfunc = np.correlate(_data_meas-np.nanmean(_data_meas),
                                _data_ref-np.nanmean(_data_ref),
                                "full")
        return corrfunc
    except Exception as e:
        print("Error:sigproc.calc_coeffunc : {}".format(e.args[0]))
        return np.nan

""" ----------------------------------------------------------------------------
## 遅延算出
# @param data_meas 測定信号
# @param data_ref 参照信号
# @param invalid_values 無効値
# @return delay 遅延
# @brief _delay>0 : 測定信号のほうが遅い
# @brief _delay<0 : 測定信号のほうが早い
---------------------------------------------------------------------------- """
def calc_delay(data_meas, data_ref, invalid_values=[]):
    try:
        _data_meas = interpolate_invalid(data_meas, invalid_values)
        _data_ref = interpolate_invalid(data_ref, invalid_values)
        _corrfunc = np.correlate(_data_meas-np.nanmean(_data_meas),
                                 _data_ref-np.nanmean(_data_ref),
                                 "full")
        delay = (len(_data_ref)-1) - _corrfunc.argmax()
        return delay
    except Exception as e:
        print("Error:sigproc.calc_delay : {}".format(e.args[0]))
        return np.nan

""" ----------------------------------------------------------------------------
## 測定信号-参照信号間の遅延補正とデータ長の統一
# @param data_meas 測定信号
# @param data_ref 参照信号
# @param invalid_values 無効値
# @param delay 遅延
# @return aligned_meas データ長調整後の測定信号(ndarray型)
# @return aligned_ref データ長調整後の参照信号(ndarray型)
---------------------------------------------------------------------------- """
def align(data_meas, data_ref, invalid_values=[], delay=np.nan):
    try:
        if np.isnan(delay):
            _delay = calc_delay(data_meas, data_ref, invalid_values)
        else:
            _delay = delay
        _data_meas = pandas.DataFrame(data_meas,columns=["meas"]).copy()
        _data_ref = pandas.DataFrame(data_ref,columns=["ref"]).copy()
        if _delay > 0:
            _data_ref = _data_ref.drop(range(_delay)).reset_index(drop=True)
        elif _delay < 0:
            _data_meas = _data_meas.drop(range(abs(_delay))).reset_index(drop=True)
        _data_concat = pandas.concat([_data_meas,_data_ref],1).dropna()
        aligned_meas = _data_concat["meas"].values
        aligned_ref = _data_concat["ref"].values
        return aligned_meas, aligned_ref
    except Exception as e:
        print("Error:sigproc.align : {}".format(e.args[0]))
        return np.nan

""" ----------------------------------------------------------------------------
## 相関係数算出
# @param data_meas 測定信号
# @param data_ref 参照信号
# @param invalid_values 無効値
# @param delay 遅延
# @return corrcoef 相関係数
---------------------------------------------------------------------------- """
def calc_corrcoef(data_meas, data_ref, invalid_values=[], delay=np.nan):
    try:
        _data_meas, _data_ref = align(data_meas, data_ref, invalid_values, delay)
        corrcoef = np.corrcoef(_data_meas, _data_ref)[0][1]
        return corrcoef
    except Exception as e:
        print("Error:sigproc.calc_corrcoef : {}".format(e.args[0]))
        return np.nan

""" ----------------------------------------------------------------------------
## 平均誤差(ME)算出
# @param data_meas 測定信号
# @param data_ref 参照信号
# @param invalid_values 無効値
# @param delay 遅延
# @return mean_error 平均誤差
---------------------------------------------------------------------------- """
def calc_mean_error(data_meas, data_ref, invalid_values=[]):
    try:
        _data_meas, _data_ref = align(data_meas, data_ref, invalid_values, delay)
        mean_error = np.nanmean(_data_meas - _data_ref)
        return mean_error
    except Exception as e:
        print("Error:sigproc.calc_mean_error : {}".format(e.args[0]))
        return np.nan

""" ----------------------------------------------------------------------------
## 平均絶対誤差(MAE)算出
# @param data_meas 測定信号
# @param data_ref 参照信号
# @param invalid_values 無効値
# @param delay 遅延
# @return mean_abs_error 平均絶対誤差
---------------------------------------------------------------------------- """
def calc_mean_abs_error(data_meas, data_ref, invalid_values=[]):
    try:
        _data_meas, _data_ref = align(data_meas, data_ref, invalid_values, delay)
        mean_abs_error = np.nanmean(abs(_data_meas - _data_ref))
        return mean_abs_error
    except Exception as e:
        print("Error:sigproc.calc_mean_abs_error : {}".format(e.args[0]))
        return np.nan

""" ----------------------------------------------------------------------------
## 平均二乗誤差(RSE)算出
# @param data_meas 測定信号
# @param data_ref 参照信号
# @param invalid_values 無効値
# @param delay 遅延
# @return mean_sq_error 平均二乗誤差
---------------------------------------------------------------------------- """
def calc_mean_sq_error(data_meas, data_ref, invalid_values=[]):
    try:
        _data_meas, _data_ref = align(data_meas, data_ref, invalid_values, delay)
        mean_sq_error = np.nanmean((_data_meas - _data_ref)**2)
        return mean_sq_error
    except Exception as e:
        print("Error:sigproc.calc_mean_sq_error : {}".format(e.args[0]))
        return np.nan

""" ----------------------------------------------------------------------------
## 平均平方二乗誤差(RMSE)算出
# @param data_meas 測定信号
# @param data_ref 参照信号
# @param invalid_values 無効値
# @param delay 遅延
# @return root_mean_sq_error 平均平方二乗誤差
---------------------------------------------------------------------------- """
def calc_root_mean_sq_error(data_meas, data_ref, invalid_values=[]):
    try:
        _data_meas, _data_ref = align(data_meas, data_ref, invalid_values, delay)
        root_mean_sq_error = np.sqrt(np.nanmean((_data_meas - _data_ref)**2))
        return root_mean_sq_error
    except Exception as e:
        print("Error:sigproc.calc_root_mean_sq_error : {}".format(e.args[0]))
        return np.nan