import numpy as np
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess
import numpy.polynomial.polynomial as poly
from scipy.optimize import curve_fit

import fe3_incoming


class DataProcessing:
    """ Class for FE3 data processing methods that will be used in both
        display and routine or batch calculations. """

    fe3cals = fe3_incoming.FE3_cals()
    fe3curves = fe3_incoming.FE3_cal_curves()
    fe3db = fe3_incoming.FE3_db()

    def __init__(self):
        self.cals = self.fe3cals.cals
        self.calcurves = self.fe3curves.calcurves_df

    def detrend_response(self, df0, mol, ssv_norm_port, lowess=True):
        """ Method detrends the response data either with a lowess smooth or
            point-by-point linear interpolation.
            Returns the detrended series. """

        df = df0.copy()
        det = f'{mol}_det'
        df[det] = np.nan
        flags = f'{mol}_flag'
        resp = f'{mol}_ht'      # hardcoded to ht

        np_resp = df.loc[(df['port'] == ssv_norm_port) & (df[flags] is False)][resp]
        if lowess:
            df[det] = self.make_lowess(np_resp)
        else:
            df[det] = np_resp

        df[det] = df[resp]/df[det].interpolate(method='quadratic')
        return df[det]

    @staticmethod
    def make_lowess(series, frac=.4):
        # from https://gist.github.com/AllenDowney/818f6153ef316aee80467c51faee80f8
        endog = series.values
        exog = series.index.values

        smooth = lowess(endog, exog, frac=frac)
        index, data = smooth.T

        ds = pd.Series(data, index=pd.to_datetime(index))
        return ds.tz_localize('utc')   # to match utc time

    def cal_column(self, pid, mol):
        """ Returns the cal tank mole fraction value for the selected molecule (mol)
            identified by port_id. This method is usually called by 'apply' """
        try:
            val = self.cals.loc[pid, mol]
        except KeyError:
            val = np.nan
        return val

    @staticmethod
    def reduce_df(df, mol):
        """ Reduces a FE3_merged dataframe to only columns for the molecule
            selected (mol) """
        keep = list(df.columns[0:5])    # keep first 5 columns
        for col in df.columns[5:]:
            if col.find(f'{mol}_') >= 0:
                keep.append(col)
        return df[keep]

    def calculate_calcurve(self, fit_function, x, y, scale0=False):
        """ Method to fit x and y data to either polyfit or a function for
            curve_fit.
            Set scale0 to True to return x and y fits to 0 mole fraction. """

        if fit_function.find('line') >= 0:
            degree = 1
        elif fit_function.find('quad') >= 0:
            degree = 2
        else:
            degree = 3

        if len(x) > 1:
            # calculate fit
            if fit_function.find('exp') >= 0:
                coefs, pcov = curve_fit(self.exp_func, x, y, p0=[0., 1., .01])
            else:
                coefs = poly.polyfit(x, y, degree)

            minx, maxx = min(x), max(x)
            if scale0:
                minx = 0
            range = maxx - minx
            x_fit = np.linspace(minx-range*0.05, maxx+range*0.05, 100)

            if fit_function.find('exp') >= 0:
                y_fit = self.exp_func(x_fit, *coefs)
            else:
                y_fit = poly.polyval(x_fit, coefs)
        else:
            coefs = [0, 0]
            x_fit, y_fit = [], []

        return coefs, x_fit, y_fit

    def calculate_calcurve_run(self, df, run, mol, ssv_norm_port=1, scale0=False, save=True):
        """ This method calculates a fit for a specific run (or directory) for
            a molecule and returns the coefs and x_fit, y_fit arrays. """

        dir_df = df.loc[df.dir == run].copy()
        # dir_df = self.reduce_df(dir_df, mol)

        meth_for_run = dir_df[f'{mol}_meth'].values[0]
        detrend, fit_function = self.parse_method(meth_for_run)

        lowess = True if detrend == 'lowess' else False

        # add cal and det columns to dir_df dataframe
        cal = f'{mol}_cal'
        det = f'{mol}_det'
        dir_df[cal] = dir_df['port_id'].apply(self.cal_column, args=[mol])
        dir_df[det] = self.detrend_response(dir_df, mol, ssv_norm_port, lowess=lowess)

        # fit to unflagged data
        good = dir_df.loc[dir_df[f'{mol}_flag'] is False][[cal, det]].dropna()
        good = good.sort_values([cal, det])
        x = good[cal].values
        y = good[det].values

        coefs, x_fit, y_fit = self.calculate_calcurve(fit_function, x, y, scale0=scale0)

        # saves to calcurve db
        if save:
            self.save_calcurve(dir_df, mol, coefs)

        return coefs, x_fit, y_fit

    def save_calcurve(self, dir_df, mol, coefs):
        # only save cal curves of type='other' not type='flask'
        if dir_df['type'].values[0] != 'other':
            return

        run = dir_df['dir'].values[0]
        meth = dir_df[f'{mol}_meth'].values[0]
        # if the run is not in the calcurve database then add a line for it
        if run not in self.calcurves.index.values:
            self.calcurves = self.calcurves.append(pd.Series(name=run))

        # update calcurve method and save coefs
        self.calcurves.loc[self.calcurves.index == run, f'{mol}_meth'] = meth
        coefstr = ''.join(f'{i};' for i in coefs)
        self.calcurves.loc[self.calcurves.index == run, f'{mol}_coefs'] = coefstr
        self.calcurves.sort_index(inplace=True)
        self.fe3curves.calcurves_df = self.calcurves
        self.fe3curves.save()

    def save_calcurve_allmols(self, df, run, norm_port=1):
        for mol in self.fe3db.mols:
            self.calculate_calcurve_run(df, run, mol, ssv_norm_port=norm_port, save=True)

    @staticmethod
    def exp_func(x, a, b, c):
        """ exponential fit function """
        return a + b * np.exp(c * x)

    @staticmethod
    def parse_method(method):
        detrend, fit_function = method.split(';')
        return detrend, fit_function

    def mf_onepoint(self, df, mol, norm_port):
        """ Mole fraction calculation, one point cal through the norm_port
            Todo: add uncertainty estimate """
        det, cal = f'{mol}_det', f'{mol}_cal'
        value, unc = f'{mol}_value', f'{mol}_unc'
        calvalue = df.loc[(df['port'] == norm_port)][cal].values[0]
        df[value] = df[det] * calvalue
        return df

    def mf_twopoint(self, df, mol, *ports):
        """ Mole fraction calculation, two point cal through the norm_port
            Todo: add uncertainty estimate """
        p0, p1 = ports
        # column names
        det, cal = f'{mol}_det', f'{mol}_cal'
        value, unc, flags = f'{mol}_value', f'{mol}_unc', f'{mol}_flag'

        df1 = df.copy()     # temporary dataframe
        cal0 = df.loc[(df['port'] == p0)][cal].values[0]  # cal val on p0
        cal1 = df.loc[(df['port'] == p1)][cal].values[0]  # cal val on p1
        # detrended response on p0
        df1['r0'] = df.loc[(df['port'] == p0) & (df[flags] is False)][det]
        df1['r0'] = df1['r0'].interpolate()
        # detrended response on p1
        df1['r1'] = df.loc[(df['port'] == p1) & (df[flags] is False)][det]
        df1['r1'] = df1['r1'].interpolate()
        # slope
        df1['m'] = (df1['r0']-df1['r1']) / (cal0-cal1)
        # intercept
        df1['b'] = df1['r0'] - df1['m'] * cal0
        df[value] = (df1[det] - df1['b'])/df1['m']
        return df
