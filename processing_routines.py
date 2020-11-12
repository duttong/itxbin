import numpy as np
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.optimize import curve_fit, least_squares

import fe3_incoming


class FE3config:
    """ Base class with FE3 instrument specific variables also
        used to load the various data sources.  """

    fe3cals = fe3_incoming.FE3_cals()
    fe3curves = fe3_incoming.FE3_cal_curves()
    fe3db = fe3_incoming.FE3_db()

    def __init__(self):
        self.MAX_N_SSVports = 10     # number of SSV ports for cal tanks
        self.MAX_N_Flasks = 8        # number of SSV ports for flasks
        self.ssv_norm_port = 1      # port number to normalize to
        self.ssv_flask_port = 2     # port number flasks are analyzed on
        self.second_cal_port = 3    # port number for second calibration tank (two point cal)
        self.cals = self.fe3cals.cals
        self.calcurves = self.fe3curves.calcurves_df


class DataProcessing(FE3config):
    """ Class for FE3 data processing methods that will be used in both
        display and routine or batch calculations. """

    def __init__(self):
        super().__init__()

    @staticmethod
    def dir_to_datetime(dir):
        """ Returns a datatime from a run dir name.
            for example: 20200207-183526
        """
        dt = pd.to_datetime(dir, infer_datetime_format=True)
        return dt

    def detrend_response(self, df0, mol, lowess=True):
        """ Method detrends the response data either with a lowess smooth or
            point-by-point linear interpolation.
            Returns the detrended series. """

        df = df0.copy()
        det = f'{mol}_det'
        df[det] = np.nan
        flags = f'{mol}_flag'
        resp = f'{mol}_ht'      # hardcoded to ht

        np_resp = df.loc[(df['port'] == self.ssv_norm_port) & (df[flags] == False)][resp]
        if lowess:
            df[det] = self.make_lowess(np_resp)
        else:
            df[det] = np_resp

        df[det] = df[resp]/df[det].interpolate(method='quadratic')
        return df[det]

    @staticmethod
    def make_lowess(series, frac=.25):
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

        if mol[-1] == 'b':
            mol = mol[:-1]

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

    def unflagged_data(self, fit_function, dir_df, mol):
        """ Method returns the unflagged data depending on the fit_function. """

        # determine which data to use
        flag = f'{mol}_flag'
        unflagged = (dir_df[flag] == False)
        if fit_function == 'two-points':
            mask = unflagged & ((dir_df['port'] == self.ssv_norm_port) | (dir_df['port'] == self.second_cal_port))
        else:
            mask = unflagged

        cal = f'{mol}_cal'
        det = f'{mol}_det'
        good = dir_df.loc[mask][[cal, det]].dropna()
        good = good.sort_values([cal, det])
        x = good[cal].values
        y = good[det].values
        return x, y

    def calculate_calcurve(self, fit_function, dir_df, mol, scale0=False):
        """ Method to fit x and y data to either polyfit or a function for
            curve_fit.
            Set scale0 to True to return x and y fits to 0 mole fraction. """

        x, y = self.unflagged_data(fit_function, dir_df, mol)

        # no data to fit to
        if len(x) == 0:
            if fit_function == 'linear':
                return [0, 0], [], []
            elif fit_function == 'quadratic' or fit_function == 'exponential':
                return [0, 0, 0], [], []
            else:
                return [0, 0, 0, 0], [], []

        minx, maxx = min(x), max(x)
        if scale0:
            minx = 0
        range = maxx - minx
        x_fit = np.linspace(minx-range*0.05, maxx+range*0.05, 100)

        # calculate fit
        if fit_function == 'linear':
            coefs, _ = curve_fit(self.linear, x, y, p0=[0., 1.])
            y_fit = self.linear(x_fit, *coefs)
        elif fit_function == 'quadratic':
            coefs, _ = curve_fit(self.quadratic, x, y, p0=[0., 1., .01])
            y_fit = self.quadratic(x_fit, *coefs)
        elif fit_function == 'cubic':
            coefs, _ = curve_fit(self.cubic, x, y, p0=[0., 1., .01, .001])
            y_fit = self.cubic(x_fit, *coefs)
        elif fit_function == 'exponential':
            coefs, _ = curve_fit(self.exponential, x, y, p0=[0., 1., .001], maxfev=20000)
            y_fit = self.exponential(x_fit, *coefs)
        elif fit_function == 'one-point':
            cal = f'{mol}_cal'
            calval = dir_df.loc[dir_df['port'] == self.ssv_norm_port, cal].values[0]
            coefs = [0, 1/calval]
            y_fit = self.linear(x_fit, *coefs)
        elif fit_function == 'two-points':
            coefs, _ = curve_fit(self.linear, x, y, p0=[0., 1.])
            y_fit = self.linear(x_fit, *coefs)
        else:
            print(f'Unknown curve fit type: {fit_function} using linear instead.')
            coefs, _ = curve_fit(self.linear, x, y, p0=[0., 1.])
            y_fit = self.linear(x_fit, *coefs)

        return coefs, x_fit, y_fit

    def calculate_calcurve_run(self, df, run, mol, scale0=False, save=True):
        """ This method calculates a fit for a specific run (or directory) for
            a molecule and returns the coefs and x_fit, y_fit arrays. """

        dir_df = df.loc[df.dir == run].copy()
        dir_df = self.add_det_cal_columns(dir_df, mol)

        fit_function = dir_df[f'{mol}_methcal'].values[0]
        coefs, x_fit, y_fit = self.calculate_calcurve(fit_function, dir_df, mol, scale0=scale0)

        # saves to calcurve db
        if save:
            self.save_calcurve(dir_df, mol, coefs)

        return coefs, x_fit, y_fit

    def add_det_cal_columns(self, dir_df, mol):
        """ Method to add the detrended response and cal tank value columns
            to the dir_df dataframe. """

        df = dir_df.copy()
        # dir_df = self.reduce_df(dir_df, mol)

        detrend = df[f'{mol}_methdet'].values[0]
        lowess = True if detrend == 'lowess' else False

        # add cal and detrend columns to dir_df dataframe
        cal = f'{mol}_cal'
        det = f'{mol}_det'
        df[cal] = df['port_id'].apply(self.cal_column, args=[mol])
        df[det] = self.detrend_response(df, mol, lowess=lowess)
        return df

    def save_calcurve(self, dir_df, mol, coefs):
        """ Method to save a cal curve type (name) and its coefficients to a
            a cal curve db file.
            dir_df is the run or dir dataframe """

        # Only save cal curves of type='other' not type='flask'
        if dir_df['type'].values[0] != 'other':
            return

        run = dir_df['dir'].values[0]
        meth = dir_df[f'{mol}_methcal'].values[0]
        # if the run is not in the calcurve database then add a line for it
        if run not in self.calcurves.index.values:
            self.calcurves = self.calcurves.append(pd.Series(name=run))

        # update calcurve method and save coefs
        self.calcurves.loc[self.calcurves.index == run, f'{mol}_methcal'] = meth
        coefstr = ''.join(f'{i};' for i in coefs)
        coefstr = coefstr[0:-1]     # drop trailing ';'
        self.calcurves.loc[self.calcurves.index == run, f'{mol}_coefs'] = coefstr
        self.calcurves.sort_index(inplace=True)
        self.fe3curves.calcurves_df = self.calcurves
        self.fe3curves.save()

    def nearest_calcurves(self, run, n=4):
        """ Returns a list of the n nearest cal curves, split on either
            side of the run date """
        cc = self.calcurves.copy().reset_index().set_index('dir_time')
        cc.sort_index(inplace=True)
        # print(cc.index)

        dt = pd.to_datetime(run, infer_datetime_format=True)
        idx = cc.index.get_loc(dt, method='ffill')

        caldates = []
        for i in range(1-n//2, n//2+1):
            try:
                c = cc.iloc[idx+i]['dir']
                caldates.append(c)
            except IndexError:
                pass

        return caldates

    def calcurve_params(self, calrun, mol):
        """ returns the fit function and coefficients of a calcurve
            on the date of 'calrun' for the given molecule 'mol' """

        cc = self.calcurves.copy().reset_index().set_index('dir_time')
        cc.sort_index(inplace=True)

        try:
            c_lst = cc[cc['dir'] == calrun][f'{mol}_coefs'].values[0]
            meth = cc[cc['dir'] == calrun][f'{mol}_methcal'].values[0]
            # a floating point list of coefficients
            coefs = [float(c) for c in c_lst.split(';')]
            return meth, coefs
        except IndexError:
            # this occurs if calrun can't be found or a problem with calcurves db
            return 'unknown', []

    def calcurve_values(self, calrun, mol, *x_values):
        """ Takes a cal curve date and mol with a list of x values
            and returns the y values for the cal curve function and coefficients
            in the cal curve db. """

        meth, coefs = self.calcurve_params(calrun, mol)
        try:
            f = getattr(self, meth)
            y_values = f(*x_values, *coefs)
        except AttributeError:
            y_values = []

        return y_values

    """
    def save_calcurve_allmols(self, df, run):
        for mol in self.fe3db.mols:
            self.calculate_calcurve_run(df, run, mol, save=True)
    """

    @staticmethod
    def linear(x, *coefs):
        """ linear fit function """
        return coefs[0] + coefs[1] * x

    @staticmethod
    def quadratic(x, *coefs):
        """ quadratic fit function """
        return coefs[0] + coefs[1] * x + coefs[2] * x**2

    @staticmethod
    def cubic(x, *coefs):
        """ cubic fit function """
        return coefs[0] + coefs[1] * x + coefs[2] * x**2 + coefs[3] * x**3

    @staticmethod
    def exponential(x, *coefs):
        """ exponential fit function """
        return coefs[0] + coefs[1] * np.exp(coefs[2] * x)

    """ The mole fraction methods below use a run/dir dataframe that already
        has the det and cal columns added. """

    def molefraction_calc(self, dir_df, mol, meth):

        df = dir_df.copy()

        if meth == 'one-point':
            df = self.mf_onepoint(self.mol_select, df)
        elif meth == 'two-points':
            df = self.mf_twopoint(self.mol_select, df)
        elif meth.find('-') > 0:    # cal curves specified by cal run date
            df = self.mf_calcurve(self.mol_select, df, meth)
        else:
            coefs, _, _ = self.calculate_calcurve(meth, df, self.mol_select)
            value = f'{mol}_value'
            df[value] = df.apply(self.solve_meth, args=([meth, coefs, self.mol_select]), axis=1)

        return df

    def mf_onepoint(self, mol, dir_df):
        """ Mole fraction calculation, one point cal through the norm_port
            Todo: add uncertainty estimate """
        det, cal = f'{mol}_det', f'{mol}_cal'
        value = f'{mol}_value'
        # unc = f'{mol}_unc'
        # normalizing cal tank value
        calvalue = dir_df.loc[(dir_df['port'] == self.ssv_norm_port)][cal].values[0]
        dir_df[value] = dir_df[det] * calvalue
        return dir_df

    def mf_twopoint(self, mol, dir_df):
        """ Mole fraction calculation, two point cal through the norm_port and
            a second port (p1).
            Todo: add uncertainty estimate """
        p0, p1 = self.ssv_norm_port, self.second_cal_port
        # column names
        det, cal = f'{mol}_det', f'{mol}_cal'
        value, flags = f'{mol}_value', f'{mol}_flag'
        # unc = f'{mol}_unc'

        dir_df1 = dir_df.copy()     # temporary dataframe
        cal0 = dir_df.loc[(dir_df['port'] == p0)][cal].values[0]  # cal val on p0
        cal1 = dir_df.loc[(dir_df['port'] == p1)][cal].values[0]  # cal val on p1
        # detrended response on p0
        dir_df1['r0'] = dir_df.loc[(dir_df['port'] == p0) & (dir_df[flags] == False)][det]
        dir_df1['r0'] = dir_df1['r0'].interpolate()
        # detrended response on p1
        dir_df1['r1'] = dir_df.loc[(dir_df['port'] == p1) & (dir_df[flags] == False)][det]
        dir_df1['r1'] = dir_df1['r1'].interpolate()
        # slope
        dir_df1['m'] = (dir_df1['r0']-dir_df1['r1']) / (cal0-cal1)
        # intercept
        dir_df1['b'] = dir_df1['r0'] - dir_df1['m'] * cal0
        dir_df[value] = (dir_df1[det] - dir_df1['b'])/dir_df1['m']
        return dir_df

    def mf_calcurve(self, mol, dir_df, caldate):
        """ Method uses a specified calibration date (caldate). The fit type
            and coefficients are stored in the calibration db. """
        value = f'{mol}_value'
        # unc = f'{mol}_unc'

        df = dir_df.copy()
        df[value] = df.apply(self.solve_caldate, args=([caldate, mol]), axis=1)
        return df

    def solve_caldate(self, df, caldate, mol):
        """ Method to be called by pandas apply function
            Calculates mole fraction using cal curve specified by caldate """

        det = df[f'{mol}_det']      # detrended response
        if pd.isna(det):
            return np.nan

        meth, coefs = self.calcurve_params(caldate, mol)
        f = getattr(self, meth)     # cal curve function

        cc = coefs.copy()
        cc[0] -= det            # subtract y value from constant offset
        initial_guess = 300     # works well
        res = least_squares(f, x0=initial_guess, args=(cc), bounds=(0, 3000))
        return res.x[0]

    def solve_meth(self, df, meth, coefs, mol):
        """ Method to be called by pandas apply function
            Calculates mole fraction using fitmethod and coefs """

        det = df[f'{mol}_det']      # detrended response
        if pd.isna(det):
            return np.nan

        f = getattr(self, meth)     # cal curve function

        cc = coefs.copy()
        cc[0] -= det            # subtract y value from constant offset
        initial_guess = 300     # works well
        res = least_squares(f, x0=initial_guess, args=(cc), bounds=(0, 3000))
        return res.x[0]
