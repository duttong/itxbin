from PyQt5 import QtCore, QtWidgets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import num2date, AutoDateFormatter, AutoDateLocator, DateFormatter

import fe3_panel
from processing_routines import DataProcessing

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)


class FE3_Process(QtWidgets.QMainWindow, fe3_panel.Ui_MainWindow, DataProcessing):

    def __init__(self):
        super().__init__()
        DataProcessing.__init__(self)
        self.fe3data = self.fe3db.db
        self.cals = self.fe3cals.cals
        self.calcurves = self.fe3curves.calcurves_df
        self.mols = [c[0:c.find('_ht')] for c in self.fe3data.columns if c.find('_ht') > 0]
        self.mol_select = 'CFC11'
        self.sub = pd.DataFrame()
        self.run_selected = ''
        self.fits = ['linear', 'quadratic', 'cubic', 'exponential']
        self.methods = ['one-point', 'two-points']
        self.madechanges = False    # if True, save the DB when the app quits
        # different color for each SSV port (port 2 is for flask SSV)
        self.colors = {0: 'cornflowerblue', 1: 'green', 2: 'red', 3: 'cyan', 4: 'pink',
                       5: 'gray', 6: 'orange', 7: 'darkgreen', 8: 'darkred', 9: 'orange'}
        self.flaggedcolor = (0, 0, 0, 1)
        self.flaskbgcolor = 'moccasin'
        self.flasktextcolor = 'maroon'
        # different marker for each flask port
        self.markers = {0: '$1$', 1: '$2$', 2: '$3$', 3: '$4$', 4: '$5$',
                        5: '$6$', 6: '$7$', 7: '$8$'}
        self.markersize = 75    # points

        self.setupUi(self)

        # pick event setup for flagging data
        self.mpl_plot.canvas.mpl_connect('pick_event', self.onpick)

        # buttons for molecule selection
        self.buttonGroup_mols1.buttonClicked.connect(self.mol_clicked)
        self.buttonGroup_mols2.buttonClicked.connect(self.mol_clicked)

        # tables in flask/port tab
        self.table_ports.setColumnCount(2)
        self.table_ports.setRowCount(self.MAX_N_SSVports)
        self.table_ports.setHorizontalHeaderLabels(('SSV ports', 'Tank value'))
        self.table_flasks.setColumnCount(1)
        self.table_flasks.setRowCount(self.MAX_N_Flasks)
        self.table_flasks.setHorizontalHeaderLabels(('Flask ID',))

        # display control
        self.fig_response.setChecked(True)
        self.buttonGroup_figs.buttonClicked.connect(self.update_data_fig)
        self.checkBox_one2one.toggled.connect(self.update_data_fig)
        self.checkBox_scale0.toggled.connect(self.update_data_fig)

        self.comboBox_calcurve.currentIndexChanged.connect(self.update_method_field)

        self.actionDelete_Selected_Run.triggered.connect(self.delete_run)

        self.initialize_run_filtering()

    def mol_clicked(self, button):
        self.mol_select = button.text().replace('-', '')
        self.mol_select = self.mol_select.replace(' (b)', 'b')
        # make both buttons change together
        mainbutton = getattr(self, f'button_{self.mol_select}')
        mainbutton.setChecked(True)
        altbutton = getattr(self, f'button_{self.mol_select}_2')
        altbutton.setChecked(True)
        self.update_data_fig()

    # Run selection code below
    def initialize_run_filtering(self):
        """ Code for selecting a subset of runs from runs_df
            box_runs: are the current list of runs to view
            box_types: Select a run type like flask or calibration
            box_dates: Choose a period of time for runs to be selected.
        """
        # initial box_types set to 'flask' (the run types are stored in runs_df)
        # runtypes = sorted(list(set(self.fe3data['type'])))
        runtypes = ['flask', 'calibration', 'all']
        self.box_types.addItems(runtypes)
        idx = runtypes.index('flask')
        self.box_types.setCurrentIndex(idx)

        # initial box_dates is set to past-two-weeks
        date_items = ['last-two-weeks', 'last-two-months', 'last-year', 'all']
        self.box_dates.addItems(date_items)
        self.box_dates.setCurrentIndex(0)

        # setup signals
        self.box_types.currentIndexChanged.connect(self.create_run_list)
        self.box_dates.currentIndexChanged.connect(self.create_run_list)
        self.box_runs.currentIndexChanged.connect(self.create_run_list)

        self.create_run_list()

    def create_run_list(self):
        """ Set up the list of runs using box_types and box_dates criterias """
        type = self.box_types.currentText()
        dates = self.box_dates.currentText()

        # create 'series' subset of all dirs
        if type == 'all':
            series = self.fe3data.dir
        if type == 'calibration':
            series = self.fe3data.loc[self.fe3data.type == 'other'].dir
        else:
            series = self.fe3data.loc[self.fe3data.type == type].dir

        if dates == 'last-two-weeks':
            series = series.last('2W')
        elif dates == 'last-two-months':
            series = series.last('2M')
        elif dates == 'last-year':
            series = series.last('Y')

        runs = series.drop_duplicates().tolist()    # use the dir (directory) column
        runs.sort(reverse=True)
        self.box_runs.currentIndexChanged.disconnect()
        self.box_runs.clear()
        self.box_runs.addItems(runs)
        self.box_runs.setCurrentIndex(0)
        self.box_runs.currentIndexChanged.connect(self.update_data_fig)

        self.update_data_fig()

    # code below for plotting figures
    def unique_ports(self, df, remove_flask_port=False):
        """ Returns a list of port numbers (with or
            without the flask_port) """
        ports = list(df['port'].unique())
        if remove_flask_port:
            try:
                ports.remove(self.ssv_flask_port)
            except ValueError:
                pass
        return sorted(ports)

    def portlist(self, df):
        """ Creates a list of tanks on the SSV for each port """
        plist = []
        for p in range(self.MAX_N_SSVports):
            if p == 0:
                val = df.loc[df['port'] == 10, 'port_id'][0]
            elif p == 2:
                val = 'Stop port'
            else:
                try:
                    val = df.loc[df['port'] == p, 'port_id'][0]
                except IndexError:
                    val = ''
            plist.append(val)

        return plist

    def flasklist(self, df):
        """ Creates a list of flask_id on the flask SSV """
        sub = df.dropna()   # all flask_port are nan unless there is a flask on
        flist = []
        for p in range(self.MAX_N_Flasks):
            try:
                val = sub.loc[sub['flask_port'] == p, 'port_id'][0]
            except IndexError:
                val = ''
            flist.append(val)
        return flist

    def subset_fe3data(self):
        """ a subset of the full FE3 dataframe based on the selected run
            and molecule. """

        self.run_selected = self.box_runs.currentText()
        df = self.fe3data.loc[self.fe3data['dir'] == self.run_selected]
        df = self.reduce_df(df, self.mol_select)
        self.sub = self.add_det_cal_columns(df, self.mol_select)

    def update_method_field(self):
        """ Save changes to DB if the lowess or p2p detrend
            buttons are toggled. """

        dir = self.sub['dir'].values[0]
        meth = f'{self.mol_select}_methdet'
        det = 'lowess' if self.detrend_lowess.isChecked() else 'p2p'
        self.fe3data.loc[self.fe3data['dir'] == dir, meth] = f'{det}'

        meth = f'{self.mol_select}_methcal'
        self.fe3data.loc[self.fe3data['dir'] == dir, meth] =    \
            self.comboBox_calcurve.currentText()

        self.madechanges = True     # set to True to save changes to db
        self.update_data_fig()

    def update_data_fig(self):
        """ Sets toggle and comboBoxes to stored values """

        self.subset_fe3data()

        # use method columns to set detrend and cal curve fields
        det = self.sub[f'{self.mol_select}_methdet'].values[0]
        fit = self.sub[f'{self.mol_select}_methcal'].values[0]
        type = self.box_types.currentText()

        # update detrend toggle buttons
        self.detrend_lowess.disconnect()
        if det == 'lowess':
            self.detrend_lowess.setChecked(True)
        else:
            self.detrend_linear.setChecked(True)
        self.detrend_lowess.toggled.connect(self.update_method_field)

        # update cal curve comboBox which depends on run type.
        self.comboBox_calcurve.currentIndexChanged.disconnect()
        self.comboBox_calcurve.clear()
        if type == 'flask':
            # which cal curve to use for mole fraction calculation
            methods = self.methods + self.nearest_calcurves(self.sub['dir'].values[0])
            fit = 'two-points' if fit == 'quadratic' else fit   # a fix if db is wrong
            idx = methods.index(fit)
            self.comboBox_calcurve.addItems(methods)
        else:
            idx = self.fits.index(fit)
            self.comboBox_calcurve.addItems(self.fits)
        try:
            self.comboBox_calcurve.setCurrentIndex(idx)   # update pull down menu
        except ValueError:
            self.comboBox_calcurve.setCurrentIndex(1)     # update pull down menu
        self.comboBox_calcurve.currentIndexChanged.connect(self.update_method_field)

        # choose which figure to display
        if self.fig_detrend.isChecked():
            self.detrend_response_fig(self.run_selected)
        elif self.fig_response.isChecked():
            self.response_fig(self.run_selected)
        elif self.fig_calibration.isChecked():
            self.calibration_curve_fig(self.run_selected)
        elif self.fig_molefractions.isChecked():
            self.molefractions_fig(self.run_selected)
        else:
            print('not coded')

        self.update_tables()

    def response_fig(self, title_text):
        df = self.sub.copy()

        port_list = self.portlist(df)

        resp = f'{self.mol_select}_ht'      # hardcoded to height
        flags = f'{self.mol_select}_flag'

        self.checkBox_scale0.setEnabled(False)
        self.checkBox_one2one.setEnabled(False)
        self.fig_calibration.setChecked(False)

        self.mpl_plot.canvas.ax1.clear()
        self.mpl_plot.canvas.ax2.clear()
        self.mpl_plot.canvas.ax1.set_visible(False)
        self.mpl_plot.canvas.ax2.set_position([0.11, 0.1, .68, .85])
        # this is needed for refreshing from shared xaxis from calibration fig
        shax = self.mpl_plot.canvas.ax1.get_shared_x_axes()
        shax.remove(self.mpl_plot.canvas.ax1)

        # plot all ports except flask port
        for p in self.unique_ports(df, remove_flask_port=True):
            x = df.loc[df['port'] == p].index
            y = df.loc[df['port'] == p][resp]

            c = self.colors[p % 10]
            facecolors = [c if flag == False else self.flaggedcolor for flag in df.loc[df['port'] == p][flags]]
            edgecolors = c
            port_label = f'({p}) {port_list[p % 10]}'

            self.mpl_plot.canvas.ax2.scatter(x, y,
                color=facecolors, edgecolors=edgecolors, s=self.markersize,
                label=port_label, picker=1)

        # individual flask data on flask port
        if self.ssv_flask_port in self.unique_ports(df):
            for n in range(self.MAX_N_Flasks):
                x = df.loc[df['flask_port'] == n].index
                y = df.loc[df['flask_port'] == n][resp]

                facecolors = [self.flaskbgcolor if flag == False else self.flaggedcolor for flag in df.loc[df['flask_port'] == n][flags]]
                edgecolors = self.flaskbgcolor

                # square marker behind numbered flask markers, this is used for flagging (the picker)
                self.mpl_plot.canvas.ax2.scatter(x, y,
                    marker='s', color=facecolors, edgecolors=edgecolors,
                    s=self.markersize, picker=1)
                # numbered flask markers
                label = df.loc[df['flask_port'] == n].port_id.values[0]
                self.mpl_plot.canvas.ax2.scatter(x, y,
                    marker=self.markers[n],
                    color=self.flasktextcolor, label=label)

        # detrend linear or lowess smooth line
        norm = df.loc[(df['port'] == self.ssv_norm_port) & (df[flags] == False)][resp]
        if self.detrend_linear.isChecked():
            sm = norm
        else:
            sm = self.make_lowess(norm)
        self.mpl_plot.canvas.ax2.plot(sm, color='darkgreen', linestyle='dashed')

        xtick_locator = AutoDateLocator()
        xtick_formatter = AutoDateFormatter(xtick_locator)
        xtick_formatter = DateFormatter('%H:%M:%S')
        # xtick_formatter.scaled[1/(24.*60.)] = '%H:%M:%S'
        self.mpl_plot.canvas.ax2.xaxis.set_major_locator(xtick_locator)
        self.mpl_plot.canvas.ax2.xaxis.set_major_formatter(xtick_formatter)

        uppermargin = .6
        plt.setp(self.mpl_plot.canvas.ax2.get_xticklabels(), rotation=15)
        self.mpl_plot.canvas.ax2.set_ylabel('height')
        yyyymmdd = f'{self.run_selected[0:4]}-{self.run_selected[4:6]}-{self.run_selected[6:8]}'
        self.mpl_plot.canvas.ax2.set_xlabel(yyyymmdd)
        self.mpl_plot.canvas.ax2.set_title(f'Response: {title_text}', loc='right')
        self.mpl_plot.canvas.ax2.legend(fontsize=8, loc=(1.04, uppermargin))
        self.mpl_plot.canvas.ax2.tick_params(labelsize=8)
        # self.mpl_plot.canvas.fig.tight_layout()
        self.mpl_plot.canvas.draw()
        self.mpl_plot.toolbar.update()

    def detrend_response_fig(self, title_text):
        """ Figure used to display detrended response data """
        df = self.sub.copy()
        flags = f'{self.mol_select}_flag'
        det = f'{self.mol_select}_det'

        port_list = self.portlist(df)

        self.checkBox_scale0.setEnabled(False)
        self.checkBox_one2one.setEnabled(False)

        self.mpl_plot.canvas.ax1.clear()
        self.mpl_plot.canvas.ax2.clear()
        self.mpl_plot.canvas.ax1.set_visible(False)
        self.mpl_plot.canvas.ax2.set_position([0.1, 0.1, .68, .85])
        # this is needed for refreshing from shared xaxis from calibration fig
        shax = self.mpl_plot.canvas.ax1.get_shared_x_axes()
        shax.remove(self.mpl_plot.canvas.ax1)

        for p in self.unique_ports(df, remove_flask_port=True):
            # all data for a selected port
            x = df.loc[df['port'] == p].index
            y = df.loc[df['port'] == p][det]
            # all unflagged data for a selected port
            ygood = df.loc[(df['port'] == p) & (df[flags] == False)][det]

            avg, std = ygood.mean(), ygood.std()
            if np.isnan(avg):
                continue

            c = self.colors[p % 10]
            facecolors = [c if flag == False else self.flaggedcolor for flag in df.loc[df['port'] == p][flags]]
            edgecolors = c

            port_label = f'({p}) {port_list[p%10]}'
            port_label = f'{port_label}\n{avg:0.3f} ± {std:0.3f}'

            self.mpl_plot.canvas.ax2.scatter(x, y,
                color=facecolors, edgecolors=edgecolors, s=self.markersize,
                label=port_label, picker=1)

        # individual flask data on flask port
        if self.ssv_flask_port in self.unique_ports(df):
            for n in range(self.MAX_N_Flasks):
                # all data for a selected port
                x = df.loc[df['flask_port'] == n].index
                y = df.loc[df['flask_port'] == n][det]
                # all unflagged data for a selected port
                ygood = df.loc[(df['flask_port'] == n) & (df[flags] == False)][det]

                avg, std = ygood.mean(), ygood.std()
                if np.isnan(avg):
                    continue

                port_label = df.loc[df['flask_port'] == n].port_id.values[0]
                port_label = f'{port_label}\n{avg:0.3f} ± {std:0.3f}'
                facecolors = [self.flaskbgcolor if flag == False else self.flaggedcolor for flag in df.loc[df['flask_port'] == n][flags]]
                edgecolors = self.flaskbgcolor

                # square marker behind numbered flask markers, this is used for flagging (the picker)
                self.mpl_plot.canvas.ax2.scatter(x, y,
                    marker='s', color=facecolors, edgecolors=edgecolors,
                    s=self.markersize, picker=1)
                # numbered flask markers
                self.mpl_plot.canvas.ax2.scatter(x, y,
                    marker=self.markers[n],
                    color=self.flasktextcolor, label=port_label)

        xtick_locator = AutoDateLocator()
        xtick_formatter = AutoDateFormatter(xtick_locator)
        xtick_formatter = DateFormatter('%H:%M:%S')
        # xtick_formatter.scaled[1/(24.*60.)] = '%H:%M:%S'
        self.mpl_plot.canvas.ax2.xaxis.set_major_locator(xtick_locator)
        self.mpl_plot.canvas.ax2.xaxis.set_major_formatter(xtick_formatter)

        uppermargin = .3
        plt.setp(self.mpl_plot.canvas.ax2.get_xticklabels(), rotation=15)
        self.mpl_plot.canvas.ax2.set_ylabel('normalized response')
        yyyymmdd = f'{self.run_selected[0:4]}-{self.run_selected[4:6]}-{self.run_selected[6:8]}'
        self.mpl_plot.canvas.ax2.set_xlabel(yyyymmdd)
        self.mpl_plot.canvas.ax2.set_title(f'Detrended: {title_text}', loc='right')
        self.mpl_plot.canvas.ax2.legend(fontsize=8, loc=(1.04, uppermargin))
        self.mpl_plot.canvas.ax2.tick_params(labelsize=8)
        self.mpl_plot.canvas.draw()
        self.mpl_plot.toolbar.update()

    def calibration_curve_fig(self, title_text):
        df = self.sub.copy()
        flags = f'{self.mol_select}_flag'
        cal = f'{self.mol_select}_cal'
        det = f'{self.mol_select}_det'
        type = self.box_types.currentText()

        self.checkBox_scale0.setEnabled(True)
        self.checkBox_one2one.setEnabled(True)

        self.mpl_plot.canvas.ax1.set_visible(True)
        self.mpl_plot.canvas.ax1.clear()
        self.mpl_plot.canvas.ax2.clear()
        # share x axis
        self.mpl_plot.canvas.ax2.get_shared_x_axes().join(self.mpl_plot.canvas.ax1, self.mpl_plot.canvas.ax2)
        self.mpl_plot.canvas.ax1.set_position([0.11, 0.70, .64, .24])
        self.mpl_plot.canvas.ax2.set_position([0.11, 0.08, .64, .60])
        self.mpl_plot.canvas.ax1.tick_params(axis='x', which='both', length=0)

        cc = self.comboBox_calcurve.currentText()

        # which type of run is displayed?
        # if flasks, then use the cal curves that have already been calculated
        # if other, then calculate and save the cal curve.
        if type == 'flask':
            if cc == 'one-point' or cc == 'two-points':
                fit = cc
                x, y = self.unflagged_data(fit, df, self.mol_select)
                coefs, x_fit, y_fit = self.calculate_calcurve(cc, df, self.mol_select, scale0=self.checkBox_scale0.isChecked())
            else:
                fit, coefs = self.calcurve_params(cc, self.mol_select)
                x, y = self.unflagged_data(fit, df, self.mol_select)
                xmin = 0 if self.checkBox_scale0.isChecked() else min(x)
                span = max(x) - xmin
                x_fit = np.linspace(xmin-span*0.05, max(x)+span*0.05, 100)
                y_fit = self.calcurve_values(cc, self.mol_select, x_fit)
        else:
            # not flask, calibration run instead. Calculate a cal curve instead.
            fit = self.sub[f'{self.mol_select}_methcal'].values[0]
            x, y = self.unflagged_data(fit, df, self.mol_select)
            coefs, x_fit, y_fit = self.calculate_calcurve(fit, df, self.mol_select, scale0=self.checkBox_scale0.isChecked())
            self.save_calcurve(df, self.mol_select, coefs)

        if len(x) > 1:
            # one-to-one fit
            calval = df.loc[df['port'] == self.ssv_norm_port, cal].values[0]
            x_one2one = np.linspace(min(x_fit), max(x_fit), 100)
            y_one2one = x_one2one / calval
        else:
            x_one2one, y_one2one = [], []

        # plot residuals
        self.mpl_plot.canvas.ax1.grid(True)
        for p in self.unique_ports(df, remove_flask_port=True):
            x = df.loc[(df['port'] == p) & (df[flags] == False), cal]
            y = df.loc[(df['port'] == p) & (df[flags] == False), det]
            if fit == 'one-point' or fit == 'two-points':
                f = self.linear
            else:
                f = getattr(self, fit)
            resid = f(x, *coefs) - y        # residual

            c = self.colors[p % 10]
            if p != 10:
                avg, std = resid.mean(), resid.std()
                port_label = f'({p}) {avg:0.3f} ± {std:0.3f}'

                self.mpl_plot.canvas.ax1.scatter(x, resid,
                    color=c, edgecolors=c, s=75,
                    label=port_label)

        # ax2 (bottom figure) legend label
        if fit.find('exponential') >= 0:
            fitlabel = 'exponential fit\n' + r'$a + b e^{cx}$' + '\n'
            vars = ['a', 'b', 'c', 'd']
            for n, coef in enumerate(coefs):
                fitlabel += f'({vars[n]}) {coef:0.6f}\n'
        else:
            fitlabel = f'{fit} fit\n'
            for n, coef in enumerate(coefs):
                fitlabel += f'($x^{n}$) {coef:0.6f}\n'

        # add fits to figure
        self.mpl_plot.canvas.ax2.plot(x_fit, y_fit, 'k-', label=fitlabel)
        if self.checkBox_one2one.isChecked():
            self.mpl_plot.canvas.ax2.plot(x_one2one, y_one2one, c='grey', ls='--', label='one-to-one')

        # plot detrended response vs cals
        self.mpl_plot.canvas.ax2.grid(True)
        port_list = self.portlist(df)
        for p in self.unique_ports(df, remove_flask_port=True):
            x = df.loc[df['port'] == p, cal]
            y = df.loc[df['port'] == p, det]

            c = self.colors[p % 10]
            facecolors = [c if flag == False else self.flaggedcolor for flag in df.loc[df['port'] == p][flags]]
            edgecolors = c
            if p != 10:
                port_label = f'({p}) {port_list[p%10]}'

                self.mpl_plot.canvas.ax2.scatter(x, y,
                    color=facecolors, edgecolors=edgecolors, s=75,
                    label=port_label, picker=1)

        uppermargin = .5
        self.mpl_plot.canvas.ax1.set_title(f'Calibration Curve: {title_text}', loc='right')
        self.mpl_plot.canvas.ax1.set_ylabel('residual')
        self.mpl_plot.canvas.ax2.set_ylabel('normalized response')
        self.mpl_plot.canvas.ax2.set_xlabel('mole fraction')
        # see below on controlling the legend order
        # https://stackoverflow.com/questions/22263807/how-is-order-of-items-in-matplotlib-legend-determined
        self.mpl_plot.canvas.ax1.legend(fontsize=8, loc=(1.04, uppermargin))
        self.mpl_plot.canvas.ax1.tick_params(labelsize=8)
        self.mpl_plot.canvas.ax2.legend(fontsize=8, loc=(1.04, uppermargin))
        self.mpl_plot.canvas.ax2.tick_params(labelsize=8)
        # hide ticklabels on ax1
        plt.setp(self.mpl_plot.canvas.ax1.get_xticklabels(), visible=False)
        self.mpl_plot.canvas.draw()

        self.mpl_plot.canvas.ax1.set_zorder(0)
        self.mpl_plot.canvas.ax2.set_zorder(1)
        self.mpl_plot.toolbar.update()  # store default settings

    def molefractions_fig(self, title_text):
        """ Figure used to display mole fractions """
        self.checkBox_scale0.setEnabled(False)
        self.checkBox_one2one.setEnabled(False)

        df = self.sub.copy()
        flags = f'{self.mol_select}_flag'
        det = f'{self.mol_select}_det'
        value = f'{self.mol_select}_value'

        df[det] = self.detrend_response(df, self.mol_select)
        port_list = self.portlist(df)

        # which cal curve method to use.
        meth = self.comboBox_calcurve.currentText()
        df = self.molefraction_calc(df, self.mol_select, meth)

        # update full dataframe
        self.fe3data.loc[self.fe3data['dir'] == self.run_selected, value] = df[value]
        self.madechanges = True

        # setup plot axes
        self.mpl_plot.canvas.ax1.clear()
        self.mpl_plot.canvas.ax2.clear()
        self.mpl_plot.canvas.ax1.set_visible(False)
        self.mpl_plot.canvas.ax2.set_position([0.1, 0.1, .68, .85])
        # this is needed for refreshing from shared xaxis from the calibration fig
        shax = self.mpl_plot.canvas.ax1.get_shared_x_axes()
        shax.remove(self.mpl_plot.canvas.ax1)

        # non flask data ports
        for p in self.unique_ports(df, remove_flask_port=True):
            # all data for a selected port
            x = df.loc[df['port'] == p].index
            y = df.loc[df['port'] == p][value]
            # all unflagged data for a selected port
            ygood = df.loc[(df['port'] == p) & (df[flags] == False)][value]

            avg, std = ygood.mean(), ygood.std()
            if np.isnan(avg):
                continue

            c = self.colors[p % 10]
            facecolors = [c if flag == False else self.flaggedcolor for flag in df.loc[df['port'] == p][flags]]
            edgecolors = c

            port_label = f'({p}) {port_list[p % 10]}'
            port_label = f'{port_label}\n{avg:0.2f} ± {std:0.2f}'

            self.mpl_plot.canvas.ax2.scatter(x, y,
                color=facecolors, edgecolors=edgecolors, s=self.markersize,
                label=port_label, picker=1)

        # individual flask data on flask port
        if self.ssv_flask_port in self.unique_ports(df):
            for n in range(self.MAX_N_Flasks):
                # all data for a selected port
                x = df.loc[df['flask_port'] == n].index
                y = df.loc[df['flask_port'] == n][value]
                # all unflagged data for a selected port
                ygood = df.loc[(df['flask_port'] == n) & (df[flags] == False)][value]

                avg, std = ygood.mean(), ygood.std()
                if np.isnan(avg):
                    continue

                port_label = df.loc[df['flask_port'] == n].port_id.values[0]
                port_label = f'{port_label}\n{avg:0.2f} ± {std:0.2f}'
                facecolors = [self.flaskbgcolor if flag == False else self.flaggedcolor for flag in df.loc[df['flask_port'] == n][flags]]
                edgecolors = self.flaskbgcolor

                # square marker behind numbered flask markers, this is used for flagging (the picker)
                self.mpl_plot.canvas.ax2.scatter(x, y,
                    marker='s', color=facecolors, edgecolors=edgecolors,
                    s=self.markersize, picker=1)
                # numbered flask markers
                self.mpl_plot.canvas.ax2.scatter(x, y,
                    marker=self.markers[n],
                    color=self.flasktextcolor, label=port_label)

        xtick_locator = AutoDateLocator()
        xtick_formatter = AutoDateFormatter(xtick_locator)
        xtick_formatter = DateFormatter('%H:%M:%S')
        self.mpl_plot.canvas.ax2.xaxis.set_major_locator(xtick_locator)
        self.mpl_plot.canvas.ax2.xaxis.set_major_formatter(xtick_formatter)

        plt.setp(self.mpl_plot.canvas.ax2.get_xticklabels(), rotation=15)
        self.mpl_plot.canvas.ax2.set_ylabel(f'mole fraction ({self.units()})')
        yyyymmdd = f'{self.run_selected[0:4]}-{self.run_selected[4:6]}-{self.run_selected[6:8]}'
        self.mpl_plot.canvas.ax2.set_xlabel(yyyymmdd)
        self.mpl_plot.canvas.ax2.set_title(f'{meth} mole fractions: {title_text}', loc='right')
        self.mpl_plot.canvas.ax2.legend(fontsize=8, loc=(1.04, 0.3))
        self.mpl_plot.canvas.ax2.tick_params(labelsize=8)
        self.mpl_plot.canvas.draw()
        self.mpl_plot.toolbar.update()

    def units(self):
        """ Method returns abbreviated mole fraction units. """
        if self.mol_select.upper() == 'N2O':
            return 'ppb'
        return 'ppt'

    def onpick(self, event):
        """ The picker tolerance set in scatter(), the onpick method will
            set and unset flags. """
        artist = event.artist
        point = event.ind
        # the picker may choose more than one point. Use first point.
        point = point.T[0]
        x, y = artist.get_offsets()[point].T
        # picked from calibration figures (x units are in mole fraction
        # instead of date)
        calfig = True if x < 2000 else False

        # print(artist.properties())     # method to see what is in the artist
        fc = artist.properties()['facecolor']
        ec = artist.properties()['edgecolor']

        flagcol = f'{self.mol_select}_flag'
        cal = f'{self.mol_select}_cal'
        det = f'{self.mol_select}_det'
        ht = f'{self.mol_select}_ht'

        # get the current flag value
        if calfig:
            # xaxis units in cal mol fraction
            # need to locate det,cal pair in the whole dataframe (self.fe3data)
            dir = self.sub.iloc[0].dir
            height = self.sub.loc[(self.sub[cal] == x) & (self.sub[det] == y), ht].values[0]
            pt = self.fe3data.loc[(self.fe3data['dir'] == dir) & (self.fe3data[ht] == height)]
            flag = pt[flagcol].values
        else:
            # xaxis units in pandas datetime
            flag = self.fe3data.loc[self.fe3data.index == num2date(x)][flagcol].values

        if flag == True:
            # toggled to no flag (False)
            if calfig:
                self.fe3data.loc[(self.fe3data['dir'] == dir) & (self.fe3data[ht] == height), flagcol] = False
            else:
                self.fe3data.loc[self.fe3data.index == num2date(x), flagcol] = False
            fc[point, :] = ec      # edgecolor never changes, can use to set facecolor
            artist.set_facecolors(fc)
        else:
            # toggled to flagged
            if calfig:
                self.fe3data.loc[(self.fe3data['dir'] == dir) & (self.fe3data[ht] == height), flagcol] = True
            else:
                self.fe3data.loc[self.fe3data.index == num2date(x), flagcol] = True
            fc[point, :] = self.flaggedcolor
            artist.set_facecolors(fc)

        self.madechanges = True
        self.update_data_fig()

    def update_tables(self):
        """ Updates text in flask and port data tables """
        port_list = self.portlist(self.sub)
        flask_list = self.flasklist(self.sub)

        # flask id numbers
        for row in range(self.MAX_N_Flasks):
            cell = QtWidgets.QTableWidgetItem(f'{flask_list[row]}')
            cell.setTextAlignment(QtCore.Qt.AlignCenter)
            self.table_flasks.setItem(row, 0, cell)

        self.table_ports.setHorizontalHeaderLabels(('SSV ports', f'{self.mol_select} value'))

        # cal tanks
        for row in range(self.MAX_N_SSVports):
            # first column is tank ID
            cell0 = QtWidgets.QTableWidgetItem(f'{port_list[row]}')
            cell0.setTextAlignment(QtCore.Qt.AlignCenter)

            # second column is cal value
            try:
                mol = self.mol_select if self.mol_select[-1] != 'b' else self.mol_select[:-1]
                calvalue = self.cals.loc[f'{port_list[row]}', mol]
            except KeyError:
                calvalue = ''
            cell1 = QtWidgets.QTableWidgetItem(f'{calvalue}')
            cell1.setTextAlignment(QtCore.Qt.AlignCenter)
            listrow = row-1 if row != 0 else self.MAX_N_SSVports-1
            self.table_ports.setItem(listrow, 0, cell0)
            self.table_ports.setItem(listrow, 1, cell1)
            self.table_ports.setItemDelegate(FloatDelegate(2, self.table_ports))

    def delete_run(self):
        msg = QtWidgets.QMessageBox()
        msg.setWindowTitle('Deleting an FE3 run')
        msg.setText('Permanently delete all data for this run?')
        msg.setIcon(QtWidgets.QMessageBox.Warning)
        msg.setStandardButtons(QtWidgets.QMessageBox.Cancel | QtWidgets.QMessageBox.Yes)
        msg.setDefaultButton(QtWidgets.QMessageBox.Cancel)

        returnValue = msg.exec()
        if returnValue == QtWidgets.QMessageBox.Yes:
            print(f'Deleting {self.run_selected}. Not implemented yet.')

            # update db
            # self.fe3data = self.fe3data.loc[self.fe3data['dir'] != self.run_selected]
            # self.fe3db.save_db_file()

            # delete run directory
            # yy = self.run_selected[2:4]
            # p = self.fe3db.basepath / yy / self.run_selected


class FloatDelegate(QtWidgets.QStyledItemDelegate):
    """ To report floating point numbers in Qt table """

    def __init__(self, decimals, parent=None):
        super(FloatDelegate, self).__init__(parent=parent)
        self.nDecimals = decimals

    def displayText(self, value, locale):
        try:
            number = float(value)
        except ValueError:
            return super(FloatDelegate, self).displayText(value, locale)
        else:
            return locale.toString(number, format="f", precision=self.nDecimals)
