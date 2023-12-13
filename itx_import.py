#! /usr/bin/env python

''' Added the 'wide_spike_filter' routine.  GSD 150102
    Improved the speed of parse_chroms() by factor of 3.  GSD 150218
    Capture IndexError on wide spike filter.  GSD 150508
    Now uses python3  GSD 170104
    linted GSD 191212
    
    Added smoothfile option and pathlib. 220901
    Added box smooth algo. Need to check that it works through the smooth file. 231020
'''
VERSION = 2.00

import argparse
import numpy as np
import pandas as pd
import gzip
from pathlib import Path
import shutil
from datetime import date
import re

# smoothing algos
from scipy.ndimage import convolve1d
from scipy.signal import savgol_filter


class ITX:
    """ Igor Pro Text File Class """

    def __init__(self, itxfile, saveorig=False):
        self.file = str(itxfile)
        if itxfile.stat().st_size < 100:
            print(f'File too small: {self.file}')
            self.data = None
            return

        self.data = self.load()
        self.chans = self.countchans()
        self.datafreq = self.samplefreq()       # Hz
        self.name = self.chromname()
        self.chroms = self.parse_chroms()
        if saveorig is True:
            self.org = np.copy(self.chroms)     # save original data

    def load(self):
        """ Loads content of file into memory.
            Handles gziped files too.
        """
        if self.file[-3:] == '.gz' or self.file[-2:] == '.Z':
            return [line.strip().decode() for line in gzip.GzipFile(self.file)]
        else:
            return [line.strip() for line in open(self.file)]

    @staticmethod
    def compress_to_Z(file):
        """ Compresses a file using gzip and renames to use .Z extention 
            file is a pathlib object
        """

        # rename .gz file to .Z files
        if file.suffix == '.gz':
            file.rename(file.with_suffix('.Z'))
            return
        elif file.suffix == '.Z':
            return

        # compress .itx file to .Z
        with open(file, 'rb') as f_in, gzip.open(f'{file}.Z', 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
            file.unlink()

    def countchans(self):
        """ Returns the number of columns of data.
            It is currently hard coded to line 20 of an itx file.
        """
        try:
            cols = len(self.data[20].split())
        except IndexError:
            cols = 0
        return cols

    def wavenote(self):
        """ returns date and time of injection and the SSV positions.
            this method could also return injection press and temp, etc.
            example data: X note chr4_00004, " 4; 5412; 22:21:09; 12-17-2007; 8; 31.2; 2.9; 3.0; 0.7; "
        """
        line = self.data[-2]
        if line.find('note') > -1:
            ll = line.split(';')
            date = ll[2].strip()
            time = ll[3].strip()
            ssv = int(ll[4].strip())
            return [date, time, ssv]
        else:
            return ['01-01-2001', '00:00:00', -1]

    def samplefreq(self):
        """ reads the last line of .itx file for SetScale command
            example data: X SetScale /P x, 0, 0.25, chr1_00004, chr2_00004, chr3_00004, chr4_00004
        """
        line = self.data[-1]
        if line.find('SetScale') > -1:
            rate = float(line.split(',')[2])
            return int(1/rate)
        else:
            return None

    def chromname(self):
        """ returns a string used for GCwerks file naming.  Format:  YYMMDD.HHMM.X """
        time, ddate, ssv = self.wavenote()
        mn, dd, yyyy = ddate.split('-')
        hh, mm, ss = time.split(':')
        return f'{yyyy[2:4]}{mn}{dd}.{hh}{mm}.{ssv}'
    
    def parse_chroms(self):
        """ parses data into chroms array """
        lastrow = self.chans + 2
        raw = self.data[4:-lastrow]     # string data for all channels
        try:
            return np.array([[int(x) for x in r.split()] for r in raw]).transpose()
        except ValueError:
            # handle some early fe3 files that were floats instead of ints
            return np.array([[int(float(x)*1000) for x in r.split()] for r in raw]).transpose()

    def write(self, stdout=True):
        """ writes chroms to a string.
            The string can be sent with subprocess.run to GCwerks.
            Write to stdout if piping processes together.
        """
        output = f'name {self.name}\nhz {self.datafreq}\n'
        for r in range(self.chroms.shape[1]):
            line = ''.join([str(self.chroms[c, r])+' ' for c in range(self.chans)])
            output += f'{line}\n'
        if stdout:
            print(output)
        return output

    def spike_filter(self, ch, thresh=500):
        """ Applies a spike filter.  A spike is one data point wide.
        """
        if ch == 'all':
            for c in range(self.chans):
                self.spike_filter(c)
        else:
            for pt in range(1, self.chroms.shape[1]-1):
                if (self.chroms[ch, pt-1] + self.chroms[ch, pt+1] - 2*self.chroms[ch, pt]) > thresh:
                    self.chroms[ch, pt] = (self.chroms[ch, pt-1] + self.chroms[ch, pt+1])/2

    @staticmethod
    def findgroups(indx):
        """ Returns first and last point pairs in a group of indices.  """

        if len(indx) < 2:
            return []

        indxthresh = 3     # largest gap in points
        indxdiff = [indx[i+1]-indx[i] for i in range(len(indx)-1)]

        pt1 = indx[0]
        groups = []
        for i, v in enumerate(indxdiff):
            if v > indxthresh:
                pt2 = indx[i]
                groups.append((pt1, pt2))
                pt1 = indx[i+1]
        pt2 = indx[-1]
        groups.append((pt1, pt2))

        return groups

    def wide_spike_filter(self, ch, start=40):
        # import matplotlib.pyplot as plt
        """ Removes spikes that are wider than one-point (ie the other spike filter)
              that occur after the 'start' time (seconds after injection).
            Finds spikes using 2nd derivative.  Spikes are about 2 seconds wide.
        """
        thresh = 3000

        if ch == 'all':
            for c in range(self.chans):
                self.wide_spike_filter(c, start=start)
        else:
            # print(f'wide filter, {ch} {start}')
            y = self.chroms[ch, :]
            ydd = abs(np.gradient(np.gradient(y)))       # abs of 2nd derivative
            # plt.plot(ydd)
            # plt.show()
            startpt = int(start)*self.datafreq
            idx = [i for i, v in enumerate(ydd[startpt:]) if v > thresh]      # index of spikes
            groups = self.findgroups(idx)           # group of spikes
            for spike in groups:
                pt0 = startpt + (spike[0]-1)
                pt0 = 0 if pt0 < 0 else pt0
                pt1 = startpt + (spike[1]+10)
                pt1 = len(self.chroms[ch])-1 if pt1 >= len(self.chroms[ch]) else pt1
                # replace spike with a line
                m = (self.chroms[ch, pt1] - self.chroms[ch, pt0]) / float(pt1-pt0)
                b = self.chroms[ch, pt0] - m*pt0
                self.chroms[ch, pt0:pt1] = [m*x+b for x in range(pt0, pt1)]

    def savitzky_golay(self, ch, winsize=21, order=4):
        """ applies the savitzky golay smoothing algo """
        
        if ch == 'all':
            for c in range(self.chans):
                self.savitzky_golay(c, winsize=winsize, order=order)
        else:
            # print(f'Savitzky Golay {ch} {winsize} {order}')
            y = self.chroms[ch, :]
            self.chroms[ch] = savgol_filter(y, winsize, order)

    def box_smooth(self, ch, winsize=25):
        """ applies a box smooth of winsize points on a single channel (ch) or 'all' channels """
        
        box_kernel = np.ones(winsize) / winsize

        if ch == 'all':
            for c in range(self.chans):
                self.chroms[c] = list(convolve1d(self.chroms[c], box_kernel, mode='nearest'))
        else:
            self.chroms[ch] = list(convolve1d(self.chroms[ch], box_kernel, mode='nearest'))

    """ Need to finish this
    def variable_window_smooth(self, ch, winsize):
        box_kernel0 = np.ones(winsize) / winsize
        box_kernel1 = np.ones(winsize*2) / (winsize*2)
        box_kernel2 = np.ones(winsize*4) / (winsize*4)
        p1 = 2000
        p2 = 5000

        self.chroms[ch][:p1] = list(convolve1d(self.chroms[ch][:p1], box_kernel0, mode='nearest'))
        self.chroms[ch][p1:p2] = list(convolve1d(self.chroms[ch][p1:p2], box_kernel1, mode='nearest'))
        self.chroms[ch][p2:] = list(convolve1d(self.chroms[ch][p2:], box_kernel2, mode='nearest'))
    
    def running_average(self, ch, winsize):
        if ch == 'all':
            for c in range(self.chans):
                self.variable_window_smooth(c, winsize)
        else:
            self.variable_window_smooth(ch, winsize)
    """
           
    def display(self, ch):
        import matplotlib.pyplot as plt
        num = self.chroms.shape[1]
        t = np.linspace(0, num/self.datafreq, num)
        y = self.chroms[ch, :]
        fig = plt.figure(figsize=(14, 7))
        fig.suptitle('ITX Chromatogram')
        ax = fig.add_subplot(111)
        ax.plot(t, self.org[ch, :], label='original chrom')
        ax.plot(t, y, 'r', label='smoothed chrom')
        ax.set_title(f'Channel {ch}')
        ax.set_ylabel('Response (hz)')  
        ax.set_xlabel('Time (s)')
        ax.legend()
        #ax.set_ylim([164000, 172000])
        #ax.axvline(x=2000/20, color='gainsboro')
        #ax.axvline(x=5000/20, color='gainsboro')
        # bx = fig.add_subplot(212)
        # bx.plot(np.diff(self.org[ch, :]))
        # bx.set_xlabel('Time (s)')
        plt.show()

        
class ITX_smoothfile:
    """ Class to read and parse the smoothing parameters file. This file provides a table
        of parameters to apply to ITX chromatograms per channel and by a date range. 
        
        The method smoothfile_parameters returns a pandas dataframe with the parameters and dates.
    """
    
    def __init__(self, smoothfile):
        self.smoothfile = smoothfile
        self.params_df = self.smoothfile_parameters()
        
    def process_smoothfile(self, line):
        """ Parses a line in the smoothing setup file and returns a dictionary with smoothing options.
            A typical line with smoothing parameters:
                220801: -c 0 -g -gw 61 -go 4 -W 52
            valid_options = ['-c', '-s', '-W', '-b', '-g', '-gw', '-go']
        """
        pt = line.find(':')
        date = line[:pt]
        # split command up at spaces and digits
        opts = re.split('\s+|(\d+)', line[pt+1:])
        # clean up the list (remove Nones and '')
        opts = [o for o in opts if o != None and o != '']

        opts_dict = {}
        opts_dict['date'] = date

        # get channel (required option)
        chan = 0
        try:
            chan = opts[opts.index('-c')+1]
        except ValueError:
            print('missing option -c using "-c 0" (channel 0)')
        opts_dict['chan'] = int(chan)

        # apply Savitzky Golay smoothing
        opts_dict['sg'] = False
        if '-g' in opts:
            opts_dict['sg'] = True
            opts_dict['sg_win'] = 51
            opts_dict['sg_ord'] = 4
            if '-gw' in opts:
                sg_win = opts[opts.index('-gw')+1]
                opts_dict['sg_win'] = int(sg_win)
            if '-go' in opts:
                sg_ord = opts[opts.index('-go')+1]
                opts_dict['sg_ord'] = int(sg_ord)
            #print(f'Savitzky Golay filter, {sg_win}, {sg_ord} on channel {chan} starting on {date}')

        # box smoothing
        if '-b' in opts:
            opts_dict['boxwidth'] = opts[opts.index('-b')+1]

        # apply 1 second spike filter
        opts_dict['spike'] = False
        if '-s' in opts:
            opts_dict['spike'] = True
            #print(f'spike filter on channel {chan} starting on {date}')

        # apply wide spike filter
        opts_dict['wide_spike'] = False
        if '-W' in opts:
            opts_dict['wide_spike'] = True
            wide_start = opts[opts.index('-W')+1]
            opts_dict['wide_start'] = int(wide_start)
            #print(f'wide spike filter on channel {chan} start_time = {wide_start} starting on {date}')

        return opts_dict

    def smoothfile_parameters(self):
        """ load smoothfile parameters and return a dataframe """

        print(self.smoothfile)
        with open(self.smoothfile) as file:
            df_line = []
            n = 0
            while (line := file.readline().lstrip().rstrip()):
                # skip comment lines starting with #
                if line[0] != '#':
                    params = self.process_smoothfile(line)
                    df_line.append(pd.DataFrame(params, index=[n]))
                    n += 1

        df = pd.concat(df_line)

        # Either 6 or 8 character dates are allowed. They have to be consistant throughout the smoothfile.
        dateformat = len(df.iloc[0]['date'])
        if dateformat == 6:
            df['dt'] = pd.to_datetime(df['date'], format='%y%m%d')
        else:
            df['dt'] = pd.to_datetime(df['date'], format='%Y%m%d')

        df = df.set_index('dt')
        df = df.drop('date', axis=1)
        df = df.sort_index()

        return df    


if __name__ == '__main__':

    SGwin, SGorder = 21, 4      # Savitzky Golay default variables
    WSTART = -1                 # Wide spike filter start time
    yyyy = date.today().year

    parser = argparse.ArgumentParser(description='Smooth and display chromatograms \
        that are in the Igor Pro Text file format (.itx).')
    parser.add_argument('-s', action='store_true', default=False,
                        help='Apply 1-point spike filter (default off)')
    parser.add_argument('-W', action='store', dest='ws_start', default=WSTART, type=int,
                        help='Apply wide spike filter (default off)')
    parser.add_argument('-b', action='store', dest='boxwidth', metavar='Win', type=int,
                        help=f'Apply a Box smooth with window width in points')
    parser.add_argument('-g', action='store_true', default=False,
                        help='Apply Savitzky Golay smoothing (default off)')
    parser.add_argument('-gw', action='store', dest='SGwin', metavar='Win', default=SGwin, type=int,
                        help=f'Sets Savitzky Golay smoothing window (default = {SGwin} points)')
    parser.add_argument('-go', action="store", dest='SGorder', metavar='Order', default=SGorder, type=int,
                        help=f'Sets Savitzky Golay order of fit (default = {SGorder})')
    parser.add_argument('-file', action='store', dest='smoothfile', type=str,
                        help='Use the smoothing parameters defined in this file.')
    parser.add_argument('-d', action='store', dest='chan', type=int, default=-1,
                        help='Display original chrom and exported data for a channel')
    parser.add_argument(dest='itxfile', help='ITX chromatogram file to process')

    args = parser.parse_args()
    itxfile = Path(args.itxfile)
    
    if args.chan >= 0:
        # keeps a copy of the original data to compare when displayed
        chroms = ITX(itxfile, saveorig=True)
    else:
        chroms = ITX(itxfile)

    # use either the smoothfile or the parameters defined on the command line (one or the other)
    if args.smoothfile:
        sm = ITX_smoothfile(Path(args.smoothfile))
        itxtime = pd.to_datetime(chroms.name[:11], format='%y%m%d.%H%M')

        for ch in range(chroms.chans):
            # find the smoothing parameter date appropriate for the itxfile
            params = sm.params_df.loc[(sm.params_df.index < itxtime) & (sm.params_df.chan == ch)].iloc[-1]
            
            if params.spike:
                chroms.spike_filter(ch)
            if params.wide_spike:
                chroms.wide_spike_filter(ch, start=params.wide_start)
            if params.sg:
                chroms.savitzky_golay(ch, winsize=params.sg_win, order=params.sg_ord)
                
        # display chrom?
        if args.chan >= 0:
            chroms.display(args.chan)
        quit()


    # apply spike filter before SG smoothing
    if args.s:
        chroms.spike_filter('all')

    # apply wide spike filter?
    if args.ws_start > 0:
        chroms.wide_spike_filter('all', start=args.ws_start)

    # apply Savitzky Golay smoothing
    if args.g:
        chroms.savitzky_golay('all', winsize=args.SGwin, order=args.SGorder)
    if args.boxwidth:
        chroms.box_smooth('all', args.boxwidth)

    # display chrom?
    if args.chan >= 0:
        chroms.display(args.chan)
        quit()

    chroms.write()
