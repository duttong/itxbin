#! /home/hats/gdutton/anaconda3/bin/python
''' Added the 'wide_spike_filter' routine.  GSD 150102
    Improved the speed of parse_chroms() by factor of 3.  GSD 150218
    Capture IndexError on wide spike filter.  GSD 150508
    Now uses python3  GSD 170104
    linted GSD 191212
'''

import argparse
import numpy as np
import gzip
import os.path
import os
import shutil

import gmd_smoothing

VERSION = 1.23


class ITX():
    """ Igor Pro Text File Class """

    def __init__(self, itxfile, saveorig=False):
        self.file = str(itxfile)
        if os.path.getsize(self.file) < 100:
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
        time, date, ssv = self.wavenote()
        mn, dd, yyyy = date.split('-')
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
        #import matplotlib.pyplot as plt
        """ Removes spikes that are wider than one-point (ie the other spike filter)
              that occur after the 'start' time (seconds after injection).
            Finds spikes using 2nd derivative.  Spikes are about 2 seconds wide.
        """
        thresh = 3000

        if ch == 'all':
            for c in range(self.chans):
                self.wide_spike_filter(c, start=args.ws_start)
        else:
            y = self.chroms[ch, :]
            ydd = abs(np.gradient(np.gradient(y)))       # abs of 2nd derivative
            #plt.plot(ydd)
            #plt.show()
            startpt = int(start)*self.datafreq
            idx = [i for i,v in enumerate(ydd[startpt:]) if v > thresh]      # index of spikes
            groups = self.findgroups(idx)           # group of spikes
            for spike in groups:
                pt0 = startpt + (spike[0]-1)
                pt0 = 0 if pt0 < 0 else pt0
                pt1 = startpt + (spike[1]+10)
                pt1 = len(self.chroms[ch])-1 if pt1 >= len(self.chroms[ch]) else pt1
                # replace spike with a line
                m = (self.chroms[ch, pt1] - self.chroms[ch, pt0]) / float(pt1-pt0)
                b = self.chroms[ch, pt0] - m*pt0
                self.chroms[ch, pt0:pt1] = [m*x+b for x in range(pt0,pt1)]

    def savitzky_golay(self, ch, winsize=21, order=4):
        """ applies the savitzky golay smoothing algo """
        from scipy.signal import savgol_filter

        if ch == 'all':
            for c in range(self.chans):
                self.savitzky_golay(c, winsize=winsize, order=order)
        else:
            y = self.chroms[ch, :]
            self.chroms[ch] = savgol_filter(y, winsize, order)

    @staticmethod
    def compress_to_Z(file):
        file = str(file)    # file is a pathlib filetype
        with open(file, 'rb') as f_in, gzip.open(file+'.Z', 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
            os.remove(file)

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
        ax.set_title('Channel '+str(ch))
        ax.set_ylabel('Response (hz)')
        ax.set_xlabel('Time (s)')
        ax.legend()
        #bx = fig.add_subplot(212)
        #bx.plot(np.diff(self.org[ch, :]))
        #bx.set_xlabel('Time (s)')
        plt.show()


if __name__ == '__main__':

    SGwin, SGorder = 21, 4      # Savitzky Golay default variables
    WSTART = -1                 # Wide spike filter start time

    parser = argparse.ArgumentParser(description='Import chromatograms \
        in the Igor Text File (.itx) format')
    parser.add_argument('-s', action="store_true", default=False,
                        help='Apply 1-point spike filter (default=False)')
    parser.add_argument('-W', action="store", dest='ws_start', default=WSTART,
                        help='Apply wide spike filter (default off)')
    parser.add_argument('-g', action="store_true", default=False,
                        help='Apply Savitzky Golay smoothing (default=False)')
    parser.add_argument('-gw', action="store", dest='SGwin', metavar='Win',
                        default=SGwin, type=int,
                        help='Sets Savitzky Golay smoothing window (default = '+str(SGwin)+' points)')
    parser.add_argument('-go', action="store", dest='SGorder', metavar='Order',
                        default=SGorder, type=int,
                        help='Sets Savitzky Golay order of fit (default = '+str(SGorder)+')')
    parser.add_argument('-d', action="store", dest='chan', type=int, default=-1,
                        help='Display original chrom and exported data for a channel')
    parser.add_argument(dest='itxfile', help='ITX chromatogram file to process')

    args = parser.parse_args()

    if args.chan >= 0:
        chroms = ITX(args.itxfile, saveorig=True)
    else:
        chroms = ITX(args.itxfile)

    # apply spike filter before SG smoothing
    if args.s:
        chroms.spike_filter('all')

    # apply wide spike filter?
    if args.ws_start > 0:
        chroms.wide_spike_filter('all', start=args.ws_start)

    # apply Savitzky Golay smoothing
    if args.g:
        chroms.savitzky_golay('all', winsize=args.SGwin, order=args.SGorder)

    # display chrom?
    if args.chan >= 0:
        chroms.display(args.chan)
        quit()

    chroms.write()
