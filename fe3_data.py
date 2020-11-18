#! /usr/bin/env python

from PyQt5 import QtWidgets
import sys
import argparse

# Local Module Imports
import fe3_gui as gui

if __name__ == '__main__':

    opt = argparse.ArgumentParser(
        description='Program for processing FE3 data.'
    )
    opt.add_argument('--export', dest='exp_duration', metavar='Duration',
        help='Export summary data for flask runs (duration such as 2W, 1M, 2M, 1Y, etc.)')
    opt.add_argument('--batch', dest='batch_duration', metavar='Duration',
        help='Batch process a portion of FE3 data (duration such as 2W, 1M, 2M, 1Y, etc.')

    options = opt.parse_args()

    # Create GUI application
    app = QtWidgets.QApplication(sys.argv)
    frontend = gui.FE3_Process()

    # batch processing
    if options.batch_duration:
        frontend.flask_batch(duration=options.batch_duration)
    elif options.exp_duration:
        frontend.flask_export(duration=options.exp_duration)
    else:
        # display application
        frontend.show()
        frontend.button_CFC11.setChecked(True)
        app.exec_()

    if frontend.madechanges:
        frontend.fe3db.save_db_file()
