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
    opt.add_argument('--batch', dest='duration',
        help='Batch process a portion of FE3 data (durations can be 1M, 2M, 3M, 1Y, etc.')

    options = opt.parse_args()

    # Create GUI application
    app = QtWidgets.QApplication(sys.argv)
    frontend = gui.FE3_Process()

    # batch processing
    if options.duration:
        frontend.flask_batch(duration=options.duration)
    else:
        # display application
        frontend.show()
        frontend.button_CFC11.setChecked(True)
        app.exec_()

    if frontend.madechanges:
        frontend.fe3db.save_db_file()
