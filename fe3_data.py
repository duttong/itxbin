#! /usr/bin/env python

from PyQt5 import QtWidgets
import sys

# Local Module Imports
import fe3_gui as gui

# Create GUI application
app = QtWidgets.QApplication(sys.argv)
frontend = gui.FE3_Process()
frontend.show()
frontend.button_CFC11.setChecked(True)
app.exec_()

if frontend.madechanges:
    print(f'Saving changes to {frontend.fe3db.dbfile}')
    frontend.fe3db.save_db_file()
