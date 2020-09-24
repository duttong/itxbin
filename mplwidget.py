# Imports
import matplotlib
matplotlib.use('QT5Agg')    # noqa

from PyQt5 import QtWidgets
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar


class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax1 = self.fig.add_subplot(211)       # top figure
        self.ax2 = self.fig.add_subplot(212)       # bottom figure
        super(MplCanvas, self).__init__(self.fig)


# Matplotlib widget with NavigationToolbar
class MplWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent)   # Inherit from QWidget
        self.canvas = MplCanvas()                  # Create canvas object
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.vbl = QtWidgets.QVBoxLayout()         # Set box for plotting
        self.vbl.addWidget(self.canvas)
        self.vbl.addWidget(self.toolbar)
        self.setLayout(self.vbl)
