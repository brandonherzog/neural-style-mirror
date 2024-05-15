# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

try:
    from PySide.QtCore import *
    from PySide.QtGui import *
except ImportError:
    from PyQt5.QtGui import *
    from PyQt5.QtCore import *
    Signal = pyqtSignal
    Slot = pyqtSlot

def handle_capture(window, capture):
    file_format = 'png'
    initial_path = QDir.currentPath() + '/untitled.' + str(file_format)
    filter = '%s Files (*.%s);;All Files (*)' % (str(file_format).upper(), file_format)
    file_name,_ = QFileDialog.getSaveFileName(window, 'Save As', initial_path, filter)
    if file_name:
        capture.save(file_name, str(file_format))