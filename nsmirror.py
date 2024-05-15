#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, division, absolute_import
import PyQt5
import time
import os
import sys
from pydoc import locate
from PyQt5.QtWidgets import *
import numpy as np

import cv2
import chainer
chainer.using_config
# location for these differ from opencv 2 vs 3
try: # 2
    CV_CAP_PROP_FRAME_WIDTH = cv2.cv.CV_CAP_PROP_FRAME_WIDTH
    CV_CAP_PROP_FRAME_HEIGHT = cv2.cv.CV_CAP_PROP_FRAME_HEIGHT
    CV_BGR2RGB = cv2.cv.CV_BGR2RGB
except AttributeError: # 3
    CV_CAP_PROP_FRAME_WIDTH = cv2.CAP_PROP_FRAME_WIDTH
    CV_CAP_PROP_FRAME_HEIGHT = cv2.CAP_PROP_FRAME_HEIGHT
    CV_BGR2RGB = cv2.COLOR_BGR2RGB

try:
    from PySide.QtCore import *
    from PySide.QtGui import *
except ImportError:
    from PyQt5.QtGui import *
    from PyQt5.QtCore import *
    Signal = pyqtSignal
    Slot = pyqtSlot

from PIL import Image

import settings
from style import Style, load_styles

class FrameGrabber(QObject):
    last_frame_signal = Signal(object)

    def __init__(self, size_str, *args, **kwargs):
        QObject.__init__(self, *args, **kwargs)
        self.active = True
        self.setup_camera()
        self.set_camera_size(size_str)

    def setup_camera(self):
        """Initialize camera."""
        self.capture = cv2.VideoCapture(settings.WEBCAM)

    def set_camera_size(self, size_str):
        width, height = map(int, size_str.split('x'))
        self.capture.set(CV_CAP_PROP_FRAME_WIDTH, width)
        self.capture.set(CV_CAP_PROP_FRAME_HEIGHT, height)

    @Slot()
    def grab(self):
        while self.active:
            _, frame = self.capture.read()
            frame = cv2.cvtColor(frame, CV_BGR2RGB)
            last_frame = Image.fromarray(frame)
            self.last_frame_signal.emit(last_frame)
            #check for stop/changes
            QApplication.processEvents()

    @Slot(str)
    def change_size(self, size_str):
        self.set_camera_size(size_str)

    @Slot()
    def stop_work(self):
        self.active = False

class ImageProcessor(QObject):
    image_signal = Signal(object)

    def __init__(self, style, *args, **kwargs):
        QObject.__init__(self, *args, **kwargs)
        self.style = None
        self.last_frame = None
        self.active = True
        self.change_style(style)

    @Slot()
    def monitor_images(self):
        while self.active:
            if self.last_frame:
                image_array = self.style.stylize(self.last_frame)
                image = QImage(image_array.data, image_array.shape[1], image_array.shape[0], 
                               image_array.strides[0], QImage.Format_RGB888)

                rotating = QTransform()
                rotating.scale(-1, 1) # mirror
                rotating.rotate(settings.ROTATE_IMAGE)
                image = image.transformed(rotating)

                self.image_signal.emit(image)

            #check for stop/changes
            QApplication.processEvents()

    @Slot()
    def stop_work(self):
        self.active = False

    @Slot(object)
    def change_style(self, style):
        if self.style:
            self.style.unload()
        self.style = style
        self.style.preload()

    @Slot(object)
    def change_last_frame(self, last_frame):
        self.last_frame = last_frame

class ViewBase(QWidget):
    style_changed = Signal(object)
    toggle_fullscreen_signal = Signal()
    quality_changed = Signal(str)

    def __init__(self, styles, selected_style=None, size=None, *args, **kwargs):
        QWidget.__init__(self, *args, **kwargs)
        self.styles = styles
        self._selected_style = selected_style
        if self._selected_style is None:
            self._selected_style = self.styles[0]
        self._quality = size
        if self._quality is None:
            self._quality = settings.SIZES[0]
        self.setup_ui()

    @property
    def quality(self):
        return self._quality

    @quality.setter
    def quality(self, value):
        self._quality = value
        self.size_combo.setCurrentIndex(settings.SIZES.index(self.quality))

    @property
    def selected_style(self):
        return self._selected_style

    @selected_style.setter
    def selected_style(self, value):
        self._selected_style = value
        self.style_buttons[self.styles.index(self.selected_style)].setChecked(True)

    def setup_ui():
        raise NotImplementedError()

    def style_button_clicked(self, style):
        self.selected_style = style
        self.style_changed.emit(style)

    def toggle_fullscreen(self):
        self.toggle_fullscreen_signal.emit()

    def quality_choice(self, quality):
        self.quality = quality
        self.quality_changed.emit(quality)

    def set_image(self, image):
        pixmap = QPixmap.fromImage(image)
        pixmap_scaled = pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(pixmap_scaled)

class LandscapeView(ViewBase):
    def setup_ui(self):
        """Initialize widgets."""
        info_label = QLabel()
        info_label.setText("Hit enter to capture your image!")

        self.image_label = QLabel()
        self.image_label.setMinimumSize(320,240)
        self.image_label.setScaledContents(True)

        button_layout = QVBoxLayout()
        button_layout.setAlignment(Qt.AlignLeft)
        self.style_buttons = [
            QRadioButton(settings.STYLE_SHORTCUTS[i] + ". " + style.name
                         if i < len(settings.STYLE_SHORTCUTS) else style.name) 
            for i, style in enumerate(self.styles)
            ]

        self.style_buttons[self.styles.index(self.selected_style)].setChecked(True)
        self.style_button_group = QButtonGroup()
        for i, btn in enumerate(self.style_buttons):
            button_layout.addWidget(btn)
            self.style_button_group.addButton(btn, i)
            btn.clicked.connect(lambda x=i: self.style_button_clicked(self.styles[x]))

        button_layout.addStretch(1)

        ctrl_layout = QHBoxLayout()
        if not settings.KIOSK:
            fullscreen_button = QPushButton('[ ]')
            fullscreen_button.setMaximumWidth(
                fullscreen_button.fontMetrics().boundingRect('[ ]').width() + 10
                )
            fullscreen_button.clicked.connect(self.toggle_fullscreen)
            ctrl_layout.addWidget(fullscreen_button)
        ctrl_layout.addStretch(1)

        self.size_combo = QComboBox()
        for s in settings.SIZES:
            self.size_combo.addItem(s)
        self.size_combo.setCurrentIndex(settings.SIZES.index(self.quality))
        ctrl_layout.addWidget(self.size_combo)
        self.size_combo.activated[str].connect(self.quality_choice)

        button_layout.addLayout(ctrl_layout)

        sub_layout = QHBoxLayout()
        sub_layout.addLayout(button_layout)
        sub_layout.addWidget(self.image_label, 1)

        main_layout = QVBoxLayout()
        main_layout.addWidget(info_label)
        main_layout.addLayout(sub_layout)
        self.setLayout(main_layout)

class PortraitView(ViewBase):
    def setup_ui(self):
        """Initialize widgets."""
        info_label = QLabel()
        info_label.setText("Hit enter to capture your image!")

        self.image_label = QLabel()
        self.image_label.setMinimumSize(240, 320)
        self.image_label.setScaledContents(True)

        button_layout = QGridLayout()
        self.style_buttons = [
            QRadioButton(settings.STYLE_SHORTCUTS[i] + ". " + style.name 
                         if i < len(settings.STYLE_SHORTCUTS) else style.name) 
            for i, style in enumerate(self.styles)
            ]
        self.style_buttons[self.styles.index(self.selected_style)].setChecked(True)
        self.style_button_group = QButtonGroup()
        for i, btn in enumerate(self.style_buttons):
            button_layout.addWidget(btn, i // 3, i % 3)
            self.style_button_group.addButton(btn, i)
            btn.clicked.connect(lambda x=i: self.style_button_clicked(self.styles[x]))

        ctrl_layout = QHBoxLayout()
        if not settings.KIOSK:
            fullscreen_button = QPushButton('[ ]')
            fullscreen_button.setMaximumWidth(
                fullscreen_button.fontMetrics().boundingRect('[ ]').width() + 10
                )
            fullscreen_button.clicked.connect(self.toggle_fullscreen)
            ctrl_layout.addWidget(fullscreen_button)
        ctrl_layout.addStretch(1)

        self.size_combo = QComboBox()
        for s in settings.SIZES:
            self.size_combo.addItem(s)
        self.size_combo.setCurrentIndex(settings.SIZES.index(self.quality))
        ctrl_layout.addWidget(self.size_combo)
        self.size_combo.activated[str].connect(self.quality_choice)

        button_layout.addLayout(ctrl_layout, (len(self.style_buttons) - 1) // 3 + 1, 0, 1, 3)

        main_layout = QVBoxLayout()
        main_layout.addWidget(info_label)
        main_layout.addWidget(self.image_label, 1)
        main_layout.addLayout(button_layout)
        self.setLayout(main_layout)

class MainApp(QWidget):
    LANDSCAPE = 0
    PORTRAIT = 1

    stop_signal = Signal()
    quality_changed = Signal(str)
    style_changed = Signal(object)
    last_frame_changed = Signal(object)

    def __init__(self):
        QWidget.__init__(self)

        self.styles = load_styles()
        self.image = QImage(256, 256, QImage.Format_RGB888) # placeholder
        self.freeze = None
        if isinstance(settings.CAPTURE_HANDLER, str):
            self.capture_handler = locate(settings.CAPTURE_HANDLER)
        else:
            self.capture_handler = settings.CAPTURE_HANDLER
        self.setup_ui()

        self.frame_grabber = FrameGrabber(settings.SIZES[0])
        self.frame_thread = QThread()
        #self.frame_grabber.image_signal.connect(self.display_frame)
        self.frame_grabber.last_frame_signal.connect(self.last_frame)
        self.frame_grabber.moveToThread(self.frame_thread)
        self.frame_thread.started.connect(self.frame_grabber.grab)
        self.stop_signal.connect(self.frame_grabber.stop_work)
        self.quality_changed.connect(self.frame_grabber.change_size)
        self.frame_thread.start()

        self.image_processor = ImageProcessor(self.styles[0])
        self.image_thread = QThread()
        self.image_processor.image_signal.connect(self.display_frame)
        self.image_processor.moveToThread(self.image_thread)
        self.image_thread.started.connect(self.image_processor.monitor_images)
        self.stop_signal.connect(self.image_processor.stop_work)
        self.style_changed.connect(self.image_processor.change_style)
        self.last_frame_changed.connect(self.image_processor.change_last_frame)
        self.image_thread.start()

    def closeEvent(self, event):
        self.stop_signal.emit()
        self.frame_thread.quit()
        self.image_thread.quit()
        self.frame_thread.wait()
        self.image_thread.wait()

    def setup_ui(self):
        """Initialize widgets."""
        def switch_style(i):
            view = self.landscape_view if self.view_mode == MainApp.LANDSCAPE else self.portrait_view
            self.style_changed.emit(self.styles[i])
            view.selected_style = self.styles[i]

        for i in range(min(len(self.styles), len(settings.STYLE_SHORTCUTS))):
            QShortcut(QKeySequence(settings.STYLE_SHORTCUTS[i]), self, lambda x=i: switch_style(x))

        self.landscape_view = LandscapeView(self.styles)
        self.landscape_view.style_changed.connect(self.style_button_clicked)
        self.landscape_view.toggle_fullscreen_signal.connect(self.toggle_fullscreen)
        self.landscape_view.quality_changed.connect(self.quality_choice)
        self.portrait_view = PortraitView(self.styles)
        self.portrait_view.style_changed.connect(self.style_button_clicked)
        self.portrait_view.toggle_fullscreen_signal.connect(self.toggle_fullscreen)
        self.portrait_view.quality_changed.connect(self.quality_choice)

        self.main_layout = QStackedLayout()
        self.main_layout.addWidget(self.landscape_view)
        self.main_layout.addWidget(self.portrait_view)
        self.setLayout(self.main_layout)

        self.view_mode = MainApp.LANDSCAPE

        self.setStyleSheet('background-color:black;'
                           'font-family: Arial;'
                           'font-style: normal;'
                           'font-size: 12pt;'
                           'font-weight: bold;'
                           'color:white;'
                           )
        self.setWindowTitle('Neural Style Mirror')

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape and not settings.KIOSK:
            if self.windowState() & Qt.WindowFullScreen:
                self.showNormal()
        elif event.key() == Qt.Key_Enter or event.key() == Qt.Key_Return:
            self.image_capture()

    def last_frame(self, frame):
        self.last_frame_changed.emit(frame)

    def display_frame(self, image):
        self.image = image
        if not self.freeze:
            if self.view_mode == MainApp.LANDSCAPE:
                self.landscape_view.set_image(self.image)
            else:
                self.portrait_view.set_image(self.image)

    def resizeEvent(self, event):
        super(MainApp, self).resizeEvent(event)
        new_view_mode = MainApp.LANDSCAPE if self.width() >= self.height() else MainApp.PORTRAIT

        if self.view_mode != new_view_mode:
            old_view = self.portrait_view if new_view_mode == MainApp.LANDSCAPE else self.landscape_view
            new_view = self.landscape_view if new_view_mode == MainApp.LANDSCAPE else self.portrait_view

            new_view.quality = old_view.quality
            new_view.selected_style = old_view.selected_style

            self.view_mode = new_view_mode
            self.main_layout.setCurrentIndex(self.view_mode)

    def style_button_clicked(self, style):
        self.style_changed.emit(style)

    def toggle_fullscreen(self):
        if self.windowState() & Qt.WindowFullScreen:
            self.showNormal()
        else:
            self.showFullScreen()

    def quality_choice(self, quality):
        self.quality_changed.emit(quality)

    def image_capture(self):
        self.freeze = self.image.copy() # prevent background update
        try:
            self.capture_handler(self, self.freeze)
        except:
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Critical)
            msg.setText('Error during capture.')
            msg.setWindowFlags(Qt.FramelessWindowHint | Qt.Popup)
            msg.exec_()

        self.freeze = None

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = MainApp()
    if settings.KIOSK:
        win.showFullScreen()
    else:
        win.show()
    sys.exit(app.exec_())