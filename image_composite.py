import os
import sys
from os.path import join
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.QtWidgets import QApplication, QMainWindow,QWidget, QAction, QFileDialog,QLabel, QPushButton, QSlider, QGridLayout, QGroupBox, QListWidget
from PyQt5.QtGui import QIcon, QPixmap, QImage
import numpy as np
import matplotlib.pyplot as plt
import cv2
import evaluation
import numbergen as ng
import imagen as ig
from datetime import datetime
from drag_widget import drag_img

class composite_gui(QMainWindow):
    def __init__(self):
        super(composite_gui, self).__init__()
        self.title = 'Shadow Compositing'
        self.left, self.top = 400,400
        self.width, self.height = 1640, 1080

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left,self.top, self.width, self.height)
        self.setFixedSize(self.width, self.height)

        self.set_menu()

        # canvas
        self.canvas = QLabel(self)
        self.canvas.move(10,40)
        self.read_img('imgs/x.jpg', self.canvas, (1024, 1024))

        # cutout layer
        self.cutout_layer = []

        # shadow layer
        self.shadow_layer = []

        self.show()

    def set_menu(self):
        main_menu = self.menuBar()
        file_menu = main_menu.addMenu('File')

        load_canvas_button = QAction(QIcon(":/images/open.png"), 'load canvas', self)
        load_canvas_button.triggered.connect(self.load_canvas)

        load_cutout_button = QAction(QIcon(":/images/open.png"), 'load cutout', self)
        load_cutout_button.triggered.connect(self.load_cutout)

        file_menu.addAction(load_canvas_button)
        file_menu.addAction(load_cutout_button)

    def set_drag_widget(self):
        self.setAcceptDrops(True)
        self.drag_img_wgt = drag_img(self)
        self.drag_img_wgt.set_image('imgs/000_final.png')
        self.drag_img_wgt.move(99, 65)

    def dragEnterEvent(self, e):
        e.accept()

    def dropEvent(self, e):
        position = e.pos()
        self.drag_img_wgt.move(position)

        e.setDropAction(Qt.MoveAction)
        e.accept()

    def dragMoveEvent(self, e):
        self.drag_img_wgt.move(e.pos())
        e.accept()

    """ Utilities
    """
    def read_img(self, file, label, size=None):
        def to_qt_img(np_img):
            if np_img.dtype != np.uint8:
                np_img = np.clip(np_img, 0.0, 1.0)
                np_img = np_img * 255.0
                np_img = np_img.astype(np.uint8)

            h, w, c = np_img.shape
            # bytesPerLine = 3 * w
            return QImage(np_img.data, w, h, QImage.Format_RGB888)

        if not os.path.exists(file):
            print('cannot find file', file)
            return

        img = plt.imread(file)
        if len(img.shape) == 2:
            img = np.repeat(img[:,:,np.newaxis], 3, axis=2)
        h,w,_ = img.shape
        pixmap = QPixmap(to_qt_img(img))
        if size is not None:
            pixmap = pixmap.scaled(size[0], size[1])
        label.setPixmap(pixmap)
        label.adjustSize()

    def load_file(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        fname = QFileDialog.getOpenFileName(self, 'Open file', os.path.join(dir_path,'imgs'))
        return fname[0]

    def add_cutout(self, filename):
        cutout_label = QLabel(self)
        self.read_img(filename, cutout_label)
        cutout_label.show()
        
        self.cutout_layer.append(cutout_label)

    #################### Actions ##############################
    @pyqtSlot()
    def load_canvas(self):
        canvas_file = self.load_file()
        print('load file', canvas_file)
        self.read_img(canvas_file, self.canvas, (1024, 1024))

    @pyqtSlot()
    def load_cutout(self):
        cutout_file = self.load_file()
        print('load file', cutout_file)
        self.add_cutout(cutout_file)

    @pyqtSlot()
    def render_layers(self):
        pass

if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = composite_gui()
    sys.exit(app.exec_())