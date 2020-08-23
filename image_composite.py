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
        self.cutout_count = 0

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left,self.top, self.width, self.height)
        self.setFixedSize(self.width, self.height)
        self.setAcceptDrops(True)
        self.set_menu()

        # canvas
        self.canvas = QLabel(self)
        self.canvas.move(10,40)
        self.read_img('imgs/x.jpg', self.canvas, (1024, 1024))

        # buttons
        self.render_btn = QPushButton("render", self)
        self.render_btn.move(1480, 40)
        self.render_btn.clicked.connect(self.render_layers)

        # cutout layer
        self.cutout_layer = []

        # shadow layer
        self.shadow_layer = []

        # # layouts
        # canvas_group = QGroupBox("canvas")
        # canvas_layout = QtWidgets.QHBoxLayout()
        # canvas_layout.addWidget(self.canvas)
        # canvas_group.setLayout(canvas_layout)
        #
        # control_group = QGroupBox('control')
        # control_layout = QtWidgets.QVBoxLayout()
        # control_layout.addWidget(self.render_btn)
        # control_group.setLayout(control_layout)
        #
        # grid = QGridLayout()
        # grid.addWidget(canvas_group, 0, 0)
        # # grid.addWidget(control_group, 0, 1)
        # self.setLayout(grid)

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

    def dragEnterEvent(self, e):
        e.accept()

    def dropEvent(self, e):
        e.setDropAction(Qt.MoveAction)
        e.accept()

    def dragMoveEvent(self, e):
        e.accept()
        self.cur_cutout.move(e.pos() - self.cur_cutout_offset)

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
        self.canvas_img = cv2.resize(img, (size[0], size[1]))

        self.set_img(to_qt_img(self.canvas_img), label)


    def set_img(self, img, label):
        pixmap = QPixmap(img)
        label.setPixmap(pixmap)
        label.adjustSize()


    def load_file(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        fname = QFileDialog.getOpenFileName(self, 'Open file', os.path.join(dir_path,'imgs'))
        return fname[0]

    def add_cutout(self, filename):
        cutout_label = drag_img(self)
        cutout_label.set_id(self.cutout_count)
        self.cutout_count += 1

        # self.read_img(filename, cutout_label)
        cutout_label.read_img(filename)
        cutout_label.show()

        self.cutout_layer.append(cutout_label)
        self.render_layers()

    def get_cutout_label(self, id):
        for label in self.cutout_layer:
            if label.get_id() == id:
                return label

        print('cannot find the label')
        return None

    def set_cur_label(self, id, offset_pos):
        self.cur_cutout = self.get_cutout_label(id)
        self.cur_cutout_offset = offset_pos

    #################### Actions ##############################
    @pyqtSlot()
    def load_canvas(self):
        canvas_file = self.load_file()
        print('load file', canvas_file)
        self.read_img(canvas_file, self.canvas, (1024, 1024))

        self.render_layers()

    @pyqtSlot()
    def load_cutout(self):
        cutout_file = self.load_file()
        print('load file', cutout_file)
        self.add_cutout(cutout_file)

    @pyqtSlot()
    def render_layers(self):
        # shadow layer composite
        tmp = self.canvas_img
        print('canvas shape: ', tmp.shape)

        # composite result with cutout
        for cutout in self.cutout_layer:
            cutout_img = cutout.get_img()
            h, w,_ = cutout_img.shape
            x, y = cutout.pos().x(), cutout.pos().y()
            mask = cutout_img[:,:,3]
            mask = np.repeat(mask[:,:,np.newaxis], 3, axis=2)
            print('mask: {}, tmp: {}, cutout: {}'.format(mask.shape, tmp[y:y+h,x:x+w,:].shape, cutout_img[:,:,:3].shape))

            tmp = (1.0-mask) * tmp[y:y+h,x:x+w,:] + mask * cutout_img[:,:,:3]

        self.canvas_img = tmp


if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = composite_gui()
    sys.exit(app.exec_())