import os
import sys
from os.path import join
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QSlider, QGridLayout, QGroupBox, QListWidget
from PyQt5.QtGui import QIcon, QPixmap, QImage  
import numpy as np
import matplotlib.pyplot as plt
import cv2
import evaluation
import numbergen as ng
import imagen as ig
from datetime import datetime
from drag_widget import drag_img

class composite_gui(QWidget):
    def __init__(self):
        super(composite_gui, self).__init__()
        self.title = 'Shadow Compositing'
        self.left, self.top = 400,400
        self.width, self.height = 1280, 720

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left,self.top, self.width, self.height)
        self.setFixedSize(self.width, self.height)

        self.setAcceptDrops(True)
        self.drag_img_wgt = drag_img(self)
        self.drag_img_wgt.set_image('imgs/001_final.png')
        self.drag_img_wgt.move(100, 65)
        self.show()

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

if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = composite_gui()
    sys.exit(app.exec_())