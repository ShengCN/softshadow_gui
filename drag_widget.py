import sys

from PyQt5.QtCore import Qt, QMimeData
from PyQt5.QtGui import QDrag, QPixmap
from PyQt5.QtWidgets import QLabel, QApplication
from PyQt5.QtGui import QPixmap, QImage
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2

class drag_img(QLabel):
    def __init__(self, parent):
        super().__init__(parent)
        self.resize_flag = False

    def mouseMoveEvent(self, e):
        if e.buttons() == Qt.LeftButton:
            # resize window
            if self.resize_flag:
                QApplication.setOverrideCursor(Qt.WaitCursor)
                w,h = e.pos().x(), e.pos().y()
                print('resize window size {}, {}'.format(w,h))
                np_img = cv2.resize(self.img, (w, h))
                self.set_img(np_img)

        if e.buttons() == Qt.RightButton:
            mimeData = QMimeData()

            drag = QDrag(self)
            drag.setMimeData(mimeData)
            drag.setHotSpot(e.pos() - self.rect().topLeft())

            dropAction = drag.exec_(Qt.MoveAction)

    def mousePressEvent(self, e):
        super().mousePressEvent(e)

        if e.button() == Qt.LeftButton:
            print('press')
            if self.check_resize(e.pos()):
                print('inside resize position')
                self.resize_flag = True

        if e.button() == Qt.RightButton:
            self.parent().set_cur_label(self.id, e.pos() - self.rect().topLeft())

    def mouseReleaseEvent(self, e):
        self.resize_flag = False

    def check_resize(self, pos):
        x,y = pos.x(), pos.y()
        print('x: {}, y: {}'.format(x, y))
        w,h = self.width(), self.height()

        resize_region = 20
        return w - x < resize_region and h - y < resize_region

    def set_id(self, id):
        self.id = id

    def get_id(self):
        return self.id

    def get_img(self):
        return self.img

    def set_img(self, np_img):
        pixmap = QPixmap(self.to_qt_img(np_img))
        self.setPixmap(pixmap)
        self.adjustSize()

    def to_qt_img(self, np_img):
        if np_img.dtype != np.uint8:
            np_img = np.clip(np_img, 0.0, 1.0)
            np_img = np_img * 255.0
            np_img = np_img.astype(np.uint8)

        h, w, c = np_img.shape
        # bytesPerLine = 3 * w
        tmp = np_img[:,:,:3].copy()
        return QImage(tmp.data, w, h, QImage.Format_RGB888)

    def read_img(self, file):
        """
            assumption: image file has 4 dimensions, last dimension is the alpha channel
        """
        if not os.path.exists(file):
            print('cannot find file', file)
            return

        self.img = plt.imread(file)
        print('min: {}, max: {}'.format(np.min(self.img), np.max(self.img)), self.img.shape)
        self.set_img(self.img)
