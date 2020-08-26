import sys

from PyQt5.QtCore import Qt, QMimeData
from PyQt5.QtGui import QDrag, QPixmap
from PyQt5.QtWidgets import QLabel, QApplication
from PyQt5.QtGui import QPixmap, QImage
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
import imagen as ig

class ibl():
    def __init__(self, pos=(0,0), radius=0.04, scale=1.0):
        self.pos = pos
        self.radius = radius
        self.scale = scale

class ibl_widget(QLabel):
    def __init__(self, parent):
        super().__init__(parent)

        # initialize ibl
        self.ibl_img = np.zeros((256, 512, 3))
        self.set_img(self.ibl_img)

        self.ibls = [ibl(pos=(0.5, 0.5))]
        self.update_ibl()
        self.cur_ibl = 0

    def to_qt_img(self, np_img):
        if np_img.dtype != np.uint8:
            np_img = np.clip(np_img, 0.0, 1.0)
            np_img = np_img * 255.0
            np_img = np_img.astype(np.uint8)

        h, w = np_img.shape[0], np_img.shape[1]
        return QImage(np_img.data, w, h, QImage.Format_RGB888)

    def get_ibl_numpy(self):
        num = len(self.ibls)
        gs = ig.Composite(operator=np.add,
                          generators=[ig.Gaussian(
                              size=self.ibls[i].radius,
                              scale=self.ibls[i].scale,
                              x=self.ibls[i].pos[0] - 0.5,
                              y=self.ibls[i].pos[1] * 0.3 + 0.2,
                              aspect_ratio=1.0,
                          ) for i in range(num)],
                          xdensity=512)

        # rotate by 180 degree
        tmp = gs()
        h,w = tmp.shape[0], tmp.shape[1]

        ret = tmp.copy()
        ret[:,:w//2], ret[:,w//2:] = tmp[:,w//2:], tmp[:,:w//2]
        return ret

    def mousePressEvent(self, e):
        self.update_ibl_event(e)

    def mouseReleaseEvent(self, e):
        pass

    def mouseMoveEvent(self, e):
        self.update_ibl_event(e)

    def update_ibl_event(self, e):
        x,y = e.pos().x() / self.width(), e.pos().y() / self.height()
        self.ibls[self.cur_ibl].pos = (x,1.0 - y)
        self.update_ibl()

    def set_img(self, np_img):
        pixmap = QPixmap(self.to_qt_img(np_img))
        self.setPixmap(pixmap)
        self.adjustSize()

    def update_ibl(self):
        num = len(self.ibls)
        gs = ig.Composite(operator=np.add,
                        generators=[ig.Gaussian(
                                    size=self.ibls[i].radius,
                                    scale=self.ibls[i].scale,
                                    x=self.ibls[i].pos[0] - 0.5,
                                    y=self.ibls[i].pos[1] -0.5,
                                    aspect_ratio=1.0,
                                    ) for i in range(num)],
                            xdensity=512)

        self.ibl_img = np.repeat(gs()[:,:,np.newaxis], 3, axis=2)
        self.set_img(self.ibl_img)
        self.parent().render_layers()
