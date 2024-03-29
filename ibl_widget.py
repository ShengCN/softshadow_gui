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
    def __init__(self, pos=(0,0), radius=0.09, scale=1.0):
        self.pos = pos
        self.radius = radius
        self.scale = scale

class ibl_widget(QLabel):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent_handle = parent

        # initialize ibl
        self.ibl_img = np.zeros((256, 512, 3))
        self.set_img(self.ibl_img)

        self.ibls = []
        # self.update_ibl()
        self.cur_ibl = 0
        self.setFixedSize(512,256)

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
        if e.buttons() == Qt.LeftButton:
            print('current ibl: ', self.cur_ibl)
            self.update_ibl_event(e)
        elif e.buttons() == Qt.RightButton:
            # delete current ibl
            self.ibls = self.ibls[:self.cur_ibl] + self.ibls[self.cur_ibl+1:]
            self.cur_ibl = min(self.cur_ibl, len(self.ibls) - 1)
            self.update_ibl()
            self.parent_handle.update_list(self.cur_ibl)
            self.parent_handle.render_layers()


    def mouseReleaseEvent(self, e):
        pass

    def mouseMoveEvent(self, e):
        self.update_ibl_event(e)

    def update_ibl_event(self, e):
        if len(self.ibls) == 0:
            return

        x,y = e.pos().x() / self.width(), e.pos().y() / self.height()
        self.ibls[self.cur_ibl].pos = (x,1.0 - y)
        self.update_ibl()

    def add_light(self):
        self.ibls.append(ibl((0.5,0.5)))
        self.parent_handle.update_list(self.cur_ibl)
        self.update_ibl()

    def set_cur_light(self, cur_light):
        self.cur_ibl = cur_light

    def get_light_num(self):
        return len(self.ibls)

    def get_cur_light_state(self):
        return self.ibls[self.cur_ibl].radius, self.ibls[self.cur_ibl].scale

    def set_img(self, np_img):
        pixmap = QPixmap(self.to_qt_img(np_img))
        self.setPixmap(pixmap)
        self.adjustSize()

    def update_ibl(self):
        num = len(self.ibls)
        if num == 0:
            black = np.zeros((256,512,3))
            self.set_img(black)
            return

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

        w = self.ibl_img.shape[1]
        tmp = self.ibl_img.copy()
        ret = tmp.copy()
        ret[:,:w//2], ret[:,w//2:] = tmp[:,w//2:], tmp[:,:w//2]

        plt.imsave("test_sharp.png", np.clip(ret, 0.0, 1.0))
        self.set_img(self.ibl_img)
        self.parent_handle.render_layers()

    def set_cur_size(self, radius):
        if (len(self.ibls)) == 0:
            return

        self.ibls[self.cur_ibl].radius = radius
        self.update_ibl()

    def set_cur_scale(self, scale):
        if (len(self.ibls)) == 0:
            return

        self.ibls[self.cur_ibl].scale = scale
        self.update_ibl()