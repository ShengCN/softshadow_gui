import sys
from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtWidgets import QWidget, QMainWindow,QLabel, QApplication
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon, QPixmap, QImage
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

class painter_widget(QLabel):
    def __init__(self, h, w, parent):
        super().__init__(parent)
        
        self.buffer = np.zeros((h, w,3))
        img = self.to_qt_img(self.buffer)
        self.set_img(img, self)
        self.last_x, self.last_y = None, None
        self.begin_paint = False
        self.left = True

    """ Utilities
    """
    def to_qt_img(self, np_img):
        if np_img.dtype != np.uint8:
            np_img = np.clip(np_img, 0.0, 1.0)
            np_img = np_img * 255.0
            np_img = np_img.astype(np.uint8)

        h, w, c = np_img.shape
        # bytesPerLine = 3 * w
        return QImage(np_img.data, w, h, QImage.Format_RGB888)
    
    def update_pred(self, img):
        self.buffer = img
        self.update_display()

    def set_img(self, img, label):
        pixmap = QPixmap(img)
        label.setPixmap(pixmap)
        w,h = img.width(), img.height()
        label.adjustSize()
    
    def get_img(self):
        return self.buffer[:,:,0]
    
    def update_display(self):
        self.set_img(self.to_qt_img(self.buffer), self)
        self.update()
    
    def mouseMoveEvent(self, e):
        if not self.begin_paint:
            return

        if self.last_x is None:
            self.last_x = e.x()
            self.last_y = e.y()
            return
            
        pil_img = Image.fromarray(np.uint8(self.buffer * 255))
        draw = ImageDraw.Draw(pil_img)
        if self.left:
            fill = (255,255,255)
            width = 3
        else:
            fill = 0
            width = 10

        draw.line((self.last_x, self.last_y, e.x(), e.y()), fill=fill, width=width)
        self.buffer = np.array(pil_img) 
        if self.buffer.dtype == np.uint8:
            self.buffer = self.buffer / 255.0 

        self.update_display()

        # Update the origin for next time.
        self.last_x = e.x()
        self.last_y = e.y()

    def mouseReleaseEvent(self, e):
        self.last_x = None
        self.last_y = None
        self.begin_paint = False
    
    def mousePressEvent(self, e):
        # print('begin paint')
        self.left = e.button() == Qt.LeftButton
        self.begin_paint = True

if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    wnd = QMainWindow()
    painter = painter_widget(500,500,wnd)
    wnd.setCentralWidget(painter)
    wnd.show()
    app.exec_()