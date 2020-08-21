import sys

from PyQt5.QtCore import Qt, QMimeData
from PyQt5.QtGui import QDrag, QPixmap
from PyQt5.QtWidgets import QLabel

class drag_img(QLabel):
    def __init__(self, parent):
        super().__init__(parent)

    def mouseMoveEvent(self, e):

        if e.buttons() != Qt.RightButton:
            return

        mimeData = QMimeData()

        drag = QDrag(self)
        drag.setMimeData(mimeData)
        drag.setHotSpot(e.pos() - self.rect().topLeft())

        dropAction = drag.exec_(Qt.MoveAction)

    def mousePressEvent(self, e):

        super().mousePressEvent(e)

        if e.button() == Qt.LeftButton:
            print('press')

    def set_image(self, img_path):
        pixmap = QPixmap(img_path)
        self.setPixmap(pixmap)
