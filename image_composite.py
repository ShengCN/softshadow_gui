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
from drag_widget import drag_img
from ibl_widget import ibl_widget

class composite_gui(QMainWindow):
    def __init__(self):
        super(composite_gui, self).__init__()
        self.title = 'Shadow Compositing'
        self.left, self.top = 400,400
        self.width, self.height = 1720, 1080
        self.cutout_count = 0

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left,self.top, self.width, self.height)
        self.setFixedSize(self.width, self.height)
        self.setAcceptDrops(True)
        self.set_menu()

        # cutout layer
        self.cutout_layer = []

        # shadow layer
        self.shadow_layer = []

        # canvas
        self.canvas = QLabel(self)
        self.canvas_offset = (10,40)
        self.canvas.move(self.canvas_offset[0], self.canvas_offset[1])
        self.read_img('imgs/x.jpg', self.canvas, (1024, 1024))

        self.ibl = ibl_widget(self)
        self.ibl.move(self.canvas.pos().x() + self.canvas.width() + 100, self.canvas.pos().y())

        # buttons
        self.render_btn = QPushButton("render", self)
        self.render_btn.move(1480, self.ibl.pos().y() + self.ibl.height() + 10)
        self.render_btn.clicked.connect(self.render_layers)

        self.save_btn = QPushButton("save", self)
        self.save_btn.move(1380, self.render_btn.pos().y())
        self.save_btn.clicked.connect(self.save_result)

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
    def to_qt_img(self, np_img):
        if np_img.dtype != np.uint8:
            np_img = np.clip(np_img, 0.0, 1.0)
            np_img = np_img * 255.0
            np_img = np_img.astype(np.uint8)

        h, w, c = np_img.shape
        # bytesPerLine = 3 * w
        return QImage(np_img.data, w, h, QImage.Format_RGB888)

    def read_img(self, file, label, size=None):
        if not os.path.exists(file):
            print('cannot find file', file)
            return

        img = plt.imread(file)
        if len(img.shape) == 2:
            img = np.repeat(img[:,:,np.newaxis], 3, axis=2)
        h,w,_ = img.shape
        self.canvas_img = cv2.resize(img, (size[0], size[1]))

        self.set_img(self.to_qt_img(self.canvas_img), label)


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

    def get_cutout_label(self, id):
        for label in self.cutout_layer:
            if label.get_id() == id:
                return label

        print('cannot find the label')
        return None

    def set_cur_label(self, id, offset_pos):
        self.cur_cutout = self.get_cutout_label(id)
        self.cur_cutout_offset = offset_pos

    def composite_layer_result(self, cur_canvas, cur_widget):
        """
            input:   canvas image, widget
            outout:  alpha blending compsite result
        """
        tmp, cutout = cur_canvas.copy(), cur_widget
        cutout_img = cutout.get_render_img()/255.0
        canvas_h, canvas_w = tmp.shape[0], tmp.shape[1]
        x, y = cutout.pos().x() - self.canvas.pos().x(), cutout.pos().y() - self.canvas.pos().y()
        h, w, _ = cutout_img.shape

        mask_x, mask_y = 0, 0
        mask_h, mask_w = h, w
        tmp_x, tmp_y = x, y

        # boundary case
        if x < 0:
            tmp_x, mask_x = 0, -x

        if y < 0:
            tmp_y, mask_y = 0, -y

        if x + w > canvas_w:
            mask_w = canvas_w - x
            tmp_x = x

        if y + h > canvas_h:
            mask_h = canvas_h - y
            tmp_y = y

        tmp_h, tmp_w = mask_h - mask_y, mask_w - mask_x
        mask = cutout_img[mask_y: mask_h, mask_x:mask_w, 3]
        mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        print('original shape: ', tmp.shape)
        print('tmp: {}, tmp x: {}, tmp y: {}, tmp w:{}, tmp h: {}'.format(
            tmp[tmp_y:tmp_y + tmp_h, tmp_x:tmp_x + tmp_w, :].shape, tmp_x, tmp_y, tmp_w, tmp_h))
        print(
            'mask: {}, mask: x: {}, mask y: {}, mask w: {}, mask h: {}'.format(mask.shape, mask_x, mask_y, mask_w, mask_h))

        tmp[tmp_y:tmp_y + tmp_h, tmp_x:tmp_x + tmp_w, :] = (1.0 - mask) * tmp[tmp_y:tmp_y + tmp_h, tmp_x:tmp_x + tmp_w,
                                                                          :] + mask * cutout_img[mask_y: mask_y + mask_h,
                                                                                      mask_x:mask_x + mask_w, :3]
        return tmp

    def render_cutout(self, cur_canvas):
        tmp = cur_canvas.copy()
        canvas_h, canvas_w,_ = tmp.shape
        # composite result with cutout
        for cutout in self.cutout_layer:
            tmp = self.composite_layer_result(tmp, cutout)

        return tmp

    def render_shadow(self, cur_canvas):
        """
            Render shadow to cutout layers
        """
        if len(self.cutout_layer) == 0:
            return

        # h x w
        ibl_np = self.ibl.get_ibl_numpy()

        # before passed into net, some modification needs to be done on ibl
        ibl_np = cv2.flip(ibl_np, 0)
        if np.sum(ibl_np) > 1e-3:
            ibl_np = ibl_np * 30.0 / np.sum(ibl_np)

        ibl_np = np.transpose(np.expand_dims(cv2.resize(ibl_np, (32, 16)), axis=2), (2,0,1))
        ibl_np = np.repeat(ibl_np[np.newaxis,:,:,:], len(self.cutout_layer), axis=0)

        # convert to predict format
        mask_input = []
        for cutout in self.cutout_layer:
            cur_curout = cutout.get_img() # h x w x 4
            cur_curout = cv2.resize(cur_curout,(256,256))
            cur_curout = np.transpose(cur_curout[:,:,3:], (2,0,1))
            mask_input.append(cur_curout)

        # b x c x h x w
        mask_input = np.array(mask_input)
        print('mask input: {}, {}, ibl: {}, {}, {}'.format(np.min(mask_input), np.max(mask_input), np.min(ibl_np), np.max(ibl_np), np.sum(ibl_np)))
        shadow_pred = evaluation.net_render_np(mask_input, ibl_np)
        print('shadow pred: {}, {}'.format(np.min(shadow_pred), np.max(shadow_pred)))
        for i, shadow in enumerate(shadow_pred):
            shadow = np.transpose(shadow, (1,2,0))
            print('shadow shape: ', shadow.shape)
            cv2.normalize(shadow[:,:,0], shadow[:,:,0], 0.0,1.0,cv2.NORM_MINMAX)
            h,w = shadow.shape[0], shadow.shape[1]
            shadow_out = np.zeros((h,w,4))
            shadow_out[:,:,3:] = shadow

            self.cutout_layer[i].composite_shadow(shadow_out)

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
        print('canvas shape: ', self.canvas_img.shape)

        self.render_shadow(self.canvas_img)  # render to cutout layers
        # self.canvas_img = self.render_cutout(self.canvas_img) # composite cutout with canvas
        # self.set_img(self.to_qt_img(self.canvas_img), self.canvas)

    @pyqtSlot()
    def save_result(self):
        # get save file name
        dir_path = os.path.dirname(os.path.realpath(__file__))
        save_fname = QFileDialog.getSaveFileName(self, 'Open file', os.path.join(dir_path,'output'))
        out_img = self.render_cutout(self.canvas_img)

        if cv2.imwrite(save_fname[0], out_img):
            print('file {} saved succeed'.format(save_fname[0]))
        else:
            print('file {} save fail'.format(save_fname[0]))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = composite_gui()
    sys.exit(app.exec_())