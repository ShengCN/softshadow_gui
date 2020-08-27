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
        # self.setGeometry(self.left,self.top, self.width, self.height)
        # self.setFixedSize(self.width, self.height)
        self.setAcceptDrops(True)
        self.set_menu()

        # cutout layer
        self.cutout_layer = []

        # shadow layer
        self.shadow_layer = []

        # center widget
        wid = QWidget(self)
        self.setCentralWidget(wid)

        # canvas
        self.canvas = QLabel(self)
        self.canvas_img = np.ones((720,680,3))
        self.set_img(self.to_qt_img(self.canvas_img), self.canvas)
        # self.read_img('imgs/x.jpg', self.canvas, (1024, 1024))

        self.ibl = ibl_widget(self)
        self.light_list = QListWidget(self)
        self.light_list.itemClicked.connect(self.light_item_clicked)

        # sliders
        self.shadow_intensity_label = QLabel('intensity', self)
        self.shadow_intensity_slider = QSlider(Qt.Horizontal)
        self.shadow_intensity_slider.valueChanged.connect(self.shadow_intensity_change)
        self.shadow_intensity_slider.setValue(99)

        self.size_label = QLabel('size', self)
        self.size_slider = QSlider(Qt.Horizontal)
        self.size_slider.valueChanged.connect(self.shadow_size_change)

        self.scale_label = QLabel('scale', self)
        self.scale_slider = QSlider(Qt.Horizontal)
        self.scale_slider.valueChanged.connect(self.shadow_scale_change)

        # buttons
        self.save_btn = QPushButton("save", self)
        self.save_btn.move(1300, self.ibl.pos().y() + self.ibl.height() + 10)
        self.save_btn.clicked.connect(self.save_result)

        # layouts
        self.canvas_group = QGroupBox("canvas", self)
        canvas_layout = QtWidgets.QHBoxLayout()
        canvas_layout.addWidget(self.canvas)
        self.canvas_group.setLayout(canvas_layout)

        control_group = QGroupBox('control', self)
        control_layout = QtWidgets.QVBoxLayout()
        control_layout.addWidget(self.ibl)
        control_layout.addWidget(self.light_list)
        control_layout.addWidget(self.shadow_intensity_label)
        control_layout.addWidget(self.shadow_intensity_slider)
        control_layout.addWidget(self.size_label)
        control_layout.addWidget(self.size_slider)
        control_layout.addWidget(self.scale_label)
        control_layout.addWidget(self.scale_slider)
        control_layout.addWidget(self.save_btn)
        control_group.setLayout(control_layout)

        grid = QGridLayout()
        grid.addWidget(self.canvas_group, 0, 0)
        grid.addWidget(control_group, 0, 1)
        wid.setLayout(grid)

        self.setFocusPolicy(Qt.StrongFocus)

        # init smooth mask
        h,w = 256, 256
        mask, padding = np.zeros((h, w, 1)), 5
        mask[padding:h - padding, padding:w - padding] = 1.0
        self.soft_mask = cv2.GaussianBlur(mask, (padding * 4 + 1, padding * 4 + 1), 0)
        self.soft_mask = self.soft_mask[:,:,np.newaxis]

        self.init_state()
        self.show()

    def init_state(self):
        self.add_light()

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
        self.render_layers()

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

        self.canvas_img = img[:,:,:3]
        if size is not None:
            self.canvas_img = cv2.resize(img, (size[0], size[1]))
            if self.canvas_img.dtype == np.uint8:
                self.canvas_img = self.canvas_img/255.0

        # print('self canvas: {}, {}'.format(np.min(self.canvas_img), np.max(self.canvas_img)))
        self.set_img(self.to_qt_img(self.canvas_img.copy()), label)

    def set_img(self, img, label):
        pixmap = QPixmap(img)
        label.setPixmap(pixmap)
        w,h = img.width(), img.height()
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

    def composite_region(self, canvas_xy, canvas_wh, xy, wh):
        canvas_xxyy = [canvas_xy[0] + canvas_wh[0], canvas_xy[1] + canvas_wh[1]]
        xxyy = [xy[0] + wh[0], xy[1] + wh[1]]

        # com_xy, com_xxyy = canvas_xy, canvas_xxyy
        com_xy, com_xxyy = [canvas_xy[0], canvas_xy[1]], [canvas_xy[0], canvas_xy[1]]
        com_xy[0] = max(canvas_xy[0], xy[0])
        com_xy[1] = max(canvas_xy[1], xy[1])
        com_xxyy[0] = min(canvas_xxyy[0], xxyy[0])
        com_xxyy[1] = min(canvas_xxyy[1], xxyy[1])

        # print('canvas xy: {}, xy: {}'.format(canvas_xy, xy))
        # print('canvas xxyy: {}, xxyy: {}'.format(canvas_xxyy, xxyy))
        # print('com xy: {}, com xxyy: {}'.format(com_xy, com_xxyy))

        canvas_region = (com_xy[1] - canvas_xy[1], com_xxyy[1] - canvas_xy[1], com_xy[0] - canvas_xy[0], com_xxyy[0] - canvas_xy[0])
        widget_region = (com_xy[1] - xy[1], com_xxyy[1] - xy[1], com_xy[0] - xy[0], com_xxyy[0] - xy[0])
        return canvas_region, widget_region

    def composite_layer_result(self, cur_canvas, xy, wh, composite_img, composite_operator='lerp'):
        """
            input:   canvas image, composite image top-left corner xy, width and height, ...
            outout:  alpha blending compsite result
        """
        tmp = cur_canvas
        cutout_img =  composite_img
        canvas_region, widget_region = self.composite_region([self.canvas.pos().x(),
                                                              self.canvas.pos().y()],
                                                             [self.canvas.width(),
                                                              self.canvas.height()],
                                                             xy,
                                                             wh)
        # print('canvas region: {}, h: {}, w: {}, widget region: {}, h: {}, w: {}'.format(canvas_region,
        #                                                                                 canvas_region[1],
        #                                                                                 cur_canvas.shape[1],
        #                                                                                 widget_region,
        #                                                                                 wh[1],
        #                                                                                 wh[0]))
        if composite_operator == 'lerp':
            mask = cutout_img[widget_region[0]:widget_region[1], widget_region[2]:widget_region[3], 3:]
            mask = np.repeat(mask, 3, axis=2)

            tmp[canvas_region[0]:canvas_region[1], canvas_region[2]:canvas_region[3], :] = \
                (1.0 - mask) * tmp[canvas_region[0]:canvas_region[1], canvas_region[2]:canvas_region[3], :] + \
                mask * cutout_img[widget_region[0]:widget_region[1], widget_region[2]:widget_region[3], :3]
        else:
            tmp[canvas_region[0]:canvas_region[1], canvas_region[2]:canvas_region[3], :] = \
                tmp[canvas_region[0]:canvas_region[1], canvas_region[2]:canvas_region[3],:] * composite_img[widget_region[0]:widget_region[1], widget_region[2]:widget_region[3],:]

        return tmp

    def render_cutout(self, cur_canvas):
        tmp = cur_canvas
        canvas_h, canvas_w,_ = tmp.shape
        # composite result with cutout
        for cutout in self.cutout_layer:
            xy = (cutout.pos().x(), cutout.pos().y())
            wh = (cutout.width(), cutout.height())
            tmp = self.composite_layer_result(tmp, xy, wh, cutout.get_img())

        return tmp

    def render_shadow(self, cur_canvas):
        """
            Render shadow to canvas
        """
        if len(self.cutout_layer) == 0:
            return cur_canvas

        # h x w
        ibl_np = self.ibl.get_ibl_numpy()

        # before passed into net, some modification needs to be done on ibl
        ibl_np = ibl_np[:80, :]
        ibl_np = cv2.flip(ibl_np, 0)

        ibl_np = np.transpose(np.expand_dims(cv2.resize(ibl_np, (32, 16), cv2.INTER_LINEAR), axis=2), (2,0,1))
        if np.sum(ibl_np) > 1e-3:
            ibl_np = ibl_np * 30.0 / np.sum(ibl_np)
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
        # print('mask input: {}, {}, ibl: {}, {}, {}'.format(np.min(mask_input), np.max(mask_input), np.min(ibl_np), np.max(ibl_np), np.sum(ibl_np)))
        shadow_pred = evaluation.net_render_np(mask_input, ibl_np)
        # print('shadow pred: {}, {}'.format(np.min(shadow_pred), np.max(shadow_pred)))
        tmp = cur_canvas
        # print('canvas range: {}, {}'.format(np.min(tmp), np.max(tmp)))
        for i, shadow in enumerate(shadow_pred):
            shadow = np.transpose(shadow, (1,2,0))
            # print('shadow shape: ', shadow.shape)
            cv2.normalize(shadow[:,:,0], shadow[:,:,0], 0.0, 1.0, cv2.NORM_MINMAX)
            h,w = shadow.shape[0], shadow.shape[1]
            shadow = self.soft_shadow_boundary(shadow) * self.cur_shadow_itensity_fract
            shadow_out = np.zeros((h,w,3))
            shadow_out[:,:,0] = shadow_out[:,:,1] = shadow_out[:,:,2] = 1.0 - shadow[:,:,0]
            shadow_out = cv2.resize(shadow_out, (self.cutout_layer[i].width(), self.cutout_layer[i].height()))

            # composite shadow with canvas
            xy, wh = (self.cutout_layer[i].pos().x(), self.cutout_layer[i].pos().y()), (self.cutout_layer[i].width(), self.cutout_layer[i].height())

            tmp = self.composite_layer_result(tmp, xy, wh, shadow_out, 'prod')

        return tmp

    def soft_shadow_boundary(self, shadow_img):
        return np.clip(np.multiply(self.soft_mask, shadow_img), 0.0, 1.0)

    def keyPressEvent(self, event):
        print(event.key())
        if event.key() == Qt.Key_A:
            print('Pressed A')
            self.add_light()

    #################### Actions ##############################
    @pyqtSlot()
    def load_canvas(self):
        canvas_file = self.load_file()
        print('load file', canvas_file)
        self.read_img(canvas_file, self.canvas, (1024, 1024))

    @pyqtSlot()
    def load_cutout(self):
        cutout_file = self.load_file()
        if not os.path.exists(cutout_file):
            return
        print('load file', cutout_file)
        self.add_cutout(cutout_file)
        self.render_layers()

    @pyqtSlot()
    def render_layers(self):
        if len(self.cutout_layer) == 0:
            return self.canvas_img.copy()

        # print('canvas shape: ', self.canvas_img.shape)

        shadow_canvas = self.render_shadow(self.canvas_img.copy())  # render to cutout layers
        shadow_canvas = self.render_cutout(shadow_canvas) # composite cutout with canvas
        self.set_img(self.to_qt_img(shadow_canvas), self.canvas)
        return shadow_canvas

    @pyqtSlot()
    def save_result(self):
        # get save file name
        dir_path = os.path.dirname(os.path.realpath(__file__))
        save_fname = QFileDialog.getSaveFileName(self, 'Open file', os.path.join(dir_path,'output'))
        print(save_fname)

        out_img = self.render_layers()
        if cv2.imwrite(save_fname[0], out_img*255.0):
            print('file {} saved succeed'.format(save_fname[0]))
        else:
            print('file {} save fail'.format(save_fname[0]))

    def light_item_clicked(self, item):
        # self.ibl.set_cur_ibl()
        cur_ibl = self.light_list.currentRow()
        self.update_list(cur_ibl)

    @pyqtSlot()
    def add_light(self):
        self.ibl.add_light()
        self.update_list(self.ibl.get_light_num()-1)

    @pyqtSlot()
    def update_list(self, cur_ibl):
        self.light_list.clear()
        light_num = self.ibl.get_light_num()
        for i in range(light_num):
            self.light_list.insertItem(i, 'light {}'.format(i))

        self.light_list.setCurrentRow(cur_ibl)
        self.ibl.set_cur_light(cur_ibl)

        if cur_ibl>=0:
            radius, scale = self.ibl.get_cur_light_state()
            self.set_slider_state(radius, scale)
        else:
            self.set_slider_state(0.008, 0)

    def set_slider_state(self, radius, scale):
        size_value = (radius-0.008)/(0.1-0.008)*99.0
        self.size_slider.setValue(int(size_value))

        scale_value = scale * 99.0
        self.scale_slider.setValue(scale_value)

    @pyqtSlot()
    def shadow_intensity_change(self):
        self.cur_shadow_itensity_fract = self.shadow_intensity_slider.value()/99.0
        self.render_layers()

    @pyqtSlot()
    def shadow_scale_change(self):
        fract = self.scale_slider.value()/99.0
        self.ibl.set_cur_scale(fract)

    @pyqtSlot()
    def shadow_size_change(self):
        fract = self.size_slider.value()/99.0
        min_value = 0.008
        max_value = 0.1
        radius = (1.0-fract) * min_value + fract * max_value
        self.ibl.set_cur_size(radius)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = composite_gui()
    sys.exit(app.exec_())