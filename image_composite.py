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

        self.scale_label = QLabel('scale', self)
        self.scale_slider = QSlider(Qt.Horizontal)

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

        self.canvas_img = img
        if size is not None:
            self.canvas_img = cv2.resize(img, (size[0], size[1]))/255.0

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

    def composite_region(self, cur_canvas, xy, wh):
        tmp = cur_canvas
        canvas_h, canvas_w = tmp.shape[0], tmp.shape[1]

        x, y = xy[0] - self.canvas.pos().x(), xy[1] - self.canvas.pos().y()
        w, h = wh[0], wh[1]

        mask_x, mask_y = 0, 0
        mask_h, mask_w = h, w
        tmp_x, tmp_y = x, y

        # boundary case
        if x < 0:
            tmp_x, mask_x = 0, -x
            mask_w = mask_w - mask_x

        if y < 0:
            tmp_y, mask_y = 0, -y
            mask_h = mask_h - mask_y

        if x + w > canvas_w:
            mask_w = canvas_w - x
            tmp_x = x

        if y + h > canvas_h:
            mask_h = canvas_h - y
            tmp_y = y

        tmp_h, tmp_w = mask_h, mask_w
        canvas_region = (tmp_y, tmp_y + tmp_h, tmp_x, tmp_x + tmp_w)
        widget_region = (mask_y, mask_y + mask_h, mask_x, mask_x + mask_w)
        return canvas_region, widget_region


    def composite_layer_result(self, cur_canvas, xy, wh, composite_img, composite_operator='lerp'):
        """
            input:   canvas image, composite image top-left corner xy, width and height, ...
            outout:  alpha blending compsite result
        """
        tmp = cur_canvas
        cutout_img =  composite_img
        canvas_region, widget_region = self.composite_region(cur_canvas, xy, wh)
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
            plt.imsave('test_shadow.png', shadow_out)

        return tmp

    def soft_shadow_boundary(self, shadow_img):
        return np.clip(np.multiply(self.soft_mask, shadow_img), 0.0, 1.0)


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
        self.render_layers()

    @pyqtSlot()
    def render_layers(self):
        if len(self.cutout_layer) == 0:
            return

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
        out_img = self.render_layers()

        if cv2.imwrite(save_fname[0], out_img*255.0):
            print('file {} saved succeed'.format(save_fname[0]))
        else:
            print('file {} save fail'.format(save_fname[0]))

    @pyqtSlot()
    def light_item_clicked(self, item):
        # self.ibl.set_cur_ibl()
        print(self.light_list.currentRow())

    @pyqtSlot()
    def add_light(self):
        self.ibl.add_light()

    @pyqtSlot()
    def update_list(self, cur_ibl):
        self.light_list.clear()
        light_num = self.ibl.get_light_num()
        for i in range(light_num):
            self.light_list.addItem('light {}'.format(i))

        self.light_list.setCurrentRow(cur_ibl)

    @pyqtSlot()
    def shadow_intensity_change(self):
        self.cur_shadow_itensity_fract = self.shadow_intensity_slider.value()/99.0
        self.render_layers()



if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = composite_gui()
    sys.exit(app.exec_())