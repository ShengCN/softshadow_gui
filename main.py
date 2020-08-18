import sys
sys.path.append("..")

import os
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

class App(QWidget):

    def __init__(self):
        super(App, self).__init__()
        self.title = 'Soft Shadow Generation'
        self.left = 400
        self.top = 400
        self.width = 1280
        self.height = 480
        self.ibl_counter = 0
        
        self.template_label = QLabel(self)

        self.cutout_label = QLabel(self)
        self.cutout_label.move(256, 0)

        self.ibl_label = QLabel(self)
        self.ibl_label.move(256 * 2, 0)
        
        self.next_btn = QPushButton("next")
        self.next_btn.move(256 * 4, 0)
        self.next_btn.clicked.connect(self.next_scene)

        # light list
        self.light_list = QListWidget()
        self.light_list.itemClicked.connect(self.light_item_clicked)

        # size slider
        self.size_slider_label = QLabel(self)
        self.size_slider_label.setText('light size')
        self.size_slider = QSlider(Qt.Horizontal)
        self.size_slider.setSingleStep(1)
        self.size_slider.valueChanged.connect(self.size_change)

        # scale slider
        self.scale_slider_label = QLabel(self)
        self.scale_slider_label.setText('scale size')
        self.scale_slider = QSlider(Qt.Horizontal)
        self.scale_slider.setSingleStep(1)
        self.scale_slider.valueChanged.connect(self.scale_change)

        label_group = QGroupBox("Composite")
        self.hlayout = QtWidgets.QHBoxLayout()
        self.hlayout.addWidget(self.template_label)
        self.hlayout.addWidget(self.cutout_label)
        self.hlayout.addWidget(self.ibl_label)
        label_group.setLayout(self.hlayout)

        button_group = QGroupBox("Light control")
        self.vlayout = QtWidgets.QVBoxLayout()
        self.vlayout.addWidget(self.light_list)
        self.vlayout.addWidget(self.size_slider_label)
        self.vlayout.addWidget(self.size_slider)
        self.vlayout.addWidget(self.scale_slider_label)
        self.vlayout.addWidget(self.scale_slider)
        self.vlayout.addWidget(self.next_btn)
        button_group.setLayout(self.vlayout)

        grid = QGridLayout()
        grid.addWidget(label_group, 0, 0)
        grid.addWidget(button_group, 0, 1)
        self.setLayout(grid)

        self.cur_ibl = np.zeros((256,512,3))
        self.cur_size_min, self.cur_size_max = 0.03, 0.2
        self.cur_scale_min, self.cur_scale_max = 0.0, 3.0

        self.cur_ibl_cmd = -1
        self.ibl_cmds = []
        self.init_ibl_state()
        self.add_cur_ibl_state()

        # test imgs
        self.num_test = 4
        self.final_imgs = [join('imgs', '{:03d}_final.png'.format(i)) for i in range(1,self.num_test+1)]
        self.mask_imgs = [join('imgs','{:03d}_mask.png'.format(i)) for i in range(1,self.num_test+1)]
        self.cutout_imgs = [join('imgs','{:03d}_cutout.png'.format(i)) for i in range(1,self.num_test+1)]
        self.cur_exp = 0

        # output folder
        self.cache_folder = 'us_cache'
        os.makedirs(self.cache_folder, exist_ok=True)
        self.initUI()

    def init_ibl_state(self):
        self.cur_x, self.cur_y, self.cur_size, self.cur_scale = 0.0, 0.0, 0.04, 1.0
        self.update_slider()

    def update_slider(self):
        size_factor = (self.cur_size - self.cur_size_min)/(self.cur_size_max - self.cur_size_min)
        scale_factor = (self.cur_scale - self.cur_scale_min)/(self.cur_scale_max-self.cur_scale_min)

        self.scale_slider.setValue(scale_factor * 99.0)
        self.size_slider.setValue(size_factor * 99.0)

    def reset_ibl_cmds(self):
        self.ibl_cmds = []
        self.init_ibl_state()
        self.cur_ibl_cmd = -1
        self.light_list.clear()

    def add_cur_ibl_state(self):
        self.cur_ibl_cmd = len(self.ibl_cmds)
        self.ibl_counter += 1
        self.light_list.addItem('{}'.format(self.ibl_counter))
        self.light_list.setCurrentItem(self.light_list.item(self.cur_ibl_cmd))

        self.ibl_cmds.append((self.cur_x, self.cur_y, self.cur_size, self.cur_scale))

    def light_item_clicked(self, item):
        # self.cur_ibl_cmd = int(item.text())
        self.cur_ibl_cmd = self.light_list.currentRow()
        print('select current item: ', self.cur_ibl_cmd)
        
        # update current status
        self.cur_x, self.cur_y, self.cur_size, self.cur_scale = self.ibl_cmds[self.cur_ibl_cmd]

        # update slider
        self.update_slider()


    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.update_cur_labels()
        self.setFixedSize(self.width, self.height)

        self.show()

    def set_image(self, label, img_path):
        # Create widget
        pixmap = QPixmap(img_path)
        label.setPixmap(pixmap)

    def set_np_img(self, label, img_np):
        if len(img_np.shape) == 2:
            img_np = np.repeat(img_np[:,:,np.newaxis], 3, axis=2)

        pixmap = QPixmap(self.to_qt_img(img_np))
        label.setPixmap(pixmap)

    def to_qt_img(self, np_img):
        if np_img.dtype != np.uint8:
            np_img = np.clip(np_img, 0.0, 1.0)
            np_img = np_img * 255.0
            np_img = np_img.astype(np.uint8)

        h,w,c = np_img.shape
        # bytesPerLine = 3 * w
        return QImage(np_img.data, w,h, QImage.Format_RGB888)

    def update_cur_labels(self):
        self.cur_final, self.cur_cutout, self.cur_mask = self.load_cur_np(self.cur_exp)
        self.set_np_img(self.template_label, self.cur_final)
        self.set_np_img(self.cutout_label, self.cur_cutout)
        self.set_np_img(self.ibl_label, self.cur_ibl)
        
        self.cur_user_comopsite = np.copy(self.cur_cutout)
        self.cur_user_ibl = np.copy(self.cur_ibl)
        self.ibl_label.setFocus()

    def save_current(self):
        cutout_pix, ibl_pix = self.cur_user_comopsite, self.cur_user_ibl
        cur_time = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

        print('cutout path: ', join(self.cache_folder, '{}_cutout.png'.format(cur_time)))
        cv2.imwrite(join(self.cache_folder, '{}_cutout.png'.format(cur_time)), cutout_pix*255.0)
        cv2.imwrite(join(self.cache_folder, '{}_ibl.png'.format(cur_time)), ibl_pix*255.0)

    def next_scene(self):
        self.save_current()

        print('go to next scene')
        self.cur_exp += 1
        self.cur_exp = self.cur_exp % self.num_test
        print(self.cur_exp)
        self.update_cur_labels()
        # self.ibl_cmds = []
        # self.init_ibl_state()
        self.reset_ibl_cmds()

    def load_cur_np(self, cur_exp):
        final, cutout, mask = cv2.imread(self.final_imgs[cur_exp]), cv2.imread(self.cutout_imgs[cur_exp]), cv2.imread(self.mask_imgs[cur_exp])
        return final, cutout, mask
    
    def lerp(self, a, b, f):
        return (1.0-f) * a + f * b

    def size_change(self):
        fract = self.size_slider.value()/99.0
        self.cur_size = self.lerp(self.cur_size_min, self.cur_size_max, fract)

        # print('size value: ', )
        # self.cur_size = np.clip(self.cur_size, 0.001, 1.0)
        if len(self.ibl_cmds) != 0:
            self.ibl_cmds[self.cur_ibl_cmd] = (self.cur_x, self.cur_y, self.cur_size, self.cur_scale)
            self.update_composite()

        self.ibl_label.setFocus()

    def scale_change(self):
        fract = self.scale_slider.value()/99.0
        self.cur_scale = self.lerp(self.cur_scale_min, self.cur_scale_max, fract)

        if len(self.ibl_cmds) != 0:
            self.ibl_cmds[self.cur_ibl_cmd] = (self.cur_x, self.cur_y, self.cur_size, self.cur_scale)
            self.update_composite()

        self.ibl_label.setFocus()

    def keyPressEvent(self, event):      
        if event.key() == QtCore.Qt.Key_A:
            print('add an area light')
            self.add_cur_ibl_state()
            self.init_ibl_state()
            self.update_composite()

        if event.key() == QtCore.Qt.Key_Q:
            self.cur_size += 0.01
            self.cur_size = np.clip(self.cur_size, self.cur_size_min, self.cur_size_max)

            if len(self.ibl_cmds) != 0:
                self.ibl_cmds[self.cur_ibl_cmd] = (self.cur_x, self.cur_y, self.cur_size, self.cur_scale)
                self.update_composite()

        if event.key() == QtCore.Qt.Key_W:
            self.cur_size -= 0.01
            self.cur_size = np.clip(self.cur_size, self.cur_size_min, self.cur_size_max)
            print( self.cur_size)

            if len(self.ibl_cmds) != 0:
                self.ibl_cmds[self.cur_ibl_cmd] = (self.cur_x, self.cur_y, self.cur_size, self.cur_scale)
                self.update_composite()

    def check_mouse(self, x, y):
        if x > 1.0 or x <=0 or y > 1.0 or y <= 0:
            return False
        
        # dis = np.sqrt((x-self.cur_x) ** 2 + (y-self.cur_y) ** 2) 
        # print('cur dis', dis)
        # if dis > 0.1:
        #     return False
        
        return True

    def mouseMoveEvent(self, event):
        if event.buttons() == QtCore.Qt.LeftButton:
            x,y = self.get_relative_pos(event)
            if not self.check_mouse(x,y):
                return

            self.cur_x, self.cur_y = x, y
            if len(self.ibl_cmds) != 0:
                self.ibl_cmds[self.cur_ibl_cmd] = (self.cur_x, self.cur_y, self.cur_size, self.cur_scale)
            self.update_composite()

        self.ibl_label.setFocus()
        super(App, self).mouseMoveEvent(event)

    def set_ibl(self, ibl_np):
        self.set_np_img(self.ibl_label, ibl_np)
    
    def composite(self, final, mask, shadow):
        shadow = np.clip(shadow, 0.0, 1.0)
        composite = mask * final + (1.0 - mask) * (1.0-shadow)
        return composite

    def update_composite(self):
        # if len(self.ibl_cmds) != 0:
        #     self.ibl_cmds[-1] = (self.cur_x, self.cur_y, self.cur_size)

        cur_ibl = self.get_cur_ibl()
        self.cur_user_ibl = cur_ibl

        self.set_ibl(cur_ibl)
        if np.sum(cur_ibl) < 1e-3:
            shadow_result = np.zeros((256,256,3))
        else:
            shadow_result = evaluation.net_render_np(self.cur_mask, cur_ibl)
            shadow_result = np.repeat(shadow_result[:,:,np.newaxis], 3, axis=2)
        final_composite = self.composite(self.cur_final/255.0, self.cur_mask/255.0, shadow_result)
        self.cur_user_comopsite = final_composite
        self.set_np_img(self.cutout_label, final_composite)

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            x,y = self.get_relative_pos(event)
            # print("Press! {}, {}".format(x, y))

            if not self.check_mouse(x,y):
                return

            self.cur_x, self.cur_y = x, y
            if len(self.ibl_cmds) != 0:
                self.ibl_cmds[self.cur_ibl_cmd] = (self.cur_x, self.cur_y, self.cur_size, self.cur_scale)
                self.update_composite()

        if event.button() == QtCore.Qt.RightButton:
            print('Delete last IBL')
            if len(self.ibl_cmds) != 0:
                # self.ibl_cmds = self.ibl_cmds[:-1]
                del self.ibl_cmds[self.cur_ibl_cmd]
                self.light_list.takeItem(self.cur_ibl_cmd)

                self.cur_ibl_cmd = self.cur_ibl_cmd - 1
                self.update_composite() 

        super(App, self).mousePressEvent(event)

    def get_relative_pos(self, mouse_event):
        ibl_pos = self.ibl_label.pos()
        mouse_pos = mouse_event.pos()
        ibl_w, ibl_h = 512, 256
        relative = mouse_pos - ibl_pos
        x,y = relative.x()/ibl_w, 1.0 - (relative.y() - 100)/ibl_h
        print('x: {}, y: {}'.format(x,y))
        return x,y 

    def get_cur_ibl(self):
        num = len(self.ibl_cmds)
        if num == 0:
            return np.zeros((256,512,3))
        gs = ig.Composite(operator=np.add,
                generators=[ig.Gaussian(
                            size=self.ibl_cmds[i][2],
                            scale=self.ibl_cmds[i][3],
                            x=self.ibl_cmds[i][0]-0.5,
                            y=self.ibl_cmds[i][1]-0.5,
                            aspect_ratio=1.0,
                            ) for i in range(num)],
                    xdensity=512)
        return np.repeat(gs()[:,:,np.newaxis], 3, axis=2) 

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
