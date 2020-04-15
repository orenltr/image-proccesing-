# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'texture_synthesis.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication,QWidget, QVBoxLayout, QPushButton, QFileDialog , QLabel, QTextEdit
import sys
from PyQt5.QtGui import QPixmap
import non_parametric_sampling
from non_parametric_sampling import *
import cv2
import os
import numpy as np





class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.setEnabled(True)
        MainWindow.resize(900, 500)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        MainWindow.setMinimumSize(QtCore.QSize(900, 500))
        MainWindow.setMaximumSize(QtCore.QSize(900, 500))
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setContentsMargins(6, 6, 6, 6)
        self.gridLayout.setSpacing(4)
        self.gridLayout.setObjectName("gridLayout")
        self.input_tolerance = QtWidgets.QLineEdit(self.centralwidget)
        self.input_tolerance.setCursor(QtGui.QCursor(QtCore.Qt.IBeamCursor))
        self.input_tolerance.setObjectName("input_tolerance")
        self.gridLayout.addWidget(self.input_tolerance, 10, 2, 1, 1)
        self.sample = QtWidgets.QLabel(self.centralwidget)
        self.sample.setText("")
        self.sample.setScaledContents(True)
        self.sample.setObjectName("sample")
        self.gridLayout.addWidget(self.sample, 2, 0, 1, 3)
        self.input_width = QtWidgets.QLineEdit(self.centralwidget)
        self.input_width.setCursor(QtGui.QCursor(QtCore.Qt.IBeamCursor))
        self.input_width.setObjectName("input_width")
        self.gridLayout.addWidget(self.input_width, 4, 2, 2, 1)
        spacerItem = QtWidgets.QSpacerItem(201, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem, 1, 6, 1, 1)
        self.label_result_size = QtWidgets.QLabel(self.centralwidget)
        self.label_result_size.setObjectName("label_result_size")
        self.gridLayout.addWidget(self.label_result_size, 3, 0, 1, 1)
        self.frame = QtWidgets.QFrame(self.centralwidget)
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.gridLayout.addWidget(self.frame, 4, 1, 2, 1)
        self.progressBar = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBar.setProperty("value", 0)
        self.progressBar.setTextVisible(True)
        self.progressBar.setObjectName("progressBar")
        self.gridLayout.addWidget(self.progressBar, 13, 4, 1, 3)
        self.input_method = QtWidgets.QComboBox(self.centralwidget)
        self.input_method.setObjectName("input_method")
        self.input_method.addItem("")
        self.input_method.addItem("")
        self.gridLayout.addWidget(self.input_method, 8, 2, 2, 1)
        self.label_overlap = QtWidgets.QLabel(self.centralwidget)
        self.label_overlap.setEnabled(True)
        self.label_overlap.setObjectName("label_overlap")
        self.gridLayout.addWidget(self.label_overlap, 11, 0, 2, 1)
        self.generate = QtWidgets.QPushButton(self.centralwidget)
        self.generate.setObjectName("generate")
        self.generate.setDisabled(True)
        self.gridLayout.addWidget(self.generate, 13, 0, 1, 3)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem1, 1, 4, 1, 1)
        self.label_hight = QtWidgets.QLabel(self.centralwidget)
        self.label_hight.setObjectName("label_hight")
        self.gridLayout.addWidget(self.label_hight, 6, 0, 1, 1)
        self.input_overlap = QtWidgets.QLineEdit(self.centralwidget)
        self.input_overlap.setObjectName("input_overlap")
        self.gridLayout.addWidget(self.input_overlap, 11, 2, 2, 1)
        spacerItem2 = QtWidgets.QSpacerItem(131, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem2, 1, 5, 1, 1)
        self.insert = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.insert.sizePolicy().hasHeightForWidth())
        self.insert.setSizePolicy(sizePolicy)
        self.insert.setObjectName("insert")
        self.gridLayout.addWidget(self.insert, 1, 0, 1, 3)
        self.label_tolerance = QtWidgets.QLabel(self.centralwidget)
        self.label_tolerance.setScaledContents(False)
        self.label_tolerance.setObjectName("label_tolerance")
        self.gridLayout.addWidget(self.label_tolerance, 10, 0, 1, 1)
        spacerItem3 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout.addItem(spacerItem3, 2, 3, 1, 1)

        self.input_hight = QtWidgets.QLineEdit(self.centralwidget)
        self.input_hight.setObjectName("input_hight")


        self.gridLayout.addWidget(self.input_hight, 6, 2, 2, 1)
        self.label_method = QtWidgets.QLabel(self.centralwidget)
        self.label_method.setObjectName("label_method")
        self.gridLayout.addWidget(self.label_method, 8, 0, 1, 1)
        self.label_width = QtWidgets.QLabel(self.centralwidget)
        self.label_width.setObjectName("label_width")
        self.gridLayout.addWidget(self.label_width, 4, 0, 1, 1)
        self.result = QtWidgets.QLabel(self.centralwidget)
        self.result.setText("")
        self.result.setScaledContents(True)
        self.result.setObjectName("result")
        self.gridLayout.addWidget(self.result, 2, 4, 10, 3)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionnew = QtWidgets.QAction(MainWindow)
        self.actionnew.setObjectName("actionnew")
        self.actionsave_as = QtWidgets.QAction(MainWindow)
        self.actionsave_as.setObjectName("actionsave_as")
        self.actionsettings = QtWidgets.QAction(MainWindow)
        self.actionsettings.setObjectName("actionsettings")

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        # defaults
        self.input_hight.setText('50')
        self.input_width.setText('50')
        self.input_overlap.setText('0.1666')
        self.input_tolerance.setText('0.1')

        # insert sample with file dialog
        self.insert.clicked.connect(self.browseImage)

        # generate sample
        self.generate.clicked.connect(lambda: self.run(sample_path))


    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label_result_size.setText(_translate("MainWindow", "Result Size"))
        self.input_method.setItemText(0, _translate("MainWindow", "Image Quilting"))
        self.input_method.setItemText(1, _translate("MainWindow", "Non-parametric Sampling"))
        self.label_overlap.setText(_translate("MainWindow", "overlap:"))
        self.generate.setText(_translate("MainWindow", "Generate Texture"))
        self.label_hight.setText(_translate("MainWindow", "Hight:"))
        self.insert.setText(_translate("MainWindow", "Insert Sample"))
        self.label_tolerance.setText(_translate("MainWindow", "Tolerance:"))
        self.label_method.setText(_translate("MainWindow", "method:"))
        self.label_width.setText(_translate("MainWindow", "Width:"))
        self.actionnew.setText(_translate("MainWindow", "new"))
        self.actionsave_as.setText(_translate("MainWindow", "save as"))
        self.actionsettings.setText(_translate("MainWindow", "settings"))


    def run(self, sample_path):

        # sample = cv2.imread(sample_path)
        window_height = int(self.input_hight.text())
        window_width = int(self.input_width.text())
        overlap = float(self.input_overlap.text())
        tolerance = float(self.input_tolerance.text())
        # window_height = int(self.input_kernel.text())
        args = Args(sample_path,hight=window_height,width=window_width,overlap=overlap,tolerance=tolerance)
        if self.input_method.currentIndex() == 1:
            self.nonParametricSampling(args)


    def browseImage(self):
        global sample_path
        fname = QFileDialog.getOpenFileName(None, 'Open file', filter="Image files (*.jpg *.png)")
        sample_path = fname[0]
        self.sample.setPixmap(QPixmap(sample_path))
        self.generate.setDisabled(False)



    def nonParametricSampling(self,args):
        sample = cv2.imread(sample_path)
        # sample = cv2.imread(r'textures\t5.png')
        non_parametric_sampling.validate_args(args)

        synthesized_texture = self.synthesize_texture(original_sample=sample,
                                                 window_size=(args.window_height, args.window_width),
                                                 kernel_size=args.kernel_size)
        x=1

    def synthesize_texture(self,original_sample, window_size, kernel_size, visualize=True):
        global gif_count
        (sample, window, mask, padded_window,
         padded_mask, result_window) = initialize_texture_synthesis(original_sample, window_size, kernel_size)

        # Synthesize texture until all pixels in the window are filled.
        while texture_can_be_synthesized(mask):
            # Get neighboring indices
            neighboring_indices = get_neighboring_pixel_indices(mask)

            # Permute and sort neighboring indices by quantity of 8-connected neighbors.
            neighboring_indices = permute_neighbors(mask, neighboring_indices)

            for ch, cw in zip(neighboring_indices[0], neighboring_indices[1]):

                window_slice = padded_window[ch:ch + kernel_size, cw:cw + kernel_size]
                mask_slice = padded_mask[ch:ch + kernel_size, cw:cw + kernel_size]

                # Compute SSD for the current pixel neighborhood and select an index with low error.
                ssd = normalized_ssd(sample, window_slice, mask_slice)
                indices = get_candidate_indices(ssd)
                selected_index = select_pixel_index(ssd, indices)

                # Translate index to accommodate padding.
                selected_index = (selected_index[0] + kernel_size // 2, selected_index[1] + kernel_size // 2)

                # Set windows and mask.
                window[ch, cw] = sample[selected_index]
                mask[ch, cw] = 1
                result_window[ch, cw] = original_sample[selected_index[0], selected_index[1]]

                if visualize:
                    cv2.imshow('synthesis window', result_window)
                    height, width, channel = result_window.shape
                    bytesPerLine = 3 * width
                    qImg = QtGui.QImage(result_window.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888).rgbSwapped()
                    # result_qimage = QtGui.QImage(result_window.data, result_window.shape[1], result_window.shape[0],
                    #                           QtGui.QImage.Format_RGB888).rgbSwapped()
                    self.result.setPixmap(QtGui.QPixmap.fromImage(qImg))
                    key = cv2.waitKey(1)
                    if key == 27:
                        cv2.destroyAllWindows()
                        return result_window

        if visualize:
            cv2.imshow('synthesis window', result_window)
            # self.image = QtGui.QImage(self.image.data, self.image.shape[1], self.image.shape[0],
            #                           QtGui.QImage.Format_RGB888).rgbSwapped()
            # self.image_frame.setPixmap(QtGui.QPixmap.fromImage(self.image))
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return result_window



class Args(object):
    def __init__(self,sample_path,hight=50,width=50,overlap = 1/6,tolerance=0.1, kernel=11):
        self.__window_height = hight
        self.__window_width = width
        self.__kernel_size = kernel
        self.__overlap = overlap
        self.__tolerance = tolerance
        self.__sample_path = sample_path

    @property
    def window_height(self):
        return self.__window_height
    @property
    def window_width(self):
        return self.__window_width
    @property
    def kernel_size(self):
        return self.__kernel_size
    @property
    def overlap(self):
        return self.__overlap
    @property
    def tolerance(self):
        return self.__tolerance
    @property
    def sample_path(self):
        return self.__sample_path

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
