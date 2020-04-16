import sys
import os
import cv2
import numpy as np
from math import ceil
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication,QWidget, QVBoxLayout, QPushButton, QFileDialog , QLabel, QTextEdit
from PyQt5.QtGui import QPixmap
import non_parametric_sampling
from non_parametric_sampling import *
from generate import *
from itertools import product




class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.setEnabled(True)
        MainWindow.resize(1000, 500)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        MainWindow.setMinimumSize(QtCore.QSize(1000, 500))
        MainWindow.setMaximumSize(QtCore.QSize(1000, 515))
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setContentsMargins(6, 6, 6, 6)
        self.gridLayout.setSpacing(4)
        self.gridLayout.setObjectName("gridLayout")
        self.insert = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.insert.sizePolicy().hasHeightForWidth())
        self.insert.setSizePolicy(sizePolicy)
        self.insert.setObjectName("insert")
        self.gridLayout.addWidget(self.insert, 0, 0, 1, 2)
        self.input_path = QtWidgets.QTextBrowser(self.centralwidget)
        self.input_path.setMaximumSize(QtCore.QSize(16777215, 20))
        self.input_path.setReadOnly(True)
        self.input_path.setObjectName("input_path")
        self.gridLayout.addWidget(self.input_path, 0, 2, 1, 3)
        spacerItem = QtWidgets.QSpacerItem(18, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem, 0, 5, 2, 1)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem1, 0, 6, 1, 1)
        spacerItem2 = QtWidgets.QSpacerItem(131, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem2, 0, 7, 1, 1)
        spacerItem3 = QtWidgets.QSpacerItem(201, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem3, 0, 8, 1, 1)
        self.output = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.output.sizePolicy().hasHeightForWidth())
        self.output.setSizePolicy(sizePolicy)
        self.output.setObjectName("output")
        self.gridLayout.addWidget(self.output, 1, 0, 1, 2)
        self.output_path = QtWidgets.QTextBrowser(self.centralwidget)
        self.output_path.setMaximumSize(QtCore.QSize(16777215, 20))
        self.output_path.setObjectName("output_path")
        self.gridLayout.addWidget(self.output_path, 1, 2, 1, 3)
        self.result = QtWidgets.QLabel(self.centralwidget)
        self.result.setMinimumSize(QtCore.QSize(600, 400))
        self.result.setMaximumSize(QtCore.QSize(600, 400))
        self.result.setAutoFillBackground(True)
        self.result.setText("")
        self.result.setScaledContents(True)
        self.result.setObjectName("result")
        self.gridLayout.addWidget(self.result, 1, 6, 10, 3)
        self.sample = QtWidgets.QLabel(self.centralwidget)
        self.sample.setMinimumSize(QtCore.QSize(300, 200))
        self.sample.setMaximumSize(QtCore.QSize(300, 200))
        self.sample.setAutoFillBackground(True)
        self.sample.setText("")
        self.sample.setScaledContents(True)
        self.sample.setObjectName("sample")
        self.gridLayout.addWidget(self.sample, 2, 0, 1, 5)
        spacerItem4 = QtWidgets.QSpacerItem(20, 49, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout.addItem(spacerItem4, 3, 0, 1, 2)
        self.label_result_size = QtWidgets.QLabel(self.centralwidget)
        self.label_result_size.setObjectName("label_result_size")
        self.gridLayout.addWidget(self.label_result_size, 4, 0, 1, 2)
        self.label_width = QtWidgets.QLabel(self.centralwidget)
        self.label_width.setObjectName("label_width")
        self.gridLayout.addWidget(self.label_width, 5, 1, 1, 1)
        self.input_width = QtWidgets.QLineEdit(self.centralwidget)
        self.input_width.setMaximumSize(QtCore.QSize(50, 16777215))
        self.input_width.setCursor(QtGui.QCursor(QtCore.Qt.IBeamCursor))
        self.input_width.setObjectName("input_width")
        self.gridLayout.addWidget(self.input_width, 5, 3, 1, 1)
        spacerItem5 = QtWidgets.QSpacerItem(131, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem5, 5, 4, 2, 1)
        self.label_hight = QtWidgets.QLabel(self.centralwidget)
        self.label_hight.setObjectName("label_hight")
        self.gridLayout.addWidget(self.label_hight, 6, 1, 1, 1)
        self.input_hight = QtWidgets.QLineEdit(self.centralwidget)
        self.input_hight.setMaximumSize(QtCore.QSize(50, 16777215))
        self.input_hight.setObjectName("input_hight")
        self.gridLayout.addWidget(self.input_hight, 6, 3, 1, 1)
        self.label_method = QtWidgets.QLabel(self.centralwidget)
        self.label_method.setObjectName("label_method")
        self.gridLayout.addWidget(self.label_method, 7, 0, 1, 1)
        self.input_method = QtWidgets.QComboBox(self.centralwidget)
        self.input_method.setObjectName("input_method")
        self.input_method.addItem("")
        self.input_method.addItem("")
        self.gridLayout.addWidget(self.input_method, 7, 1, 1, 3)
        self.label_kernel = QtWidgets.QLabel(self.centralwidget)
        self.label_kernel.setScaledContents(False)
        self.label_kernel.setObjectName("label_kernel")
        self.gridLayout.addWidget(self.label_kernel, 8, 1, 1, 2)
        self.input_kernel = QtWidgets.QLineEdit(self.centralwidget)
        self.input_kernel.setMaximumSize(QtCore.QSize(50, 16777215))
        self.input_kernel.setCursor(QtGui.QCursor(QtCore.Qt.IBeamCursor))
        self.input_kernel.setObjectName("input_kernel")
        self.gridLayout.addWidget(self.input_kernel, 8, 3, 1, 1)
        self.label_tolerance = QtWidgets.QLabel(self.centralwidget)
        self.label_tolerance.setScaledContents(False)
        self.label_tolerance.setObjectName("label_tolerance")
        self.gridLayout.addWidget(self.label_tolerance, 9, 1, 1, 1)
        self.input_tolerance = QtWidgets.QLineEdit(self.centralwidget)
        self.input_tolerance.setMaximumSize(QtCore.QSize(50, 16777215))
        self.input_tolerance.setCursor(QtGui.QCursor(QtCore.Qt.IBeamCursor))
        self.input_tolerance.setObjectName("input_tolerance")
        self.gridLayout.addWidget(self.input_tolerance, 9, 3, 1, 1)
        self.label_overlap = QtWidgets.QLabel(self.centralwidget)
        self.label_overlap.setEnabled(True)
        self.label_overlap.setObjectName("label_overlap")
        self.gridLayout.addWidget(self.label_overlap, 10, 1, 1, 1)
        self.input_overlap = QtWidgets.QLineEdit(self.centralwidget)
        self.input_overlap.setMaximumSize(QtCore.QSize(50, 16777215))
        self.input_overlap.setObjectName("input_overlap")
        self.gridLayout.addWidget(self.input_overlap, 10, 3, 1, 1)
        self.generate = QtWidgets.QPushButton(self.centralwidget)
        self.generate.setMinimumSize(QtCore.QSize(0, 30))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.generate.setFont(font)
        self.generate.setObjectName("generate")
        self.gridLayout.addWidget(self.generate, 11, 0, 1, 4)
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
        self.input_hight.setText('100')
        self.input_width.setText('150')
        self.input_overlap.setText('0.1666')
        self.input_tolerance.setText('0.1')
        self.input_kernel.setText('21')

        # input file dialog
        self.insert.clicked.connect(self.browseImage)

        # output file dialog
        self.output.clicked.connect(self.saveImage)

        # generate sample
        self.generate.clicked.connect(self.run)

        self.input_method.currentIndexChanged.connect(self.overlapStatus)




    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.insert.setStatusTip(_translate("MainWindow", "no spaces in path!"))
        self.insert.setText(_translate("MainWindow", "Insert Sample"))
        self.input_path.setStatusTip(_translate("MainWindow", "no spaces in path!"))
        self.output.setStatusTip(_translate("MainWindow", "no spaces in path!"))
        self.output.setText(_translate("MainWindow", "Output Location"))
        self.output_path.setStatusTip(_translate("MainWindow", "no spaces in path!"))
        self.label_result_size.setText(_translate("MainWindow", "Result Size"))
        self.label_width.setText(_translate("MainWindow", "Width:"))
        self.input_width.setStatusTip(_translate("MainWindow", "width of the result image in pixels"))
        self.label_hight.setText(_translate("MainWindow", "Hight:"))
        self.input_hight.setStatusTip(_translate("MainWindow", "hight of the result image in pixels"))
        self.label_method.setText(_translate("MainWindow", "method:"))
        self.input_method.setItemText(0, _translate("MainWindow", "Image Quilting"))
        self.input_method.setItemText(1, _translate("MainWindow", "Non-parametric Sampling"))
        self.label_kernel.setText(_translate("MainWindow", "Kernel/Block Size:"))
        self.input_kernel.setStatusTip(_translate("MainWindow", "image quilting requires block size. non_parametric sampling requiers odd number kernel size (pixels)"))
        self.label_tolerance.setText(_translate("MainWindow", "Tolerance:"))
        self.input_tolerance.setStatusTip(_translate("MainWindow", "rms tolerance (0-1)"))
        self.label_overlap.setText(_translate("MainWindow", "overlap:"))
        self.input_overlap.setStatusTip(_translate("MainWindow", "block overlap for image quilting (0-1)"))
        self.generate.setStatusTip(_translate("MainWindow", "create "))
        self.generate.setText(_translate("MainWindow", "Generate Texture"))
        self.actionnew.setText(_translate("MainWindow", "new"))
        self.actionsave_as.setText(_translate("MainWindow", "save as"))
        self.actionsettings.setText(_translate("MainWindow", "settings"))


    def run(self):

        for p,path in enumerate(sample_paths):
            self.result.clear()
            self.sample.clear()
            self.sample.setPixmap(QPixmap(path))
            # self.generate.setDisabled(True)
            # arguments for
            window_height = int(self.input_hight.text())
            window_width = int(self.input_width.text())
            overlap = float(self.input_overlap.text())
            tolerance = float(self.input_tolerance.text())
            kernel_size = int(self.input_kernel.text())
            args = Args(path,hight=window_height,width=window_width,overlap=overlap,tolerance=tolerance, kernel=kernel_size)

            # which synthesis method
            if self.input_method.currentIndex() == 0:
                result = self.imageQuilting(args)
            if self.input_method.currentIndex() == 1:
                result = self.nonParametricSampling(args)

            if len(sample_paths) > 1:
                cv2.imwrite(output_directory+"/synth_texture_"+str(p)+".jpg", result)

            else:
                cv2.imwrite(output_file, result)
            # cv2.imwrite(output_file, result)


    def browseImage(self):
        global sample_paths
        fname = QFileDialog.getOpenFileNames(None, 'Open file', filter="Image files (*.jpg *.png)")
        sample_paths = fname[0]
        self.sample.setPixmap(QPixmap(sample_paths[0]))
        self.generate.setDisabled(False)
        self.input_path.setText(sample_paths[0])

    def saveImage(self):
        global output_file
        global output_directory
        if len(sample_paths) > 1:
            fname = QFileDialog.getExistingDirectory(None,"Select Directory")
            output_directory = fname
            self.output_path.setText(output_directory)

        else:
            fname = QFileDialog.getSaveFileName(None, 'Open file', filter="Image files (*.jpg *.png)")
            output_file = fname[0]
            self.output_path.setText(output_file)

    def overlapStatus(self):
        if self.input_method.currentIndex()==1:
            self.input_overlap.setDisabled(True)
        else:
            self.input_overlap.setDisabled(False)

    def nonParametricSampling(self,args):
        sample = cv2.imread(args.sample_path)
        non_parametric_sampling.validate_args(args)

        return self.synthesizeTexture(original_sample=sample,
                                                 window_size=(args.window_height, args.window_width),
                                                 kernel_size=args.kernel_size)

    def imageQuilting(self, args):
        sample = cv2.imread(args.sample_path)
        sample = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB) / 255.0
        # non_parametric_sampling.validate_args(args)
        if args.overlap > 0:
            args.overlap = int(args.kernel_size * args.overlap)
        else:
            args.overlap = int(args.kernel_size / 6.0)

        return self.generateTextureMap(sample, args.kernel_size, args.overlap, args.window_height, args.window_width, args.tolerance)

    def synthesizeTexture(self, original_sample, window_size, kernel_size, visualize=True):
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

                # visualize
                cv2.imshow('synthesis window', result_window)
                height, width, channel = result_window.shape
                bytesPerLine = 3 * width
                qImg = QtGui.QImage(result_window.data, width, height, bytesPerLine,
                                    QtGui.QImage.Format_RGB888).rgbSwapped()
                self.result.setPixmap(QtGui.QPixmap.fromImage(qImg))
                key = cv2.waitKey(1)

                # if visualize:
                #     cv2.imshow('synthesis window', result_window)
                #     height, width, channel = result_window.shape
                #     bytesPerLine = 3 * width
                #     qImg = QtGui.QImage(result_window.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888).rgbSwapped()
                #     self.result.setPixmap(QtGui.QPixmap.fromImage(qImg))
                #     key = cv2.waitKey(1)
                #     if key == 27:
                #         cv2.destroyAllWindows()
                #         return result_window

        # if visualize:
        #     cv2.imshow('synthesis window', result_window)
        #     height, width, channel = result_window.shape
        #     bytesPerLine = 3 * width
        #     qImg = QtGui.QImage(result_window.data, width, height, bytesPerLine,
        #                         QtGui.QImage.Format_RGB888).rgbSwapped()
        #     self.result.setPixmap(QtGui.QPixmap.fromImage(qImg))
        #     cv2.waitKey(1)
        #     cv2.destroyAllWindows()

        return result_window

    def generateTextureMap(self, image, blocksize, overlap, window_height, window_width, tolerance, visualize=True):
        nH = int(ceil((window_height - blocksize) * 1.0 / (blocksize - overlap)))  # number of blocks in output image
        nW = int(ceil((window_width - blocksize) * 1.0 / (blocksize - overlap)))

        result_window = np.zeros(((blocksize + nH * (blocksize - overlap)),
                                  (blocksize + nW * (blocksize - overlap)), image.shape[2]))


        # Starting index and block
        H, W = image.shape[:2]
        randH = np.random.randint(H - blocksize)
        randW = np.random.randint(W - blocksize)

        startBlock = image[randH:randH + blocksize, randW:randW + blocksize]
        result_window[:blocksize, :blocksize, :] = startBlock

        # Fill the first row 
        for i, blkIdx in enumerate(range((blocksize - overlap), result_window.shape[1] - overlap, (blocksize - overlap))):
            # Find horizontal error for this block
            # Calculate min, find index having tolerance
            # Choose one randomly among them
            # blkIdx = block index to put in
            refBlock = result_window[:blocksize, (blkIdx - blocksize + overlap):(blkIdx + overlap)]
            patchBlock = findPatchHorizontal(refBlock, image, blocksize, overlap, tolerance)
            minCutPatch = getMinCutPatchHorizontal(refBlock, patchBlock, blocksize, overlap)
            result_window[:blocksize, (blkIdx):(blkIdx + blocksize)] = minCutPatch

            # visualize
            result_view = (255 * result_window).astype(np.uint8)
            result_view = cv2.cvtColor(result_view, cv2.COLOR_RGB2BGR)
            cv2.imshow('synthesis window', result_view)
            height, width, channel = result_view.shape
            bytesPerLine = 3 * width
            qImg = QtGui.QImage(result_view.data, width, height, bytesPerLine,
                                QtGui.QImage.Format_RGB888).rgbSwapped()
            self.result.setPixmap(QtGui.QPixmap.fromImage(qImg))
            key = cv2.waitKey(1)

            # if visualize:
            #     result_view = (255 * result_window).astype(np.uint8)
            #     result_view = cv2.cvtColor(result_view, cv2.COLOR_RGB2BGR)
            #     cv2.imshow('synthesis window', result_view)
            #     height, width, channel = result_view.shape
            #     bytesPerLine = 3 * width
            #     qImg = QtGui.QImage(result_view.data, width, height, bytesPerLine,
            #                         QtGui.QImage.Format_RGB888).rgbSwapped()
            #     self.result.setPixmap(QtGui.QPixmap.fromImage(qImg))
            #     key = cv2.waitKey(1)
            #     if key == 27:
            #         cv2.destroyAllWindows()
            #         return result_view
            
        # print("{} out of {} rows complete...".format(1, nH + 1))
        
        ### Fill the first column
        for i, blkIdx in enumerate(range((blocksize - overlap), result_window.shape[0] - overlap, (blocksize - overlap))):
            # Find vertical error for this block
            # Calculate min, find index having tolerance
            # Choose one randomly among them
            # blkIdx = block index to put in
            refBlock = result_window[(blkIdx - blocksize + overlap):(blkIdx + overlap), :blocksize]
            patchBlock = findPatchVertical(refBlock, image, blocksize, overlap, tolerance)
            minCutPatch = getMinCutPatchVertical(refBlock, patchBlock, blocksize, overlap)

            result_window[(blkIdx):(blkIdx + blocksize), :blocksize] = minCutPatch

            # visualize
            result_view = (255 * result_window).astype(np.uint8)
            result_view = cv2.cvtColor(result_view, cv2.COLOR_RGB2BGR)
            cv2.imshow('synthesis window', result_view)
            height, width, channel = result_view.shape
            bytesPerLine = 3 * width
            qImg = QtGui.QImage(result_view.data, width, height, bytesPerLine,
                                QtGui.QImage.Format_RGB888).rgbSwapped()
            self.result.setPixmap(QtGui.QPixmap.fromImage(qImg))
            key = cv2.waitKey(1)

            # if visualize:
            #     result_view = (255 * result_window).astype(np.uint8)
            #     result_view = cv2.cvtColor(result_view, cv2.COLOR_RGB2BGR)
            #     cv2.imshow('synthesis window', result_view)
            #     height, width, channel = result_view.shape
            #     bytesPerLine = 3 * width
            #     qImg = QtGui.QImage(result_view.data, width, height, bytesPerLine,
            #                         QtGui.QImage.Format_RGB888).rgbSwapped()
            #     self.result.setPixmap(QtGui.QPixmap.fromImage(qImg))
            #     key = cv2.waitKey(1)
            #     if key == 27:
            #         cv2.destroyAllWindows()
            #         return result_view

        ### Fill in the other rows and columns
        for i in range(1, nH + 1):
            for j in range(1, nW + 1):
                # Choose the starting index for the texture placement
                block_index_i = i * (blocksize - overlap)
                block_index_j = j * (blocksize - overlap)
                # Find the left and top block, and the min errors independently
                refBlockLeft = result_window[(block_index_i):(block_index_i + blocksize),
                               (block_index_j - blocksize + overlap):(block_index_j + overlap)]
                refBlockTop = result_window[(block_index_i - blocksize + overlap):(block_index_i + overlap),
                              (block_index_j):(block_index_j + blocksize)]

                patchBlock = findPatchBoth(refBlockLeft, refBlockTop, image, blocksize, overlap, tolerance)
                minCutPatch = getMinCutPatchBoth(refBlockLeft, refBlockTop, patchBlock, blocksize, overlap)

                result_window[(block_index_i):(block_index_i + blocksize), (block_index_j):(block_index_j + blocksize)] = minCutPatch

                # visualize
                result_view = (255 * result_window).astype(np.uint8)
                result_view = cv2.cvtColor(result_view, cv2.COLOR_RGB2BGR)
                cv2.imshow('synthesis window', result_view)
                height, width, channel = result_view.shape
                bytesPerLine = 3 * width
                qImg = QtGui.QImage(result_view.data, width, height, bytesPerLine,
                                    QtGui.QImage.Format_RGB888).rgbSwapped()
                self.result.setPixmap(QtGui.QPixmap.fromImage(qImg))
                key = cv2.waitKey(1)

                # if visualize:
                #     result_view = (255 * result_window).astype(np.uint8)
                #     result_view = cv2.cvtColor(result_view, cv2.COLOR_RGB2BGR)
                #     cv2.imshow('synthesis window', result_view)
                #     height, width, channel = result_view.shape
                #     bytesPerLine = 3 * width
                #     qImg = QtGui.QImage(result_view.data, width, height, bytesPerLine,
                #                         QtGui.QImage.Format_RGB888).rgbSwapped()
                #     self.result.setPixmap(QtGui.QPixmap.fromImage(qImg))
                #     key = cv2.waitKey(1)
                #     if key == 27:
                #         cv2.destroyAllWindows()
                #         return result_view

            # print("{} out of {} rows complete...".format(i + 1, nH + 1))



        return result_view


class Args(object):
    def __init__(self,sample_path,hight=50,width=50,overlap = 1/6,tolerance=0.1, kernel=20):
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
    @overlap.setter
    def overlap(self, value):
        self.__overlap = value
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
