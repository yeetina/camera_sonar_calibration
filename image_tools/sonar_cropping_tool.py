import cv2
import sys
import os
import json
import pickle
import numpy as np
from PyQt5 import QtGui, QtWidgets, QtCore
from PyQt5.QtGui import QPixmap

import matplotlib
import matplotlib.figure
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT

class NavigationToolbar(NavigationToolbar2QT):
    """
    Only display relevant buttons.
    Solution copied from:
    https://stackoverflow.com/questions/12695678/how-to-modify-the-navigation-toolbar-easily-in-a-matplotlib-figure-window
    """
    toolitems = [
        t
        for t in NavigationToolbar2QT.toolitems
        if t[0] in ("Home", "Pan", "Zoom", "Save")
    ]


class EnterPointDialog(QtWidgets.QDialog):
    def __init__(self, point_cb):
        super(EnterPointDialog, self).__init__()
        # This callback will be called with the label that should be removed
        self.setWindowTitle("Select Label")
        self.point_cb = point_cb
        self.setup_layout()

    def setup_layout(self):
        text_label = QtWidgets.QLabel("Label:")
        ok_button = QtWidgets.QPushButton("OK")
        ok_button.clicked.connect(self.handle_ok)
        cancel_button = QtWidgets.QPushButton("Cancel")
        cancel_button.clicked.connect(self.handle_cancel)
        self.select_postion = QtWidgets.QComboBox()
        self.select_postion.addItems(['Origin', 'Left', 'Right'])

        text_row = QtWidgets.QHBoxLayout()
        text_row.addWidget(text_label)
        
        dropdown_row = QtWidgets.QHBoxLayout()
        dropdown_row.addWidget(self.select_postion)

        button_row = QtWidgets.QHBoxLayout()
        button_row.addWidget(ok_button)
        button_row.addWidget(cancel_button)

        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(text_row)
        layout.addLayout(dropdown_row)
        layout.addLayout(button_row)
        self.setLayout(layout)

    def handle_cancel(self):
        # DO nothing, just close the window
        self.done(0)

    def handle_ok(self):
        option = self.select_postion.currentText()
        self.point_cb(option)
        self.done(0)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, sonar_file, json_filename = "sonar_cropping_params.json"):
        """
        GUI for selecting the 
        """
        super(MainWindow, self).__init__()
        self._main = QtWidgets.QWidget()
        self.setCentralWidget(self._main)
        self.raw_sonar_im = cv2.imread(sonar_file, cv2.IMREAD_GRAYSCALE)
        self.json_filename = json_filename
        self.arc_points = {}
        self.im_height, self.im_width = self.raw_sonar_im.shape
        self.cropping_params = {"crop_left": 0, 
                                "crop_right": self.im_width,
                                "crop_top": 0,
                                "crop_bottom": self.im_height,
                                "angle_start": 0,
                                "angle_end": 0,
                                "center": 0,
                                "radius": 0}
        
        self.setup_layout()
        self.plot_sonar_im(False)

    def setup_layout(self):
        self.layout = QtWidgets.QHBoxLayout(self._main)

        self.sonar_image_fig = matplotlib.figure.Figure()
        self.sonar_image_ax = self.sonar_image_fig.add_axes([0.0, 0.0, 1.0, 1.0])
        self.sonar_image_artist = None
        self.sonar_image_canvas = FigureCanvas(self.sonar_image_fig)
        self.sonar_image_canvas.mpl_connect(
            "button_press_event", self.handle_sonar_click
        )
        self.sonar_image_canvas.mpl_connect(
            "button_release_event", self.handle_sonar_release
        )
        raw_sonar_col = QtWidgets.QVBoxLayout()
        raw_sonar_col_header = QtWidgets.QHBoxLayout()
        raw_sonar_col_label = QtWidgets.QLabel("Raw Sonar Image")
        self.raw_sonar_col_toolbar = NavigationToolbar(self.sonar_image_canvas, self)
        raw_sonar_col_header.addWidget(raw_sonar_col_label)
        raw_sonar_col_header.addWidget(self.raw_sonar_col_toolbar)
        raw_sonar_col.addLayout(raw_sonar_col_header, stretch=0)
        raw_sonar_col.addWidget(self.sonar_image_canvas)

        # button
        self.button_row = QtWidgets.QHBoxLayout()
        self.save_button = QtWidgets.QPushButton("Save selections")
        self.save_button.setStyleSheet("padding: 3px;")
        self.save_button.clicked.connect(self.handle_save_button)
        self.button_row.addWidget(self.save_button)
        raw_sonar_col.addLayout(self.button_row)
        sonar_col = QtWidgets.QWidget()
        sonar_col.setLayout(raw_sonar_col)

        instructions_layout = QtWidgets.QVBoxLayout()
        instructions_heading = QtWidgets.QLabel("How to use this tool")
        instructions_layout.addWidget(instructions_heading)
        diagram_im = QPixmap("./sonar_arc_diagram.png")
        diagram = QtWidgets.QLabel()
        diagram.setPixmap(diagram_im)
        image_grid = QtWidgets.QGridLayout()
        image_grid.addWidget(diagram,1,1)
        instructions_layout.addLayout(image_grid)
        instructions_text = QtWidgets.QLabel()
        instructions_label = (f"Click anywhere on the sonar image to add a point.\n" 
                    "Use the dropdown menu to select a label for the point.\n"
                    "If you want to move or replace a point, just click on \n"
                    "a new spot and select the label again.")
        instructions_text.setText(instructions_label)
        instructions_layout.addWidget(instructions_text)
        instructions = QtWidgets.QWidget()
        instructions.setLayout(instructions_layout)
        print(instructions.sizeHint())
        print(sonar_col.sizeHint())

        # main_layout = QtWidgets.QHBoxLayout()
        self.layout.addWidget(instructions, stretch=2)
        self.layout.addWidget(sonar_col, stretch=8)
        print("main layout created")
        # label_sonar_splitter = QtWidgets.QSplitter()
        # label_sonar_splitter.addWidget(instructions)
        # label_sonar_splitter.addWidget(sonar_col)
        # label_sonar_splitter.setSizes([150,750])  
        # label_sonar_splitter.setStretchFactor(0, 1)
        # label_sonar_splitter.setStretchFactor(1, 10)

        # # Finally, add all columns to the main window!
        # label_sonar_widget = QtWidgets.QWidget()
        # label_sonar_widget.setLayout(raw_sonar_col)
        #self.layout.addWidget(label_sonar_splitter)
        #self.setLayout(main_layout)

    def handle_sonar_click(self, event):
        """
        Store the location of the previous click in order to determine
        whether the release is likely to have belonged to a pan/zoom and should be ignored
        """
        self.click_event = event

    def handle_sonar_release(self, event):
        """
        If this mouse release did not correspond to a drag, it's probably meant
        to label the image (rather than resize), so pop up dialog input a label.
        """
        if (
            abs(self.click_event.x - event.x) > 2
            or abs(self.click_event.y - event.y) > 2
        ):
            return
        dialog = EnterPointDialog(lambda x, event=event: self.add_point(event, x))
        dialog.exec_()

    def handle_save_button(self):
        with open(self.json_filename, 'w') as json_file:
            json.dump(self.cropping_params, json_file, indent=4)
        print(f"Sonar cropping parameters have been saved to {self.json_filename}")
        self.close()

    def add_point(self, event, label):
        """
        User selection of points
        """
        self.arc_points[label] = (event.xdata, event.ydata)
        self.plot_sonar_im()
        print(f"{label} point added", event.xdata, ", ", event.ydata)

    def plot_sonar_im(self, keep_limits=True):
        xlim = self.sonar_image_ax.get_xlim()
        ylim = self.sonar_image_ax.get_ylim()
        self.sonar_image_ax.cla()

        image = self.raw_sonar_im

        if len(self.arc_points) == 3:
            # all three points have been selscted
            self.origin_point = self.arc_points["Origin"]
            self.left_point = self.arc_points["Left"]
            self.right_point = self.arc_points["Right"]
            image = self.mask_arc(image, self.origin_point, self.left_point, self.right_point)
            
        self.sonar_image_ax.imshow(image,             
            cmap="gray",
            aspect="equal",
            interpolation="none",)
        
        for label, coord in self.arc_points.items():
            x, y = coord
            self.sonar_image_ax.plot(
                x, y, scalex=False, marker=".", ms=8, c="red", fillstyle="none"
            )
            self.sonar_image_ax.text(x+1, y+2, label, c="red", fontsize=8)
        
        if keep_limits:
            self.sonar_image_ax.set_xlim(xlim)
            self.sonar_image_ax.set_ylim(ylim)
        self.sonar_image_canvas.draw()

    def mask_arc(self, image, origin, left, right):
        
        """
        Isolate the sonar display from images saved by oculus software
        """
        #sonar_im = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2GRAY)
        ox, oy = origin[0], origin[1]
        lx, ly = left[0], left[1]
        rx, ry = right[0], right[1]
        #radius is the average distance from left/right point to the origin
        radius = round(np.mean([np.sqrt((lx-ox)**2+(ly-oy)**2), np.sqrt((rx-ox)**2+(ry-oy)**2),]))
        self.cropping_params["center"] = centerpoint
        self.cropping_params["radius"] = radius
        self.cropping_params["crop_left"] = round(lx)
        self.cropping_params["crop_right"] = round(rx)
        self.cropping_params["crop_top"] = round(oy - radius)
        self.cropping_params["crop_bottom"] = round(oy)
        centerpoint = (round(ox-lx), radius)
        angle_start = np.rad2deg(np.atan((ox-lx)/(ly-oy)))
        angle_end = np.rad2deg(np.atan((ox-rx)/(ry-oy)))
        self.cropping_params["angle_start"] = angle_start
        self.cropping_params["angle_end"] = angle_end
        
        mask = np.zeros(image.shape[:2], dtype="uint8")
        cv2.ellipse(mask, centerpoint, (radius, radius), 0.0, angle_start+270, angle_end+270, (255), -1)
        masked = cv2.bitwise_and(image, image, mask=mask)
        return cv2.addWeighted(image, 0.25, masked, 0.75, 0)
    
if __name__ == "__main__":
    rootdir = "C:/Users/corri/OneDrive/Documents/SonarExperimentData/07-23-2025"
    rootdir += "/sonar"
    # first sonar image in sonar folder
    sonar_file = os.path.join(rootdir, os.listdir(rootdir)[0])
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow(sonar_file)
    window.showMaximized()  
    sys.exit(app.exec_())