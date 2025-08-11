# Author: Laura Lindzey
# Copyright 2020-2022 University of Washington Applied Physics Laboratory
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
# THE POSSIBILITY OF SUCH DAMAGE.

import cv2
import sys
import os
import json
import cycler
import pickle
import numpy as np
from PyQt5 import QtGui, QtWidgets, QtCore

import matplotlib
import matplotlib.figure
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT

import charuco_utils
import image_sonar_utils as isc
import data_analysis_tools as dtools

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

class Camera():
    def __init__(self, mtx, dst):
        self.K = mtx
        self.D = dst

class EnterPointDialog(QtWidgets.QDialog):
    def __init__(self, point_cb, removing=False):
        super(EnterPointDialog, self).__init__()
        # This callback will be called with the label that should be removed
        self.setWindowTitle("Select Label")
        self.point_cb = point_cb
        self.removing = removing
        self.setup_layout()

    def setup_layout(self):
        text_label = QtWidgets.QLabel("Label:")
        self.text_edit = QtWidgets.QLineEdit()
        ok_button = QtWidgets.QPushButton("OK")
        ok_button.clicked.connect(self.handle_ok)
        cancel_button = QtWidgets.QPushButton("Cancel")
        cancel_button.clicked.connect(self.handle_cancel)
        self.delete_all = False

        checkbox_row = QtWidgets.QHBoxLayout()
        if self.removing:
            self.delete_all_box = QtWidgets.QCheckBox(text="Delete all?")
            self.delete_all_box.stateChanged.connect(self.onStateChanged)
            checkbox_row.addWidget(self.delete_all_box)

        text_row = QtWidgets.QHBoxLayout()
        text_row.addWidget(text_label)
        text_row.addWidget(self.text_edit)

        button_row = QtWidgets.QHBoxLayout()
        button_row.addWidget(ok_button)
        button_row.addWidget(cancel_button)

        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(text_row)
        layout.addLayout(checkbox_row)
        layout.addLayout(button_row)
        self.setLayout(layout)

    def handle_cancel(self):
        # DO nothing, just close the window
        self.done(0)

    def handle_ok(self):
        self.point_cb(self.text_edit.text(), self.delete_all)
        self.done(0)

    def onStateChanged(self):
        if self.delete_all_box.isChecked():
            self.delete_all = True
        else:
            self.delete_all = False

class AnnotatedCanvas(QtWidgets.QWidget):
    """Widget that gives canvas a title and a `?` with mouseover help"""

    def __init__(self, title_text, help_text, canvas):
        super(AnnotatedCanvas, self).__init__()

        title_label = QtWidgets.QLabel(title_text)

        help_button = QtWidgets.QPushButton("?")
        help_button.setDisabled(False)
        help_button.setToolTip(help_text)
        # TODO: This button is too big; I'd like to make it tight to the "?".
        # Ideally, it would calculated the desired height normally, then set width
        # to that, but I'm in a time crunch so taking this shortcut.
        help_button.setStyleSheet("padding: 3px;")

        title_row = QtWidgets.QHBoxLayout()
        title_row.addStretch(1)
        title_row.addWidget(title_label)
        title_row.addWidget(help_button)
        title_row.addStretch(1)

        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(title_row)
        layout.addWidget(canvas)
        self.setLayout(layout)

class SensorWindow(QtWidgets.QMainWindow):
    def __init__(self, rootdir):
        super(SensorWindow, self).__init__()
        self.rootdir = rootdir
        self.outdir = f"{rootdir}/output"

        try:
            if not os.path.exists(self.outdir):
                os.makedirs(self.outdir)
        except Exception as ex:
            print("Could not create directory: {}".format(self.outdir))
            raise (ex)

        self._main = QtWidgets.QWidget()
        self.setCentralWidget(self._main)
        self.setup_layout()
        print("GUI setup complete")
        self.setup_data()
        self.initialize_camera()

        (self.aruco_dict, self.charuco_board, self.sonar_coords) = isc.init_charuco_sonar() 

        self.handle_next_button()

        # Only called once; the rest of the draw* are called on every refresh
        self.plot_charuco_target(self.charuco_board, self.sonar_coords)

    def setup_data(self):
        json_file_path = f'{self.rootdir}/inputparams.json'
        try:
            with open(json_file_path, 'r') as file: # Read the JSON file
                json_data = json.load(file)
        except Exception as ex:
            print("Could not open json file: {}".format(json_file_path))
            print("Creating new json file now")
            sonar_in = float(input("Range (as a float): "))
            wide_in = input("Using wide field of view? (True/False) ")
            wide_bool = True if wide_in.lower() == "true" else False
            t_in = input("External translation vector: ")
            t_data = [float(num) for num in list(t_in.split(", "))] 
            r_in = input("External rotation vector: ")
            r_data = [float(num) for num in list(r_in.split(", "))] 
            json_data = {"sonar_range": sonar_in, "sonar_wide": wide_bool, 
                    "ext_t": t_data, "ext_r": r_data}
            with open(json_file_path, 'w') as file:
                json.dump(json_data, file, indent=4)

        sonar_range = json_data["sonar_range"]
        sonar_wide = json_data["sonar_wide"]
        self.ext_rvec = tuple(json_data["ext_r"])
        self.ext_tvec = tuple(json_data["ext_t"])

        sonar_cropping_params = self.rootdir + "/sonar_cropping_params.json"
        self.sonar_params = isc.SonarInfo(sonar_range, sonar_wide, sonar_cropping_params)
        self.range_m = sonar_range
        self.polar_transform = isc.create_transform_map(self.sonar_params)
        print("Sonar parameters loaded successfully")

        self.paired_data = isc.SensorData(self.rootdir, self.sonar_params)
        (self.good_timestamps, self.skip_timestamps, self.sonar_labels, self.calibration_results, self.camera_poses) = self.load_state()
        # good_timestamps and skip_timestamps are sets of timestamps
        # sonar_labels is a dictionary mapping labels to user-selected sonar points

        # calibration_results is a dict mapping timestamps to calibration results and the
        # sonar and camera points used to calculate the calibrataion

        # camera_poses is a dict mapping timestamp to the charuco board's location in the
        # camera frame, where location is given as a (rvec, tvec) tuple.
        
    def initialize_camera(self, json_file_path = None):
        if json_file_path is not None:
            with open(json_file_path, 'r') as file: # Read the JSON file
                json_data = json.load(file)

            mtx = np.array(json_data['mtx'])
            dst = np.array(json_data['dist'])
        else:
            # from chessboard
            mtx = np.array([[1.02082611e+03, 0.00000000e+00, 7.69307527e+02],
                [0.00000000e+00, 1.02245381e+03, 2.90583592e+02],
                [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
            dst = np.array([[-3.76227154e-01,  1.94912143e-01, 
                             -2.04912328e-03,  7.63774994e-05, -5.57738640e-02]])
        self.camera_info = Camera(mtx, dst)

    def setup_layout(self):
        self.layout = QtWidgets.QVBoxLayout(self._main)

        ###################
        # Label row
        self.header_row = QtWidgets.QHBoxLayout()

        self.timestamp_label = QtWidgets.QLabel("Timestamp: None")
        self.good_label = QtWidgets.QLabel("Label: unknown")
        self.good_label.setAutoFillBackground(True)

        self.header_row.addStretch(5)
        self.header_row.addWidget(self.timestamp_label)
        self.header_row.addStretch(1)
        self.header_row.addWidget(self.good_label)
        self.header_row.addStretch(5)

        ###################
        # Matplotlib figures for displaying the Camera views

        # Figure showing the (idealized) charuco board + sonar targets
        self.target_fig = matplotlib.figure.Figure(figsize=(5, 5))
        self.target_ax = self.target_fig.add_axes([0, 0, 1, 1])
        self.target_ax.axis("off")
        self.target_artist = None
        self.target_canvas = FigureCanvas(self.target_fig)

        target_help = (
            "Charuco board used for the calibration. "
            "Sonar targets are installed at the center of "
            "the labeled black squares."
        )
        target_widget = AnnotatedCanvas("Target", target_help, self.target_canvas)

        # Figure showing the raw camera image
        self.raw_camera_fig = matplotlib.figure.Figure(figsize=(5, 3))
        self.raw_camera_ax = self.raw_camera_fig.add_axes([0, 0, 1, 1])
        self.raw_camera_ax.axis("off")
        self.raw_camera_artist = None
        self.raw_camera_canvas = FigureCanvas(self.raw_camera_fig)

        raw_camera_help = "Raw camera image from bagfile"
        raw_camera_widget = AnnotatedCanvas(
            "Raw Camera Image", raw_camera_help, self.raw_camera_canvas
        )

        # Figure showing aruco/charuco detections on the camera image
        self.charuco_annotated_camera_fig = matplotlib.figure.Figure(figsize=(5, 3))
        self.charuco_annotated_camera_ax = self.charuco_annotated_camera_fig.add_axes([0, 0, 1, 1])
        self.charuco_annotated_camera_ax.axis("off")
        self.charuco_annotated_camera_canvas = FigureCanvas(self.charuco_annotated_camera_fig)

        charuco_annotated_help = (
            "Detected aruco markers (green squares) and " "charuco corners (red dots)"
        )
        charuco_annotated_widget = AnnotatedCanvas(
            "Detected Aruco / Charuco",
            charuco_annotated_help,
            self.charuco_annotated_camera_canvas,
        )

        # Figure showing expected locations of the sonar targets on the
        # camera image, using the camera-derived board position
        self.camera_annotated_camera_fig = matplotlib.figure.Figure(figsize=(5, 3))
        self.camera_annotated_camera_ax = self.camera_annotated_camera_fig.add_axes([0, 0, 1, 1])
        self.camera_annotated_camera_ax.axis("off")
        self.camera_annotated_camera_canvas = FigureCanvas(self.camera_annotated_camera_fig)

        camera_camera_help = (
            "Red dots show the expected locations of sonar "
            "targets based on the detected charuco board"
        )
        camera_camera_widget = AnnotatedCanvas(
            "Charuco-Derived Locations",
            camera_camera_help,
            self.camera_annotated_camera_canvas,
        )

        # Figure showing expected locations of the sonar targets on the
        # camera image, using the sonar-derived board position
        self.camera_annotated_sonar_fig = matplotlib.figure.Figure(figsize=(5, 3))
        self.camera_annotated_sonar_ax = self.camera_annotated_sonar_fig.add_axes([0, 0, 1, 1])
        self.camera_annotated_sonar_ax.axis("off")
        self.camera_annotated_sonar_canvas = FigureCanvas(self.camera_annotated_sonar_fig)

        camera_sonar_help = (
            "Camera image annotated with arcs showing "
            "projection of labeled sonar points, \n"
            "using the current calibration.\n\n"
            "The displayed arc corresponds to a +/- 15 degree "
            "vertical field of view. \n"
            "The instrument's reported FOV is the angle at "
            "which the beam pattern drops by N dB. \n"
            "So, it is possible to see bright targets even "
            "if they are outside the nominal FOV.\n\n"
            "NB: The displayed image is NOT undistorted, "
            "but the sonar points have been projected onto "
            "the image using the CameraInfo."
        )
        camera_sonar_widget = AnnotatedCanvas(
            "Sonar-Derived Locations",
            camera_sonar_help,
            self.camera_annotated_sonar_canvas,
        )

        ###################
        # Matplotlib figures for displaying the sonar views

        # Figure showing the raw sonar image
        self.cartesian_sonar_fig = matplotlib.figure.Figure(figsize=(5, 3))
        self.cartesian_sonar_ax = self.cartesian_sonar_fig.add_axes(
            [0.05, 0.05, 0.9, 0.9]) 
        self.cartesian_sonar_canvas = FigureCanvas(self.cartesian_sonar_fig)
        cartesian_sonar_help = "Cartesian image of sonar arc after removing background"
        cartesian_sonar_widget = AnnotatedCanvas(
            "Cartesian Sonar Image", cartesian_sonar_help, self.cartesian_sonar_canvas
        )

        # Figure showing expected locations of the sonar targets on the
        # sonar image, using the charuco-derived board position
        self.sonar_annotated_camera_fig = matplotlib.figure.Figure(figsize=(5, 3))
        self.sonar_annotated_camera_ax = self.sonar_annotated_camera_fig.add_axes(
            [0.05, 0.05, 0.9, 0.9]
        )
        self.sonar_annotated_camera_canvas = FigureCanvas(
            self.sonar_annotated_camera_fig
        )
        sonar_camera_help = (
            "Expected locations of targets on sonar image, "
            "based on the charuco detection. \n\n"
            "Yellow circles use the input initial calibration. \n"
            "Red Xs use the currently-computed calibration."
        )
        sonar_camera_widget = AnnotatedCanvas(
            "Camera-Derived Locations",
            sonar_camera_help,
            self.sonar_annotated_camera_canvas,
        )

        transform_col = QtWidgets.QVBoxLayout()
        transform_col_label = QtWidgets.QLabel("COMPUTED TRANSFORMATIONS")
        self.charuco_pose_label = QtWidgets.QLabel("Camera -> Target: ")
        self.sonar_pose_label = QtWidgets.QLabel("Oculus -> Target: ")
        self.final_pose_label = QtWidgets.QLabel("Camera -> Sonar: ")

        transform_col.addWidget(transform_col_label)
        transform_col.addWidget(self.charuco_pose_label)
        transform_col.addWidget(self.sonar_pose_label)
        transform_col.addWidget(self.final_pose_label)

        ###################
        # Buttons for stepping through the data
        self.button_row = QtWidgets.QHBoxLayout()

        # For moving on to the next image pair, without labeling it "bad"
        self.next_button = QtWidgets.QPushButton("Next")
        self.next_button.setStyleSheet("padding: 3px;")
        self.next_button.clicked.connect(self.handle_next_button)
        self.next10_button = QtWidgets.QPushButton("Next10")
        self.next10_button.setStyleSheet("padding: 3px;")

        # Mark this image as "good" (saves to disk)
        self.good_button = QtWidgets.QPushButton("Mark Good")
        self.good_button.setStyleSheet("padding: 3px;")
        self.good_button.clicked.connect(self.handle_good_button)
        self.ungood_button = QtWidgets.QPushButton("UN Mark Good")
        self.ungood_button.setStyleSheet("padding: 3px;")
        self.ungood_button.clicked.connect(self.handle_unmark_good_button)

        # Updates the display to the next/prev data that had ben marked "good"
        self.next_good_button = QtWidgets.QPushButton("Next Good")
        self.next_good_button.setStyleSheet("padding: 3px;")
        self.next_good_button.clicked.connect(self.handle_next_good_button)
        self.prev_good_button = QtWidgets.QPushButton("Prev Good")
        self.prev_good_button.setStyleSheet("padding: 3px;")
        self.prev_good_button.clicked.connect(self.handle_prev_good_button)

        # Mark this image pair as unusable (don't show again)
        self.skip_button = QtWidgets.QPushButton("Skip")
        self.skip_button.setStyleSheet("padding: 3px;")
        self.skip_button.clicked.connect(self.handle_skip_button)

        # Remove labels
        self.remove_label_button = QtWidgets.QPushButton("Remove Label")
        self.remove_label_button.setStyleSheet("padding: 3px;")
        self.remove_label_button.clicked.connect(self.handle_remove_label_button)

        # Save all images as png
        # self.print_button = QtWidgets.QPushButton("Print")
        # self.print_button.setStyleSheet("padding: 3px;")
        # self.print_button.clicked.connect(self.handle_print_button)
        self.recalibrate_button = QtWidgets.QPushButton("Recalibrate")
        self.recalibrate_button.setStyleSheet("padding: 3px;")
        self.recalibrate_button.clicked.connect(self.handle_recalibrate_button)

        self.button_row.addWidget(self.skip_button)
        self.button_row.addWidget(self.prev_good_button)
        self.button_row.addWidget(self.good_button)
        self.button_row.addWidget(self.ungood_button)
        self.button_row.addWidget(self.next_good_button)
        self.button_row.addWidget(self.next_button)
        self.button_row.addWidget(self.remove_label_button)
        self.button_row.addWidget(self.recalibrate_button)

        ###################
        # Matplotlib figures for displaying the sonar views

        # Main figure for labeling the sonar image
        self.sonar_image_fig = matplotlib.figure.Figure(figsize=(5, 3))
        self.sonar_image_ax = self.sonar_image_fig.add_axes([0.025, 0.025, 0.95, 0.95])
        self.sonar_image_artist = None
        self.sonar_image_canvas = FigureCanvas(self.sonar_image_fig)
        self.sonar_image_canvas.mpl_connect(
            "button_press_event", self.handle_sonar_click
        )
        self.sonar_image_canvas.mpl_connect(
            "button_release_event", self.handle_sonar_release
        )

        # Create the central column used for labeling the sonar targets.
        # Nothing within this layout needs resizing, so just use a layout
        # and wrap it in a widget to add to the main splitter.
        raw_sonar_help_button = QtWidgets.QPushButton("?")
        raw_sonar_help_button.setDisabled(False)
        help_text = (
            "Main display for labeling targets in sonar data.\n\n"
            "* Left-click to select a pixel, and enter corresponding "
            "label from the Target image. (e.g. A1)\n"
            "* Use Next/Prev buttons to step through the bagfile (10 will skip ahead by 10 images) \n"
            "* Click Skip to indicate that this image should never be shown again\n"
            "* Click Remove Label and then enter the ID (e.g. D5) to remove an annotation"
        )
        raw_sonar_help_button.setToolTip(help_text)

        raw_sonar_col = QtWidgets.QVBoxLayout()
        raw_sonar_col_header = QtWidgets.QHBoxLayout()
        raw_sonar_col_label = QtWidgets.QLabel("Polar Sonar Image")
        self.raw_sonar_col_toolbar = NavigationToolbar(self.sonar_image_canvas, self)
        raw_sonar_col_header.addWidget(raw_sonar_col_label)
        raw_sonar_col_header.addWidget(self.raw_sonar_col_toolbar)
        raw_sonar_col_header.addWidget(raw_sonar_help_button)
        raw_sonar_col.addLayout(raw_sonar_col_header, stretch=0)
        raw_sonar_col.addWidget(self.sonar_image_canvas, stretch=1)
        raw_sonar_col.addLayout(self.button_row)

        # Create the left-most column.
        # Since we want a grid of annotated camera images, need to nest splitters
        aruco_splitter = QtWidgets.QSplitter()
        aruco_splitter.setOrientation(QtCore.Qt.Orientation.Horizontal)
        aruco_splitter.addWidget(raw_camera_widget)
        aruco_splitter.addWidget(charuco_annotated_widget)

        reproj_camera_splitter = QtWidgets.QSplitter()
        reproj_camera_splitter.setOrientation(QtCore.Qt.Orientation.Horizontal)
        reproj_camera_splitter.addWidget(camera_camera_widget)
        reproj_camera_splitter.addWidget(camera_sonar_widget)

        camera_splitter = QtWidgets.QSplitter()
        camera_splitter.setOrientation(QtCore.Qt.Orientation.Vertical)
        camera_splitter.addWidget(target_widget)
        camera_splitter.addWidget(aruco_splitter)
        camera_splitter.addWidget(reproj_camera_splitter)
        transform_widget = QtWidgets.QWidget()
        transform_widget.setLayout(transform_col)
        camera_splitter.addWidget(transform_widget)
        
        sonar_splitter = QtWidgets.QSplitter()
        sonar_splitter.setOrientation(QtCore.Qt.Orientation.Vertical)
        sonar_splitter.addWidget(cartesian_sonar_widget)
        sonar_splitter.addWidget(sonar_camera_widget)
        self.sonar_err_label = QtWidgets.QLabel("Reprojection Error: ")
        sonar_splitter.addWidget(self.sonar_err_label)

        # Finally, add all columns to the main window!
        main_splitter = QtWidgets.QSplitter()
        main_splitter.setOrientation(QtCore.Qt.Orientation.Horizontal)
        main_splitter.addWidget(camera_splitter)
        label_sonar_widget = QtWidgets.QWidget()
        label_sonar_widget.setLayout(raw_sonar_col)
        main_splitter.addWidget(label_sonar_widget)
        main_splitter.addWidget(sonar_splitter)
        main_splitter.setStretchFactor(0, 2)
        main_splitter.setStretchFactor(1, 3)
        main_splitter.setStretchFactor(2, 3)

        self.layout.addLayout(self.header_row)
        self.layout.addWidget(main_splitter, stretch=1)

    def plot_charuco_target(self, charuco_board, sonar_coords):
        """
        Draw charuco board on the target_ax.

        I like having data input be explicit, because it makes it easier to
        reason about ordering dependencies in the init steps.
        """
        square_pixels = 100
        ncols, nrows = charuco_board.getChessboardSize()
        xpixels = ncols * square_pixels
        ypixels = nrows * square_pixels
        board_img = charuco_board.generateImage((xpixels, ypixels))
        self.target_ax.imshow(
            board_img, cmap="gray", aspect="equal", interpolation="none"
        )

        scale = square_pixels / charuco_board.getSquareLength()
        for label, coord in sonar_coords.items():
            xx, yy = coord
            x = xx * scale
            # coordinate system conversion: board's origin is lower-left, <-I corrected this in init_charuco_sonar
            # but image's is upper-left.
            y = yy * scale
            self.target_ax.text(
                x,
                y,
                label,
                color="red",
                fontsize=10,
                horizontalalignment="center",
                verticalalignment="center",
            )
    
    def plot_raw_camera_data(self, camera_data):
        """
        Update the figure that shows the bare camera image.

        Since it was easy in this case, save the artist to help make
        updates faster.
        """
        if self.raw_camera_artist is None:
            self.raw_camera_artist = self.raw_camera_ax.imshow(camera_data, cmap="gray")
        else:
            self.raw_camera_artist.set_data(camera_data)
        self.raw_camera_canvas.draw()
    
    def plot_charuco_detections(self, camera_data, charuco_corners, charuco_ids):
        """
        Update the figure that shows detected aruco markers and charuco
        corners on top of the camera data.

        * aruco_corners -- np.ndarray (empty for no data)
        * charuco_corners -- None or array (None for no data)

        This function only accepts charuco_corners for now
        """
        self.charuco_annotated_camera_ax.cla()
        self.charuco_annotated_camera_ax.axis("off")
        self.charuco_annotated_camera_ax.imshow(camera_data, cmap="gray")

        # if len(aruco_corners) > 0:
        #     for corner, _corner_id in zip(aruco_corners, aruco_ids):
        #         pt1, pt2, pt3, pt4 = corner[0]
        #         xx, yy = zip(pt1, pt2, pt3, pt4, pt1)
        #         self.charuco_annotated_camera_ax.plot(xx, yy, "c")
        #         # Removed because too noisy in GUI -- only useful on saved images.
        #         # self.charuco_annotated_camera_ax.text(np.mean(xx), np.mean(yy),
        #         #                                      "{}".format(corner_id[0]), color='c')
        if charuco_corners is not None:
            for corner, _corner_id in zip(charuco_corners, charuco_ids):
                xx, yy = corner[0]
                circle = matplotlib.patches.Circle((xx, yy), 10, color="r")
                self.charuco_annotated_camera_ax.add_artist(circle)
                # Removed because too noisy in GUI -- only useful on saved images.
                # self.charuco_annotated_camera_ax.text(
                #         xx+10, yy+10, "{}".format(corner_id[0]), color='r')

        self.charuco_annotated_camera_canvas.draw()

    def plot_camera_targets_from_sonar(self, camera_data, camera_info, cs_rot, cs_trans):
        """
        Plot name: sonar-derived locations
        Update the figure that shows the position of the labeled sonar
        targets superimposed on the camera data, using the calculated
        camera/sonar transformation.

        * cs_rot -- rotation matrix from sonar to camera frames
        * cs_trans -- translation vector from sonar to camera frames
        """
        self.camera_annotated_sonar_ax.cla()
        self.camera_annotated_sonar_ax.axis("off")
        self.camera_annotated_sonar_ax.imshow(camera_data, cmap="gray")

        color_cycler = cycler.cycler(color=matplotlib.cm.plasma(np.linspace(0, 1, 10)))
        my_cycler = color_cycler()

        if self.current_timestamp in self.sonar_labels:
            points = self.sonar_labels[self.current_timestamp]
            for label, coord in points.items():
                azi_deg, rr = isc.pixel_to_polar(coord, self.sonar_params) 
                azi_rad = np.radians(azi_deg)
                elev_rads = np.arange(np.radians(-10.0), np.radians(10.0), np.radians(0.25)) 
                label_color = next(my_cycler)["color"]

                for elev_rad in elev_rads:
                    #Changed coordinate system to match camera
                    yy = -rr * np.sin(elev_rad)
                    zz = rr * np.cos(elev_rad) * np.cos(azi_rad)
                    xx = rr * np.cos(elev_rad) * np.sin(azi_rad)
                    sonar_point = np.reshape([xx, yy, zz], (3, 1))
                    
                    # cs_{trans, rot} give transformation from camera to sonar frame
                    # We need the opposite here ...
                    camera_coord = np.transpose(cs_rot) @ (sonar_point - cs_trans)

                    image_coord, _ = cv2.projectPoints(
                        camera_coord,
                        0 * cs_trans,
                        0 * cs_trans,
                        camera_info.K,
                        camera_info.D,
                        )
                    
                    ix, iy = image_coord[0][0]
                    ix, iy = int(ix), int(iy)
                    # don't plot points outside the FOV
                    nrows, ncols = camera_data.shape[:2]
                    if ix >= 0 and ix < ncols and iy >= 0 and iy < nrows:
                        self.camera_annotated_sonar_ax.plot(
                            ix, iy, marker=".", ms=1, color=label_color
                        )

                # Intentionally plot the label for the last point drawn
                self.camera_annotated_sonar_ax.text(ix, iy, label, color=label_color)

        self.camera_annotated_sonar_canvas.draw()

    def plot_camera_targets_from_camera(self, camera_data, rvec, tvec):
        """
        Plot name: Charuco-derived locations
        Update the figure that shows the inferred position of the sonar
        targets superimposed on the camera data.

        This serves to check that the transformation is correct from
        coordinates on the target to the image frame.
        """
        self.camera_annotated_camera_ax.cla()
        self.camera_annotated_camera_ax.axis("off")

        if rvec is not None and tvec is not None:

            cv2.drawFrameAxes(camera_data, self.camera_info.K, self.camera_info.D, rvec, tvec, .1, 3)
            self.camera_annotated_camera_ax.imshow(camera_data, cmap="gray")
            
            rot, _ = cv2.Rodrigues(rvec)
            pose_text = (
                "Camera -> Target: \n"
                "rvec = [{:.2f}, {:.2f}, {:.2f}] \n"
                "tvec = [{:.2f}, {:.2f}, {:.2f}] \n"
                "fwd = {:.2f}, right = {:.2f}, down = {:.2f}".format(
                    rvec[0][0],
                    rvec[1][0],
                    rvec[2][0],
                    tvec[0][0],
                    tvec[1][0],
                    tvec[2][0],
                    tvec[2][0],
                    tvec[0][0],
                    tvec[1][0],
                )
            )
            self.charuco_pose_label.setText(pose_text)
            for _label, (cx, cy) in self.sonar_coords.items():
                target_point = np.reshape([cx, cy, 0], (3, 1))
                # Option 1: object points relative to the board, using
                # rvec/tvec from estimatePose to transform into camera frame
                # image_coord, _  = cv2.projectPoints(object_points, rvec, tvec,
                #                                    camera_matrix, camera_info.D)
                # xx, yy = image_coord[0][0]
                # Option 2: use rvec tvec to transform points into camera frame,
                # project only corrects for distortion
                world_coord = tvec + rot @ target_point
                image_coord, _ = cv2.projectPoints(
                    world_coord, 0 * rvec, 0 * tvec, self.camera_info.K, self.camera_info.D
                )
                xx, yy = image_coord[0][0]
                
                # don't plot points outside the FOV
                nrows, ncols = camera_data.shape
                if xx >= 0 and xx < ncols and yy >= 0 and yy < nrows:
                    y = nrows - yy
                    self.camera_annotated_camera_ax.plot(int(xx), int(yy), "r.")
                    # Removed because too noisy in GUI -- only useful on saved images.
                    # self.camera_annotated_camera_ax.text(xx, yy, label, color='black')
        else:
            self.charuco_pose_label.setText("Camera -> Target: None")

        self.camera_annotated_camera_canvas.draw()

    def plot_sonar_image(self, data, keep_limits):
        """
        removed params extent, rvec, tvec
        Plot sonar image in rectangular plot.
        This is the axis that will be used for human annotations.
        """
        xlim = self.sonar_image_ax.get_xlim()
        ylim = self.sonar_image_ax.get_ylim()
        self.sonar_image_ax.cla()

        self.sonar_image_ax.imshow(
            data,
            cmap="inferno",
            aspect="auto",
            interpolation="none",
            origin="lower",
        )

        # Plot the human-provided labels
        if self.current_timestamp in self.sonar_labels:
            points = self.sonar_labels[self.current_timestamp]
            for label, coord in points.items():
                theta_deg, rr = coord
                self.sonar_image_ax.plot(
                    theta_deg, rr, scalex=False, marker="o", ms=8, c="white", fillstyle="none"
                )
                self.sonar_image_ax.text(theta_deg+1, rr+2, label, c="white", fontsize=8)

        if keep_limits:
            self.sonar_image_ax.set_xlim(xlim)
            self.sonar_image_ax.set_ylim(ylim)
        self.sonar_image_canvas.draw()

    def plot_cartesian_sonar_image(self, data):
        """
        
        """
        self.cartesian_sonar_ax.cla()
        self.cartesian_sonar_ax.imshow(self.sonar_image, cmap="inferno")
        self.cartesian_sonar_canvas.draw()

    def plot_sonar_targets_from_camera(self, data, rvec, tvec, agg_rvec, agg_tvec, cam_rvec, cam_tvec, err, agg_err):
        """
        Plot name: Camera-derived locations

        rvec, tvec are between sonar and target
        cam_rvec, cam_tvec are from camera to target
        """

        self.sonar_annotated_camera_ax.cla()

        self.sonar_annotated_camera_ax.imshow(data, cmap="inferno", 
            aspect="auto",
            interpolation="none",
            origin="lower",
        )

        target_data = np.transpose(
            np.array([[coord[0], coord[1], 0, label] for label, coord in self.sonar_coords.items()]))
        
        target_points = target_data[0:3, :].astype(np.float32)
        
        if rvec is not None:
            rot, _ = cv2.Rodrigues(rvec)
            label_text = (
                "Sonar -> Target: \n"
                "rvec = [{:.2f}, {:.2f}, {:.2f}]\n"
                "tvec = [{:.2f}, {:.2f}, {:.2f}]\n"
                "fwd = {:.2f}, right = {:.2f}, down = {:.2f}".format(
                    rvec[0][0],
                    rvec[1][0],
                    rvec[2][0],
                    tvec[0][0],
                    tvec[1][0],
                    tvec[2][0],
                    tvec[0][0],
                    tvec[1][0],
                    tvec[2][0],
                )
            )
            self.sonar_pose_label.setText(label_text)
            sonar_points = tvec + rot @ target_points
            sonar_coords = isc.polar_from_3d(sonar_points)
            plottable = isc.polar_to_pixel(sonar_coords, self.sonar_params)
            self.sonar_annotated_camera_ax.plot(
                plottable[0, :], plottable[1, :], "rx", fillstyle="none", label="current frame calibration"
            )
            # self.sonar_annotated_camera_ax.text(
            #     plottable[0, :], plottable[1, :], target_labels[:], color="r")
        if agg_rvec is not None:
            agg_rot, _ = cv2.Rodrigues(agg_rvec) 
            sonar_points = agg_tvec + agg_rot @ target_points
            sonar_coords = isc.polar_from_3d(sonar_points)
            plottable = isc.polar_to_pixel(sonar_coords, self.sonar_params)
            self.sonar_annotated_camera_ax.plot(
                plottable[0, :], plottable[1, :], "yx", fillstyle="none", label="aggregate calibration"
            )
            
            self.sonar_annotated_camera_ax.legend(loc="lower right")

        self.sonar_err_label.setText("Reprojection error: {:.2f}\nAggregate reprojection error: {:.2f}".format(err, agg_err))
        self.sonar_annotated_camera_canvas.draw()

    def handle_next_button(self, skip=False, reverse=False):
        try:
            self.current_timestamp, self.sonar_image, self.camera_data = self.paired_data.next(reverse=reverse)
            if self.current_timestamp in self.skip_timestamps:
                print("Skipping timestamp: {}".format(self.current_timestamp))
                self.handle_next_button(skip=True, reverse=reverse)
        except Exception as ex:
            # We expect to hit this at the end of the pairs
            print(ex)
            if skip:
                print("Returning to previous timestamp")
                self.handle_next_button(skip=True, reverse=True)

        self.update_plots(keep_limits=False, recalibrate=True)

    def handle_good_button(self):
        self.good_timestamps.add(self.current_timestamp)
        time_str = isc.timestamp_tostr(self.current_timestamp)
        # Using the camera's timestamp for both
        camera_filename = "{}/{}_camera.png".format(self.outdir, time_str)
        sonar_filename = "{}/{}_sonar.jpg".format(self.outdir, time_str)
        print("Trying to save data to {}".format(camera_filename))
        if not os.path.exists(camera_filename):
            cv2.imwrite(camera_filename, self.camera_data)
        if not os.path.exists(sonar_filename):
            cv2.imwrite(sonar_filename, self.sonar_image)

        self.save_state()
        self.update_good_label()
        print("Saved successfully")

    def handle_unmark_good_button(self):
        self.good_timestamps.discard(self.current_timestamp)
        self.save_state()
        self.update_good_label()

    def load_from_timestamp(self, timestamp):
        time_str = isc.timestamp_tostr(timestamp)
        camera_filename = "{}/{}_camera.png".format(self.outdir, time_str)
        sonar_filename = "{}/{}_sonar.jpg".format(self.outdir, time_str)

        try:
            self.camera_data = cv2.imread(camera_filename)
        except Exception as ex:
            print(
                "Could not load camera data for timestamp: {}, file: {}".format(
                    timestamp, camera_filename
                )
            )
            print(ex)
            return

        try:
            self.sonar_image = cv2.imread(sonar_filename, cv2.IMREAD_GRAYSCALE)
        except Exception as ex:
            print(
                "Could not load sonar data for timestamp: {}, file: {}".format(
                    timestamp, sonar_filename
                )
            )
            print(ex)
            return

    def handle_next_good_button(self):
        ts_array = np.array(list(self.good_timestamps))
        (future_idxs,) = np.where(ts_array > self.current_timestamp)
        if future_idxs.size == 0:
            print("No future data has been labeled good")
            return
        next_timestamp = np.min(ts_array[future_idxs])
        self.current_timestamp = next_timestamp
        
        self.load_from_timestamp(next_timestamp)
        self.update_plots(keep_limits=False, recalibrate=True)

    def handle_prev_good_button(self):
        ts_array = np.array(list(self.good_timestamps))
        (past_idxs,) = np.where(ts_array < self.current_timestamp)
        if past_idxs.size == 0:
            print("No previous data has been labeled good")
            return
        prev_timestamp = np.max(ts_array[past_idxs])
        self.load_from_timestamp(prev_timestamp)
        self.update_plots(keep_limits=False, recalibrate=True)

    def handle_skip_button(self):
        self.skip_timestamps.add(self.current_timestamp)
        self.save_state()
        self.handle_next_button()

    def handle_remove_label_button(self):
        dialog = EnterPointDialog(self.remove_point, True)
        time_str = isc.timestamp_tostr(self.current_timestamp)
        target_filename = "{}/{}_target.png".format(self.outdir, time_str)
        self.target_fig.savefig(target_filename)
        dialog.exec_()

    def handle_recalibrate_button(self):
        self.update_plots(keep_limits=False, recalibrate=True)

    def remove_point(self, text, remove_all):
        if self.current_timestamp in self.sonar_labels:
            if remove_all:
                del self.sonar_labels[self.current_timestamp]
            else:
                label = text.strip().upper()
                if label in self.sonar_labels[self.current_timestamp]:
                    self.sonar_labels[self.current_timestamp].pop(label, None)
                    if len(self.sonar_labels[self.current_timestamp]) == 0:
                        del self.sonar_labels[self.current_timestamp]
      
            self.save_state()
            self.update_plots(recalibrate=True) 
    
    def add_point(self, event, text, remove_all):
        """
        User selection of points
        """
        label = text.strip().upper()
        if label not in self.sonar_coords.keys():
            print("Cannot add point {}; not valid label".format(label))
            return

        if self.current_timestamp not in self.sonar_labels:
            self.sonar_labels[self.current_timestamp] = {}

        self.sonar_labels[self.current_timestamp][label] = (event.xdata, event.ydata)
        self.save_state()
        self.update_plots()
        print("point added", event.xdata, ", ", event.ydata)

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
        dialog = EnterPointDialog(lambda x, delete, event=event: self.add_point(event, x, False))
        dialog.exec_()
        
    def update_good_label(self):
        palette = QtGui.QPalette()
        if self.current_timestamp in self.good_timestamps:
            palette.setColor(QtGui.QPalette.Window, QtGui.QColor("green"))
            self.good_label.setText("label: Good")
        else:
            palette.setColor(QtGui.QPalette.Window, QtGui.QColor("gray"))
            self.good_label.setText("label: Unknown")
        self.good_label.setPalette(palette)

    def calibrate_sonar(self, camera_rvec, camera_tvec):
        """
        I've found that its more stable to direcly solve for the camera->sonar
        transformation than to try to solve for the sonar->board transform
        and then chain them. I think this is because we can start the former
        optimization with a much better prior, so it's likely to land in the
        correct local minimum.

        I've also found it more stable if we first solve for the translation,
        holding rotation constant, and then solve for the full pose.

        * sonar_points: locations of the detected points in the sonar frame
        * cs_rvec: rotation to align
        * cs_tvec: vector from camera frame to sonar frame
        """
        have_labels = self.current_timestamp in self.sonar_labels
        if not have_labels or camera_rvec is None:
            return None, None, None, None, None
        
        labeled_points = self.sonar_labels[self.current_timestamp]
        sonar_points, target_points = isc.get_sonar_target_correspondences(labeled_points, self.sonar_params)
        
        # Transform from target's coordinate frame to camera coordinate frame
        camera_rot, _ = cv2.Rodrigues(camera_rvec)
        camera_points = camera_tvec + camera_rot @ target_points
        
        #print("init err ", isc.calc_projection_error(camera_points, sonar_points, self.ext_rvec, self.ext_tvec, self.sonar_params, True))

        cs_err, cs_rvec, cs_tvec = isc.calibrate_sonar(
            sonar_points,
            camera_points,
            self.sonar_params,
            self.ext_rvec,
            self.ext_tvec,
        )

        cs_rot, _ = cv2.Rodrigues(cs_rvec)
        dx, dy, dz = cs_tvec[0][0], cs_tvec[1][0], cs_tvec[2][0]
        yaw, pitch, roll = cs_rvec[0][0], cs_rvec[1][0], cs_rvec[2][0]
        
        pose_text = ("Final cs_tvec: {:03f} {:03f} {:03f} \n"
        "Final cs_rvec: {:03f} {:03f} {:03f}".format(dx, dy, dz, yaw, pitch, roll))
        self.final_pose_label.setText(pose_text)

        # Calculate sonar-> target from camera->sonar and camera->target
        camera_rot, _ = cv2.Rodrigues(camera_rvec)
        sonar_tvec = cs_tvec + cs_rot @ camera_tvec
        sonar_rot = cs_rot @ camera_rot
        sonar_rvec, _ = cv2.Rodrigues(sonar_rot)

        vectors = (cs_err, cs_rvec, cs_tvec, sonar_rvec, sonar_tvec)
        self.calibration_results[self.current_timestamp] = (vectors, sonar_points, camera_points)
        self.save_state()

        return cs_rvec, cs_tvec, sonar_rvec, sonar_tvec, cs_err

    def multi_calibration(self, timestamps, current_time = None):
        if not self.calibration_results:
            return -1, None, None
        
        all_sonar_points = []
        all_camera_points = []
        current_son_pts, current_cam_pts = None, None
        for timestamp in timestamps:
            try:
                results, sonar_pts, camera_pts = self.calibration_results[timestamp]
                if timestamp == current_time:
                    current_son_pts = sonar_pts
                    current_cam_pts = camera_pts
                # sonar_pts is 2xN numpy array
                # camera_pts is a 3xN numpy array
                all_sonar_points.append(sonar_pts)
                all_camera_points.append(camera_pts)
            except IndexError("One or more timestamps do not exist. Cancelling operation"):
                return
            
        concat_sonar = np.concat(all_sonar_points, axis=1)
        concat_camera = np.concat(all_camera_points, axis=1)

        agg_cs_err, agg_cs_rvec, agg_cs_tvec = isc.calibrate_sonar(
            concat_sonar,
            concat_camera,
            self.sonar_params,
            self.ext_rvec,
            self.ext_tvec,
        )

        if current_cam_pts is not None and current_son_pts is not None:
            current_err = isc.calc_projection_error(current_cam_pts, current_son_pts, agg_cs_rvec, agg_cs_tvec, self.sonar_params)
            current_err/=(current_son_pts.shape[1])
            return current_err, agg_cs_rvec, agg_cs_tvec
        else:
            return agg_cs_err, agg_cs_rvec, agg_cs_tvec

    
    def update_plots(self, keep_limits=True, recalibrate=False):
        self.timestamp_label.setText(f"Sonar timestamp: {self.current_timestamp}")
        self.update_good_label()

        charucoCorners, charucoIds, camera_tvec, camera_rvec = charuco_utils.detect_charuco_board(
                        self.charuco_board, self.camera_data, self.camera_info.K, self.camera_info.D
                        )
        
        if camera_rvec is not None:
            self.camera_poses[self.current_timestamp] = (camera_rvec, camera_tvec)
        else:
            print("No charuco detections. Automatically skipping")
            self.handle_skip_button()

        ######################
        # Update the figures
        camera_gray = cv2.cvtColor(self.camera_data.copy(), cv2.COLOR_RGB2GRAY)
        self.plot_raw_camera_data(camera_gray) #TODO: display color image with true colors
        self.plot_charuco_detections(camera_gray, charucoCorners, charucoIds)
        self.plot_camera_targets_from_camera(camera_gray, camera_rvec, camera_tvec)

        sonar_matrix = cv2.remap(self.sonar_image, *self.polar_transform, cv2.INTER_LINEAR)
        sonar_matrix = cv2.flip(sonar_matrix, 0)
        
        cc = self.calibrate_sonar(camera_rvec, camera_tvec)
        cs_rvec, cs_tvec, sonar_rvec, sonar_tvec, cs_err = cc
        if recalibrate:
            self.agg_err, self.agg_rvec, self.agg_tvec = self.multi_calibration(
                self.calibration_results.keys(), current_time=self.current_timestamp) #all timestamps
                
        # Code for saving data to plot number of images vs calibration accuracy:
        # combos = dtools.generate_calibration_groups(list(self.calibration_results.keys()))
        # my_data = []
        # for group in combos:
        #     err, r, t = self.multi_calibration(group)
        #     rlin = [r[0][0], r[1][0], r[2][0]]
        #     tlin = [t[0][0], t[1][0], t[2][0]]
        #     my_data.append([len(group), err, *rlin, *tlin])
        # #print(my_data)
        # my_data = np.array(my_data)
        # np.savetxt("data5.csv", my_data, 
        #         delimiter = ",")    
        # print("data5 saved")
        
        if self.agg_rvec is None:
            agg_cs_rot, agg_son_tvec, agg_son_rot, agg_son_rvec = None, None, None, None
        else:
            agg_cs_rot, _ = cv2.Rodrigues(self.agg_rvec)
            camera_rot, _ = cv2.Rodrigues(camera_rvec)
            agg_son_tvec = self.agg_tvec + agg_cs_rot @ camera_tvec
            agg_son_rot = agg_cs_rot @ camera_rot
            agg_son_rvec, _ = cv2.Rodrigues(agg_son_rot)
        print("overall calibration value\nRvec: ", self.agg_rvec, "Tvec: \n", self.agg_tvec)

        self.plot_sonar_image(sonar_matrix, keep_limits=keep_limits)
        self.plot_cartesian_sonar_image(sonar_matrix)

        if sonar_rvec is None:
            cs_err = -1.0

        if cs_rvec is None:
            cs_rotation = None
        else:
            cs_rotation, _ = cv2.Rodrigues(cs_rvec)

        #agg_err = isc.calc_projection_error(camera_points, sonar_points, agg_cs_rvec, agg_cs_tvec, self.sonar_params)
        self.plot_camera_targets_from_sonar(camera_gray, self.camera_info, cs_rotation, cs_tvec)

        self.plot_sonar_targets_from_camera(
            sonar_matrix, sonar_rvec, sonar_tvec, 
            agg_son_rvec, agg_son_tvec, 
            camera_rvec, camera_tvec, cs_err, self.agg_err
        )
        self.save_state()

    def load_state(self):
        labels_filename = "{}/calibration_labels.pkl".format(self.outdir)
        calibration_filename = "{}/calibration_data.pkl".format(self.outdir)
        poses_filename = "{}/camera_poses.pkl".format(self.outdir)
        if os.path.exists(labels_filename):
            try:
                with open(labels_filename, "rb") as fp:
                    labels = pickle.load(fp)
                with open(calibration_filename, "rb") as fp:
                    points = pickle.load(fp)
                with open(poses_filename, "rb") as fp:
                    poses = pickle.load(fp)    
                
                return labels["good_timestamps"], labels["skip_timestamps"], labels["sonar_labels"], points, poses

            except Exception as ex:
                print("Could not load labels from: {}".format(labels_filename))
                # Intentionally not trying to recover, since we don't wan't
                # to accidentally overwrite a file that the human has
                # already started!
                raise (ex)
        else:
            return set(), set(), {}, {}, {}

    def save_state(self):
        """
        Save all of the human-generated metadata.
        For now, this is the skipped/skipped timestamps and the labeled points.

        Other possibly-relevant state data:
        * Generate index of bagfile to enable stepping through backwards
        * Actually save the charuco board locations so the pickle files are
          sufficient for running calibration independently.
        """
        filename = "{}/calibration_labels.pkl".format(self.outdir)
        with open(filename, "wb") as fp:
            pickle.dump(
                {
                    "good_timestamps": self.good_timestamps,
                    "skip_timestamps": self.skip_timestamps,
                    "sonar_labels": self.sonar_labels,
                },
                fp,
            )

        cam_filename = "{}/camera_poses.pkl".format(self.outdir)
        with open(cam_filename, "wb") as fp:
            pickle.dump(self.camera_poses, fp)

        calibration_filename = "{}/calibration_data.pkl".format(self.outdir)
        with open(calibration_filename, "wb") as fp:
            pickle.dump(self.calibration_results, fp)

if __name__ == "__main__":
    description = "GUI for manually labeling imaging sonar calibration data"
    # Name of folder containing sonar and camera images. Data will be saved here
    rootdir = "C:/Users/corri/OneDrive/Documents/SonarExperimentData/07-23-2025"

    app = QtWidgets.QApplication(sys.argv)
    window = SensorWindow(rootdir)
    window.showMaximized()         
    sys.exit(app.exec_())