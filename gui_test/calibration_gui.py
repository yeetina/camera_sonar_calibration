import cv2
import sys
import os
import json
import cycler
import pickle
import tensorflow as tf
import numpy as np
from PyQt5 import QtGui, QtWidgets, QtCore

import matplotlib
import matplotlib.figure
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT

import charuco_utils
import image_sonar_utils as isc

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
    def __init__(self, point_cb):
        super(EnterPointDialog, self).__init__()
        # This callback will be called with the label that should be removed
        self.setWindowTitle("Select Label")
        self.point_cb = point_cb
        self.setup_layout()

    def setup_layout(self):
        text_label = QtWidgets.QLabel("Label:")
        self.text_edit = QtWidgets.QLineEdit()
        ok_button = QtWidgets.QPushButton("OK")
        ok_button.clicked.connect(self.handle_ok)
        cancel_button = QtWidgets.QPushButton("Cancel")
        cancel_button.clicked.connect(self.handle_cancel)

        text_row = QtWidgets.QHBoxLayout()
        text_row.addWidget(text_label)
        text_row.addWidget(self.text_edit)

        button_row = QtWidgets.QHBoxLayout()
        button_row.addWidget(ok_button)
        button_row.addWidget(cancel_button)

        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(text_row)
        layout.addLayout(button_row)
        self.setLayout(layout)

    def handle_cancel(self):
        # DO nothing, just close the window
        self.done(0)

    def handle_ok(self):
        self.point_cb(self.text_edit.text())
        self.done(0)

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
    def __init__(self, outdir):
        super(SensorWindow, self).__init__()
        self.outdir = outdir

        self._main = QtWidgets.QWidget()
        self.setCentralWidget(self._main)

        self.setup_layout()
        self.setup_data()

        phone_mtx = np.array([[942.6847133455641, 0.0, 229.60291730101702],
                            [0.0, 4077.110071945852, 1408.7761995830372],
                            [0.0, 0.0, 1.0]])
        phone_dst = np.array([[-1.1701606614778872,
                    2.27606959770457,
                    -0.016377618529930534,
                    0.10323614752890109,
                    -2.4914094424655855]])
        self.initialize_camera()

        (self.aruco_dict, self.charuco_board, self.sonar_coords) = isc.init_charuco_sonar() #can change length params here

        # Calls that assume more than one of the setup_* have been called
        self.handle_next_button()

        # Only called once; the rest of the draw* are called on every refresh
        self.plot_charuco_target(self.charuco_board, self.sonar_coords)

    def setup_data(self):
        self.sonar_params = isc.SonarInfo(1.6, wide=True)
        main_folder = "C:/Users/corri/OneDrive/Documents/SonarExperimentData/testpairs"

        self.paired_data = isc.SensorData(main_folder, self.sonar_params)
        self.current_timestamp = ""
        self.range_m = self.sonar_params.range
        self.polar_transform = isc.create_transform_map(self.sonar_params)

        (self.good_timestamps, self.skip_timestamps, self.sonar_labels) = self.load_state()
        
        # Dict mapping SONAR timestamp to the charuco board's location in the
        # camera frame, where location is given as a (rvec, tvec) tuple.
        self.camera_poses = {}
        # TODO: camera_poses
        
        # TODO: Make board properties into parameters
        
        
    def initialize_camera(self, json_file_path='./phonecalibration.json'):
        with open(json_file_path, 'r') as file: # Read the JSON file
            json_data = json.load(file)

        mtx = np.array(json_data['mtx'])
        dst = np.array(json_data['dist'])

        self.camera_info = Camera(mtx, dst)

    def setup_layout(self):
        #rospy.logwarn("SensorWindow.setup_layout")
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
            "Detected aruco / charuco",
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
            "Charuco-derived locations",
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
            "Sonar-derived locations",
            camera_sonar_help,
            self.camera_annotated_sonar_canvas,
        )

        ###################
        # Matplotlib figures for displaying the sonar views

        # Polar plot of sonar data, with labels
        self.polar_sonar_fig = matplotlib.figure.Figure(figsize=(5, 3))
        self.polar_sonar_ax = self.polar_sonar_fig.add_axes(
            [0.05, 0.05, 0.9, 0.9]) #polar=True
        self.polar_sonar_canvas = FigureCanvas(self.polar_sonar_fig)
        polar_sonar_help = "Polar plot of sonar image"
        polar_sonar_widget = AnnotatedCanvas(
            "Polar Sonar Image", polar_sonar_help, self.polar_sonar_canvas
        )

        # Figure showing expected locations of the sonar targets on the
        # sonar image, using the charuco-derived board position
        self.sonar_annotated_camera_fig = matplotlib.figure.Figure(figsize=(5, 3))
        self.sonar_annotated_camera_ax = self.sonar_annotated_camera_fig.add_axes(
            [0, 0, 1, 1]
        )
        self.sonar_annotated_camera_ax.axis("off")
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
            "Camera-derived locations",
            sonar_camera_help,
            self.sonar_annotated_camera_canvas,
        )

        transform_col = QtWidgets.QVBoxLayout()
        transform_col_label = QtWidgets.QLabel("COMPUTED TRANSFORMATIONS")
        self.charuco_pose_label = QtWidgets.QLabel("Camera -> Target: ")
        self.sonar_pose_label = QtWidgets.QLabel("Oculus -> Target: ")

        transform_col.addWidget(transform_col_label)
        transform_col.addWidget(self.charuco_pose_label)
        transform_col.addWidget(self.sonar_pose_label)

        ###################
        # Buttons for stepping through the data
        self.button_row = QtWidgets.QHBoxLayout()

        # For moving on to the next image pair, without labeling it "bad"
        self.next_button = QtWidgets.QPushButton("Next")
        self.next_button.setStyleSheet("padding: 3px;")
        self.next_button.clicked.connect(self.handle_next_button)
        # Skip 10 images
        self.next10_button = QtWidgets.QPushButton("Next10")
        self.next10_button.setStyleSheet("padding: 3px;")
        #self.next10_button.clicked.connect(self.handle_next10_button)

        # Mark this image as "good" (saves to disk)
        self.good_button = QtWidgets.QPushButton("Mark Good")
        self.good_button.setStyleSheet("padding: 3px;")
        self.good_button.clicked.connect(self.handle_good_button)
        self.ungood_button = QtWidgets.QPushButton("UN Mark Good")
        self.ungood_button.setStyleSheet("padding: 3px;")
        self.ungood_button.clicked.connect(self.handle_unmark_good_button)

        # Updates the display to the next/prev data that had ben marked "good"
        # NB: there are two very confusing ways of stepping through the data.
        # 1) Sequentially, using next/skip/next10
        # 2) back/fwd through saved frames, using good/next/prev
        # clicking "next" will always jump you back to whatever comes out of
        # the bag file next; next/prev good buttons are always relative to
        # the current timestamp.
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
        self.print_button = QtWidgets.QPushButton("Print")
        self.print_button.setStyleSheet("padding: 3px;")
        #self.print_button.clicked.connect(self.handle_print_button)

        self.button_row.addWidget(self.print_button)
        self.button_row.addWidget(self.prev_good_button)
        self.button_row.addWidget(self.good_button)
        self.button_row.addWidget(self.ungood_button)
        self.button_row.addWidget(self.next_good_button)
        self.button_row.addWidget(self.next_button)
        self.button_row.addWidget(self.next10_button)
        self.button_row.addWidget(self.remove_label_button)
        self.button_row.addWidget(self.skip_button)

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
        raw_sonar_col_label = QtWidgets.QLabel("Raw Sonar Image")
        self.raw_sonar_col_toolbar = NavigationToolbar(self.sonar_image_canvas, self)
        raw_sonar_col_header.addWidget(raw_sonar_col_label)
        raw_sonar_col_header.addWidget(self.raw_sonar_col_toolbar)
        raw_sonar_col_header.addWidget(raw_sonar_help_button)
        raw_sonar_col.addLayout(raw_sonar_col_header, stretch=0)
        raw_sonar_col.addWidget(self.sonar_image_canvas, stretch=1)
        raw_sonar_col.addLayout(self.button_row)

        # Create the left-most column.
        # Since we want a agrid of annotated camera images, need to nest splitters
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
        sonar_splitter.addWidget(polar_sonar_widget)
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

        NB: It's not my fault that these have different forms for "no data" ...
            that can be traced back to the function signatures of cv2.aruco's
            detectMarkers vs. interpolateCornersCharuco
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
        Update the figure that shows the position of the labeled sonar
        targets superimposed on the camera data, using the calculated
        camera/sonar transformation.

        Labeled as sonar-derived location I think

        * cs_rot -- rotation matrix from sonar to camera frames
        * cs_trans -- translation vector from sonar to camera frames
        """
        self.camera_annotated_sonar_ax.cla()
        self.camera_annotated_sonar_ax.axis("off")
        self.camera_annotated_sonar_ax.imshow(camera_data, cmap="gray")

        # color_cycler = cycler.cycler(color=matplotlib.cm.inferno(np.linspace(0,1,10)))
        color_cycler = cycler.cycler(color=matplotlib.cm.plasma(np.linspace(0, 1, 10)))
        my_cycler = color_cycler()

        if self.current_timestamp in self.sonar_labels:
            points = self.sonar_labels[self.current_timestamp]
            for label, coord in points.items():
                azi_deg, rr = isc.pixel_to_polar(coord, self.sonar_params) #convert from pixels to real units
                azi_rad = np.radians(azi_deg)
                elev_rads = np.arange(np.radians(-10.0), np.radians(10.0), np.radians(0.25)) #change back to .25
                label_color = next(my_cycler)["color"]

                #print("theta, rr, color: ", azi_deg, rr, label_color)
                for elev_rad in elev_rads:
                    # X is forward
                    zz = rr * np.sin(elev_rad)
                    xx = rr * np.cos(elev_rad) * np.cos(azi_rad)
                    yy = rr * np.cos(elev_rad) * np.sin(azi_rad)
                    sonar_point = np.reshape([xx, yy, zz], (3, 1))
                    
                    # cs_{trans, rot} give transformation from camera to sonar frame
                    # We need the opposite here ...
                    camera_coord = np.transpose(cs_rot) @ (sonar_point - cs_trans)
                    #print(f'sonar: {sonar_point},\ncamera: {camera_coord}') #these seem reasonable
                    
                    image_coord, _ = cv2.projectPoints(
                        camera_coord,
                        0 * cs_trans,
                        0 * cs_trans,
                        camera_info.K,
                        camera_info.D,
                        )
                    
                    ix, iy = image_coord[0][0]
                    ix, iy = int(ix), int(iy)
                    #print("camera sonar", camera_coord, ix, iy, "r ", rr)
                    # don't plot points outside the FOV
                    nrows, ncols = camera_data.shape[:2]
                    if ix >= 0 and ix < ncols and iy >= 0 and iy < nrows:
                        self.camera_annotated_sonar_ax.plot(
                            ix, iy, marker=".", ms=2, color=label_color
                        )

                # Intentionally plot the label for the last point drawn
                print("last point", ix, iy)
                self.camera_annotated_sonar_ax.text(ix, iy, label, color=label_color)

        self.camera_annotated_sonar_canvas.draw()

    def plot_camera_targets_from_camera(self, camera_data, camera_info, rvec, tvec):
        """
        Update the figure that shows the inferred position of the sonar
        targets superimposed on the camera data.

        This serves to check that I have the transformations correct from
        coordinates on the target to the image frame.
        """
        self.camera_annotated_camera_ax.cla()
        self.camera_annotated_camera_ax.axis("off")
        self.camera_annotated_camera_ax.imshow(camera_data, cmap="gray")
        print("cam tvec", tvec)

        if rvec is not None:
            rot, _ = cv2.Rodrigues(rvec)
            #camera_matrix = np.reshape(camera_info.K, (3, 3))
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
                    world_coord, 0 * rvec, 0 * tvec, camera_info.K, camera_info.D
                )
                xx, yy = image_coord[0][0]
                
                # don't plot points outside the FOV
                nrows, ncols = camera_data.shape
                #print(nrows, ncols, xx, yy)
                if xx >= 0 and xx < ncols and yy >= 0 and yy < nrows:
                    y = nrows - yy
                    self.camera_annotated_camera_ax.plot(int(xx), int(yy), "r.")
                    # Removed because too noisy in GUI -- only useful on saved images.
                    # self.camera_annotated_camera_ax.text(xx, yy, label, color='black')
        else:
            self.charuco_pose_label.setText("Camera -> Target: None")

        self.camera_annotated_camera_canvas.draw()

    def plot_sonar_image(self, data, rvec, tvec, extent, keep_limits):
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
        #removed extent, vmin=0, vmax=128,

        # Plot the human-provided labels
        if self.current_timestamp in self.sonar_labels:
            points = self.sonar_labels[self.current_timestamp]
            for label, coord in points.items():
                theta_deg, rr = coord
                self.sonar_image_ax.plot(
                    theta_deg, rr, scalex=False, marker="o", ms=8, c="white", fillstyle="none"
                )
                self.sonar_image_ax.text(theta_deg+1, rr+2, label, c="white", fontsize=8)

        # Plot the projected location of the targets
        if rvec is not None:
            rot, _ = cv2.Rodrigues(rvec)
            # Project points in the charuco board's coordinate frame
            for label, (cx, cy) in self.sonar_coords.items():
                target_point = np.reshape([cx, cy, 0], (3, 1))
                pt_sf = tvec + rot @ target_point  # sonar instrument frame
                pt_if = isc.image_from_sonar3d(pt_sf)  # sonar image frame
                th_deg = np.degrees(pt_if[0])
                rr = pt_if[1]
                min_th, max_th, min_range, max_range = extent
                theta_in_range = (th_deg >= min_th) and (th_deg <= max_th)
                range_in_range = (rr >= min_range) and (rr <= max_range)
                if theta_in_range and range_in_range:
                    self.sonar_image_ax.plot(th_deg, rr, "g.", markersize=4)
                    self.sonar_image_ax.text(th_deg, rr, label, c="white", fontsize=6)

        if keep_limits:
            self.sonar_image_ax.set_xlim(xlim)
            self.sonar_image_ax.set_ylim(ylim)
        self.sonar_image_canvas.draw()

    def plot_polar_sonar_image(self, data):
        """
        Might improve this later, but not using it now

        Plot sonar image in polar plot, including the annotated points.

        Degrees/Radians is annoyingly inconsistent here.
        Min/max are specified in degrees.
        Actually plotting data, angles need to be in radians.
        """
        self.polar_sonar_ax.cla()
        self.polar_sonar_ax.imshow(self.sonar_image, cmap="inferno")
        self.polar_sonar_canvas.draw()

    def plot_sonar_targets_from_camera(self, data, rvec, tvec, cam_rvec, cam_tvec, err):
        """
        rvec, tvec are between sonar and target
        cam_rvec, cam_tvec are from camera to target
        """

        self.sonar_annotated_camera_ax.cla()

        # min_th = self.sonar_image.azimuth_angles[0]
        # max_th = self.sonar_image.azimuth_angles[-1]
        # min_range = self.sonar_image.ranges[0]
        # max_range = self.sonar_image.ranges[-1]
        # extent_radians = [min_th, max_th, min_range, max_range]

        self.sonar_annotated_camera_ax.imshow(data, cmap="inferno", 
            aspect="auto",
            interpolation="none",
            origin="lower",
        )

        target_data = np.transpose(
            np.array([[coord[0], coord[1], 0, label] for label, coord in self.sonar_coords.items()]))
        target_points = target_data[0:3, :].astype(np.float32)
        target_labels = target_data[3, :]
        

        # target_points = np.transpose(
        #     np.array([[coord[0], coord[1], 0] for coord in self.sonar_coords.values()])
        # ) #these coordinates are in target frame
        # print(target_p, "labels", target_points)
        # print("rvec and tvec", rvec, tvec)
        if rvec is not None:
            print("Using calibration data from this frame")
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
            #print("points", sonar_points)
            sonar_coords = isc.polar_from_3d(sonar_points)
            #print("polar from 3D", sonar_coords)
            plottable = isc.polar_to_pixel(sonar_coords, self.sonar_params)
            #print(plottable)
            self.sonar_annotated_camera_ax.plot(
                plottable[0, :], plottable[1, :], "rx", fillstyle="none"
            )
            # self.sonar_annotated_camera_ax.text(
            #     plottable[0, :], plottable[1, :], target_labels[:], color="r")
        elif cam_rvec is not None:
            print("Using calibration data from another frame")
            # TODO: Actually *always* show the aggregate calibration?
            #   (This requires actually calculating one from more than one frame...)
            # Use calibration value from another frame ... just to show that it generalizes.

            # Final ROS XYZ: 0.006632 -0.055447 0.075904
            # Final ROS YPR: -3.101167 -1.401133
            # [1.33891995, 1.32590053, 1.1214456 ]
            cam_rot, _ = cv2.Rodrigues(cam_rvec)
            init_rvec = np.reshape([1.33891995, 1.32590053, 1.1214456], (3, 1))
            init_rot, _ = cv2.Rodrigues(init_rvec)
            init_tvec = np.reshape([0.006632, -0.055447, 0.075904], (3, 1))
            init_sonar_points = init_tvec + init_rot @ (
                cam_tvec + cam_rot @ target_points
            )
            init_sonar_coords = isc.polar_from_3d(init_sonar_points)
            self.sonar_annotated_camera_ax.plot(
                init_sonar_coords[0, :], init_sonar_coords[1, :], "rx", fillstyle="none"
                #add text?
            )

        # This is an ugly set of magic numbers...
        # In order to plot the "as-initialized", use the camera pose + initial
        # camera->sonar transformation.
        if cam_rvec is not None:
            cam_rot, _ = cv2.Rodrigues(cam_rvec)
            init_rvec = np.reshape([1.2092, 1.2092, 1.2092], (3, 1))
            init_rot, _ = cv2.Rodrigues(init_rvec)

            # Plot prior that camera and sonar are aligned
            # init_tvec = np.reshape([0, 0, 0], (3,1))
            # init_sonar_points = init_tvec + init_rot @ (cam_tvec + cam_rot @ target_points)
            # init_sonar_coords = isc.image_from_sonar3d(init_sonar_points)
            # self.sonar_annotated_camera_ax.plot(init_sonar_coords[0,:],
            #                                    init_sonar_coords[1,:],
            #                                    'go', fillstyle='none')

            # This uses the values from the URDF [0, -0.06, 0.095]
            init_tvec = np.reshape([-.6, .15, -.07], (3, 1))
            init_sonar_points = init_tvec + init_rot @ (
                cam_tvec + cam_rot @ target_points
            )
            init_sonar_coords = isc.polar_from_3d(init_sonar_points)
            init_sonar_coords = isc.polar_to_pixel(init_sonar_coords, self.sonar_params)
            # self.sonar_annotated_camera_ax.plot(
            #     init_sonar_coords[0, :], init_sonar_coords[1, :], "yo", fillstyle="none"
            # )

            # self.sonar_annotated_camera_ax.set_xlim(extent_radians[0:2])
            # self.sonar_annotated_camera_ax.set_ylim(extent_radians[2:4])

        self.sonar_err_label.setText("Reprojection error: {:.2f}".format(err))
        self.sonar_annotated_camera_canvas.draw()

    def handle_next_button(self):
        try:
            # TODO: Our handling of camera info is sketchy: it is NOT being
            #   saved alongside the images. This is OK for now, since it's
            # fixed throughout the bagfiles, but should be revised
            # if/when I go back and change how we step through the data.
            self.current_pair, self.sonar_image, self.camera_data = self.paired_data.next()
            
            self.current_timestamp = self.current_pair.timestamp
            #self.sonar_image, self.camera_info, image_mono = next(self.data_gen)
            # while self.sonar_image.header.stamp.to_sec() in self.skip_timestamps:
            #     time_sec = self.sonar_image.header.stamp.to_sec()
            #     print("Skipping timestamp: {}".format(time_sec))
            #     self.sonar_image, _, image_mono = next(self.data_gen)
        except Exception as ex:
            # We expect to hit this at the end of the bagfile.
            print(ex)
        #self.current_timestamp = "14:45:02"
        #= self.sonar_image.header.stamp.to_sec()

        # NB: We don't every explicitly undistort the image for display;
        #     instead, any calls projecting aruco markers to/from the image
        #     use camera_info to do the conversion to image coordinates.
        #self.camera_data = _

        # This is duplicated in update_plots, but I want to automatically skip
        # frames that don't have sufficient charuco tags.
        # aruco_corners, aruco_ids, charuco_corners, charuco_ids, camera_rvec, camera_tvec = self._detect_charuco()
        # if camera_rvec is None:
        #     print("Automatically skipping.")
        #     self.handle_skip_button()
        # else:
        #     self.update_plots(False)
        self.update_plots(False)
#handle_next10_button
    def handle_good_button(self):
        self.good_timestamps.add(self.current_timestamp)
        time_str = isc.timestamp_tostr(self.current_timestamp)
        # NB: using the _camera's_ timestamp for both of 'em, for easier playback.
        camera_filename = "{}/{}_camera.jpg".format(self.outdir, time_str)
        sonar_filename = "{}/{}_sonar.jpg".format(self.outdir, time_str)
        print("Trying to save data to {}".format(camera_filename))
        if not os.path.exists(camera_filename):
            cv2.imwrite(camera_filename, self.camera_data)
        if not os.path.exists(sonar_filename):
            cv2.imwrite(sonar_filename, self.sonar_image)
            # with open(sonar_filename, "wb") as fp:
            #     # NB: Pickle does not play nicely with the rosbag API, since
            #     #     rosbag relies on duck typing. You can re-load the pickled
            #     #     file from the same process, but attempts to load it from a
            #     #     different one will always fail with an error like:
            #     #     > ModuleNotFoundError: No module named 'tmpwujkj9ik'
            #     # So, use rosmsg's serialization/deserialization to save images
            #     self.sonar_image.serialize(fp)

        self.save_state()
        self.update_good_label()
        print("Saved successfully")

    def handle_unmark_good_button(self):
        self.good_timestamps.discard(self.current_timestamp)
        self.save_state()
        self.update_good_label()

    def load_from_timestamp(self, timestamp):
        time_str = isc.timestamp_tostr(timestamp)
        camera_filename = "{}/{}_camera.jpg".format(self.outdir, time_str)
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
        #good_timestamps is a set
        ts_array = np.array(list(self.good_timestamps))
        (future_idxs,) = np.where(ts_array > self.current_timestamp)
        if future_idxs.size == 0:
            print("No future data has been labeled good")
            return
        next_timestamp = np.min(ts_array[future_idxs])
        
        self.load_from_timestamp(next_timestamp)
        self.update_plots(False)

    def handle_prev_good_button(self):
        ts_array = np.array(list(self.good_timestamps))
        (past_idxs,) = np.where(ts_array < self.current_timestamp)
        if past_idxs.size == 0:
            print("No previous data has been labeled good")
            return
        prev_timestamp = np.max(ts_array[past_idxs])
        self.load_from_timestamp(prev_timestamp)
        self.update_plots(False)

    def handle_skip_button(self):
        self.skip_timestamps.add(self.current_timestamp)
        self.save_state()
        self.handle_next_button()

#handle_print_button
    def handle_remove_label_button(self):
        dialog = EnterPointDialog(self.remove_point)
        time_str = isc.timestamp_tostr(self.current_timestamp)
        target_filename = "{}/{}_target.png".format(self.outdir, time_str)
        self.target_fig.savefig(target_filename)
        dialog.exec_()

    def remove_point(self, text):
        if self.current_timestamp in self.sonar_labels:
            label = text.strip().upper()
            # Technically, pop doesn't require checking if the dict contains
            # the key, but we only want to redraw if necessary.
            if label in self.sonar_labels[self.current_timestamp]:
                self.sonar_labels[self.current_timestamp].pop(label, None)
                if len(self.sonar_labels[self.current_timestamp]) == 0:
                    del self.sonar_labels[self.current_timestamp]
                self.save_state()
                self.update_plots(keep_limits=False)
    
    def add_point(self, event, text):
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
        self.update_plots(keep_limits=False)
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
        dialog = EnterPointDialog(lambda x, event=event: self.add_point(event, x))
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

    def get_plottable_sonar(self):
        # type: None -> (np.ndarray, List]
        """
        This doesn't really need to be a separate function
        Reshape the most recent sonar image into the right format to
        sent to plt.imshow, and determine the appropriate extent.

        * sonar_matrix: intensities
        * extent_degrees: [min_th, max_th, min_range, max_range]
        """
        # assert self.sonar_image.data_size == 1
        # # Convert from bytes in message field to numpy array
        # n_ranges = len(self.sonar_image.ranges)
        # n_angles = len(self.sonar_image.azimuth_angles)
        # sonar_matrix = np.array(
        #     [bb for bb in self.sonar_image.intensities]  # noqa: C416
        # )
        # sonar_matrix = np.reshape(sonar_matrix, (n_ranges, n_angles))

        # # Annoyingly, polar plots in matplotlib live in degrees,
        # # So before plotting, I'm converting everything to degrees.
        # # This includes the labeled points...
        # extent_degrees = [
        #     np.degrees(self.sonar_image.azimuth_angles[0]),
        #     np.degrees(self.sonar_image.azimuth_angles[-1]),
        #     self.sonar_image.ranges[0],
        #     self.sonar_image.ranges[-1],
        # ]
        #sonar_matrix = isc.raw_sonar_to_rectangular(self.sonar_image, 2.0, wide=True)
        sonar_matrix = cv2.remap(self.sonar_image, *self.polar_transform, cv2.INTER_LINEAR)
        azimuth_min, azimuth_max = -65, 65
        extent_degrees = [azimuth_min, azimuth_max, 0.0, 2.0]
        return sonar_matrix, extent_degrees

    def calibrate_sonar(self, camera_rvec, camera_tvec):
        # type: None
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
        have_camera = camera_rvec is not None
        if not have_labels or not have_camera:
            return None, None, None, None, None, None

        labeled_points = self.sonar_labels[self.current_timestamp]
        sonar_points, target_points = isc.get_sonar_target_correspondences(labeled_points, self.sonar_params)
        # Now these are correct

        # Transform from target's coordinate frame to camera coordinate frame
        camera_rot, _ = cv2.Rodrigues(camera_rvec)
        camera_points = camera_tvec + camera_rot @ target_points

        range_resolution, angle_resolution = self.sonar_params.r_res, self.sonar_params.th_res
        #isc.get_sonar_resolution(self.sonar_image)

        # TODO: These initial vectors should become input params!
        init_rvec = (1.2092, 1.2092, 1.2092)
        init_tvec = (0, -0.06, 0.095)  # I think this is from the CAD drawing

        cs_err, cs_rvec, cs_tvec = isc.calibrate_sonar(
            sonar_points,
            camera_points,
            self.sonar_params,
            init_rvec,
            init_tvec,
        )

        cs_rot, _ = cv2.Rodrigues(cs_rvec)
        #yaw, pitch, roll = tf.transformations.euler_from_matrix(np.transpose(cs_rot))
        #tf.geometry.transformation.euler.from_rotation_matrix(np.transpose(cs_rot))
        dx, dy, dz = cs_tvec[0][0], cs_tvec[1][0], cs_tvec[2][0]
        print("Final XYZ: {:03f} {:03f} {:03f}".format(dx, dy, dz))
        #print("Final ROS YPR: {:03f} {:03f} {:03f}".format(yaw, pitch, roll))
        print()

        # TODO: Calculate sonar-> target from camera->sonar and camera->target
        camera_rot, _ = cv2.Rodrigues(camera_rvec)
        sonar_tvec = cs_tvec + cs_rot @ camera_tvec
        sonar_rot = cs_rot @ camera_rot
        sonar_rvec, _ = cv2.Rodrigues(sonar_rot)

        print("camera to sonar", cs_tvec, "\n", cs_rot, "\n")
        print("sonar to target", sonar_tvec, "\n", sonar_rot, "\n")

        return sonar_points, target_points, cs_rvec, cs_tvec, sonar_rvec, sonar_tvec

#good below here
    def update_plots(self, keep_limits=True):
        #print("Showing sonar data at timestamp: {}".format(self.current_timestamp))
        self.timestamp_label.setText(f"Sonar timestamp: {self.current_timestamp}")

        self.update_good_label()
        charucoCorners, charucoIds, camera_tvec, camera_rvec = charuco_utils.detect_charuco_board(
                        self.charuco_board, self.camera_data, self.camera_info.K, self.camera_info.D)
        
        # Save these! The current timestamp is actually the SONAR's header
        # stamp, but that's what we're using to index into the sensor pairs.
        if camera_rvec is not None:
            self.camera_poses[self.current_timestamp] = (camera_rvec, camera_tvec)

        ######################
        # Update the figures
        camera_gray = cv2.cvtColor(self.camera_data.copy(), cv2.COLOR_RGB2GRAY)
        self.plot_raw_camera_data(self.camera_data)
        self.plot_charuco_detections(camera_gray, charucoCorners, charucoIds)
        self.plot_camera_targets_from_camera(camera_gray, self.camera_info, camera_rvec, camera_tvec)

        sonar_matrix, extent_degrees = self.get_plottable_sonar()
        sonar_matrix = cv2.flip(sonar_matrix, 0) 

        cc = self.calibrate_sonar(camera_rvec, camera_tvec)
        sonar_points, target_points, cs_rvec, cs_tvec, sonar_rvec, sonar_tvec = cc

        self.sonar_image_ax.imshow(self.sonar_image)
        self.plot_sonar_image(sonar_matrix, None, None, extent_degrees, keep_limits=keep_limits)
        self.plot_polar_sonar_image(sonar_matrix)

        if sonar_rvec is not None:
            rres, ares = self.sonar_params.r_res, self.sonar_params.th_res
            sonar_err = isc.calc_projection_error(
                target_points, sonar_points, sonar_rvec, sonar_tvec, self.sonar_params
            )
        else:
            sonar_err = -1.0

        if cs_rvec is None:
            cs_rotation = None
        else:
            cs_rotation, _ = cv2.Rodrigues(cs_rvec)

        self.plot_camera_targets_from_sonar(camera_gray, self.camera_info, cs_rotation, cs_tvec)

        self.plot_sonar_targets_from_camera(
            sonar_matrix, sonar_rvec, sonar_tvec, camera_rvec, camera_tvec, sonar_err
        )

    def load_state(self):
        filename = "{}/calibration_labels.pkl".format(self.outdir)
        if os.path.exists(filename):
            try:
                with open(filename, "rb") as fp:
                    data = pickle.load(fp)
                # backwards-compatible with previous format
                # TODO: Maybe move this out of the try/except block?
                if "good_timestamps" in data:
                    good = data["good_timestamps"]
                else:
                    good = set()
                return good, data["skip_timestamps"], data["sonar_labels"]

            except Exception as ex:
                print("Could not load labels from: {}".format(filename))
                print("")
                # Intentionally not trying to recover, since we don't wan't
                # to accidentally overwrite a file that the human has
                # already started!
                raise (ex)
        else:
            return set(), set(), {}

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

if __name__ == "__main__":
    description = "GUI for manually labeling imaging sonar calibration data"
    # outdir_help = (
    #     "name of file to save manually generated labels. If file "
    #     "already exists, will resume from previous session"
    # )
    outdir = "C:/Users/corri/OneDrive/Documents/SonarExperimentData/testpairs/output"
    try:
        if not os.path.exists(outdir):
            os.makedirs(outdir)
    except Exception as ex:
        print("Could not create directory: {}".format(outdir))
        print("")
        raise (ex)

    app = QtWidgets.QApplication(sys.argv)
    window = SensorWindow(outdir)
    window.show()
    sys.exit(app.exec_())