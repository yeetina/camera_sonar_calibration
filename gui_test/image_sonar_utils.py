import cv2

import numpy as np
import scipy
import scipy.optimize
import os
import matplotlib.pyplot as plt

import charuco_utils

class SensorData():
    def __init__(self, main_folder, sonar):
        self.camera_folder = f'{main_folder}/camera'
        self.sonar_folder = f'{main_folder}/sonar'
        self.sonar_params = sonar
        self.current_index = None
        self.sorted_pairs = []

        image_file_names = {f for f in os.listdir(self.camera_folder)}
        sonar_file_names = {f for f in os.listdir(self.sonar_folder)}
        sonar_sorted = sorted(list(sonar_file_names)) 
        camera_sorted = sorted(list(image_file_names))
        
        for sname, cname in zip(sonar_sorted, camera_sorted):
            dt = cname[:-4].replace('_' ,'T')
            dt = dt[:4] + "-" + dt[4:6] + "-" + dt[6:11] + ":" + dt[11:13] + ":" + dt[13:]
            timestamp = np.datetime64(dt)
            self.sorted_pairs.append(SensorPair(sname, cname, timestamp))
        
        self.length = len(self.sorted_pairs)
        print("Data structure created successfully")
 
        
    def get_pair(self, idx):
        current_pair = self.sorted_pairs[idx]
        sonar = cv2.imread(f"{self.sonar_folder}/{current_pair.sonarfile}", cv2.IMREAD_GRAYSCALE)
        sonar = crop_sonar_arc(sonar, self.sonar_params)
        image = cv2.imread(f"{self.camera_folder}/{current_pair.imagefile}")
        #image = cv2.resize(image, None, None, fx = .25, fy = .25)
        return current_pair, sonar, image

    def next(self):
        if self.length > 0:
            if self.current_index is None:
                self.current_index = 0
            elif self.current_index+1 < self.length:
                self.current_index += 1
            else:
                raise Exception("You've reached the end of the image data")
            return self.get_pair(self.current_index)
        else:
            raise Exception("No data to display")

    def go_to(self, index):
        self.current_index = index

    def sort_key(self, name, char="c"):
        one, two = name.split(char)
        #print(name, one[1:],two[:-4])
        return int(one[1:]), int(two[:-4])

class SensorPair():
    def __init__(self, sonarfile, imagefile, ID):
        self.sonarfile = sonarfile
        self.imagefile = imagefile
        self.flag = None
        self.timestamp = ID
    
    def get_timestamp(self):
        return self.timestamp

class SonarInfo():
    def __init__(self, range_m, wide):
        self.range = range_m
        self.wide = wide
        if range_m >= 1.5:
            self.r_res = .0025
        else:
            self.r_res = .002

        if wide:
            self.aper = 130
            self.th_res = 0.6
            self.theta_bins = 216 #(aper/th_res)
            self.range_bins = int(range_m/self.r_res) 
            self.x_center = 808
        else:
            self.aper = 40
            self.th_res = 0.4
            self.theta_bins = int(self.aper/self.th_res) #columns
            self.range_bins = int(range_m/self.r_res)
            self.x_center = 305

def timestamp_tostr(timestamp):
    return str(timestamp).replace(":","-")

def crop_sonar_arc(sonar_im, sonarinfo):
    """
    Isolate the sonar display from images saved by oculus software
    """
    #sonar_im = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2GRAY)
    if sonarinfo.wide:
        angle_start, angle_end = 205, 335
        centerpoint = (808,892)
        cropped = sonar_im[50:942, 117:1733]
    else:
        angle_start, angle_end = 249.5, 289.4
        centerpoint = (305,892)
        cropped = sonar_im[50:942, 620:1232]
    
    mask = np.zeros(cropped.shape[:2], dtype="uint8")
    cv2.ellipse(mask, centerpoint, (892, 892), 0.0, angle_start, angle_end, (255), -1)
    return cv2.bitwise_and(cropped, cropped, mask=mask)

def create_transform_map(sonar):
    range_m = sonar.range
    r_res = sonar.r_res
    aper = sonar.aper
    th_res = sonar.th_res
    theta_bins = sonar.theta_bins
    range_bins = sonar.range_bins
    x_center = sonar.x_center
    
    x_map = np.zeros((range_bins, theta_bins), dtype=np.float32)
    y_map = np.zeros((range_bins, theta_bins), dtype=np.float32)
    
    #x is theta_bin, y is range_bin
    for y in range(range_bins):
        for x in range(theta_bins):
            theta_rad = (x*th_res - aper/2) * np.pi/180
            r_pix = (range_bins-y-1) * r_res * 892/range_m
            dx = r_pix*np.sin(theta_rad)
            dy = r_pix*np.cos(theta_rad)
            #print(x, y, dx, dy)
            xfinal = x_center + dx
            yfinal = 892-dy
            x_map[y][x] = xfinal
            y_map[y][x] = yfinal

    return x_map, y_map

def get_polar_coords(pixelx, pixely, originx, originy):
    """Radius is in pixels and theta is in radians
    """
    rad = np.sqrt((pixely-originy)**2 + (pixelx-originx)**2)
    try:
        theta = np.atan((pixelx-originx)/(pixely-originy)) #in radians I think
    except:
        theta = 0
    return rad, theta

def pixel_to_polar(coord, sonar):
    """
    Given pixel coordinates, find the polar theta (deg) and r (m)
    """
    xpix, ypix = coord
    theta_deg = (xpix*sonar.th_res - sonar.aper/2)
    r_meters = (ypix) * sonar.r_res
    return theta_deg, r_meters

def polar_to_pixel(coords, sonar):
    """
    theta is in degrees, coords is a 2xN array
    """
    r = coords[1,:]
    th = coords[0,:]
    ypix = r/sonar.r_res
    th += .5*sonar.aper
    xpix = th/sonar.th_res
    return np.array([xpix, ypix])

def polar_from_3d(points):
    """
    image_from_sonar3d
    Project 3D points in the sonar's frame to the (R, theta) data returned
    by the sensor.

    Input parameters:
    points -- (xx, yy, zz) Coordinates of point to be transformed;
              ndarray with shape (3,N)

    Output:
    angles, ranges -- Polar coordinates of point in the sonar image;
              ndarray with shape (2,N)
    """
    xx = points[0, :]
    yy = points[1, :]
    zz = points[2, :]
    ranges = np.sqrt(xx * xx + yy * yy + zz * zz)
    angles = np.rad2deg(np.arctan2(xx, yy))
    return np.array([angles, ranges])

def get_black_squares(board):
    """
    Return coordinates (in meters) of the center of every black square
    on the input charuco board coordinate plane

    Assumes that top-left square is a black square
    Black squares are numbered in row-major order, starting with 0
    """
    ncols, nrows = board.getChessboardSize()
    ss = board.getSquareLength()
    ii = 0
    centers = {}
    for row in range(nrows):
        if row % 2 == 0:
            cols = range(0, ncols, 2)
        else:
            cols = range(1, ncols, 2)
        for col in cols:
            xx = (col + 0.5) * ss
            yy = (row + 0.5) * ss
            centers[ii] = (xx, yy)
            ii += 1

    return centers

def init_charuco_sonar(SQUARE_LENGTH = .026, MARKER_LENGTH = .0195):
    aruco_dict, charuco_board = charuco_utils.make_charuco_board(SQUARE_LENGTH, MARKER_LENGTH)

    # If the charuco board is of a different shape, then our assumptions about
    # where the targets are will be wrong.
    ncols, nrows = charuco_board.getChessboardSize()
    assert 11 == ncols
    assert 8 == nrows

    # The black squares are numbered 0-N, in row-major order starting from
    # the top left corner of the charuco board. Each sonar target is a bolt
    # in the center of a square.
    black_squares = get_black_squares(charuco_board)
    #print(len(black_squares))
    # sonar_targets = {'A1': 0, 'A2': 1, 'A3': 4, 'A4': 7,
    #                 'B1': 2, 'B2': 3, 'B3': 6, 'B4': 10,
    #                 'C1': 25, 'C2': 28, 'C3': 29, 'C4': 32,
    #                 'D1': 27, 'D2': 30, 'D3': 31, 'D4': 34,}
    sonar_targets = {'A1': 0, 'A2': 1, 'A3': 6, 'A4': 11, 'A5': 12,
                     'B1': 4, 'B2': 5, 'B3': 10, 'B4': 15, 'B5': 16, 
                     'C1': 28, 'C2': 33, 'C3': 34, 'C4': 39, 'C5': 40,
                     'D1': 32, 'D2': 37, 'D3': 38, 'D4': 42, 'D5': 43}
    sonar_coords = {label: black_squares[ss] for label, ss in sonar_targets.items()}
    
    return aruco_dict, charuco_board, sonar_coords

def get_sonar_target_correspondences(labeled_points, sonar):
    """
    Find corresponding points in sonar and target frames, using the
    label to look up target coords.

    Inputs:
    * labeled_points: dict mapping label to (pixel_x, pixel_y) tuple

    # TODO: Would be good to stop passing these around as tuples/lists/etc, and actually have a class.
    Returns:
    * sonar_points: locations of labeled points in the sonar frame
    * target_points: locations of labeled points in the target's frame.
    """
    _, _, sonar_coords = init_charuco_sonar()

    # I'm sure there's a better way to do this, but I'm blanking on it
    angles = []
    ranges = []
    target_points = []
    for label, (x_pixel, y_pixel) in labeled_points.items():
        # angle_deg, pt_range = pixel_to_polar((x_pixel, y_pixel), sonar)
        # #print(f'{label}: {angle_deg}, {pt_range}')
        # angles.append(np.radians(angle_deg))
        # ranges.append(pt_range)
        angles.append(x_pixel)
        ranges.append(y_pixel)
        coord = sonar_coords[label]
        target_points.append([coord[0], coord[1], 0])

    # angles = np.array([np.radians(nn) for nn, _ in points.values()])
    # ranges = np.array([nn for _, nn in points.values()])
    sonar_points = np.array([angles, ranges])
    target_points = np.transpose(np.array(target_points))
    return sonar_points, target_points


def estimate_target_translation(camera_points, sonar_points, sonar, rvec, initial_tvec, verbose=True,):
    err0 = calc_projection_error(camera_points, sonar_points, rvec, initial_tvec, sonar)
    if verbose:
        print("Initial error: {}".format(err0))
    # noqa: E731
    fn = lambda x: calc_projection_error(camera_points, sonar_points, rvec, x, sonar)
    opt = {"maxiter": 3000}
    res = scipy.optimize.minimize(fn, initial_tvec, method="Nelder-Mead", options=opt)
    if res.status != 0 or res.fun > 100:
        print(
            "Optimization did not terminate successfully, or error too large. Trying 2nd pass"
        )
        res = scipy.optimize.minimize(fn, res.x, method="Nelder-Mead", options=opt)

    tvec = np.reshape(res.x, (3, 1))
    err_n = calc_projection_error(camera_points, sonar_points, rvec, tvec, sonar)

    if verbose:
        print("Final error: {}".format(err_n))

    return res.fun, tvec


def estimate_target_pose(
    camera_points,
    sonar_points, 
    sonar,
    initial,
    verbose=True,
):
    # Initialize the rotation s.t. the target's frame is aligned
    # with the sonar's frame; this helps avoid falling into a local
    # minimum with the target normal pointed away from the sonar
    # TODO: However, it would be better to actually put constraints on
    #    the optimization to enforce valid bounds on the rotation vector:
    #    (1) magnitude <= 2pi, (2) normal must be pointing towards the sonar.
    #    * 1 is probably best done by after-the-fact normalization
    #      (bounds would just create local extrema along the bounds)
    #    * 2 is ???
    rvec = initial[0:3]
    tvec = initial[3:6]
    err0 = calc_projection_error(camera_points, sonar_points, rvec, tvec, sonar)

    if verbose:
        print("Initial error: {}".format(err0))
        # print("(Using rvec = {}, tvec = {}".format(rvec, tvec))

    fn = lambda x: calc_projection_error(  # noqa: E731
        camera_points, sonar_points, x[0:3], x[3:6], sonar
    )
    opt = {"maxiter": 3000}
    res = scipy.optimize.minimize(fn, initial, method="Nelder-Mead", options=opt)
    if res.status != 0 or res.fun > 100:
        print(
            "Optimization did not terminate successfully, or error too large. Trying 2nd pass"
        )
        res = scipy.optimize.minimize(fn, res.x, method="Nelder-Mead", options=opt)

    rvec = np.reshape(res.x[0:3], (3, 1))
    tvec = np.reshape(res.x[3:6], (3, 1))
    err_n = calc_projection_error(camera_points, sonar_points, rvec, tvec, sonar)
    
    if verbose:
        print("Final error: {}".format(err_n))

    return res.fun, rvec, tvec


def calc_projection_error(camera_points, sonar_points, rvec, tvec, sonar):
    """
    Calculate the sum-squared reprojection error between corresponding points in the
    target frame and in the sonar image, given the input transform.

    Returns sum-squared error in pixels, using range/angle resolution
    to convert from the SI units reportd by the sonar to pixels.

    Input parameters:
    camera_points -- (3,N) np.ndarray. points from target in camera frame
                     (will usually be planar, but not required!)
                     
    sonar_points -- (2,N) np.ndarray. Corresponding points in the sonar image.
                   1st row angles, 2nd is ranges. (in pixels)
    rvec, tvec -- rotation and transformation from the sonar frame
                 (x fwd, y right, z down) to the 
    range_resolution -- (in meters)
    angle_resolution -- (in radians)
    """
    rvec = np.reshape(rvec, (3, 1))
    tvec = np.reshape(tvec, (3, 1))
    rot, _ = cv2.Rodrigues(rvec)

    # Project points from camera to sonar in xyz coords
    sonar_frame = tvec + rot @ camera_points
    # tvec and rvec from sonar to target

    #print("Targets, in sonar frame: ", target_sonar_frame)
    sonar_polar_frame = polar_from_3d(sonar_frame)
    sonar_pixels = polar_to_pixel(sonar_polar_frame, sonar)
    #print("Targets, in image coordinates: ", target_image_frame)

    # Calculate the reprojection error, in pixels
    d_angle = (sonar_points[0, :] - sonar_pixels[0, :]) #/ sonar.th_res
    #print("Angle errors: ", d_angle)
    d_range = (sonar_points[1, :] - sonar_pixels[1, :]) #/ sonar.r_res
    #print("Range errors: ", d_range)
    err = np.sum(np.sqrt(d_angle * d_angle + d_range * d_range))
    return err

def get_sonar_resolution(sonar_image):
    """
    ONLY USED for converting angle errors into pixels.

    The sonar has constant range resolution, but angle-dependent
    azimuth resolution. For now, just take average.

    * range_resolution
    * angle_resolution: in radians
    """
    rr = np.array(sonar_image.ranges)
    dr = rr[1:] - rr[0:-1]
    aa = np.array(sonar_image.azimuth_angles)
    da = aa[1:] - aa[0:-1]
    range_resolution = np.mean(dr)
    angle_resolution = np.mean(da)

    return range_resolution, angle_resolution


def calibrate_sonar(
    sonar_points,
    camera_points,
    sonar,
    init_rvec=None,
    init_tvec=None,
    verbose=True,
):
    # 
    # Initialize the sonar to be aligned with camera axis.
    # There's a rotation here because the camera has
    # X-right, but sonar is X-fwd. rvec is the Rodrigues representation
    # of the rotation required to rotate the sonar
    # frame with the camera, since we are solving for the transform
    # for X_camera = tvec + rot @ X_sonar
    # magnitude = 2*pi/3 about the [1,1,1] vector.
    if init_rvec is None:
        init_rvec = (1.2092, 1.2092, 1.2092)
    if init_tvec is None:
        init_tvec = (0, 0, 0)
    #init_rvec = np.reshape(init_rvec, (3, 1))
    #init_tvec = np.reshape(init_tvec, (3, 1))
    if verbose:
        print("Translation minimization:")
    cs_err, cs_tvec = estimate_target_translation(
        camera_points,
        sonar_points,
        sonar,
        init_rvec,
        init_tvec,
        verbose,
    )
    # print("translation-only minimization: T = {}".format(cs_tvec))

    initial = [
        init_rvec[0],
        init_rvec[1],
        init_rvec[2],
        cs_tvec[0][0],
        cs_tvec[1][0],
        cs_tvec[2][0],
    ]

    if verbose:
        print("Full minimization:")
    cs_err, cs_rvec, cs_tvec = estimate_target_pose(
        camera_points,
        sonar_points,
        sonar,
        initial,
        verbose,
    )
    # print("Full minimization: R = {}, T = {}".format(cs_rvec, cs_tvec))
    return cs_err, cs_rvec, cs_tvec

#init_charuco_sonar()
#data = SensorPairs()
#print(data.next())