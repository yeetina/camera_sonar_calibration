import numpy as np
import cv2 as cv
import glob

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*6,3), np.float32)
objp[:,:2] = np.mgrid[0:6,0:6].T.reshape(-1,2)
objp = objp*.06
print("object", objp)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('C:/Users/corri/OneDrive/Documents/SonarExperimentData/underwater_camera/chessboard/good/*.png')

for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (6,6), None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        cv.drawChessboardCorners(img, (6,6), corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(50)

cv.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
print(mtx, dist)

# result:[[1.02082611e+03 0.00000000e+00 7.69307527e+02]
#  [0.00000000e+00 1.02245381e+03 2.90583592e+02]
#  [0.00000000e+00 0.00000000e+00 1.00000000e+00]] [[-3.76227154e-01  1.94912143e-01 -2.04912328e-03  7.63774994e-05
#   -5.57738640e-02]]

# [[1.02082610e+03 0.00000000e+00 7.69307565e+02]
#  [0.00000000e+00 1.02245378e+03 2.90583438e+02]
#  [0.00000000e+00 0.00000000e+00 1.00000000e+00]] [[-3.76227152e-01  1.94912136e-01 -2.04909603e-03  7.63754306e-05
#   -5.57738687e-02]]
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
    mean_error += error
 
print( "total error: {}".format(mean_error/len(objpoints)) )