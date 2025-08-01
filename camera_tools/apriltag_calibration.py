import cv2
import glob
import numpy as np

# # load the input image and convert it to grayscale
# print("[INFO] loading image...")
# image = cv2.imread('C:/Users/corri/OneDrive/Documents/SonarExperimentData/underwater_camera/arucoboard/good/frame20250710_115014.png')
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

ARUCO_DICT_ID = cv2.aruco.DICT_APRILTAG_36h11
BOARD_ROWS = 5
BOARD_COLS = 5
SQUARE_LENGTH = .06     
MARKER_LENGTH = .045          
MARGIN_PX = 0   

dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
board = cv2.aruco.GridBoard((5, 5), MARKER_LENGTH, MARKER_LENGTH*.2, dictionary)

ARUCO_DICT = {
	"DICT_4X4_50": cv2.aruco.DICT_4X4_50,
	"DICT_4X4_100": cv2.aruco.DICT_4X4_100,
	"DICT_4X4_250": cv2.aruco.DICT_4X4_250,
	"DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
	"DICT_5X5_50": cv2.aruco.DICT_5X5_50,
	"DICT_5X5_100": cv2.aruco.DICT_5X5_100,
	"DICT_5X5_250": cv2.aruco.DICT_5X5_250,
	"DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
	"DICT_6X6_50": cv2.aruco.DICT_6X6_50,
	"DICT_6X6_100": cv2.aruco.DICT_6X6_100,
	"DICT_6X6_250": cv2.aruco.DICT_6X6_250,
	"DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
	"DICT_7X7_50": cv2.aruco.DICT_7X7_50,
	"DICT_7X7_100": cv2.aruco.DICT_7X7_100,
	"DICT_7X7_250": cv2.aruco.DICT_7X7_250,
	"DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
	"DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
	"DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
	"DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
	"DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
	"DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
}

def get_calibration_parameters():
    # Define the aruco dictionary, charuco board and detector
    #board = cv2.aruco.Board((BOARD_COLS, BOARD_ROWS), SQUARE_LENGTH, MARKER_LENGTH, dictionary)
    #cols then rows, the wrong order caused problems earlier
    #detector = cv2.aruco.CharucoDetector(board)
    
    all_corners = []
    all_ids = []
    counter = []

    # Load images from directory
    images = glob.glob('C:/Users/corri/OneDrive/Documents/SonarExperimentData/underwater_camera/arucoboard/good/*.png')
    first = True
    # Loop over images and extraction of corners
    for image_file in images:
        #print(image_file)
        image = cv2.imread(image_file)
        image_copy = image.copy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #image = cv2.resize(image, None, None, fx = .25, fy = .25)
        imgSize = image.shape
        
        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(image, dictionary)
        if first == True:
            all_corners = corners
            all_ids = ids
            first = False
        elif corners:
            print(corners)
            all_corners = np.vstack((all_corners, corners))
            all_ids = np.vstack((all_ids,ids))
            counter.append(len(ids))
        print('Found {} unique markers'.format(np.unique(ids)))
        #charucoCorners, charucoIds, marker_corners, marker_ids = detector.detectBoard(image)
        
        # for mark in marker_corners:
        #     x, y = mark[0][0]
        #     cv2.circle(image_copy, (int(x), int(y)), 4, (0, 255, 0), -1)

        # cv2.aruco.drawDetectedMarkers(image_copy, corners, ids, borderColor=(255, 0, 0))
        # cv2.imwrite("test_images/points.jpg", image_copy)

        # cv2.imshow('img', image_copy)
        # cv2.waitKey(100)

        #print(charucoCorners, marker_corners)

        # Calibrate camera with extracted information
        # if charucoCorners is not None and charucoIds is not None and len(charucoCorners) > 3:
        #     all_corners.append(charucoCorners)
        #     all_ids.append(charucoIds)
    counter = np.array(counter)
    if len(all_corners) > 0:
        #print(all_corners, "\n", all_ids, "\n", counter)
        result, mtx, dist, rvecs, tvecs = cv2.aruco.calibrateCameraAruco(all_corners, all_ids, counter, board, imgSize, None, None)
    else:
        mtx, dist = [], []

    return mtx, dist

mtx, dist = get_calibration_parameters()
print(mtx, dist)
# [[295.48016553   0.         494.4936379 ]
#  [  0.         219.98477132 369.32578704]
#  [  0.           0.           1.        ]] [[-0.13305307 -0.00186581  0.00769798  0.00511768  0.00152419]]