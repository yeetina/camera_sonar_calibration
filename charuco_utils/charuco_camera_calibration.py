import cv2
import numpy as np
import os
import json

ARUCO_DICT_ID = cv2.aruco.DICT_4X4_250
BOARD_ROWS = 8
BOARD_COLS = 11
SQUARE_LENGTH = .026       
MARKER_LENGTH = .0195           
MARGIN_PX = 0   


def get_calibration_parameters(img_dir):
    # Define the aruco dictionary, charuco board and detector
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
    board = cv2.aruco.CharucoBoard((BOARD_COLS, BOARD_ROWS), SQUARE_LENGTH, MARKER_LENGTH, dictionary)
    #cols then rows, the wrong order caused problems earlier
    detector = cv2.aruco.CharucoDetector(board)
    
    all_corners = []
    all_ids = []

    # Load images from directory
    image_files = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(".png")]
    
    # Loop over images and extraction of corners
    for image_file in image_files:
        #print(image_file)
        image = cv2.imread(image_file)
        image_copy = image.copy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #image = cv2.resize(image, None, None, fx = .25, fy = .25)
        imgSize = image.shape
        
        
        charucoCorners, charucoIds, marker_corners, marker_ids = detector.detectBoard(image)
        
        # for mark in marker_corners:
        #     x, y = mark[0][0]
        #     cv2.circle(image_copy, (int(x), int(y)), 4, (0, 255, 0), -1)

        cv2.aruco.drawDetectedMarkers(image_copy, marker_corners, marker_ids, borderColor=(255, 0, 0))
        # cv2.imwrite("test_images/points.jpg", image_copy)
        cv2.imshow('img', image_copy)
        cv2.waitKey(500)

        #print(charucoCorners, marker_corners)

        # Calibrate camera with extracted information
        if charucoCorners is not None and charucoIds is not None and len(charucoCorners) > 3:
            all_corners.append(charucoCorners)
            all_ids.append(charucoIds)

    if all_ids and all_corners:
        result, mtx, dist, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(all_corners, all_ids, board, imgSize, None, None)
    else:
        mtx, dist = [], []

    return mtx, dist

#SENSOR = 'monochrome'
#LENS = 'kowa_f12mm_F1.8'
OUTPUT_JSON = 'underwater_cam.json'

mtx, dist = get_calibration_parameters(img_dir='C:/Users/corri/OneDrive/Documents/SonarExperimentData/07-21-2025/camera')
#print(mtx, dist)
# avg_mtx = np.mean(mtx, axis=0)
# avg_dist = np.mean(dist, axis=0)
# print(mtx, avg_mtx)

data = {"mtx": mtx.tolist(), "dist": dist.tolist()} #"sensor": SENSOR, "lens": LENS, 

with open(OUTPUT_JSON, 'w') as json_file:
    json.dump(data, json_file, indent=4)

print(f'Data has been saved to {OUTPUT_JSON}')

