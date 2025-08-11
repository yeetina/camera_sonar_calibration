import cv2
import json
import numpy as np
import os

ARUCO_DICT_ID = cv2.aruco.DICT_4X4_250
BOARD_ROWS = 8
BOARD_COLS = 11
SQUARE_LENGTH = .026       
MARKER_LENGTH = .0195           
MARGIN_PX = 0     

json_file_path = './underwater_cam.json'

with open(json_file_path, 'r') as file: # Read the JSON file
    json_data = json.load(file)

mtx = np.array(json_data['mtx'])
dst = np.array(json_data['dist'])

def pos_from_image(color_image, mtx, dst):
    image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

    h,  w = image.shape[:2]
    # newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dst, (w,h), 1, (w,h))
    # image = cv2.undistort(image, mtx, dst, None, newcameramtx)

    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
    board = cv2.aruco.CharucoBoard((BOARD_COLS, BOARD_ROWS), SQUARE_LENGTH, MARKER_LENGTH, dictionary)
    detector = cv2.aruco.CharucoDetector(board)

    # cv2.aruco.drawDetectedMarkers(image_copy, marker_corners, marker_ids)
    charucoCorners, charucoIds, marker_corners, marker_ids = detector.detectBoard(image)
    
    if charucoCorners is not None and charucoIds is not None and len(charucoCorners) > 3:
        print("entered conditional")
        retval, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(np.array(charucoCorners), np.array(charucoIds), board, np.array(mtx), np.array(dst), np.empty(1), np.empty(1))
        result = color_image.copy()
        #cv2.aruco.drawDetectedMarkers(result, marker_corners, marker_ids)
        cv2.drawFrameAxes(result, np.array(mtx), np.array(dst), rvec, tvec, .1)
        #cv2.drawChessboardCorners(result, (BOARD_COLS, BOARD_ROWS), charucoCorners, retval)

        return tvec, rvec, result
    else:
        return [], [], color_image

def single_image(filepath, mtx, dst):
    image = cv2.imread(filepath)
    #image = cv2.resize(image, None, None, fx = .25, fy = .25)
    tv, rv, result = pos_from_image(image, mtx, dst)
    cv2.imwrite("frame.png", result)
    print(tv, "\n", rv)

mtx = np.array([[1.02082611e+03, 0.00000000e+00, 7.69307527e+02],
                [0.00000000e+00, 1.02245381e+03, 2.90583592e+02],
                [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
dst = np.array([[-3.76227154e-01,  1.94912143e-01, 
                             -2.04912328e-03,  7.63774994e-05, -5.57738640e-02]])
single_image("C:/Users/corri/OneDrive/Documents/SonarExperimentData/07-23-2025/camera/20250723_163603.png", mtx, dst)

def image_folder(img_dir):
    image_files = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(".png")]
    for image_file in image_files:
        image = cv2.imread(image_file)
        tvec, rvec, display = pos_from_image(image, mtx, dst)
        cv2.imshow('img', display)
        cv2.waitKey(1000)
    
def video_stream():
    cap = cv2.VideoCapture(0)#cv2.CAP_DSHOW?
    cap.set(cv2.CAP_PROP_FPS, 2)

    while True:
        # show the image
        _, frame = cap.read()
        tv, rv, result = pos_from_image(frame)
    
        cv2.imshow("Output", result)
        cv2.waitKey(3)
                    
        #press q to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
