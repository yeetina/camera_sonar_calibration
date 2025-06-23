import cv2
import json
import numpy as np


ARUCO_DICT_ID = cv2.aruco.DICT_4X4_100
BOARD_ROWS = 8
BOARD_COLS = 11
SQUARE_LENGTH = .026       
MARKER_LENGTH = .0195           
MARGIN_PX = 0     


json_file_path = './phonecalibration.json'

with open(json_file_path, 'r') as file: # Read the JSON file
    json_data = json.load(file)

mtx = np.array(json_data['mtx'])
dst = np.array(json_data['dist'])
print(mtx, "\n", dst)

def pos_from_image(color_image):
    image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

    h,  w = image.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dst, (w,h), 1, (w,h))
    image = cv2.undistort(image, mtx, dst, None, newcameramtx)

    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
    board = cv2.aruco.CharucoBoard((BOARD_COLS, BOARD_ROWS), SQUARE_LENGTH, MARKER_LENGTH, dictionary)
    detector = cv2.aruco.CharucoDetector(board)

    # cv2.aruco.drawDetectedMarkers(image_copy, marker_corners, marker_ids)
    charucoCorners, charucoIds, marker_corners, marker_ids = detector.detectBoard(image)
    print(len(charucoCorners), charucoIds)
    
    if charucoCorners is not None and charucoIds is not None and len(charucoCorners) > 3:
        print("entered conditional")
        retval, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(np.array(charucoCorners), np.array(charucoIds), board, np.array(mtx), np.array(dst), np.empty(1), np.empty(1))
        result = color_image.copy()
        cv2.drawFrameAxes(result, np.array(mtx), np.array(dst), rvec, tvec, .1)
        cv2.drawChessboardCorners(result, (BOARD_COLS, BOARD_ROWS), charucoCorners, retval)
        
        # Zx, Zy, Zz = tvec[0][0], tvec[1][0], tvec[2][0]
        # fx, fy = mtx[0][0], mtx[1][1]

        return tvec, rvec, result
    else:
        return [], [], color_image

def single_image(filepath):
    image = cv2.imread(filepath)
    image = cv2.resize(image, None, None, fx = .25, fy = .25)
    print(image.shape)
    tv, rv, result = pos_from_image(image)
    cv2.imwrite("test_images/charucodetections.jpg", result)
    print(tv, "\n", rv)

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

test_path = 'test_images/new_charuco_board/20250619_140320.jpg'
single_image(test_path)
# rvec = np.array([-0.24193354, -0.06892308, -0.10476409])
# dst, jac = cv2.Rodrigues(rvec)
# print(dst)

#video_stream()

# def perspective_function(x, Z, f): 
#     return x*Z/f

# nb_pixels = 200
# print(perspective_function(nb_pixels, Zz, fx))    