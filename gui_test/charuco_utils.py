import cv2
import cv2.aruco
import numpy as np

ARUCO_DICT_ID = cv2.aruco.DICT_4X4_250
BOARD_ROWS = 8
BOARD_COLS = 11
SQUARE_LENGTH = .026       
MARKER_LENGTH = .0195           
MARGIN_PX = 0   


def make_charuco_board(SQUARE_LENGTH, MARKER_LENGTH):
    # type: string -> Tuple[cv2.aruco_Dictionary, cv2.aruco_CharucoBoard]
    """
    Given the name of a board/plinth, return the dictionary used and the board.

    The dictionary isn't used by most callers, but is handy to have for plotting
    the individual markers for debugging.
    """
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
    board = cv2.aruco.CharucoBoard((BOARD_COLS, BOARD_ROWS), SQUARE_LENGTH, MARKER_LENGTH, dictionary) 
    return dictionary, board

def get_board_center(board):
    """
    Return coordinates (in meters) of the center of the board.
    """
    ncols, nrows = board.getChessboardSize()
    ss = board.getSquareLength()
    return 0.5 * ss * ncols, 0.5 * ss * nrows

def detect_charuco_board(board, image, mtx, dst):
    """
    Find charuco corners in the image and use them to estimate board position
    """
    #image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    detector = cv2.aruco.CharucoDetector(board)
    charucoCorners, charucoIds, marker_corners, marker_ids = detector.detectBoard(image)


    if charucoCorners is not None and charucoIds is not None and len(charucoCorners) > 3:
        retval, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(np.array(charucoCorners), np.array(charucoIds), board, np.array(mtx), np.array(dst), np.empty(1), np.empty(1))
    else:
        tvec, rvec = None, None

    return charucoCorners, charucoIds, tvec, rvec
    
def get_image_dist(aruco_corners, idx1, vertex1, idx2, vertex2):
    """
    Get distance between the two specified marker/vertex pairs.

    This takes the index into the aruco_corners array, rather than the aruco_id
    because we care about disambiguating between duplicate detections.
    """
    x1 = aruco_corners[idx1][0][vertex1][0]
    x2 = aruco_corners[idx2][0][vertex2][0]
    dx = x2 - x1
    y1 = aruco_corners[idx1][0][vertex1][1]
    y2 = aruco_corners[idx2][0][vertex2][1]
    dy = y2 - y1
    return np.sqrt(dx * dx + dy * dy)

def draw_detected_corners(img, corners, ids):
    """
    Draw all markers for a charuco board on an image.
    Return annotated image
    """
    drawn_image = img.copy()
    color = (255, 0, 0)
    drawn_image = cv2.aruco.drawDetectedCornersCharuco(drawn_image, corners, ids, color)

    return drawn_image


if __name__ == "__main__":
    test = cv2.imread("test_images/calibration/frame1749644949.png")
    dictionary, board = make_charuco_board()
    charucoCorners, charucoIds, tvec, rvec = detect_charuco_board(board, test)
    drawing = draw_detected_corners(test, charucoCorners, charucoIds)
    #board_img = board.generateImage((700,1000))
    while True:
        # show the image
        cv2.imshow("Output", drawing)
        cv2.waitKey(3)

                    
        #press q to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()