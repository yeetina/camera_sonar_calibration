import cv2
import numpy as np
import time

sonar = cv2.imread("s58c3295.jpg")
# kernel = np.array([[0, 1, 0],
#           [1, 1, 1],
#           [0, 1, 0]])
kernel = np.ones((3,3))

grayIm = cv2.cvtColor(sonar, cv2.COLOR_BGR2GRAY)
thresh = 120
ret, bw = cv2.threshold(grayIm, thresh, 255, cv2.THRESH_BINARY)
opened = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=1)
closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=1)
combined = cv2.bitwise_and(sonar, sonar, mask=closed)

#result = cv2.vconcat([cv2.hconcat([sonar, combined]), cv2.hconcat([closed, opened])])

while True:
    # show the image
    cv2.imshow("1", sonar)
    cv2.imshow("2", bw)
    cv2.imshow("3", closed)
    cv2.imshow("4", combined)
    cv2.waitKey(3)

    if cv2.waitKey(1) & 0xFF == ord('s'):
        filename = f"test_images/frame{str(round(time.time()))}.png"
        cv2.imwrite(filename, result)
        print(filename, " saved")
                
    #press q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break