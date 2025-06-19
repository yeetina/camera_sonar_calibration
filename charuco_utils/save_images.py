import cv2
import numpy as np
import time

cap = cv2.VideoCapture(0)

while True:
    # show the image
    _, frame = cap.read() 
   
    cv2.imshow("Output", frame)
    cv2.waitKey(3)

    if cv2.waitKey(10) & 0xFF == ord('s'):
        filename = f"test_images/calibration_validation/frame{str(round(time.time()))}.png"
        cv2.imwrite(filename, frame)
        print(filename, " saved")
                
    #press q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()  
cv2.destroyAllWindows()