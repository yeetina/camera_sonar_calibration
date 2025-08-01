import cv2
import numpy as np
from datetime import datetime, date, time, timezone

destination_folder = ""  #Input a folder name here
cap = cv2.VideoCapture(0) 
# ^Typically 0 to access webcam and 1 for external cam

fps = cv2.CAP_PROP_FPS
print("fps: ", fps)  
    
while True:
    ret, frame = cap.read() 
    
    if not ret:
        print("ret error")
        break

    cv2.imshow("Output", frame)
    cv2.waitKey(3)

    if cv2.waitKey(15) & 0xFF == ord('s'):
        now = datetime.now()
        timestamp = str(now).replace("-", "")
        timestamp = timestamp.replace(" ", "_")
        timestamp = timestamp.replace(":", "")
        filename = f"{destination_folder}/frame{timestamp[:15]}.png"
        cv2.imwrite(filename, frame)
        print(filename, " saved")
                
    #press q to quit
    if cv2.waitKey(15) & 0xFF == ord('q'):
        break

cap.release()  
cv2.destroyAllWindows()       