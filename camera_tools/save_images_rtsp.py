# This method doesn't work very well. When using the rtsp protocol, I had to use
# multithreading to prevent the images from being very pixelated or glitchy.  
# With continous streaming, there was the problem of extreme lag. I set it up 
# to quit after saving one image to avoid that problem, but it doesn't quit 
# cleanly and I end up having to kill the program manually. 

import cv2
import numpy as np
from datetime import datetime, date, time, timezone
import queue
import threading

rtsp = "rtsp://192.168.0.74:554/1/h264major"
print("fps", cv2.CAP_PROP_FPS)

q=queue.Queue()
quit = False

def Receive():
    print("start Reveive")
    cap = cv2.VideoCapture(rtsp)
    ret, frame = cap.read()
    q.put(frame)
    while ret:
        ret, frame = cap.read()
        q.put(frame) 
        time.sleep(10)
        if quit == True:
            cap.release()
            return


def Display():
    print("Start Displaying")
    while True:
        if q.empty() !=True:
            frame=q.get()
            cv2.imshow("frame1", frame)

        if cv2.waitKey(33) & 0xFF == ord('s'):
            now = datetime.now()
            timestamp = str(now).replace("-", "")
            timestamp = timestamp.replace(" ", "_")
            timestamp = timestamp.replace(":", "")
            filename = f"test_images/7-21-2025/camera/{timestamp[:15]}.png"
            cv2.imwrite(filename, frame)
            print(filename, " saved")
            return

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            quit = True
            return
            
if __name__=='__main__':
    p1 = threading.Thread(target=Receive)
    p2 = threading.Thread(target=Display)
    p1.start()
    p2.start()
