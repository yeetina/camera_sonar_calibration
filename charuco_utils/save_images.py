import cv2
import numpy as np
from datetime import datetime, date, time, timezone
import queue
import threading

rtsp = "rtsp://192.168.0.74:554/1/h264major"
#cap = cv2.VideoCapture(rtsp)
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
    if quit == True:
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
            #print(timestamp[:15])
            filename = f"test_images/calibration_validation/frame{timestamp[:15]}.png"
            cv2.imwrite(filename, frame)
            print(filename, " saved")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            quit = True
            return
            
if __name__=='__main__':
    p1=threading.Thread(target=Receive)
    p2 = threading.Thread(target=Display)
    p1.start()
    p2.start()


# while True:
#     # show the image
#     ret, frame = cap.read() 
    
#     if not ret:
#         print("ret error")
#         break

#     cv2.imshow("Output", frame)
#     cv2.waitKey(3)

#     if cv2.waitKey(33) & 0xFF == ord('s'):
#         now = datetime.now()
#         timestamp = str(now).replace("-", "")
#         timestamp = timestamp.replace(" ", "_")
#         timestamp = timestamp.replace(":", "")
#         #print(timestamp[:15])
#         filename = f"test_images/calibration_validation/frame{timestamp[:15]}.png"
#         cv2.imwrite(filename, frame)
#         print(filename, " saved")
                
#     #press q to quit
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()  
# cv2.destroyAllWindows()