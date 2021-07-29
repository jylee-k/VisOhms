import cv2
import numpy as np
import os
from function import evalResistance, findResistors, findBands
import time

DEBUG = False
FONT = cv2.FONT_HERSHEY_SIMPLEX


#initializing haar cascade and video source
def init(DEBUG):
    tPath = os.getcwd()
    cap = cv2.VideoCapture(0)
    haarCascade = cv2.CascadeClassifier(tPath +"/cascade/haarcascade_resistors_0.xml")
    
    return (cap,haarCascade)

def rescaleFrame(frame, percent):
    width = int(frame.shape[1] * percent/100)
    height = int(frame.shape[0] * percent/100)
    dim = (width,height)
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

cap,haarCascade = init(DEBUG)

width = cap.get(3)
height = cap.get(4)

while True:
    ret, frame = cap.read()


    if cv2.waitKey(1) & 0xFF == ord('1'):  #camera off
        print("Camera off")
        cap.release()
        cv2.destroyAllWindows()
        break


    elif cv2.waitKey(50) & 0xFF == ord('2'):  # Captures a frame & process to detect resistors
        
        print('Captured!')
        
        img_new =  frame
        resClose = findResistors(img_new, haarCascade)
        
        
        if len(resClose) == 0: #no resistors detected, outputs the original image
            copy = np.copy(frame)
            cv2.putText(copy, "No resistors in the frame", (int(np.round(width/3)), int(height-20)), FONT, 1, (255,255,255), 2, cv2.LINE_AA)
            cv2.imshow("Image", copy)
            
            
        else:
            for i in range(len(resClose)):
                sortedBands = findBands(resClose[i],DEBUG)
                final = evalResistance(sortedBands, img_new, resClose[i][1])
            print("Processed!")
        
    
            cv2.imshow("Detected Resistors", rescaleFrame(final, 70))
 
        continue
    
    elif cv2.waitKey(50) & 0xFF == ord('3'):
        cv2.destroyWindow("Image")
        continue
        
    cv2.imshow("Frame",frame)

cap.release()
cv2.destroyAllWindows()

