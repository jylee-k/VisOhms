import cv2
import numpy as np
import os
from function import evalResistance, findResistors, findBands

DEBUG = False
FONT = cv2.FONT_HERSHEY_SIMPLEX


#initializing haar cascade and video source
def init(DEBUG):
    tPath = os.getcwd()
    cap = cv2.VideoCapture(0)
    haarCascade = cv2.CascadeClassifier(tPath +"/cascade/haarcascade_resistors_0.xml")
    
    return (cap,haarCascade)

cap,haarCascade = init(DEBUG)

width = cap.get(3)
height = cap.get(4)

while True:
    ret, frame = cap.read()


    if cv2.waitKey(1) & 0xFF == ord('q'):  #camera off
        print("Camera off")
        cap.release()
        cv2.destroyAllWindows()
        break


    elif cv2.waitKey(50) & 0xFF == ord('s'):  # Captures a frame & process to detect resistors
        
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
        
    
            cv2.imshow("Detected Resistors", final)
 
        continue
        
    cv2.imshow("Frame",frame)

cap.release()
cv2.destroyAllWindows()

