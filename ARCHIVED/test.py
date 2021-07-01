import cv2
import numpy as np
import os
import imutils

DEBUG = False

COLOUR_BOUNDS = [
                [(16, 0, 0)      , (179, 255, 30)  , "BLACK"  , 0 , (0,0,0)       ],    
                [(5, 80, 63)    , (15, 250, 100)  , "BROWN"  , 1 , (0,51,102)    ],    
                [(0, 100, 80)    , (4, 255, 200)  , "RED"    , 2 , (0,0,255)     ],
                [(4, 110, 170)   , (20, 255, 255)  , "ORANGE" , 3 , (0,128,255)   ], 
                [(23, 100, 100) , (32, 255, 255)  , "YELLOW" , 4 , (0,255,255)   ],
                [(48, 125, 70)  , (65, 200, 200)   , "GREEN"  , 5 , (0,255,0)     ],  
                [(97, 100, 100)    , (110, 255, 200)  , "BLUE"   , 6 , (255,0,0)     ],  
                [(120, 40, 100) , (140, 250, 220) , "PURPLE" , 7 , (255,0,127)   ], 
                [(0, 0, 60)     , (179, 50, 90)   , "GRAY"   , 8 , (128,128,128) ],      
                [(0, 0, 150)     , (179, 15, 250)  , "WHITE"  , 9 , (255,255,255) ],
                ];

RED_TOP_LOWER = (160, 30, 80)
RED_TOP_UPPER = (179, 255, 200)
MIN_AREA = 700
FONT = cv2.FONT_HERSHEY_SIMPLEX

#required for trackbars
def empty(x): 
    pass

#initializing haar cascade and video source
def init(DEBUG):
    if (DEBUG):
        cv2.namedWindow("frame")
        cv2.createTrackbar("lh", "frame",0,179, empty)
        cv2.createTrackbar("uh", "frame",0,179, empty)
        cv2.createTrackbar("ls", "frame",0,255, empty)
        cv2.createTrackbar("us", "frame",0,255, empty)
        cv2.createTrackbar("lv", "frame",0,255, empty)
        cv2.createTrackbar("uv", "frame",0,255, empty)
    tPath = os.getcwd()
    cap = cv2.VideoCapture(0)
    rectCascade = cv2.CascadeClassifier(tPath +"/cascade/haarcascade_resistors.xml")
    
    return (cap,rectCascade)

#returns true if contour is valid, false otherwise
def validContour(cnt):
    #looking for a large enough area and correct aspect ratio
    if(cv2.contourArea(cnt) < MIN_AREA):
        return False
    else:
        x,y,w,h = cv2.boundingRect(cnt)
        aspectRatio = float(w)/h
        if (aspectRatio > 0.4):
            return False
    return True

#evaluates the resistance value based on the bands detected
def printResult(sortedBands, savedimg, resPos):
    x,y,w,h = resPos
    strVal = ""
    if (len(sortedBands) in [3,4,5]):
        for band in sortedBands[:-1]:
            strVal += str(band[3])
        intVal = int(strVal)
        intVal *= 10**sortedBands[-1][3]
        cv2.rectangle(savedimg,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(savedimg,str(intVal) + " OHMS",(x+w+10,y), FONT, 1,(255,255,255),2,cv2.LINE_AA)
        return
    #draw a red rectangle indicating an error reading the bands
    cv2.rectangle(savedimg,(x,y),(x+w,y+h),(0,0,255),2)
    
#uses haar cascade to identify resistors in the image
def findResistors(img_new, rectCascade):
    gimg_new = cv2.cvtColor(img_new, cv2.COLOR_BGR2GRAY)
    resClose = []

    #detect resistors in main frame
    ressFind = rectCascade.detectMultiScale(gimg_new,1.1,25)
    for (x,y,w,h) in ressFind: #SWITCH TO H,W FOR <CV3
        
        roi_gray = gimg_new[y:y+h, x:x+w]
        roi_color = img_new[y:y+h, x:x+w]

        #apply another detection to filter false positives
        secondPass = rectCascade.detectMultiScale(roi_gray,1.01,5)

        if (len(secondPass) != 0):
            resClose.append((np.copy(roi_color),(x,y,w,h)))
    return resClose

#analysis close up image of resistor to identify bands
def findBands(resistorInfo, DEBUG):
    if (DEBUG):
        uh = cv2.getTrackbarPos("uh","frame")
        us = cv2.getTrackbarPos("us","frame")
        uv = cv2.getTrackbarPos("uv","frame")
        lh = cv2.getTrackbarPos("lh","frame")
        ls = cv2.getTrackbarPos("ls","frame")
        lv = cv2.getTrackbarPos("lv","frame")
    #enlarge image
    resImg = cv2.resize(resistorInfo[0], (400, 200))
    resPos = resistorInfo[1]
    #apply bilateral filter and convert to hsv                                          
    pre_bil = cv2.bilateralFilter(resImg,5,80,80)
    hsv = cv2.cvtColor(pre_bil, cv2.COLOR_BGR2HSV)
    #edge threshold filters out background and resistor body
    thresh = cv2.adaptiveThreshold(cv2.cvtColor(pre_bil, cv2.COLOR_BGR2GRAY),255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,59,5)
    thresh = cv2.bitwise_not(thresh)
            
    bandsPos = []

    #if in debug mode, check only one colour
    if (DEBUG): checkColours = COLOUR_BOUNDS[0:1]
    else:       checkColours = COLOUR_BOUNDS

    for clr in checkColours:
        if (DEBUG):
            mask = cv2.inRange(hsv, (lh,ls,lv),(uh,us,uv)) #use trackbar values
        else:
            mask = cv2.inRange(hsv, clr[0], clr[1])
            if (clr[2] == "RED"): #combining the 2 RED ranges in hsv
                redMask2 = cv2.inRange(hsv, RED_TOP_LOWER, RED_TOP_UPPER)
                mask = cv2.bitwise_or(redMask2,mask,mask)
             
        mask = cv2.bitwise_and(mask,thresh,mask= mask)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        #filter invalid contours, store valid ones
        for k in range(len(contours)-1,-1,-1):
            if (validContour(contours[k])):
                leftmostPoint = tuple(contours[k][contours[k][:,:,0].argmin()][0])
                bandsPos += [leftmostPoint + tuple(clr[2:])]
                cv2.circle(pre_bil, leftmostPoint, 5, (255,0,255),-1)
            else:
                contours.pop(k)
        
        cv2.drawContours(pre_bil, contours, -1, clr[-1], 3)
        if(DEBUG):
            cv2.imshow("mask", mask)
            cv2.imshow('thresh', thresh)                                   

    cv2.imshow('Contour Display', pre_bil)#shows the most recent resistor checked.
    
    #sort by 1st element of each tuple and return
    return sorted(bandsPos, key=lambda tup: tup[0])


#MAIN
cap,rectCascade = init(DEBUG)
path = os.getcwd()

while True:
    ret, frame = cap.read()

    if cv2.waitKey(1) & 0xFF == ord('q'):  #camera off
        print("Camera off")
        cap.release()
        cv2.destroyAllWindows()
        break

    elif cv2.waitKey(1) & 0xFF == ord('s'):  # Captures a frame & process to detect resistors
        
        # os.chdir(path + '/images')
        cv2.imshow("Captured", frame)
        #cv2.imwrite(filename='saved_img.jpg', img=frame)
        print('Captured!')
        
        img_new =  frame #cv2.imread(path + '/images/saved_img.jpg')
        resClose = findResistors(img_new, rectCascade)

        for i in range(len(resClose)):
            sortedBands = findBands(resClose[i],DEBUG)
            final = printResult(sortedBands, img_new, resClose[i][1])
            
        #cv2.imwrite(filename='detected_resistors.jpg', img=final)
        #cv2.imwrite(filename='colourquantized.jpg', img=resClose[0][0])
            
        #cv2.imread(path + '/images/detected_resistors.jpg')
        cv2.imshow("detected resistors", final)
        print('processed!')
           
        continue
        
        
    
    cv2.imshow("Frame",frame)

cap.release()
cv2.destroyAllWindows()