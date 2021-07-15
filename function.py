import cv2
import numpy as np


COLOUR_BOUNDS = [[(0, 0, 0)      , (179, 255, 60)  , "BLACK"  , 0 , (0,0,0)       ],  
                [(5, 90, 10)    , (20, 250, 100)  , "BROWN"  , 1 , (0,51,102)    ],    
                [(0, 100, 100)    , (10, 255, 200)  , "RED"    , 2 , (0,0,255)     ],
                [(11, 70, 70)   , (25, 255, 200)  , "ORANGE" , 3 , (0,128,255)   ], 
                [(30, 170, 100) , (40, 250, 255)  , "YELLOW" , 4 , (0,255,255)   ],
                [(40, 40, 40)  , (70, 255, 255)   , "GREEN"  , 5 , (0,255,0)     ],  #DONE
                [(65, 50, 50)    , (115, 255, 200)  , "BLUE"   , 6 , (255,0,0)     ], #DONE 
                [(120, 70, 100) , (140, 250, 220) , "PURPLE" , 7 , (255,0,127)   ], 
                [(0, 0, 50)     , (179, 50, 80)   , "GRAY"   , 8 , (128,128,128) ],      
                [(0, 0, 90)     , (179, 15, 250)  , "WHITE"  , 9 , (255,255,255) ],
                ]

RED_TOP_LOWER = (160, 100, 100)
RED_TOP_UPPER = (179, 255, 200)

MIN_AREA = 700

FONT = cv2.FONT_HERSHEY_SIMPLEX

#returns true if contour is valid, false otherwise
def checkContour(cnt):
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
def evalResistance(sortedBands, savedimg, resPos):
    x,y,w,h = resPos
    strVal = ""
    if (len(sortedBands) in [3,4,5]):
        for band in sortedBands[:-1]:
            strVal += str(band[3])
        intVal = int(strVal)
        intVal *= 10**sortedBands[-1][3]
        cv2.rectangle(savedimg,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(savedimg,str(intVal) + " OHMS",(x+w+10,y), FONT, 1,(255,255,255),2,cv2.LINE_AA)
        return savedimg
    #draw a red rectangle indicating an error reading the bands
    cv2.rectangle(savedimg,(x,y),(x+w,y+h),(0,0,255),2)
    return savedimg
    
#uses haar cascade to identify resistors in the image
def findResistors(img_new, haarCascade):
    grayscale = cv2.cvtColor(img_new, cv2.COLOR_BGR2GRAY)
    resClose = []

    #detect resistors in main frame
    ressFind = haarCascade.detectMultiScale(grayscale,1.1,25)
    for (x,y,w,h) in ressFind: #SWITCH TO H,W FOR <CV3
        roi_gray = grayscale[y:y+h, x:x+w]
        

        #apply another detection to filter false positives
        secondPass = haarCascade.detectMultiScale(roi_gray,1.05,5)
        
        roi_color = img_new[y:y+h, x:x+w]
        
        if (len(secondPass) != 0):

            resClose.append((np.copy(roi_color),(x,y,w,h)))

    return resClose

#analysis close up image of resistor to identify bands
def findBands(resistorInfo, DEBUG):
    #enlarge image
    #resistoInfo == resClose
    resImg = cv2.resize(resistorInfo[0], (400, 200))       #resistorInfo[0] directs to the specified image of the resistor
    resPos = resistorInfo[1]                               #resistorInfo[1] directs to (x,y,w,h), position of the resistor in the image
    #apply bilateral filter and convert to hsv
    
    #Filters
    medianBlur = cv2.medianBlur(resImg,3)
    gaussianBlur = cv2.GaussianBlur(medianBlur, (3,3), 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    final = cv2.morphologyEx(gaussianBlur, cv2.MORPH_OPEN, kernel)
    
    # Display final image with filters
    cv2.imshow('Processed image', final)

    #Convert image from BGR to HSV 
    hsv = cv2.cvtColor(final, cv2.COLOR_BGR2HSV)
    
    
    #edge threshold filters out background and resistor body
    thresh = cv2.adaptiveThreshold(cv2.cvtColor(final, cv2.COLOR_BGR2GRAY),255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,59,5)
    thresh = cv2.bitwise_not(thresh)
            
    bandsPos = []

    #if in debug mode, check only one colour
    checkColours = COLOUR_BOUNDS

    for clr in checkColours:
        mask = cv2.inRange(hsv, clr[0], clr[1])
        if (clr[2] == "RED"): #combining the 2 RED ranges in hsv
            redMask2 = cv2.inRange(hsv, RED_TOP_LOWER, RED_TOP_UPPER)
            mask = cv2.bitwise_or(redMask2,mask,mask)
             
        mask = cv2.bitwise_and(mask,thresh,mask= mask)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        #filter invalid contours, store valid ones
        for k in range(len(contours)-1,-1,-1):
            if (checkContour(contours[k])):
                leftmostPoint = tuple(contours[k][contours[k][:,:,0].argmin()][0])
                bandsPos += [leftmostPoint + tuple(clr[2:])]
                cv2.circle(final, leftmostPoint, 5, (255,0,255),-1)
            else:
                contours.pop(k)
        
        cv2.drawContours(final, contours, -1, clr[-1], 3)
            

    cv2.imshow('Contour Display', final) #shows the most recent resistor checked.
    cv2.imshow('thresh', thresh) #shows the threshold
    
    #sort by 1st element of each tuple and return
    return sorted(bandsPos, key=lambda tup: tup[0])