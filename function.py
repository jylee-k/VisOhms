import cv2
import numpy as np
import pandas as pd


COLOUR_BOUNDS = [[(21, 0, 0)      , (179, 255, 100)  , "BLACK"  , 0 , (0,0,0)       ],  #DONE
                [(0, 35, 10)    , (21, 255, 150)  , "BROWN"  , 1 , (0,51,102)    ],    
                [(0, 150, 100)    , (6, 255, 255)  , "RED"    , 2 , (0,0,255)     ], #DONE
                [(5, 121, 196)   , (20, 255, 255)  , "ORANGE" , 3 , (0,128,255)   ], 
                [(20, 190, 20) , (30, 250, 255)  , "YELLOW" , 4 , (0,255,255)   ],
                [(40, 40, 40)  , (70, 255, 255)   , "GREEN"  , 5 , (0,255,0)     ],  #DONE
                [(98, 109, 20)    , (112, 255, 255)  , "BLUE"   , 6 , (255,0,0)     ], #DONE 
                [(120, 48, 93) , (164, 255, 255) , "PURPLE" , 7 , (255,0,127)   ], #DONE
                [(0, 0, 50)     , (179, 50, 80)   , "GRAY"   , 8 , (128,128,128) ],      
                [(0, 0, 90)     , (179, 15, 250)  , "WHITE"  , 9 , (255,255,255) ],
                ]

RED_TOP_LOWER = (165, 150, 100)
RED_TOP_UPPER = (179, 255, 255)
BROWN_TOP_LOWER = (167, 35, 10)
BROWN_TOP_UPPER = (179, 255, 150)
#BLACK_TOP_LOWER = (165, 0, 0)
#BLACK_TOP_UPPER = (179, 255, 100)

#reading csv file with pandas and giving names to each column
#index = ["exact_colour", "exact_colour_name", "hex", "R", "G", "B", "res_colour"]
#csv = pd.read_csv('colors.csv', names=index, header=None)

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
    #cv2.imshow('Processed image', final)

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
        if (clr[2] == "BROWN"): #combining the 2 BROWN ranges in hsv
            brownMask2 = cv2.inRange(hsv, BROWN_TOP_LOWER, BROWN_TOP_UPPER)
            mask = cv2.bitwise_or(brownMask2,mask,mask)
        #if (clr[2] == "BLACK"):
            #blackMask2 = cv2.inRange(hsv, BLACK_TOP_LOWER, BLACK_TOP_UPPER)
            #mask = cv2.bitwise_or(blackMask2,mask,mask)
             
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
            

    #cv2.imshow('Contour Display', final) #shows the most recent resistor checked.
    #cv2.imshow('thresh', thresh) #shows the threshold
    
    #sort by 1st element of each tuple and return
    return sorted(bandsPos, key=lambda tup: tup[0])