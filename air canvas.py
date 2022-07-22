import cv2
import numpy as np
import time
import os
import handtrackingmodule as htm


brushSize = 15
eraserSize = 50
folderPath = "header"
myList = os.listdir(folderPath)
#print(myList)

overlayList  = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)
    
#print(len(overlayList))

header1 = overlayList[0]
drawColor = (255,0, 255)

cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4, 720)

detector = htm.handDetector(detectionCon=0.85)
xp,yp = 0,0
imgCanvas = np.zeros((720, 1280, 3), np.uint8)

while True:
    
     # 1.import the image
    success, img = cap.read()
    img = cv2.flip(img, 1)
    
    # 2. find hand landmarks
    
    img = detector.findHands(img)
    lmList = detector.findPosition(img , draw= False)
    
    if len(lmList)!=0:
        
        #print(lmList)
        
     # tip of index and middle finger   
        x1,y1= lmList[8][1:]
        x2,y2= lmList[12][1:]
        
        
        
        
    #3. check which fingers are up
    
        fingers = detector.fingersUp()
        
        #4 selection mode- 2 fingers up
        
        
        if fingers[1] and fingers[2]:
            xp,yp = 0,0
            
            print("selection mode")
            if y1< 125:
                if 250<x1<450:
                    header1= overlayList[0]
                    drawColor = (255, 0 ,255)
                elif 550 <x1 < 750:
                        header1= overlayList[1]
                        drawColor = (0,0,255)
                elif 800 <x1 < 950:
                        header1= overlayList[2]
                        drawColor = (0,255,0)
                elif 1050 <x1 < 1200:
                        header1= overlayList[3]
                        drawColor = (0,0,0)
            cv2.rectangle(img, (x1,y1-30), (x2,y2+30),drawColor, cv2.FILLED)
            
            
        #5 drawing mode - 1 finger up
        
        
        if fingers[1] and fingers[2]==False:
            cv2.circle(img, (x1,y1), 15,(drawColor), cv2.FILLED)
            print("drawing mode")
            if xp==0 and yp==0:
                xp,yp = x1,y1
            if drawColor == (0,0,0):
                cv2.line(img, (xp,yp), (x1, y1), drawColor, eraserSize)
                cv2.line(imgCanvas, (xp,yp), (x1, y1), drawColor, eraserSize)
            else:    
                cv2.line(img, (xp,yp), (x1, y1), drawColor, brushSize)
                cv2.line(imgCanvas, (xp,yp), (x1, y1), drawColor, brushSize)
            
            
            xp,yp= x1,y1
    
    imgGray= cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)
   
    
    
    
    
    #setting the header image
    img[0:125,0:1280] = header1
    img = cv2.addWeighted(img, 0.5, imgCanvas, 0.5,0)
   
    cv2.imshow("Image", img)
    cv2.imshow("Canvas", imgCanvas)
    cv2.imshow("Inverse", imgInv)
    cv2.waitKey(1)