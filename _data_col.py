import cv2
import time
from cvzone.HandTrackingModule import HandDetector
import mediapipe
import numpy as np 
import math

offset=25
imgSize= 300
cap= cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
counter = 0
folder= "Yes"


while True:
    success, img =cap.read()
    hands,img= detector.findHands(img)
    if hands:
        hand = hands[0]
        x,y,w,h=hand['bbox']
        
        imgWhite=np.ones((imgSize,imgSize,3),np.uint8)*255
        imgCrop= img[y-offset:y+h+offset,x-offset:x+w+offset]

        imgCropShape= imgCrop.shape

        asRatio= h /w

        if asRatio <1:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize
        else:
            k= imgSize / h
            wCalc= math.ceil(k * w)
            imgResize= cv2.resize(imgCrop, (wCalc, imgSize))
            imgResizeShape= imgResize.shape
            wGap= math.ceil((imgSize - wCalc)/2)
            imgWhite[:,wGap:wCalc+wGap]= imgResize




        cv2.imshow("imgcrop",imgCrop)
        cv2.imshow("imgwhite",imgWhite)
    

    cv2.imshow("IMAGE",img)

    if cv2.waitKey(40) & 0xFF == ord('c'):
            counter += 1
            cv2.imwrite(f'assets/{folder}/Img_{time.time()}.jpg',imgWhite)
            print(f'catured = {counter}')
    if cv2.waitKey(40) & 0xFF == ord('w'): 
        break
    # cv2.destroyAllWindows()
cap.release()