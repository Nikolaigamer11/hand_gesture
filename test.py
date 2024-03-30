import cv2
from cvzone.ClassificationModule import Classifier
from cvzone.HandTrackingModule import HandDetector
import mediapipe
import numpy as np 
import math
import tensorflow
from collections import Counter
import logging
import pyttsx3


offset=15
imgSize= 300
cap= cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
values= []
Cfier = Classifier("VSCODE\hand_gesture-main\model\keras_model.h5","VSCODE\hand_gesture-main\model\labels.txt")
label=["Hello","Good Morning","Thank You"]
enigine=pyttsx3.init()

def vote(values):
    # Count occurrences of each value
    counts = Counter(values)
    
    # Find the most common value
    most_common_value = counts.most_common(1)[0][0]
    
    return most_common_value




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
            # pridictions,index =Cfier.getPrediction(img)
            # values.append(label[index])
            # if len(values) == 37:
            #     result=vote(values)
            #     print(f'the most common value is {result}')
            #     values.clear()
            # print(f'pridicted letter is {label[index]} with a pridiction rate of {pridictions}')
        else:
            k= imgSize / h
            wCalc= math.ceil(k * w)
            imgResize= cv2.resize(imgCrop, (wCalc, imgSize))
            imgResizeShape= imgResize.shape
            wGap= math.ceil((imgSize - wCalc)/2)
            imgWhite[:,wGap:wCalc+wGap]= imgResize
        pridictions,index =Cfier.getPrediction(img)
        values.append(label[index])
        if len(values) == 17:
            result=vote(values)
            print(f'the most common value is {result}with a pridiction rate of {pridictions}')
            enigine.setProperty('rate',150)
            enigine.setProperty('volumn',0.9)
            answer=(f'he said {result}')
            enigine.say(result)
            values.clear()
            enigine.runAndWait()
            # print(f'pridicted letter is {label[index]} which is at {index} with a pridiction rate of {pridictions}')




        cv2.imshow("imgcrop",imgCrop)
        cv2.imshow("imgwhite",imgWhite)
    

    cv2.imshow("IMAGE",img)
    if cv2.waitKey(40) & 0xFF == ord('w'): 
        break

cap.release()