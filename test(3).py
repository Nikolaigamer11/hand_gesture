import cv2
from cvzone.ClassificationModule import Classifier
from cvzone.HandTrackingModule import HandDetector
import numpy as np 
import math
import tensorflow
from collections import Counter
import pyttsx3


offset=15
imgSize= 300
cap= cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
values= []
Cfier = Classifier("model/keras_model.h5","model/labels.txt")

engine=pyttsx3.init()
wait_key=1
DEFAULT_GESTURE="None"
Screen_hand=False
engine.setProperty('rate',150)
engine.setProperty('volumn',1.0)
label=["Hello","Good Morning","Thank You"]

def vote(values,lable):
    counts = Counter(values)
    most_common_value = counts.most_common(1)[0][0]
    engine.say(label[most_common_value])
    values.clear()
    engine.runAndWait()
    return most_common_value




while True:
    success, img =cap.read()
    hands,img= detector.findHands(img)
    if hands== []:
            cv2.putText(img,"No gesture detected",(50,50),cv2.FONT_HERSHEY_COMPLEX,1.5,(0,0,255),3)
            values.clear()

    elif hands:
        hand = hands[0]
        x,y,w,h=hand['bbox']
        
        imgWhite=np.ones((imgSize,imgSize,3),np.uint8)*255
        imgCrop= img[y-offset:y+h+offset,x-offset:x+w+offset]

        imgCropShape= imgCrop.shape

        asRatio= h /w

        if asRatio <1:
            k = imgSize / w
            hCal = math.ceil(k * h)

            try:
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            except cv2.error as e:
                if not Screen_hand:

                    if "!ssize.empty()" in str(e):
                        print("out of the screen")
                        values.clear()
                        cv2.putText(img,"HAND OUTSIDE SCREEN",(50,50),cv2.FONT_HERSHEY_COMPLEX,2,(0,0,255),4)
                        engine.say("take hand inside screen")
                        engine.runAndWait()
                    
                    else:
                    # If it's another kind of cv2.error, you might want to handle it differently
                        print("An OpenCV error occurred:", e)
                Screen_hand=True
                continue

            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize



        else:
            k= imgSize / h
            wCalc= math.ceil(k * w)

            try:
                imgResize= cv2.resize(imgCrop, (wCalc, imgSize))
            except cv2.error as e:


                if not Screen_hand:

                    if "!ssize.empty()" in str(e):
                        print("out of the screen")
                        values.clear()
                        cv2.putText(img,"HAND OUTSIDE SCREEN",(50,50),cv2.FONT_HERSHEY_COMPLEX,2,(128,0,128),4)
                        engine.say("take hand inside screen")
                        engine.runAndWait()
                    
                    else:
                    # If it's another kind of cv2.error, you might want to handle it differently
                        print("An OpenCV error occurred:", e)
                Screen_hand=True
                continue

            imgResizeShape= imgResize.shape
            wGap= math.ceil((imgSize - wCalc)/2)
            imgWhite[:,wGap:wCalc+wGap]= imgResize
        pridictions,index =Cfier.getPrediction(imgWhite)
        cv2.putText(img,label[index],(50,50),cv2.FONT_HERSHEY_COMPLEX,2,(0,255,255),4)
        values.append(index)

        if len(values) == 17:
            result=vote(values,label)
            print(f'the most common value is {label[result]} with a pridiction rate of {pridictions}')
            
            Screen_hand=False
            

        # cv2.imshow("imgcrop",imgCrop)
        # cv2.imshow("imgwhite",imgWhite)


    

    cv2.imshow("IMAGE",img)
    if cv2.waitKey(wait_key) & 0xFF == ord('w'): 
        break

cap.release()
