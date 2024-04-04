import cv2
from cvzone.ClassificationModule import Classifier
from cvzone.HandTrackingModule import HandDetector
import numpy as np 
import math
from collections import Counter
import pyttsx3

def vote(values):
    counts = Counter(values)
    most_common_value = counts.most_common(1)[0][0]
    return most_common_value

def resize_image(imgCrop, imgSize):
    h, w, _ = imgCrop.shape
    if h / w < 1:
        k = imgSize / w
        hCal = math.ceil(k * h)
        imgResize = cv2.resize(imgCrop, (imgSize, hCal))
    else:
        k = imgSize / h
        wCalc = math.ceil(k * w)
        imgResize = cv2.resize(imgCrop, (wCalc, imgSize))
    return imgResize

def detect_gesture(hand, imgSize, offset, img):
    x, y, w, h = hand['bbox']
    imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
    imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
    imgResize = resize_image(imgCrop, imgSize)
    return imgResize, imgWhite

def main():
    cap = cv2.VideoCapture(0)
    detector = HandDetector(maxHands=1)
    values = []
    Cfier = Classifier("model/keras_model.h5", "model/labels.txt")
    label = ["Hello", "Good Morning", "Thank You"]
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.setProperty('volumn', 1.0)

    while True:
        success, img = cap.read()
        hands, img = detector.findHands(img)
        if hands == []:
            cv2.putText(img, "No gesture detected", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 0, 255), 3)
        elif hands:
            pridictions, index = Cfier.getPrediction(img)
            values.append(label[index])

            if len(values) == 17:
                result = vote(values)
                print(f'the most common value is {result} with a prediction rate of {pridictions}')
                engine.say(result)
                values.clear()
                engine.runAndWait()

        cv2.imshow("IMAGE", img)
        if cv2.waitKey(40) & 0xFF == ord('w'):
            break

    cap.release()

if __name__ == "__main__":
    main()
