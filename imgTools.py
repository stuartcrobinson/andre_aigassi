

import os

import cv2

def writeTextTopLeft(image_in, text):
    text = str(text)
    cv2.putText(img=image_in, text=text, org=(10, 30),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=[0, 0, 0], lineType=cv2.LINE_AA, thickness=4)
    cv2.putText(img=image_in, text=text, org=(10, 30),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=[100, 100, 100], lineType=cv2.LINE_AA, thickness=2)

