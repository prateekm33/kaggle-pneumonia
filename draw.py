import cv2
import numpy as np
import load_data

images, labels = load_data()
img = images[0]
x = labels[0, 0]
y = labels[0, 1]
x2 = x + labels[0, 2]
y2 = y + labels[0, 3]
cv2.rectangle(images[0],(x,y),(x2,y2),(0,255,0),3)

