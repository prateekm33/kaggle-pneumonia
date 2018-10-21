from PIL.Image import fromarray
import cv2
import numpy as np
from load_data import load_data
import pydicom

images, labels = load_data()
im = images[4]
label = labels[4]
x = int(label[1])
y = int(label[2])
x2 = x + int(label[3])
y2 = y + int(label[4])
cv2.rectangle(im,(x,y),(x2,y2),(0,0,255),3)
cv2.imshow('image_down', im)
cv2.waitKey(0)

ds = pydicom.dcmread('./stage_1_train_images/00436515-870c-4b36-a041-de91049b9ab4.dcm')
im_orig = ds.pixel_array
_x = 264
_y = 152
_x2 = _x + 213
_y2 = _y + 379
cv2.rectangle(im_orig,(_x,_y),(_x2,_y2),(0,0,255),3)

cv2.imshow('image_orig', im_orig)
cv2.waitKey(0)
cv2.destroyAllWindows()

