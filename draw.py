import cv2
from load_data import load_data
import pydicom

images, labels = load_data()
orig_labels = pd.read_csv('stage_1_train_labels.csv', dtype={'patientId': str, 'x': np.float32, 'y': np.float32, 'width': np.float32, 'height': np.float32, 'Target': np.float32}, engine="python", nrows=len(labels)).fillna(0).values

for i in range(0, len(labels)):
  im = images[i]
  label = labels[i]
  x = int(label[1])
  y = int(label[2])
  x2 = x + int(label[3])
  y2 = y + int(label[4])
  print(x, y, x2, y2)
  cv2.rectangle(im,(x,y),(x2,y2),(0,0,255),3)
  cv2.imshow('image_down', im)
  cv2.waitKey(0)

  ds = pydicom.dcmread('./stage_1_train_images/' + label[0] + '.dcm')
  im_orig = ds.pixel_array
  _x = orig_labels[0, 1]
  _y = orig_labels[0, 2]
  _x2 = _x + orig_labels[0, 3]
  _y2 = _y + orig_labels[0, 4]
  cv2.rectangle(im_orig,(_x,_y),(_x2,_y2),(0,0,255),3)

  cv2.imshow('image_orig', im_orig)
  key = cv2.waitKey(0)
  print(key)
  if key  == 27:
     cv2.destroyAllWindows()
     break