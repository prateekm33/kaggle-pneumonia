from os import listdir, path
import pandas as pd
import numpy as np
import pydicom
import scipy.misc
import cv2

def resizeIMG(arr, size=(200,200)):
  img = cv2.resize(arr, size)
  return np.reshape(img, (*size, 1))

def resizeDCM(dir, file):
  ds = pydicom.dcmread(dir + '/' + file)
  ds_ = np.array(ds.pixel_array)
  # ds_ = cv2.resize(ds_, (600,600)) / 255
  # ds_ = np.reshape(ds_, (600, 600, 1))
  ds_ = resizeIMG(ds_)
  orig_shape = ds.pixel_array.shape

  return ds_, orig_shape

def process_images(images_dir, labels_csv,  sample_size=None, bclass = False):
  labels = pd.read_csv(labels_csv, dtype={'patientId': str, 'x': np.float32, 'y': np.float32, 'width': np.float32, 'height': np.float32, 'Target': np.float32}, engine="python", nrows=sample_size).fillna(0).values
 
  if bclass == True:
    labels = labels[:, -1]
    labels = np.reshape(labels, (labels.shape[0], 1))

  # convert dicom images to np.arrays
  # images = np.zeros((len(labels), 600, 600, 1))
  for i in range(0, len(labels)):
    f = labels[i, 0] + '.dcm'
    image, shape = resizeDCM(images_dir, f)
    # images[i] = image
    np.savez_compressed('processed_train_images/' + labels[i, 0], image=image)
    if bclass == False:
      y_scale = 600. / shape[0]
      x_scale = 600. / shape[1]
      labels[i, 1] *= x_scale
      labels[i, 2] *= y_scale
      labels[i, 3] *= x_scale
      labels[i, 4] *= y_scale
      np.savez_compressed('processed_train_labels/' + labels[i, 0], label=labels[i])

  # np.reshape(labels[:,1:], (len(labels), 5))
  # return images, np.reshape(labels, (len(labels), 6))
