from os import listdir, path
import pandas as pd
import numpy as np
import pydicom
import scipy.misc
import cv2

def convertDCM_PNG(dir, file):
  ds = pydicom.dcmread(dir + '/' + file)
  ds_ = np.array(ds.pixel_array)
  # ds_ = np.uint8(ds_)
  ds_ = cv2.resize(ds_, (600,600)) / 255
  ds_ = np.reshape(ds_, (600,600,1))
  shape = ds.pixel_array.shape
  # print('ds : ', ds_.shape)
  # return np.array(ds), ds.pixel_array.shape

  # shape = ds.pixel_array.shape

  # Convert to float to avoid overflow or underflow losses.
  # image_2d = ds.pixel_array #.astype(float)

  # Rescaling grey scale between 0-255
  # image_2d_scaled = (np.maximum(image_2d,0) / image_2d.max()) * 255.0

  # Convert to uint
  # image_2d_scaled = np.uint8(image_2d_scaled)
  
  # Scale images down to 600 x 600
  # image_2d_scaled = scipy.misc.imresize(image_2d, (600, 600, 1))
  # Divide by 255 to normalize pixel data
  # image = np.array(image_2d_scaled, ndmin=3) / 255

  return ds_, shape

def process_images(images_dir, labels_csv,  sample_size=None, bclass = False):
  labels = pd.read_csv(labels_csv, dtype={'patientId': str, 'x': np.float32, 'y': np.float32, 'width': np.float32, 'height': np.float32, 'Target': np.float32}, engine="python", nrows=sample_size).fillna(0).values
 
  if bclass == True:
    labels = labels[:, -1]
    labels = np.reshape(labels, (labels.shape[0], 1))

  # convert dicom images to np.arrays
  images = np.zeros((len(labels), 600, 600, 1))
  for i in range(0, len(labels)):
    f = labels[i, 0] + '.dcm'
    image, shape = convertDCM_PNG(images_dir, f)
    images[i] = image
    y_scale = 600. / shape[0]
    x_scale = 600. / shape[1]
    if bclass == False:
      labels[i, 1] *= x_scale
      labels[i, 2] *= y_scale
      labels[i, 3] *= x_scale
      labels[i, 4] *= y_scale

  # np.reshape(labels[:,1:], (len(labels), 5))
  return images, np.reshape(labels, (len(labels), 6))
