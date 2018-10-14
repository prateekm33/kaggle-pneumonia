from os import listdir, path
import pandas as pd
import numpy as np
import pydicom
import scipy
import cv2

def convertDCM_PNG(dir, file):
  ds = pydicom.dcmread(dir + '/' + file)
  shape = ds.pixel_array.shape

  # Convert to float to avoid overflow or underflow losses.
  image_2d = ds.pixel_array.astype(float)

  # Rescaling grey scale between 0-255
  image_2d_scaled = (np.maximum(image_2d,0) / image_2d.max()) * 255.0

  # Convert to uint
  image_2d_scaled = np.uint8(image_2d_scaled)
  
  # Scale images down to 600 x 600
  image_2d_scaled = scipy.misc.imresize(image_2d_scaled, (600, 600, 1))
  # Divide by 255 to normalize pixel data
  image = np.array(image_2d_scaled, ndmin=3) / 255.0

  return image, shape

def process_images(images_dir, labels_csv,  sample_size=None, bclass = False):
  Labels = pd.read_csv(labels_csv, dtype={'patientId': str, 'x': np.float64, 'y': np.float64, 'width': np.float64, 'height': np.float64, 'Target': np.float64})
  
  if sample_size:
    Labels = Labels.loc[0:sample_size - 1,:]
  
  # convert Labels<DataFrame> to np.array
  labels = Labels.values[:, 1:]

  if bclass == True:
    labels = labels[:, -1]
    labels = np.reshape(labels, (labels.shape[0], 1))

  # convert dicom images to np.arrays
  images = []
  for i in range(0, len(Labels)):
    f = Labels.loc[i, 'patientId'] + '.dcm'
    image, shape = convertDCM_PNG(images_dir, f)
    images.append(image.T)
    y_scale = 600. / shape[0]
    x_scale = 600. / shape[1]
    if bclass == False:
      labels[i, 0] = Labels.loc[i, 'x'] * x_scale
      labels[i, 1] = Labels.loc[i, 'y'] * y_scale
      labels[i, 2] = Labels.loc[i, 'width'] * (x_scale * x_scale)
      labels[i, 3] = Labels.loc[i, 'height'] * (y_scale * y_scale)

  images = np.array(images)
  
  return images, labels