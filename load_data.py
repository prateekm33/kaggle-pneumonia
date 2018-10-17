import numpy as np

def load_data():
  images = np.load('processed_train_images.npy')
  labels = np.load('processed_train_labels.npy')
  return images, labels