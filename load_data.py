import numpy as np

def load_data():
  images = np.fromfile(open('processed_train_images', 'r'))
  labels = np.fromfile(open('processed_train_labels', 'r'))
  return images, labels