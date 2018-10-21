import numpy as np
import sys

def load_data():
  images = np.load('processed_train_images.npz')
  labels = np.load('processed_train_labels.npz')
  
  return images["images"], labels["labels"]

if len(sys.argv) > 1 and sys.argv[1] == 'debug':
  images, labels = load_data()
  print(images.shape, labels.shape)
