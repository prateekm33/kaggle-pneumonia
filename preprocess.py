from process_images import process_images
import numpy as np
import sys

sample_size = None
if len(sys.argv) > 1:
  sample_size = int(sys.argv[1])

images, labels = process_images('stage_1_train_images', 'stage_1_train_labels.csv', sample_size=sample_size)
print('shape of images : ', images.shape)
print('shape of labels : ', labels.shape)
# images_file = open('processed_train_images', 'w')
np.savez_compressed('processed_train_images', images=images)

# labels_file = open('processed_train_labels', 'w')
np.savez_compressed('processed_train_labels', labels=labels)
