from process_images import process_images
import numpy as np

images, labels = process_images('stage_1_train_images', 'stage_1_train_labels.csv', sample_size=100)

# images_file = open('processed_train_images', 'w')
np.save('processed_train_images', images)

# labels_file = open('processed_train_labels', 'w')
np.save('processed_train_labels', labels)
