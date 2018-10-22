from tensorflow.keras.models import load_model
import sys
from data_generator import DataGenerator
from get_partition import get_partition, get_labels
import numpy as np

filename = sys.argv[1]
model = load_model(filename)

params = {'dim': (600,600,1),
          'batch_size': 16,
          'n_classes': 5,
          'n_channels': 1,
          'shuffle': True}
partition = get_partition('stage_1_train_labels.csv')
labels = get_labels()
print(partition['test'], '-------')
generator = DataGenerator(['035789b1-3736-405d-9910-f8f23c62ae9f'], labels, **params)
print('beginning predictions...')
preds = model.predict_generator(generator=generator, workers=6, use_multiprocessing=True)
f = open('predictions.txt', 'w')
f.write(str(preds))
print(preds)
print('--------')

aa = np.array([np.load('processed_train_images/035789b1-3736-405d-9910-f8f23c62ae9f.npz')['image']])
print(aa.shape)
print('---actual : ', np.load('processed_train_labels/035789b1-3736-405d-9910-f8f23c62ae9f.npz')['label'][1:])
pp = model.predict(aa)
print('------->>>')
print(pp)
