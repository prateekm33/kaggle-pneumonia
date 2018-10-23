from tensorflow.keras.models import load_model
import sys
from data_generator import DataGenerator
from get_partition import get_partition, get_labels
import numpy as np

filename = sys.argv[1]
model = load_model(filename)

sample_size = 50
_input = '0100515c-5204-4f31-98e0-f35e4b00004a'

params = {'dim': (448,448,1),
          'batch_size': 1,
          'n_channels': 1,
          'shuffle': True}
partition = get_partition('stage_1_train_labels.csv', sample_size=sample_size)
labels = get_labels()

partition_test = partition['test']
print('partition : ', partition_test)
generator = DataGenerator(partition_test, labels, **params)
print('beginning predictions...')
preds = model.predict_generator(generator=generator, workers=6, use_multiprocessing=True)

f = open('predictions.txt', 'w')
f.write(str(preds))
print(preds)
print('--------')

aa = np.array([np.load('processed_train_images/' + _input + '.npz')['image']])
print(aa.shape)
print('---actual : ', np.load('processed_train_labels/' + _input + '.npz')['label'][1:])
pp = model.predict(aa)
print('------->>>')
print(pp)
