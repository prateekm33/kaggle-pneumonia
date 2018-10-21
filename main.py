import datetime
import sys
import numpy as np
from process_images import process_images
from model import create_model
from sklearn.model_selection import train_test_split
from get_partition import get_partition, get_labels
from data_generator import DataGenerator
from model_callbacks import Logger
from tensorflow.keras.models import load_model

def main(model_file=None):
  if model_file != None:
    model = load_model(model_file)
  else:
    model = create_model()
  
  run(model)


def run(pneuModel):
  params = {'dim': (600,600,1),
          'batch_size': 16,
          'n_classes': 5,
          'n_channels': 1,
          'shuffle': True}

  # Datasets
  partition = get_partition('stage_1_train_labels.csv') # IDs
  labels = get_labels() # Labels

  # Generators
  training_generator = DataGenerator(partition['train'], labels, **params)
  validation_generator = DataGenerator(partition['validation'], labels, **params)

  # Train model on dataset
  callbacks = [Logger()]
  pneuModel.fit_generator(generator=training_generator,
                      validation_data=validation_generator,
                      use_multiprocessing=True,
                      workers=6,
                      callbacks=callbacks
                      )

  # Test model
  test_label_ids = partition['test']
  test_Y = np.zeros((len(test_label_ids, 5)))
  test_X = np.zeros((len(test_label_ids, 600, 600, 1)))
  for i in range(len(test_label_ids)):
    _id = test_label_ids[i]
    test_Y[i] = labels[_id]
    test_X[i] = np.load('processed_train_images/' + _id + '.npz')['image']

  preds = pneuModel.evaluate(x=test_X, y=test_Y)
  
  ts = '.'.join(str(datetime.datetime.now()).split('.')[0])
  f = open('evaluations/%s.txt' %ts, 'w')
  f.write(str(preds))
  pneuModel.save('models/model-' + ts + '.h5')
  # pf.write('Test Accuracy = ' + str(preds[1]) + '\n')
  # pf.write('preds : \n')
  # pf.write('\t\t' + preds)


model_file = None
if (len(sys.argv) > 1):
  model_file = sys.argv[1]

run(model_file)
