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
from tensorflow.keras.callbacks import CSVLogger

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
  partition = get_partition('stage_1_train_labels.csv', 1000) # IDs
  labels = get_labels() # Labels

  # Generators
  training_generator = DataGenerator(partition['train'], labels, **params)
  validation_generator = DataGenerator(partition['validation'], labels, **params)

  # Train model on dataset
  callbacks = [Logger(), CSVLogger('training.log')]
  pneuModel.fit_generator(generator=training_generator,
                      validation_data=validation_generator,
                      use_multiprocessing=True,
                      workers=6,
                      callbacks=callbacks
                      )
  ts = '.'.join(str(datetime.datetime.now()).split('.'))
  pneuModel.save('models/model-' + ts + '.h5')

  # Test model
  evaluation_generator = DataGenerator(partition['test'], labels, **params)
  preds = pneuModel.evaluate_generator(generator=evaluation_generator, use_multiprocessing=True, workers=6)
  f = open('evaluations/model-%s.txt' %ts, 'w')
  f.write(str(preds))
  print(preds)

model_file = None
if (len(sys.argv) > 1):
  model_file = sys.argv[1]

main(model_file)
