import sys
import numpy as np
from process_images import process_images
from model import PneuModel
from sklearn.model_selection import train_test_split
# from load_data import load_data
from get_partition import get_partition, get_labels
from data_generator import DataGenerator

# Debug variables
bclass = False # True for binary_classification only. False for classification w/bb's
force = False # True to re-run preprocessing step and not to use stored data. False to use stored data
if len(sys.argv) > 1:
  if sys.argv[1] == 'true':
    force = True
  elif sys.argv[1] == 'false':
    force = False

# Meta variables
test_size = 0.3 # proportion of overall training set
sample_size = None  # number of dps to set as size for original training set
batched_sample_size = 50 # number of dps from sample_size to run at once
epochs = 5
batch_size = 16
loss = "mean_squared_logarithmic_error"
if bclass == True:
  loss = "binary_crossentropy"
optimizer = "adam"

def main(force=False):
  # Process images
  # if force:
  #   train_X_orig, train_Y_orig = process_images('stage_1_train_images', 'stage_1_train_labels.csv', sample_size=sample_size, bclass=bclass)
  # else:
  #   train_X_orig, train_Y_orig = load_data()

  train_X, test_X, train_Y, test_Y = train_test_split(train_X_orig, train_Y_orig[:, 1:], test_size=0.3)
  print("size of training set X: ", train_X.shape)
  print("size of testing set X: ", test_X.shape)
  print("size of training set Y: ", train_Y.shape)
  print("size of testing set Y: ", test_Y.shape)
  
  print("transposed : ", train_X.T.shape, train_Y.T.shape)
  
  # Create model
  pneuModel = PneuModel(train_X.shape[1:], bclass=bclass)

  # # Compile model
  pneuModel.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

  # Train model
  pneuModel.fit(x=train_X, y=train_Y, epochs=epochs, batch_size=batch_size)

  # Test model
  preds = pneuModel.evaluate(x=test_X, y=test_Y)

  print()
  print ("Loss = " + str(preds[0]))
  print ("Test Accuracy = " + str(preds[1]))




def run():
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

  # Create model
  pneuModel = PneuModel(params['dim'], bclass=bclass)

  # Compile model
  pneuModel.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

  # Train model on dataset
  history_cb = pneuModel.fit_generator(generator=training_generator,
                      validation_data=validation_generator,
                      use_multiprocessing=True,
                      workers=6)
  np.savetxt("logs.txt", history_cb, delimiter="\n")  


  # Test model
  test_label_ids = partition['test']
  test_Y = np.zeros((len(test_label_ids, 5)))
  test_X = np.zeros((len(test_label_ids, 600, 600, 1)))
  for i in range(len(test_label_ids)):
    _id = test_label_ids[i]
    test_Y[i] = labels[_id]
    test_X[i] = np.load('processed_train_images/' + _id + '.npz').image

  preds = pneuModel.evaluate(x=test_X, y=test_Y)

  print()
  print ("Loss = " + str(preds[0]))
  print("Test Accuracy = " + str(preds[1]))
  
run()
