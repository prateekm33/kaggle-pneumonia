import numpy as np
from process_images import process_images
from model import PneuModel
from sklearn.model_selection import train_test_split

# Debug: True for binary_classification only. False for classification w/bb's
bclass = False

# Meta-variables
test_size = 0.3 # proportion of overall training set
sample_size = 100 # number of dps to set as size for original training set
epochs = 5
batch_size = 16
loss = "mean_squared_logarithmic_error"
if bclass == True:
  loss = "binary_crossentropy"
optimizer = "adam"

def main():
  # Process images
  train_X_orig, train_Y_orig = process_images('stage_1_train_images', 'stage_1_train_labels.csv', sample_size=sample_size, bclass=bclass)
  train_X, test_X, train_Y, test_Y = train_test_split(train_X_orig, train_Y_orig, test_size=0.3)
  print("size of training set X: ", train_X.shape)
  print("size of testing set X: ", test_X.shape)
  print("size of training set Y: ", train_Y.shape)
  print("size of testing set Y: ", test_Y.shape)

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

main()