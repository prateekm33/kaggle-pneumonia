import numpy as np
from tensorflow.keras import layers, regularizers
from tensorflow.keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.models import Model

def PneuModel(input_shape, bclass=False, reg_lambda=0.01):
  print('input shape : ', input_shape)

  # This returns a tensor
  X_input = Input(shape=input_shape)

  # Padding layer
  X = ZeroPadding2D((1, 1))(X_input)
 
  for i in range(3):
    X = Conv2D(32, (5, 5), strides=(1,1), name = 'conv'+str(i), activity_regularizer=regularizers.l2(reg_lambda))(X)
    X = BatchNormalization(axis = 3, name = 'bn'+str(i))(X)
    X = Activation('relu')(X)
    if i > 0 and i % 5 == 0:
      X = MaxPooling2D((2, 2), name='max_pool'+str(i))(X)
 
  X = MaxPooling2D((2, 2), name='max_pool-final')(X)

  # FLATTEN X (means convert it to a vector) + FULLYCONNECTED
  num_output_layers = 5
  if bclass == True:
    num_output_layers = 1

  X = Flatten()(X)
  X = Dense(num_output_layers, activation='sigmoid', name='fc')(X)
  
  # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
  model = Model(inputs = X_input, outputs = X, name='PneuModel')

  return model

def create_model(dims=(600,600,1), optimizer='adam', loss='mean_squared_logarithmic_error', metrics=["accuracy"], bclass=False):
  # Create model
  pneuModel = PneuModel(dims, bclass=bclass)
  # Compile model
  pneuModel.compile(optimizer=optimizer, loss=loss, metrics=metrics)
  print(pneuModel, pneuModel.fit_generator)  
  return pneuModel

