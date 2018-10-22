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

  # CONV -> BN -> RELU Block applied to X
  X = Conv2D(32, (5, 5), strides=(1,1), name = 'conv0')(X)
  X = BatchNormalization(axis = 3, name = 'bn0')(X)
  X = Activation('relu')(X)

  # CONV -> BN -> RELU Block applied to X
  X = Conv2D(32, (7, 7), strides=(1,1), name = 'conv1', activity_regularizer=regularizers.l2(reg_lambda))(X)
  X = BatchNormalization(axis = 3, name = 'bn1')(X)
  X = Activation('relu')(X)
  
  # MAXPOOL
  X = MaxPooling2D((2, 2), name='max_pool')(X)
  
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
  
  return pneuModel

