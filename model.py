import numpy as np
from tensorflow.keras import layers, regularizers, losses
from tensorflow.keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.models import Model, Sequential
import tensorflow as tf

def custom_loss(y_true, y_preds):
  # y_mask = y_true[:, -1] == 0
  # y_temp = np.zeros((y_true.shape[0], y_true.shape[1] - 1))
  # y_temp = y_preds * [y_temp, y_mask]
  
  y_temp = y_preds * y_true[:, -1]
  y_temp = tf.stack([y_temp[:,:-1], y_preds[:, -1]], axis=1)
  return losses.mean_squared_error(y_true, y_temp)  


def PneuModel(input_shape, bclass=False, reg_lambda=0.01):
  print('input shape : ', input_shape)

  # This returns a tensor
  X_input = Input(shape=input_shape)

  X = Conv2D(32, (7, 7), strides=(1, 1), name='conv0')(X_input)
  # 449 --> 443x443x32
  X = Conv2D(32, (7, 7), strides=(2, 2), name='conv1')(X)
  # 443 --> 219x219x32
  X = MaxPooling2D((3,3), strides=(2,2), name='pool0')(X)
  # 219 --> 109x109x32
  X = Conv2D(64, (7,7), strides=(1, 1), name='conv2')(X)
  # 109 --> 103x103x64
  X = ZeroPadding2D((1, 1))(X)
  X = Conv2D(100, (3, 3), name='conv3')(X)
  # 103 --> 103x103x100
  X = MaxPooling2D((3,3), strides=(1,1), name='pool1')(X)
  # 103 --> 101x101x100

  X = Conv2D(256, (7, 7), strides=(2, 2), name='conv3')(X)
  # 101 --> 47x47x256

  X = Flatten()(X)
  X = Dense(282752, activation='linear', name='fc0')(X)
  X = Dense(5, activation='linear', name='fc1')(X)

  model = Model(inputs = X_input, outputs = X, name='PneuModel')
  
  # activity_regularizer=regularizers.l2(reg_lambda)
  # for i in range(3):
  #   X = Conv2D(32, (5, 5), strides=(1,1), name = 'conv'+str(i))(X)
  #   # X = BatchNormalization(axis = 3, name = 'bn'+str(i))(X)
  #   # X = Activation('relu')(X)
  #   if i > 0 and i % 5 == 0:
  #     X = MaxPooling2D((2, 2), name='max_pool'+str(i))(X)
  # print('X shape before pooling : ', X.shape) 
  # X = MaxPooling2D((2, 2), name='max_pool-final')(X)
  # print('X shape after pooling : ', X.shape)
  # # FLATTEN X (means convert it to a vector) + FULLYCONNECTED
  # num_output_layers = 5
  # if bclass == True:
  #   num_output_layers = 1

  # X = Flatten()(X)
  # print('X shape after flattening : ', X.shape)
  # X = Dense(num_output_layers, activation='linear', name='fc')(X)
  
  # # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
  # model = Model(inputs = X_input, outputs = X, name='PneuModel')

  return model

def create_model(dims=(200,200,1), optimizer='adam', loss='mean_squared_error', metrics=["accuracy"], bclass=False):
  # Create model
  pneuModel = PneuModel(dims, bclass=bclass)
  # Compile model
  print(custom_loss, 'loss fn')
  pneuModel.compile(optimizer=optimizer, loss=loss, metrics=metrics)
  return pneuModel

