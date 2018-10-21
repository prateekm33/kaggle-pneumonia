from tensorflow import keras
import numpy as np

class Logger(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        return
 
    def on_train_end(self, logs={}):
        return
 
    def on_epoch_begin(self, logs={}):
        return
 
    def on_epoch_end(self, epoch, logs={}):
        return
 
    def on_batch_begin(self, batch, logs={}):
        return
 
    def on_batch_end(self, batch, logs={}):
        logs = open('logs.txt', 'a')
        logs.write('batch : \n')
        logs.write(batch)
        logs.write('\n')
        logs.write('logs : \n')
        logs.write(logs)
        logs.write('\n')
        return
