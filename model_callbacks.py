from tensorflow import keras
import numpy as np

class Logger(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        return
 
    def on_train_end(self, logs={}):
        return
 
    def on_epoch_begin(self, epoch, logs={}):
        log_file = open('logs.txt', 'a')
        log_file.write('Epoch %d start' %epoch + '\n')
        log_file.write('Epoch logs: ' + str(logs) + '\n')
        log_file.write('----------------- \n')
        return
 
    def on_epoch_end(self, epoch, logs={}):
        log_file = open('logs.txt', 'a')
        log_file.write('Epoch %d end' %epoch + '\n')
        log_file.write('Epoch logs: ' + str(logs) + '\n')
        log_file.write('----------------- \n')
        return
 
    def on_batch_begin(self, batch, logs={}):
        return
 
    def on_batch_end(self, batch, logs={}):
        log_file = open('logs.txt', 'a')
        log_file.write('\t batch : ' + str(batch) + '\n')
        log_file.write('\t logs : ' + str(logs) + '\n')
        log_file.write('----------------- \n')
        return

