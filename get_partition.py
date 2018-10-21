import pandas as pd
import numpy as np
from os import listdir

'''
get all IDs -- read from csv
shuffle/rand perm
'''

def get_partition(labels_csv):
  IDs = np.array(pd.read_csv(
    labels_csv,
    dtype={'patientId': str, 'x': np.float32, 'y': np.float32, 'width': np.float32, 'height': np.float32, 'Target': np.float32},
    engine="python",
    usecols=['patientId']
  ).values)
  np.random.shuffle(IDs)
  partition_1 = int(np.floor(len(IDs) * 0.6))
  partition_2 = int(np.floor(len(IDs) * 0.8))
  print(partition_1, partition_2)
  return {
    'train': IDs[:partition_1,:],
    'validation': IDs[partition_1:partition_2,:],
    'test': IDs[:partition_2,:]
  }
  
def get_labels():
  labels = [np.load('processed_train_labels/' + f)['label'] for f in listdir('processed_train_labels')]

  labels_dict = {}
  for i in range(0, len(labels)):
    labels_dict[labels[i][0]] = labels[i, 1:]
  return labels_dict
