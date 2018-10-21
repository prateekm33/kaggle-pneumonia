import numpy as np
import pandas as pd

# def read_labels(sample_size):
labels = pd.read_csv('stage_1_train_labels.csv', dtype={'patientId': str, 'x': np.float32, 'y': np.float32, 'width': np.float32, 'height': np.float32, 'Target': np.float32}, engine="python", nrows=6).fillna(0).values
print(labels[5,3]) 