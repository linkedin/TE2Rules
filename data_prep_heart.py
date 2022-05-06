from data_prep import *
from sklearn.utils import shuffle
import numpy as np

np.random.seed(123)

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/spect/"
download_to = 'data/'
files = ['SPECT.test', 'SPECT.train']
column_names = [
'label',
'F1',
'F2',
'F3',
'F4',
'F5',
'F6',
'F7',
'F8',
'F9',
'F10',
'F11',
'F12',
'F13',
'F14',
'F15',
'F16',
'F17',
'F18',
'F19',
'F20',
'F21',
'F22',
]
column_names_categorical = [
'F1',
'F2',
'F3',
'F4',
'F5',
'F6',
'F7',
'F8',
'F9',
'F10',
'F11',
'F12',
'F13',
'F14',
'F15',
'F16',
'F17',
'F18',
'F19',
'F20',
'F21',
'F22',
'label'
]

download(url, download_to, files)

df = clean_missing(download_to + 'SPECT.train', column_names, sep = ",")
df = shuffle(df)
df = rename_label(df, 'label', ['0', '1'], ['0', '1'])
df.to_csv('data/train_raw.csv', index = False)
df = pd.get_dummies(data=df, columns=column_names_categorical)
df = df.drop('label_0', 1)
df.to_csv('data/train.csv', index = False)

df = clean_missing(download_to + 'SPECT.test', column_names, sep = ",")
df = shuffle(df)
df = rename_label(df, 'label', ['0', '1'], ['0', '1'])
df.to_csv('data/test_raw.csv', index = False)
df = pd.get_dummies(data=df, columns=column_names_categorical)
df = df.drop('label_0', 1)
df.to_csv('data/test.csv', index = False)


