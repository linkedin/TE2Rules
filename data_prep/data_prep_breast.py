"""
Python script to prepare Breast Cancer dataset from UCI Repository. The script downloads data, 
cleans missing values and renames labels. The data is split into training and testing data 
and is saved both in raw and one-hot encoded forms.
"""
from data_prep import download, clean_missing, rename_label
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import os 

np.random.seed(123)

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer/"
download_to = 'data/'
files = ['breast-cancer.data']
column_names = [
'label',
'age',
'menopause',
'tumor_size',
'inv_nodes',
'node_caps',
'deg_malig',
'breast',
'breast_quad',
'irradiat',
]
column_names_categorical = [
'age',
'menopause',
'tumor_size',
'inv_nodes',
'node_caps',
'deg_malig',
'breast',
'breast_quad',
'irradiat',
'label'
]

download(url, download_to, files)

df = clean_missing(download_to + 'breast-cancer.data', column_names, sep = ",")
df = shuffle(df)
df = rename_label(df, 'label', ['no_recurrence_events', 'recurrence_events'], ['0', '1'])

num_rows_train = int(0.8*len(df))

df_train = df[:num_rows_train]
df_test = df[num_rows_train:]
if not os.path.exists('data/breast'):
    os.makedirs('data/breast')
df_train.to_csv('data/breast/train_raw.csv', index = False)
df_test.to_csv('data/breast/test_raw.csv', index = False)

df = pd.get_dummies(data=df, columns=column_names_categorical)
df = df.drop(columns = ['label_0'])

df_train = df[:num_rows_train]
df_test = df[num_rows_train:]
df_train.to_csv('data/breast/train.csv', index = False)
df_test.to_csv('data/breast/test.csv', index = False)

os.remove('data/breast-cancer.data')
