"""
Python script to prepare Bank dataset from UCI Repository. The script downloads data,
cleans missing values and renames labels. The data is split into training and testing data
and is saved both in raw and one-hot encoded forms.
"""
from data_prep import download, clean_missing, rename_label
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import os
from zipfile import ZipFile

np.random.seed(123)

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/"
download_to = 'data/'
files = ['bank.zip']
column_names = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing',
                'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays',
                'previous', 'poutcome', 
                'y']
column_names_categorical = ['job', 'marital', 'education', 'default', 'housing',
                'loan', 'contact', 'month', 'poutcome',
                'label']

download(url, download_to, files)
with ZipFile(os.path.join(download_to, 'bank.zip'), 'r') as zip_ref:
    zip_ref.extractall(download_to)

df = clean_missing(download_to + 'bank-full.csv', column_names, sep = ";")
df = df.replace("admin.", "admin")
df = shuffle(df)
df = rename_label(df, 'y', ['yes', 'no'], ['1', '0'])

num_rows_train = int(0.8*len(df))

df_train = df[:num_rows_train]
df_test = df[num_rows_train:]
if not os.path.exists('data/bank'):
    os.makedirs('data/bank')
df_train.to_csv('data/bank/train_raw.csv', index = False)
df_test.to_csv('data/bank/test_raw.csv', index = False)

df = pd.get_dummies(data=df, columns=column_names_categorical)
df = df.drop(columns = ['label_0'])

df_train = df[:num_rows_train]
df_test = df[num_rows_train:]
df_train.to_csv('data/bank/train.csv', index = False)
df_test.to_csv('data/bank/test.csv', index = False)

os.remove('data/bank.zip')
os.remove('data/bank-full.csv')
os.remove('data/bank.csv')
os.remove('data/bank-names.txt')
