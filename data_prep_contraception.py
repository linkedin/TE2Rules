from data_prep import *
from sklearn.utils import shuffle
import numpy as np

np.random.seed(123)

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/cmc/"
download_to = 'data/'
files = ['cmc.data']
column_names = [
'age',
'wife_education',
'husband_education',
'num_children',
'wife_religion',
'wife_working',
'husband_occupation',
'standard_of_living',
'media_exposure', 
'label',
]
column_names_categorical = [
'wife_education','husband_education','wife_religion',
'wife_working','husband_occupation','standard_of_living','media_exposure','label'
]

download(url, download_to, files)

df = clean_missing(download_to + 'cmc.data', column_names, sep = ",")
df = shuffle(df)
df = rename_label(df, 'label', [1, 2, 3], [0, 1, 1])


num_rows_train = int(0.8*len(df))

df_train = df[:num_rows_train]
df_test = df[num_rows_train:]
df_train.to_csv('data/train_raw.csv', index = False)
df_test.to_csv('data/test_raw.csv', index = False)

df = pd.get_dummies(data=df, columns=column_names_categorical)
df = df.drop('label_0', 1)

df_train = df[:num_rows_train]
df_test = df[num_rows_train:]
df_train.to_csv('data/train.csv', index = False)
df_test.to_csv('data/test.csv', index = False)


