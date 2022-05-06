from data_prep import *
from sklearn.utils import shuffle
import numpy as np

np.random.seed(123)

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/"
download_to = 'data/'
files = ['agaricus-lepiota.data']
column_names = [
'label',
'cap_shape',
'cap_surface',
'cap_color',
'bruises',
'odor',
'gill_attachment',
'gill_spacing',
'gill_size',
'gill_color',
'stalk_shape',
'stalk_root',
'stalk_surface_above_ring',
'stalk_surface_below_ring',
'stalk_color_above_ring',
'stalk_color_below_ring',
'veil_type',
'veil_color',
'ring_number',
'ring_type',
'spore_print_color',
'population',
'habitat',
]
column_names_categorical = [
'cap_shape',
'cap_surface',
'cap_color',
'bruises',
'odor',
'gill_attachment',
'gill_spacing',
'gill_size',
'gill_color',
'stalk_shape',
'stalk_root',
'stalk_surface_above_ring',
'stalk_surface_below_ring',
'stalk_color_above_ring',
'stalk_color_below_ring',
'veil_type',
'veil_color',
'ring_number',
'ring_type',
'spore_print_color',
'population',
'habitat',
'label'
]

download(url, download_to, files)

df = clean_missing(download_to + 'agaricus-lepiota.data', column_names, sep = ",")
df = shuffle(df)
df = rename_label(df, 'label', ['p', 'e'], ["0", "1"])

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


