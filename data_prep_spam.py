from data_prep import *
from sklearn.utils import shuffle
import numpy as np

np.random.seed(123)

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/"
download_to = 'data/'
files = ['spambase.data']
column_names = [
"make", "address", "all", "3d", "our", "over", "remove", "internet",
"order", "mail", "receive", "will", "people", "report", "addresses",
"free", "business", "email", "you", "credit", "your", "font",
"000", "money", "hp", "hpl", "george", "650", "lab", "labs",
"telnet", "857", "data", "415", "85", "technology", "1999",
"parts", "pm", "direct", "cs", "meeting", "original", "project", "re", "edu", "table",
"conference", "punc;", "punc(", "punc[", "punc!", "punc$", "punc#", "length_average", "length_longest",
"length_total", "label"
]
column_names_categorical = ['label']

download(url, download_to, files)

df = clean_missing(download_to + 'spambase.data', column_names, sep = ",")
df = shuffle(df)
df = rename_label(df, 'label', ["0", "1"], ["0", "1"])

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


