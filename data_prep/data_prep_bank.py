from data_prep import *
import requests
from io import BytesIO
from zipfile import ZipFile
import pandas as pd
# from imblearn.over_sampling import SMOTE

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip"

# --NotebookApp.iopub_data_rate_limit = 10000000

with ZipFile(BytesIO(requests.get(url).content), "r") as myzip:
    with myzip.open("bank-full.csv", "r") as f_in:
        df = pd.read_csv(f_in, sep=";")

df['label'] = [int(x) for x in (df['y']=='yes')]

column_names = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing',
                'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays',
                'previous', 'poutcome',
                'label']
df = df[column_names]
column_names_categorical = ['job', 'marital', 'education', 'default', 'housing',
                'loan', 'contact', 'month', 'poutcome',
                'label']

num_rows_train = int(0.8*len(df))

df_train = df[:num_rows_train]
df_test = df[num_rows_train:]
if not os.path.exists('data/bank'):
    os.makedirs('data/bank')
df_train.to_csv('data/bank/train_raw.csv', index = False)
df_test.to_csv('data/bank/test_raw.csv', index = False)

df = pd.get_dummies(data=df, columns=column_names_categorical)
df = df.drop(columns = ['label_0'])
df.columns = [x.replace('-', '_').replace(' ', '_') for x in df.columns]

df_train = df[:num_rows_train]
# X, y = df_train.iloc[:,:-1], df_train.iloc[:,-1]
# oversample = SMOTE()
# X, y = oversample.fit_resample(X, y)
# df_train = X
# df_train['label_1'] = y

df_test = df[num_rows_train:]
df_train.to_csv('data/bank/train.csv', index = False)
df_test.to_csv('data/bank/test.csv', index = False)