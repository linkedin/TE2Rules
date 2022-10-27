"""
Python script to prepare Compas dataset from ProPublica. The script downloads data,
cleans missing values and renames labels. The data is split into training and testing data
and is saved both in raw and one-hot encoded forms.
"""
from data_prep import download, clean_missing, rename_label
from sklearn.utils import shuffle
import numpy as np
import pandas as pd
import os

np.random.seed(123)

url = "https://raw.githubusercontent.com/propublica/compas-analysis/master/"
download_to = "data/"
files = ["compas-scores-two-years.csv"]
column_names = [
    "c_charge_degree",
    "race",
    "age",
    "sex",
    "priors_count",
    "days_b_screening_arrest",
    "two_year_recid",
]
column_names_categorical = ["c_charge_degree", "race", "sex", "label"]

download(url, download_to, files)

# Custom Preprocessing
df = pd.read_csv(os.path.join(download_to, "compas-scores-two-years.csv"))
df = (
    df.loc[(df["days_b_screening_arrest"] <= 30) & (df["days_b_screening_arrest"] >= -30),:,]
    .loc[df["is_recid"] != -1, :]
    .loc[df["c_charge_degree"] != "O", :]
    .loc[df["score_text"] != "N/A", :]
)
df = df[column_names]
df.to_csv(os.path.join(download_to, 'compas-scores-two-years.csv'), index=False)

df = clean_missing(download_to + "compas-scores-two-years.csv", column_names, sep=",")
df = shuffle(df)
df = rename_label(df, "two_year_recid", ["0", "1"], ["0", "1"])

num_rows_train = int(0.8 * len(df))

df_train = df[:num_rows_train]
df_test = df[num_rows_train:]
if not os.path.exists("data/compas"):
    os.makedirs("data/compas")
df_train.to_csv("data/compas/train_raw.csv", index=False)
df_test.to_csv("data/compas/test_raw.csv", index=False)

df = pd.get_dummies(data=df, columns=column_names_categorical)
df = df.drop(columns=["label_0"])
df.columns = [x.replace("-", "_").replace(" ", "_") for x in df.columns]

df_train = df[:num_rows_train]
df_test = df[num_rows_train:]
df_train.to_csv("data/compas/train.csv", index=False)
df_test.to_csv("data/compas/test.csv", index=False)

os.remove('data/compas-scores-two-years.csv')

