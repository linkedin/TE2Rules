"""

"""
from data_prep import *
from sklearn.utils import shuffle
import numpy as np

np.random.seed(123)

dataURL = 'https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv'
dfRaw = pd.read_csv(dataURL)
dfFiltered = (dfRaw[['age', 'c_charge_degree', 'race', 'age_cat', 'score_text', 
             'sex', 'priors_count', 'days_b_screening_arrest', 'decile_score', 
             'is_recid', 'two_year_recid', 'c_jail_in', 'c_jail_out']]
             .loc[(dfRaw['days_b_screening_arrest'] <= 30) & (dfRaw['days_b_screening_arrest'] >= -30), :]
             .loc[dfRaw['is_recid'] != -1, :]
             .loc[dfRaw['c_charge_degree'] != 'O', :]
             .loc[dfRaw['score_text'] != 'N/A', :]
             )

dfFiltered['label'] = dfFiltered['two_year_recid']

column_names = ['c_charge_degree', 'race', 'age', 'sex', 'priors_count', 
                'days_b_screening_arrest', 'label']
dfFiltered = dfFiltered[column_names]
column_names_categorical = ['c_charge_degree', 'race', 'sex', 'label']

df = dfFiltered
df = shuffle(df)

num_rows_train = int(0.8*len(df))

df_train = df[:num_rows_train]
df_test = df[num_rows_train:]
if not os.path.exists('data/compas'):
    os.makedirs('data/compas')
df_train.to_csv('data/compas/train_raw.csv', index = False)
df_test.to_csv('data/compas/test_raw.csv', index = False)

df = pd.get_dummies(data=df, columns=column_names_categorical)
df = df.drop(columns = ['label_0'])
df.columns = [x.replace('-', '_').replace(' ', '_') for x in df.columns]

df_train = df[:num_rows_train]
df_test = df[num_rows_train:]
df_train.to_csv('data/compas/train.csv', index = False)
df_test.to_csv('data/compas/test.csv', index = False)