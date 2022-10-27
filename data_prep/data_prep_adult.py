"""
Python script to prepare Adult Income dataset from UCI Repository. The script downloads data, 
cleans missing values, renames labels and custom preprocesses some columns in the data. The 
data is split into training and testing data and is saved both in raw and one-hot encoded forms.
"""
from data_prep import download, clean_missing, rename_label, reduce_categories
import numpy as np
import pandas as pd
import os 

np.random.seed(123)

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/"
download_to = 'data/'
files = ['adult.data', 'adult.test']
column_names = [
	'age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status',
	'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss',
	'hours_per_week', 'native_country', 'label'
]
column_names_categorical = [
	'workclass', 'education', 'marital_status', 'occupation',
	'relationship', 'race', 'sex', 'native_country', 'label'
]

# Custom Pre-Processing for preparing Adult Income dataset:
# 1) Simplifies the categories of marital status, education, native-country.
# 2) Drops functional-weight, education-number columns 
def custom_preprocessing(df):
	df = reduce_categories(df, 'marital_status',
		['Divorced', 'Married_AF_spouse', 'Married_civ_spouse',
			'Married_spouse_absent', 'Never_married','Separated','Widowed'],
		['divorced','married','married', 
			'married', 'not_married','not_married','not_married'])

	df = reduce_categories(df, 'education',
		['Bachelors', 'Some_college', '11th', 'HS_grad',
						'Prof_school', 'Assoc_acdm', 'Assoc_voc', '9th',
						'7th_8th', '12th', 'Masters', '1st_4th', '10th',
						'Doctorate', '5th_6th', 'Preschool'],
		['Bachelors', 'Some_college', 'School', 'HS_grad',
						'Prof_school', 'Voc', 'Voc', 'School',
						'School', 'School', 'Masters', 'School', 'School',
						'Doctorate', 'School', 'School'])

	country_list = list(set(df['native_country']))
	df = reduce_categories(df, 'native_country',
		country_list, [c if c == 'United_States' else 'Other' for c in country_list])
	
	df = df.drop(columns = ['fnlwgt', 'education_num'])
	
	return df


download(url, download_to, files)

# For test data, remove the first row and remove the extra full stop at the end of each row.
with open(download_to + 'adult.test', 'r') as fin:
	data = fin.read().splitlines(True)
data = [row_text[:-2] + '\n' for row_text in data]
with open(download_to + 'adult.test', 'w') as fout:
	fout.writelines(data[1:])

df_train = clean_missing(download_to + 'adult.data', column_names)
df_train = rename_label(df_train, 'label', ["<=50K", ">50K"], ["0", "1"])
df_train = custom_preprocessing(df_train)
if not os.path.exists('data/adult'):
    os.makedirs('data/adult')
df_train.to_csv('data/adult/train_raw.csv', index = False)
df_train = pd.get_dummies(data=df_train, columns=column_names_categorical)
df_train = df_train.drop(columns = ['label_0'])
df_train.to_csv('data/adult/train.csv', index = False)


df_test = clean_missing(download_to + 'adult.test', column_names)
df_test = rename_label(df_test, 'label', ["<=50K", ">50K"], ["0", "1"])
df_test = custom_preprocessing(df_test)
df_test.to_csv('data/adult/test_raw.csv', index = False)
df_test = pd.get_dummies(data=df_test, columns=column_names_categorical)
df_test = df_test.drop(columns = ['label_0'])
df_test.to_csv('data/adult/test.csv', index = False)

# remove redundant files
os.remove('data/adult.test')
os.remove('data/adult.data')