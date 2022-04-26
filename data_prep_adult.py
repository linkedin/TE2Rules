from data_prep import *
import numpy as np

np.random.seed(123)

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/"
download_to = 'data/'
files = ['adult.data', 'adult.test']
column_names = [
	'age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 
	'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
	'hours-per-week', 'native-country', 'income'
]

def custom_preprocessing(df):
	df = reduce_categories(df, 'marital-status',
		['Divorced', 'Married-AF-spouse', 'Married-civ-spouse', 
			'Married-spouse-absent', 'Never-married','Separated','Widowed'],
		['divorced','married','married', 
			'married', 'not married','not married','not married'])

	df = reduce_categories(df, 'education',
		['Bachelors', 'Some-college', '11th', 'HS-grad', 
						'Prof-school', 'Assoc-acdm', 'Assoc-voc', '9th', 
						'7th-8th', '12th', 'Masters', '1st-4th', '10th', 
						'Doctorate', '5th-6th', 'Preschool'],
		['Bachelors', 'Some-college', 'School', 'HS-grad', 
						'Prof-school', 'Voc', 'Voc', 'School', 
						'School', 'School', 'Masters', 'School', 'School', 
						'Doctorate', 'School', 'School'])

	country_list = list(set(df['native-country']))
	df = reduce_categories(df, 'native-country', 
		country_list, [c if c == 'United-States' else 'Other' for c in country_list])
	
	df = df.drop('fnlwgt',1)
	df = df.drop('education-num',1)
	
	return df


download(url, download_to, files)

# For test data, remove the first row and remove the extra full stop at the end of each row.
with open(download_to + 'adult.test', 'r') as fin:
	data = fin.read().splitlines(True)
data = [row_text[:-2] + '\n' for row_text in data]
with open(download_to + 'adult.test', 'w') as fout:
	fout.writelines(data[1:])

df_train = clean_missing(download_to + 'adult.data', column_names)
df_train = rename_label(df_train, 'income', ["<=50K", ">50K"], ["0", "1"])
df_train = custom_preprocessing(df_train)
df_train.to_csv('data/train_raw.csv', index = False)
df_train = pd.get_dummies(data=df_train, columns=[
	'workclass', 'education', 'marital-status', 'occupation', 
	'relationship', 'race', 'sex', 'native-country', 'label'
	])
df_train = df_train.drop('label_0', 1)
feature_names = [x.replace('-', '_') for x in df_train.columns]
feature_names[feature_names.index('marital_status_not married')] = 'marital_status_not_married'
df_train.columns = feature_names
df_train.to_csv('data/train.csv', index = False)


df_test = clean_missing(download_to + 'adult.test', column_names)
df_test = rename_label(df_test, 'income', ["<=50K", ">50K"], ["0", "1"])
df_test = custom_preprocessing(df_test)
df_test.to_csv('data/test_raw.csv', index = False)
df_test = pd.get_dummies(data=df_test, columns=[
	'workclass', 'education', 'marital-status', 'occupation', 
	'relationship', 'race', 'sex', 'native-country', 'label'
	])
df_test = df_test.drop('label_0', 1)
df_test.columns = feature_names
df_test.to_csv('data/test.csv', index = False)
