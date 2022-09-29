"""
This file contains various utility functions called by data_prep scripts.
"""
import requests
import os
import pandas as pd
import numpy as np

# Downloads given files from the url to the given downloads folder.
def download(url, download_to, files):
	if(not os.path.isdir(download_to)):
		os.mkdir(download_to)
	r = requests.get(url)
	for f in files:
		r = requests.get(url + f)
		with open(os.path.join(download_to, f), 'wb') as f_handle:
			f_handle.write(r.content)

# Cleans missing values in the columns in the file by replacing 
# missing values with most frequent value in the column. For any 
# values with hyphen, the hyphen is replaced with underscore. 
def clean_missing(filename, column_names, sep = ", "):
	df = pd.read_csv(filename, sep = sep, engine = "python")
	df.columns = column_names
	
	# Replace missing value with most frequent
	for col in df.columns:
		df[col] = df[col].replace("?", np.NaN)
	df = df.apply(lambda x:x.fillna(x.value_counts().index[0]))

	# Replace hyphens with underscores
	df = df.replace('-','_', regex=True)
	return df

# For a given column from a dataframe with categorical values, 
# replaces the list of category values with an equal length list of 
# reduced_category values. The two lists are used to map original
# category values to alternate category values, preferrably to a 
# smaller set of category values.
def reduce_categories(df, col_name, categories, reduced_categories):
	df[col_name].replace(categories, reduced_categories, inplace = True)
	assert(len(df[col_name].value_counts()) == len(set(reduced_categories)))
	return df

# For a given label column from a dataframe, rename the label
# column as "label" and replace the list of label values with an equal length 
# list of binary labels. The two lists are used to map original
# label values to alternate label values, preferrably to a 
# smaller set of binary label values.
def rename_label(df, label_col, labels, binary_labels):
	column_names = df.columns
	df = df[[c for c in column_names if c != label_col] + [label_col]]
	
	df.rename(columns = {label_col:'label'}, inplace = True)
	df['label'].replace(labels, binary_labels, inplace = True)
	assert(len(df['label'].value_counts()) == len(set(binary_labels)))
	return df
