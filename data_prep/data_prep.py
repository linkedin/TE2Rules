"""

"""
import bs4
import requests
import os
import pandas as pd
import numpy as np

def download(url, download_to, files):
	if(not os.path.isdir(download_to)):
		os.mkdir(download_to)
	r = requests.get(url)
	data = bs4.BeautifulSoup(r.text, "html.parser")
	for f in files:
		r = requests.get(url + f)
		with open(download_to + f, 'w') as f_handle:
			f_handle.write(r.text)

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

def reduce_categories(df, col_name, categories, reduced_categories):
	df[col_name].replace(categories, reduced_categories, inplace = True)
	assert(len(df[col_name].value_counts()) == len(set(reduced_categories)))
	return df

def rename_label(df, label_col, labels, binary_labels):
	column_names = df.columns
	df = df[[c for c in column_names if c != label_col] + [label_col]]
	
	df.rename(columns = {label_col:'label'}, inplace = True)
	df['label'].replace(labels, binary_labels, inplace = True)
	assert(len(df['label'].value_counts()) == len(set(binary_labels)))
	return df





