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
	return df

def create_bins(x, low, high):
	if(x < low):
		return 'low'
	elif(x > high):
		return 'high'
	else:
		return 'mid'

def discretize_continuous_columns(df, continuous_cols):
	for key in continuous_cols:	
		values = df[key]
		values_min   = min(values)
		values_max   = max(values)
		# Check skew
		values_min_count   = 0
		values_max_count   = 0
		for v in values:
			if(v == values_min):
				values_min_count = values_min_count + 1
			if(v == values_max):
				values_max_count = values_max_count + 1
		if(values_min_count > len(values)/3):
			values_modified = values[values > values_min]
			values_low   = np.percentile(values_modified, 33)
			values_high  = np.percentile(values_modified, 67)
		elif(values_max_count > len(values)/3):
			values_modified = values[values < values_max]
			values_low   = np.percentile(values_modified, 33)
			values_high  = np.percentile(values_modified, 67)
		else:
			values_low   = 0.67*values_min + 0.33*values_max
			values_high  = 0.33*values_min + 0.67*values_max
		df[key] = values.apply(lambda x: create_bins(x, values_low, values_high))
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

def tensor_transform(df):
	df = pd.get_dummies(data=df, columns=df.columns)
	df = df.drop('label_0', 1)
	return df





