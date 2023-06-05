import pandas as pd
import numpy as np
import pickle
import os

print(os.getcwd())

data_dir = 'data/CVEfixes.csv'

if os.path.exists(data_dir):
    print("Directory exists")
else:
    print("Directory does not exist")

df = pd.read_csv(data_dir)

print("# of total rows: ",df.shape[0])
print(df.columns)
print(df.head(5))

nan_count = df['cwe_id'].isnull().sum()
print("Number of NaN values in 'cwe_id':", nan_count)

# convert values None/NaN to 'non-vulnerable' in cwe_id column
df['cwe_id'] = df['cwe_id'].fillna('non-vulnerable')

cwe_id_counts = df['cwe_id'].value_counts()

print("cwe_id counts:")
print(cwe_id_counts)

# drop rows based on exception_id_list
exception_id_list = ['NVD-CWE-Other', 'NVD-CWE-noinfo']

# Assuming you have a DataFrame named 'df' and a column named 'cwe_id'
for exception_id in exception_id_list:
    count = len(df[df['cwe_id'] == exception_id])
    print("Count of", exception_id, ":", count)

df = df[~df['cwe_id'].isin(exception_id_list)]

unique_values = df['cwe_id'].unique()
print(unique_values)

# load dict to map the unique values to integer indices
with open("preprocessed_datasets/total_cwe_dict.txt", "rb") as myFile:
    total_cwe_dict = pickle.load(myFile)

df['label'] = df['cwe_id'].map(total_cwe_dict)
df['vul'] = df['cwe_id'].apply(lambda x: 0 if x == 'non-vulnerable' else 1)

label_counts = df['label'].value_counts()
vul_counts = df['vul'].value_counts()

print("Label counts:")
print(label_counts)

print("\nVul counts:")
print(vul_counts)

print("# of total rows: ",df.shape[0])
print(df.columns)
print(df.head(5))

# Save to CSV, rows separated by ""
df.to_csv('preprocessed_datasets/CVEfixes_labeled.csv')


