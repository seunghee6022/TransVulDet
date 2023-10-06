# Assuming you have three datasets named 'dataset_A.csv', 'dataset_B.csv', and 'dataset_C.csv'
import pandas as pd
from sklearn.model_selection import train_test_split
import os
# import pickle

print(os.getcwd())

data_path = './datasets_'
MSR_df = pd.read_csv(f'{data_path}/MSR.csv')
MVD_df = pd.read_csv(f'{data_path}/MVD.csv')
# CVEfixes_df = pd.read_csv(f'{data_path}/CVEfixes_labeled.csv')

print(MSR_df.head(3))
print(MVD_df.head(3))
# print(CVEfixes_df.head(3))

#Add dataset_name column to each dataframe
# MSR_df['dataset_name'] = 'MSR'
# MVD_df['dataset_name'] = 'MVD'
# CVEfixes_df['dataset_name'] = 'CVEfixes'

# Remove the 'Unnamed: 0' column
MSR_df = MSR_df.drop('Unnamed: 0', axis=1)
MVD_df = MVD_df.drop('Unnamed: 0', axis=1)

print("MSR_df columns\n",MSR_df.columns)
print("MVD_df columns\n",MVD_df.columns)
# print("CVEfixes_df columns\n",CVEfixes_df.columns)

# Concatenate the datasets
# concatenated_df = pd.concat([MSR_df, MVD_df, CVEfixes_df])
concatenated_df = pd.concat([MSR_df, MVD_df])
print(f"concatenated_df: {concatenated_df.shape}\n{concatenated_df.head(3)}")

# Assuming 'concatenated_df' is your DataFrame
rows_with_na = concatenated_df[concatenated_df.isna().any(axis=1)]
print("rows_with_na: \n", rows_with_na.head(3) )

for index, row in rows_with_na.iterrows():
    print(f"NA values found in dataset_name {row['dataset_name']}  cwe_id {row['cwe_id']} row {index}:")
    if pd.isna(row['cwe_id']):
        print(f"NA case: {row['cwe_id']} --- Before {row['cwe_id']}\n{row}")
        # Handle the case where the value is 'nan' to non-vulnerable class:0
        concatenated_df.loc[index, 'cwe_id'] = 0
        concatenated_df.loc[index, 'vul']= 0
        print("After",concatenated_df.loc[index])
    else:
        concatenated_df.loc[index, 'vul']= 1


rows_with_na = concatenated_df[concatenated_df.isna().any(axis=1)]
if rows_with_na.shape[0] > 0:
     print("Successfully preprocessed NA - After relabeling rows_with_na so the # of rows should be 0")
else:
    print(f"After relabing rows_with_na: size: {rows_with_na.shape}")
    for index, row in rows_with_na.iterrows():
        print(f"NA values found in dataset_name {row['dataset_name']}  cwe_id {row['cwe_id']} row {index}:")
        for column, value in row.iteritems():
            if pd.isna(value):
                print(f"   - NA in column '{column}': value {value} ")
                        

# Set 'label' and 'vul' data type to int
concatenated_df['vul'] = concatenated_df['vul'].astype(int)
print(f"concatenated_df: {concatenated_df.shape}\n{concatenated_df.head(3)}")

# Split the data into train, validation, and test sets
train_df, test_df = train_test_split(concatenated_df, test_size=0.2, random_state=42)
train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)

print("test_df",test_df.head(3))

# Optionally, you can save the train, validation, and test sets to separate CSV files
train_df.to_csv(f'{data_path}/train_data.csv', index=False)
val_df.to_csv(f'{data_path}/val_data.csv', index=False)
test_df.to_csv(f'{data_path}/test_data.csv', index=False)
