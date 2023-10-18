import pandas as pd
from sklearn.model_selection import train_test_split
import os

print(os.getcwd())

data_path = 'datasets_'
MSR_df = pd.read_csv(f'{data_path}/MSR.csv')
# MVD_df = pd.read_csv(f'{data_path}/MVD.csv')

CVEfixes_df_chunk = pd.read_csv(f'{data_path}/CVEfixes.csv', chunksize=1000)
print("CVEfixes_df_chunk is ready")

def read_csv_chunk(chunks):
    df_temp = []
    for chunk in CVEfixes_df_chunk:
        df_temp.append(chunk)
    df = pd.concat(df_temp,ignore_index = True)
    return df

CVEfixes_df = read_csv_chunk(CVEfixes_df_chunk)
print("CVEfixes_df is done")
# List of non-existing CWE IDs
non_exist_cwe_id_list = [16, 17, 18, 19, 21, 189, 199, 254, 255, 264, 275, 310, 320, 361, 388, 399, 534, 769, 840, 1187]
print("# CVEfixes_df:",CVEfixes_df.shape)
CVEfixes_df = CVEfixes_df[~CVEfixes_df['cwe_id'].isin(non_exist_cwe_id_list)]
print("CVEfixes_df - non_exist_cwe_id_list is done",CVEfixes_df.shape)

MSR_df = MSR_df[['code','cwe_id','vul','cve_id']]
# MVD_df = MVD_df[['code','cwe_id','vul']]
CVEfixes_df = CVEfixes_df[['code','cwe_id','vul','cve_id']]

# check if there is nan in vul
CVEfixes_df = CVEfixes_df[CVEfixes_df['vul'].notna()]
CVEfixes_df['vul'] = CVEfixes_df['vul'].astype(int)
print("# CVEfixes_df:",CVEfixes_df.shape)

# check if there is nan in vul
CVEfixes_df = CVEfixes_df[CVEfixes_df['cwe_id'].notna()]
CVEfixes_df['cwe_id'] = CVEfixes_df['cwe_id'].astype(int)
print("# CVEfixes_df:",CVEfixes_df.shape)

print(MSR_df.head(3))
# print(MVD_df.head(3))
print(CVEfixes_df.head(3))

print("MSR_df columns\n",MSR_df.columns)
# print("MVD_df columns\n",MVD_df.columns)
print("CVEfixes_df columns\n",CVEfixes_df.columns)

# Concatenate the datasets
# concatenated_df = pd.concat([MSR_df, MVD_df, CVEfixes_df])
concatenated_df = pd.concat([MSR_df, CVEfixes_df])
print(f"concatenated_df: {concatenated_df.shape}\n{concatenated_df.head(3)}")

# Assuming 'concatenated_df' is your DataFrame
rows_with_na = concatenated_df[concatenated_df.isna().any(axis=1)]
print("rows_with_na: \n", rows_with_na.head(3))

nan_counts = concatenated_df.isna().sum()
print("nan_counts",nan_counts)


# for index, row in rows_with_na.iterrows():
#     print(f"NA values found in row {index}:")
#     if pd.isna(row['cwe_id']):
#         print(f"NA case: {row['cwe_id']} --- Before {row['cwe_id']}\n{row}")
#         # Handle the case where the value is 'nan' to non-vulnerable class:0
#         concatenated_df.loc[index, 'cwe_id'] = 0
#         concatenated_df.loc[index, 'vul']= 0
#         print("After",concatenated_df.loc[index])
#     elif pd.isna(row['vul']):
#         concatenated_df.loc[index, 'vul']= 1

# Delete rows with NaN values from the DataFrame
concatenated_df.dropna(inplace=True)
print("After dropping --> rows_with_na: \n", rows_with_na.shape)


rows_with_na = concatenated_df[concatenated_df.isna().any(axis=1)]
if rows_with_na.shape[0] > 0:
     print("Successfully preprocessed NA - After relabeling rows_with_na so the # of rows should be 0")
else:
    print(f"After relabing rows_with_na: size: {rows_with_na.shape}")
    for index, row in rows_with_na.iterrows():
        print(f"NA values found in cwe_id {row['cwe_id']} row {index}:")
        for column, value in row.iteritems():
            if pd.isna(value):
                print(f"   - NA in column '{column}': value {value} ")
                        

# Set 'label' and 'vul' data type to int
concatenated_df['vul'] = concatenated_df['vul'].astype(int)
print(f"concatenated_df: {concatenated_df.shape}\n{concatenated_df.head(3)}")

# Split the data into train, validation, and test sets
train_df, test_df = train_test_split(concatenated_df, test_size=0.01, random_state=42)
val_df, test_df = train_test_split(test_df, test_size=0.5, random_state=42)

print("test_df",test_df.head(3))

# Optionally, you can save the train, validation, and test sets to separate CSV files
train_df.to_csv(f'{data_path}/train_data.csv', index=False)
val_df.to_csv(f'{data_path}/val_data.csv', index=False)
test_df.to_csv(f'{data_path}/test_data.csv', index=False)

'''
print(os.getcwd())

data_path = './datasets_'
MSR_df = pd.read_csv(f'{data_path}/MSR.csv')
MVD_df = pd.read_csv(f'{data_path}/MVD.csv')

MSR_df = MSR_df[['code','cwe_id','vul']]
MVD_df = MVD_df[['code','cwe_id','vul']]

print(MSR_df.head(3))
print(MVD_df.head(3))

print("MSR_df columns\n",MSR_df.columns)
print("MVD_df columns\n",MVD_df.columns)

# Concatenate the datasets
concatenated_df = pd.concat([MSR_df, MVD_df])
print(f"concatenated_df: {concatenated_df.shape}\n{concatenated_df.head(3)}")

# Assuming 'concatenated_df' is your DataFrame
rows_with_na = concatenated_df[concatenated_df.isna().any(axis=1)]
print("rows_with_na: \n", rows_with_na.head(3))

nan_counts = concatenated_df.isna().sum()
print("nan_counts",nan_counts)


for index, row in rows_with_na.iterrows():
    print(f"NA values found in cwe_id {row['cwe_id']} row {index}:")
    if pd.isna(row['cwe_id']):
        print(f"NA case: {row['cwe_id']} --- Before {row['cwe_id']}\n{row}")
        # Handle the case where the value is 'nan' to non-vulnerable class:0
        concatenated_df.loc[index, 'cwe_id'] = 0
        concatenated_df.loc[index, 'vul']= 0
        print("After",concatenated_df.loc[index])
    elif pd.isna(row['vul']):
        concatenated_df.loc[index, 'vul']= 1

# Delete rows with NaN values from the DataFrame
concatenated_df.dropna(inplace=True)
print("After dropping --> rows_with_na: \n", rows_with_na.shape)


rows_with_na = concatenated_df[concatenated_df.isna().any(axis=1)]
if rows_with_na.shape[0] > 0:
     print("Successfully preprocessed NA - After relabeling rows_with_na so the # of rows should be 0")
else:
    print(f"After relabing rows_with_na: size: {rows_with_na.shape}")
    for index, row in rows_with_na.iterrows():
        print(f"NA values found in cwe_id {row['cwe_id']} row {index}:")
        for column, value in row.iteritems():
            if pd.isna(value):
                print(f"   - NA in column '{column}': value {value} ")
                        

# Set 'label' and 'vul' data type to int
concatenated_df['vul'] = concatenated_df['vul'].astype(int)
print(f"concatenated_df: {concatenated_df.shape}\n{concatenated_df.head(3)}")

# Split the data into train, validation, and test sets
train_df, test_df = train_test_split(concatenated_df, test_size=0.01, random_state=42)
val_df, test_df = train_test_split(test_df, test_size=0.5, random_state=42)

print("test_df",test_df.head(3))

# Optionally, you can save the train, validation, and test sets to separate CSV files
train_df.to_csv(f'{data_path}/train_data.csv', index=False)
val_df.to_csv(f'{data_path}/val_data.csv', index=False)
test_df.to_csv(f'{data_path}/test_data.csv', index=False)
'''