from collections import Counter
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import defaultdict
from sklearn.model_selection import GroupShuffleSplit
import os
import json
print(os.getcwd())

data_path = 'datasets_'
MSR_df = pd.read_csv('data_preprocessing/Bigvul/MSR.csv')
CVEfixes_df = pd.read_csv('data_preprocessing/CVEfixes/CVEfixes_new.csv')

# CVEfixes_df_chunk = pd.read_csv('data_preprocessing/CVEfixes/CVEfixes_new.csv', chunksize=1000)
# print("CVEfixes_df_chunk is ready")

# def read_csv_chunk(chunks):
#     df_temp = []
#     for chunk in CVEfixes_df_chunk:
#         df_temp.append(chunk)
#     df = pd.concat(df_temp,ignore_index = True)
#     return df

# CVEfixes_df = read_csv_chunk(CVEfixes_df_chunk)

# List of non-existing CWE IDs
non_exist_cwe_id_list = [16, 17, 18, 19, 21, 189, 199, 254, 255, 264, 275, 310, 320, 361, 388, 399, 534, 769, 840, 1187]
print("# CVEfixes_df:",CVEfixes_df.shape)
CVEfixes_df = CVEfixes_df[~CVEfixes_df['cwe_id'].isin(non_exist_cwe_id_list)]
print("CVEfixes_df - non_exist_cwe_id_list is done",CVEfixes_df.shape)

MSR_df = MSR_df[['code','cwe_id','vul','cve_id']]
CVEfixes_df = CVEfixes_df[['code','cwe_id','vul','cve_id']]

# check if there is nan in vul
CVEfixes_df = CVEfixes_df[CVEfixes_df['vul'].notna()]
CVEfixes_df['vul'] = CVEfixes_df['vul'].astype(int)
print("After drop NaN in vul - # CVEfixes_df:",CVEfixes_df.shape)

# check if there is nan in vul
CVEfixes_df = CVEfixes_df[CVEfixes_df['cwe_id'].notna()]
CVEfixes_df['cwe_id'] = CVEfixes_df['cwe_id'].astype(int)
print("After drop NaN in cwe_id - # CVEfixes_df:",CVEfixes_df.shape)

print(MSR_df.head(3))
print(CVEfixes_df.head(3))

print("MSR_df columns\n",MSR_df.columns)
unique_values = MSR_df.nunique()
print(unique_values)
print("CVEfixes_df columns\n",CVEfixes_df.columns)
unique_values = CVEfixes_df.nunique()
print(unique_values)

# Concatenate the datasets
combined_df = pd.concat([MSR_df, CVEfixes_df])
print(f"combined_df: {combined_df.shape}\n{combined_df.head(3)}")
unique_values = combined_df.nunique()
print(unique_values)
combined_df = combined_df.drop_duplicates()
print("After drop duplicates",combined_df.shape)
nan_rows = combined_df[combined_df.isna().any(axis=1)]
print("nan_rows",nan_rows)
combined_df = combined_df.dropna()
print("After drop NaN",combined_df.shape)


rows_with_na = combined_df[combined_df.isna().any(axis=1)]
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
combined_df['vul'] = combined_df['vul'].astype(int)
print(f"concatenated_df: {combined_df.shape}\n{combined_df.head(3)}")

# Split the data into train, validation, and test sets
train_df, test_df = train_test_split(combined_df, test_size=0.01, random_state=42)
val_df, test_df = train_test_split(test_df, test_size=0.5, random_state=42)

print("test_df",test_df.head(3))

# Optionally, you can save the train, validation, and test sets to separate CSV files
train_df.to_csv(f'{data_path}/train_data.csv', index=False)
val_df.to_csv(f'{data_path}/val_data.csv', index=False)
test_df.to_csv(f'{data_path}/test_data.csv', index=False)


# Count unique values in 'cwe_id' column
cwe_counts = combined_df['cwe_id'].value_counts()

# Filter out 'cwe_id's with count less than minimum_cwe_cnt
minimum_cwe_cnt = 1000
filtered_cwes = cwe_counts[cwe_counts < minimum_cwe_cnt]
print(f"filtered_cwes by {minimum_cwe_cnt}",filtered_cwes.sum())

filtered_cwes_list = filtered_cwes.index.tolist()
filtered_cwes_list = sorted([int(item) for item in filtered_cwes_list])
print("filtered_cwes_list",len(filtered_cwes_list),filtered_cwes_list)
# Load the CWE paths from your JSON (Assuming it's stored in a variable named `cwe_paths_json`)
node_paths_dir = 'data_preprocessing/preprocessed_datasets/debug_datasets'
with open(f'{node_paths_dir}/graph_all_paths.json', 'r') as f:
    cwe_paths = json.load(f)

# Function to reassign CWEs to a higher level in the hierarchy
def reassign_cwe(cwe_id, level=3):
    # Find the path for the given CWE ID
    paths = cwe_paths.get(str(cwe_id), [])
    new_paths = []

    # Reassign to the specified level up in the hierarchy
    for path in paths:
        parts = path.split("-")
        if len(parts) > level:
            new_path = "-".join(parts[:level+1])
            new_paths.append(new_path)
        else:
            print(f"len(parts) <= level:{path}")
            new_paths.append(path)  # No reassignment if path is too short

    return new_paths

# Reassign the filtered CWEs
reassigned_cwe_paths = {}
for cwe_id in filtered_cwes_list:
    reassigned_cwe_paths[cwe_id] = reassign_cwe(cwe_id, level=3)

print("reassigned_cwe_paths",reassigned_cwe_paths)


# Prepare a new mapping dictionary by getting the last element of each path
new_cwe_mapping = {k: v[-1].split('-')[-1] for k, v in reassigned_cwe_paths.items()if v}

# Use the mapping to create a new 're_cwe_id' column
# It maps the cwe_id to the last element in the path if it's in the new_cwe_mapping; otherwise, keeps the original cwe_id
combined_df['re_cwe_id'] = combined_df['cwe_id'].map(new_cwe_mapping).fillna(combined_df['cwe_id'])
combined_df = combined_df.loc[:, ~combined_df.columns.str.contains('^Unnamed')]
# combined_df['re_cwe_id']  = combined_df['re_cwe_id'].astype(int)
print(combined_df.head(5))

# Select and print rows where 'cwe_id' and 're_cwe_id' are different
different_rows = combined_df[combined_df['cwe_id'] != combined_df['re_cwe_id']]
print("different_rows[['cwe_id','re_cwe_id']]",different_rows[['cwe_id','re_cwe_id']])
# Count unique values in 'cwe_id' column
cwe_counts = combined_df['re_cwe_id'].value_counts()


# Filter out 'cwe_id's with count less than minimum_cwe_cnt
filtered_cwes = cwe_counts[cwe_counts < minimum_cwe_cnt]
print("filtered_cwes.sum()",filtered_cwes.sum())


final_filtered_cwes_list = filtered_cwes.index.tolist()
final_filtered_cwes_list = sorted([int(item) for item in final_filtered_cwes_list])
print("final_filtered_cwes_list",final_filtered_cwes_list)

total_filtered_cwes_list = sorted(list(set(final_filtered_cwes_list+filtered_cwes_list)))
print("total_filtered_cwes_list",len(total_filtered_cwes_list),total_filtered_cwes_list)

# total_filtered_cwes_list is the list of CWE IDs to filter out
final_df = combined_df[~combined_df['cwe_id'].isin(total_filtered_cwes_list)]
final_df.to_csv(f'{data_path}/MSR_CVEfixes_level3_final.csv', index=False)
print(final_df.head(5))


# Save the filtered dictionary back to the JSON file
# Filter the dictionary
final_cwe_paths = {int(k): v for k, v in cwe_paths.items() if int(k) not in total_filtered_cwes_list}
# filtered_cwe_paths[0] = ['10000-0']

with open(f'{node_paths_dir}/graph_final_cwe_paths_new.json', 'w') as file:
    json.dump(final_cwe_paths, file)


# Stratify 
df = final_df
random_state = 42
# Define the column names
# columns = ['code', 'vul', 'cwe_id','re_cwe_id']

# # Create an empty DataFrame with these columns
# train = pd.DataFrame(columns=columns)
# validation = pd.DataFrame(columns=columns)
# test = pd.DataFrame(columns=columns)

# Function to split each group
# def split_group(group, group_size):
#     gss = GroupShuffleSplit(n_splits=1, test_size=test_size + validation_size, random_state=random_state)
#     train_idx, test_validation_idx = next(gss.split(group, groups=group['cve_id']))

#     train_group = group.iloc[train_idx]
#     test_validation_group = group.iloc[test_validation_idx]

#     # Adjust the test_size for the split of validation and test
#     adjusted_test_size = test_size / (test_size + validation_size)
#     gss = GroupShuffleSplit(n_splits=1, test_size=adjusted_test_size, random_state=random_state)
#     validation_idx, test_idx = next(gss.split(test_validation_group, groups=test_validation_group['cve_id']))

#     return train_group, test_validation_group.iloc[validation_idx], test_validation_group.iloc[test_idx]

# Calculate cwe_id proportions in the original dataset
cwe_id_counts = Counter(df['re_cwe_id'])
total_count = sum(cwe_id_counts.values())

# Initialize datasets
train, validation, test = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

initial_allocation = defaultdict(lambda: {"train": 0, "validation": 0, "test": 0})

for cwe_id, group in df.groupby('re_cwe_id'):
#     print("cwe_id, group",cwe_id, group)
    # For each cwe_id, allocate the first few cve_id groups to different datasets
    for i, (cve_id, cve_group) in enumerate(group.groupby('cve_id')):
        if i % 11 < 9:
            train = pd.concat([train, cve_group])
            initial_allocation[cwe_id]["train"] += len(cve_group)
        elif i % 11 == 9:
            validation = pd.concat([validation, cve_group])
            initial_allocation[cwe_id]["validation"] += len(cve_group)
        else:
            test = pd.concat([test, cve_group])
            initial_allocation[cwe_id]["test"] += len(cve_group)

# # Split each cwe_id group
# for cwe_id, group_size in cwe_id_counts.items():
#     group = df[df['re_cwe_id'] == cwe_id]
#     train_group, validation_group, test_group = split_group(group, group_size)

#     train = pd.concat([train, train_group])
#     validation = pd.concat([validation, validation_group])
#     test = pd.concat([test, test_group])

remaining_df = df[~df['cve_id'].isin(pd.concat([train, validation, test])['cve_id'])]

if not remaining_df.empty:
    print("remaining_df is NOT EMPTY", remaining_df.shape)
    # remaining_train, remaining_validation, remaining_test = split_group(remaining_df, len(remaining_df))
    # train = pd.concat([train, remaining_train])
    # validation = pd.concat([validation, remaining_validation])
    # test = pd.concat([test, remaining_test])


train = train.sample(frac=1, random_state=random_state).reset_index(drop=True)
validation = validation.sample(frac=1, random_state=random_state).reset_index(drop=True)
test = test.sample(frac=1, random_state=random_state).reset_index(drop=True)

train.to_csv(f'{data_path}/train_dataset.csv', index=False)
validation.to_csv(f'{data_path}/validation_dataset.csv', index=False)
test.to_csv(f'{data_path}/test_dataset.csv', index=False)
print(train.shape[0], validation.shape[0], test.shape[0])

# Desired total sample size
target_sample_size = 5000

# Calculate the number of unique CWE IDs
num_unique_cwe_ids = validation['re_cwe_id'].nunique()

# Calculate how many samples to take per CWE ID
samples_per_cwe_id = target_sample_size // num_unique_cwe_ids

# Sample the rows
balanced_validation = pd.DataFrame()

for cwe_id in validation['re_cwe_id'].unique():
    # If there are not enough samples for a particular cwe_id, take all available
    samples_to_take = min(samples_per_cwe_id, len(validation[validation['re_cwe_id'] == cwe_id]))
    sampled_group = validation[validation['re_cwe_id'] == cwe_id].sample(n=samples_to_take, random_state=random_state)
    balanced_validation = pd.concat([balanced_validation, sampled_group])

# Check if we have reached the target sample size
current_sample_size = len(balanced_validation)

# If the current sample size is less than the target, fill in the remaining with random samples from the validation set
if current_sample_size < target_sample_size:
    additional_samples = validation.sample(n=target_sample_size - current_sample_size, random_state=random_state)
    balanced_validation = pd.concat([balanced_validation, additional_samples])

# Shuffle the balanced validation set
balanced_validation = balanced_validation.sample(frac=1, random_state=random_state).reset_index(drop=True)

# Update the validation DataFrame
validation = balanced_validation

# Output the sampled DataFrame
validation.to_csv(f'{data_path}/balanced_validation_dataset.csv', index=False)






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