import sqlite3 as lite
import pandas as pd
import pickle
from sqlite3 import Error
from pathlib import Path
import os

def create_connection(db_file):
    """
    create a connection to sqlite3 database
    """
    conn = None
    try:
        conn = lite.connect(db_file, timeout=10)  # connection via sqlite3
        # engine = sa.create_engine('sqlite:///' + db_file)  # connection via sqlalchemy
        # conn = engine.connect()
    except Error as e:
        print("create_connection Error - ",e)
    return conn

DATA_PATH = Path.cwd().parents[0]/ 'Data'
FIGURE_PATH = Path.cwd() / 'figures'
RESULT_PATH = Path.cwd() / 'results'

Path(DATA_PATH).mkdir(parents=True, exist_ok=True)
Path(FIGURE_PATH).mkdir(parents=True, exist_ok=True)
Path(RESULT_PATH).mkdir(parents=True, exist_ok=True)

conn = create_connection(DATA_PATH / "CVEfixes.db")

join_query = """
SELECT cc.cwe_id, mc.code
FROM file_change f
JOIN fixes fx ON f.hash = fx.hash
JOIN cve cv ON fx.cve_id = cv.cve_id
JOIN cwe_classification cc ON cv.cve_id = cc.cve_id
JOIN method_change mc ON f.file_change_id = mc.file_change_id
WHERE cc.cwe_id NOT IN ('NVD-CWE-Other', 'NVD-CWE-noinfo')
"""

# Execute the query and fetch data into a DataFrame
df = pd.read_sql_query(join_query, conn)
print(df.shape[0])

# Assuming `df` is your DataFrame
for column in df.columns:
    print(f"Column: {column}")
    print("Unique values:", df[column].unique())
    print()

# Close the connection
conn.close()


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
with open("data/total_cwe_dict.txt", "rb") as myFile:
    total_cwe_dict = pickle.load(myFile)

df['label'] = df['cwe_id'].map(total_cwe_dict)
df['vul'] = df['cwe_id'].apply(lambda x: 0 if x == 'non-vulnerable' else 1)

# Set 'label' and 'vul' data type to int
df['label'] = df['label'].astype(int)
df['vul'] = df['vul'].astype(int)

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
df.to_csv('data_preprocessing/preprocessed_datasets/CVEfixes_labeled.csv', index=False, lineterminator="")

