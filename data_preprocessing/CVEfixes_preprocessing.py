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
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")

join_query = """
SELECT cc.cwe_id, mc.code, cc.cve_id
FROM file_change f
JOIN fixes fx ON f.hash = fx.hash
JOIN cve cv ON fx.cve_id = cv.cve_id
JOIN cwe_classification cc ON cv.cve_id = cc.cve_id
JOIN method_change mc ON f.file_change_id = mc.file_change_id
WHERE cc.cwe_id NOT IN ('NVD-CWE-Other', 'NVD-CWE-noinfo')
"""
non_vul_query = """
SELECT cc.cwe_id, mc.code, cc.cve_id
FROM file_change f, fixes fx, cve cv, cwe_classification cc, method_change mc
WHERE f.hash = fx.hash 
AND fx.cve_id = cv.cve_id 
AND cv.cve_id = cc.cve_id 
AND f.file_change_id = mc.file_change_id
AND mc.before_change = 'False'
"""


vul_query = """
SELECT cc.cwe_id, mc.code, cc.cve_id
FROM file_change f, fixes fx, cve cv, cwe_classification cc, method_change mc
WHERE f.hash = fx.hash 
AND fx.cve_id = cv.cve_id 
AND cv.cve_id = cc.cve_id 
AND f.file_change_id = mc.file_change_id
AND mc.before_change = 'True'
AND cc.cwe_id IS NOT NULL
"""

# drop rows based on exception_id_list
exception_id_list = ['NVD-CWE-Other', 'NVD-CWE-noinfo']
# List of non-existing CWE IDs
non_exist_cwe_id_list = [16, 17, 18, 19, 21, 189, 199, 254, 255, 264, 275, 310, 320, 361, 388, 399, 534, 769, 840, 1187]

# Execute the query and fetch data into a DataFrame
def query_and_preprocess_dataframe(query, vul_flag):
    df = pd.read_sql_query(query, conn)

    print("# of total rows before dropping duplicates: ",df.shape[0])
    df = df.drop_duplicates()
    print("# of total rows after dropping duplicates: ",df.shape[0])

    for column in df.columns:
        print(f"Column: {column}")
        print("Unique values:", df[column].unique())
        print()

    print("# of total rows: ",df.shape[0])
    print(df.columns)
    print(df.head(5))

    nan_count = df['cwe_id'].isnull().sum()
    print("Number of NaN values in 'cwe_id':", nan_count)
 
    for exception_id in exception_id_list:
        count = len(df[df['cwe_id'] == exception_id])
        print("Count of", exception_id, ":", count)

    df = df[~df['cwe_id'].isin(exception_id_list)]

    # remove 'CWE-' and make onlu integer cwe_id and convert 'non-vulnerable' to cwe_id:0
    df['cwe_id'] = df['cwe_id'].astype(str)
    df = df[~df['cwe_id'].str.contains(',')] # remove multi labels
    df['cwe_id'] = df['cwe_id'].str.replace('CWE-', '')
    df['cwe_id'] = df['cwe_id'].replace('non-vulnerable', 0)
    df['cwe_id'] = df['cwe_id'].astype(int)
    # Remove rows where 'cwe_id' is in non_exist_cwe_id_list
    df = df[~df['cwe_id'].isin(non_exist_cwe_id_list)]

    unique_values = df['cwe_id'].unique()
    print(unique_values)

    if vul_flag:
        df['vul'] =  1 
    else:
        df['vul'] =  0
        df['cwe_id'] = 0
    
    vul_counts = df['vul'].value_counts()

    print("\nVul counts:")
    print(vul_counts)

    df = df[['code','cwe_id','cve_id','vul']]
    print("# of total rows: ",df.shape[0])
    print(df.columns)
    print(df.head(5))

    return df

vul_df = query_and_preprocess_dataframe(vul_query, True)
non_vul_df = query_and_preprocess_dataframe(non_vul_query, False)
combined_df = pd.concat([vul_df,non_vul_df])

# Save to CSV, rows separated by ""
combined_df.to_csv('../../TransVulDet/data_preprocessing/CVEfixes/CVEfixes_new_2.csv', index=False, lineterminator="")

# Close the connection
conn.close()
