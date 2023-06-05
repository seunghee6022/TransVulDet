import sqlite3 as lite
import pandas as pd
import pickle
from sqlite3 import Error
from pathlib import Path

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


join_small_query = """
SELECT cc.cwe_id, mc.code
FROM file_change f
JOIN fixes fx ON f.hash = fx.hash
JOIN cve cv ON fx.cve_id = cv.cve_id
JOIN cwe_classification cc ON cv.cve_id = cc.cve_id
JOIN method_change mc ON f.file_change_id = mc.file_change_id
WHERE cc.cwe_id NOT IN ('NVD-CWE-Other', 'NVD-CWE-noinfo')
"""


# Execute the query and fetch data into a DataFrame
df = pd.read_sql_query(join_small_query, conn)
print(df.shape[0])

# Assuming `df` is your DataFrame
for column in df.columns:
    print(f"Column: {column}")
    print("Unique values:", df[column].unique())
    print()

# Close the connection
conn.close()

# Save to CSV, rows separated by ""
df.to_csv('../../TransvulDet/data_preprocessing/preprocessed_datasets/CVEfixes.csv', index=False, lineterminator="")



