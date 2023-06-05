import pandas as pd
import sqlite3 as lite
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
        print(e)
    return conn


DATA_PATH = Path.cwd().parents[0] / 'CVEfixes'
FIGURE_PATH = Path.cwd() / 'figures'
RESULT_PATH = Path.cwd() / 'results'

Path(DATA_PATH).mkdir(parents=True, exist_ok=True)
Path(FIGURE_PATH).mkdir(parents=True, exist_ok=True)
Path(RESULT_PATH).mkdir(parents=True, exist_ok=True)

conn = create_connection(DATA_PATH / "CVEfixes.db")



'''
Columns from query

- cve(**cve_id**) ↔ Cwe_classification(**cve_id**, **cwe_id**) ↔cwe(cwe_id, is_category)
    - **cve_id, cwe_id,** is_category
- cve(**cve_id**) ↔ fixes(**cve_id, hash**) ↔ commits(**hash**) ↔file_changes (**file_change_id**, hash, code_after, code_before, programming_language) ↔ method_change (file_change_id, code)
    - **cve_id, hash, file_change_id**, hash, code_after, code_before, programming_language, code
'''

query = """
SELECT cv.cve_id, f.code_before, f.code_after, f.programming_language, cc.cwe_id, cw.is_category, cw.cwe_name, mc.code
FROM file_change f, commits c, fixes fx, cve cv, cwe_classification cc, method_change mc, cwe cw
WHERE f.hash = c.hash 
AND c.hash = fx.hash 
AND fx.cve_id = cv.cve_id 
AND cv.cve_id = cc.cve_id 
AND cc.cwe_id = cw.cwe_id
AND f.file_change_id = mc.file_change_id

"""

small_query = """
SELECT f.code_before, f.code_after, cc.cwe_id, cw.cwe_name, mc.code
FROM file_change f, commits c, fixes fx, cve cv, cwe_classification cc, method_change mc, cwe cw
WHERE f.hash = c.hash 
AND c.hash = fx.hash 
AND fx.cve_id = cv.cve_id 
AND cv.cve_id = cc.cve_id 
AND cc.cwe_id = cw.cwe_id
AND f.file_change_id = mc.file_change_id

"""


cwe_id_query = """
SELECT cc.cwe_id
FROM  cve cv, cwe_classification cc
WHERE cv.cve_id = cc.cve_id 


"""
df = pd.read_sql_query(cwe_id_query, conn)

# df.to_csv("CFD_sep_¤.csv", sep='¤')
df.to_csv("CFD_cwe_id.csv")
