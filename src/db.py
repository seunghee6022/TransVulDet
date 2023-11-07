import sqlite3 as lite
import pandas as pd
from sqlite3 import Error
from pathlib import Path
import os

print("Path.cwd()",Path.cwd())
print("Path.cwd().parents[0],",Path.cwd().parents[0])
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
FIGURE_PATH = Path.cwd() / 'CVEfixes/figures'
RESULT_PATH = Path.cwd() / 'CVEfixes/results'

print("Path.cwd()",Path.cwd())
print("Path.cwd().parents[0],",Path.cwd().parents[0])
print("DATA_PATH",DATA_PATH)