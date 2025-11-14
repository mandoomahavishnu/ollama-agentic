"""
Database connection and basic operations
"""
import mysql.connector
import pandas as pd
import streamlit as st
from mysql.connector import pooling
from config import MYSQL_CONFIG, TABLE_NAMES

def init_mysql_pool():
    """Initialize MySQL connection pool"""
    return mysql.connector.pooling.MySQLConnectionPool(
        pool_name="mypool",
        pool_size=10,
        **MYSQL_CONFIG
    )

# Global MySQL pool
MYSQL_POOL = init_mysql_pool()

def get_db_connection():
    """Get a connection from the MySQL pool"""
    return MYSQL_POOL.get_connection()

@st.cache_data(ttl=600)
def get_database_schema(table_names=None):
    """Get database schema information"""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            if table_names:
                tlist = ", ".join(f"'{t}'" for t in table_names)
                sql = f"""
                    SELECT table_name, column_name, data_type, is_nullable,
                           column_key, column_default, extra, column_comment
                    FROM information_schema.columns
                    WHERE table_schema='creme' AND table_name IN ({tlist})
                """
            else:
                sql = """
                    SELECT table_name, column_name, data_type, is_nullable,
                           column_key, column_default, extra, column_comment
                    FROM information_schema.columns
                    WHERE table_schema='creme'
                """
            cur.execute(sql)
            return cur.fetchall()
    finally:
        conn.close()

@st.cache_data(ttl=600)
def get_first_three_rows(table_name):
    """Get first three rows from a table for examples"""
    conn = get_db_connection()
    rows = []
    try:
        with conn.cursor(dictionary=True) as cur:
            cur.execute(f"SELECT * FROM {table_name} LIMIT 2")
            rows = cur.fetchall()
    finally:
        conn.close()
    return rows

def execute_sql_and_return_rows(sql_query):
    """Execute SQL query and return results"""
    rows = []
    cols = []
    err = None
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor(dictionary=True) as cursor:
            cursor.execute(sql_query)
            cols = cursor.column_names
            rows = cursor.fetchall()
    except mysql.connector.Error as sql_err:
        err = f"MySQL Error: {str(sql_err)}"
    except Exception as e:
        err = str(e)
    finally:
        if conn and conn.is_connected():
            try:
                conn.close()
            except:
                pass
    return cols, rows, err

def handle_duplicate_columns(df):
    """Handle duplicate column names in DataFrame"""
    cols = pd.Series(df.columns)
    for dup in cols[cols.duplicated()].unique():
        idxs = cols[cols == dup].index.values
        for i, idx in enumerate(idxs):
            cols[idx] = f"{dup}_dup_{i+1}"
    df.columns = cols
    return df