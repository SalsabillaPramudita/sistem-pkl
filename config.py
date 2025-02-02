# config.py
import mysql.connector

def get_db_connection():
    try:
        conn = mysql.connector.connect(
            host='localhost',
            user='root',            # 
            password='password',    # 
            database='pkl_db'       # 
        )
        return conn
    except mysql.connector.Error as err:
        print(f"Error: {err}")
        return None
