# Quick script to create sample data
import sqlite3
with sqlite3.connect("my_database.db") as conn:
    conn.execute("CREATE TABLE IF NOT EXISTS user (id INTEGER, name TEXT, email TEXT)")
    conn.execute("INSERT INTO users VALUES (1, 'Eason', 'eason@example.com')")
    conn.commit()