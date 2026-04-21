import sqlite3

def inspect_db():
    conn = sqlite3.connect("memory.db")
    cursor = conn.cursor()
    
    # 1. Get all table names
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    print(f"Tables in database: {[t[0] for t in tables]}")
    
    # 2. Inspect the 'checkpoints' table schema
    # This table stores the snapshot of your agent's memory
    print("\n--- Checkpoints Table Schema ---")
    cursor.execute("PRAGMA table_info(checkpoints);")
    for col in cursor.fetchall():
        print(col)
        
    # 3. View saved sessions (thread_ids)
    print("\n--- Saved Session IDs (Thread IDs) ---")
    cursor.execute("SELECT DISTINCT thread_id FROM checkpoints;")
    for row in cursor.fetchall():
        print(f"Active Thread: {row[0]}")
        
    conn.close()

inspect_db()