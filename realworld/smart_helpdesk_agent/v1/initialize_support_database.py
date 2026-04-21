import sqlite3

def init_db():
    with sqlite3.connect("interprise_orders_v1.db") as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS orders (
                id TEXT PRIMARY KEY, 
                status TEXT, 
                item TEXT, 
                customer_email TEXT,
                last_update TEXT
            )
        """)
        # Seed with data
        sample_orders = [
            ('ORD-101', 'Shipped', 'AI Developer Kit', 'eason@example.com', '2026-04-10'),
            ('ORD-202', 'Processing', 'Ollama Server Pro', 'tech@corp.com', '2026-04-12'),
            ('ORD-999', 'Delivered', 'Pro Developer Mac', 'eason@example.com', '2026-04-15')
        ]
        cursor.executemany("INSERT OR IGNORE INTO orders VALUES (?,?,?,?,?)", sample_orders)
        conn.commit()
    print("✅ enterprise_orders.db initialized.")


"""
This script initializes a SQLite database named "interprise_orders_v1.db" with a table called
"orders". The table has columns for order ID, status, item, customer email, and last update date.
The `init_db` function creates the table if it doesn't exist and seeds it with some sample orders.
When you run this script, it will set up the database with the necessary structure and initial data.
"""
if __name__ == "__main__":
    init_db()
