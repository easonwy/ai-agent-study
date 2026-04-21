import sqlite3


def init_transaction_db():
    conn = sqlite3.connect("transactions.db")
    cursor = conn.cursor()
    # Create a clean schema for the AI to understand
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY,
            date TEXT,
            category TEXT,
            vendor TEXT,
            amount REAL,
            status TEXT
        )
    """)

    # Seed with sample data
    sample_data = [
        ('2026-01-05', 'Income', 'TechCorp Payroll', 4200.00, 'Cleared'),
        ('2026-01-12', 'Software', 'AWS', -154.20, 'Cleared'),
        ('2026-01-20', 'Hardware', 'Apple Store', -1299.00, 'Pending'),
        ('2026-02-01', 'Rent', 'Tech City Realty', -2000.00, 'Cleared'),
        ('2026-02-15', 'Software', 'OpenAI', -20.00, 'Cleared'),
        ('2024-02-20', 'Income', 'AWS', 2750.00, 'Cleared')
    ]
    cursor.executemany("INSERT INTO transactions (date, category, vendor, amount, status) VALUES (?,?,?,?,?)", sample_data)
    conn.commit()
    conn.close()
    print("✅ transactions.db initialized.")

if __name__ == "__main__":
    init_transaction_db()