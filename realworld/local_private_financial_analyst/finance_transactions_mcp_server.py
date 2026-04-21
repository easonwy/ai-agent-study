from fastmcp import FastMCP
import sqlite3


mcp = FastMCP("FinanceDBServer")

DB_PATH = "transactions.db"

@mcp.tool
def query_transactions(sql_query: str) -> str:
    """
    Execute a SQL query on the transaction database and return the results as a string.
    Use this to find totals, averages, or specific vendor payments.
    Example: "SELECT SUM(amount) FROM transactions WHERE vendor = 'TechCorp Payroll'"
    """

    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(sql_query)
            rows = cursor.fetchall()

            if not rows:
                return "No results found."

            # Format results with headers
            headers = [description[0] for description in cursor.description]
            result =[dict(zip(headers, row)) for row in rows]
            return str(result)
    except sqlite3.Error as e:
        return f"Database Error: {str(e)}"
    

@mcp.tool
def get_spending_summary() -> str:
    """Returns a quick summary of spending by category."""
    query = "SELECT category, SUM(amount) as total FROM transactions GROUP BY category"
    return query_transactions(query)

if __name__ == "__main__":
    mcp.run(transport="stdio", show_banner=False)