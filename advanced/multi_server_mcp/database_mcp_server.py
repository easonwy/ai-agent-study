
import sqlite3
from fastmcp import FastMCP


# Initialize the MCP server
mcp = FastMCP('DatabaseServer')

# Configuration: path to your local SQLite database
DATABASE_PATH = 'my_database.db'

@mcp.tool()
def execute_query(sql_query: str):
    """Execute a SQL query against the local SQLite database and return results."""
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        cursor.execute(sql_query)

        # For SELECT queries, return the formatted results
        if sql_query.strip().upper().startswith("SELECT"):
            rows = cursor.fetchall()
            if not rows:
                return "Query executed successfully, but no results to display."
            # Format results as a string for easier display
            headers = [description[0] for description in cursor.description]
            results = [dict(zip(headers, row)) for row in rows]
            return results

        conn.commit()
        return f"Operation successful. Rows affected: {cursor.rowcount}"
    except Exception as e:
        return f"Error executing query: {e}"
    finally:
        conn.close()

def list_tables() -> str:
    """List all tables in the SQLite database."""
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        if not tables:
            return "No tables found in the database."
        tables = [row[0] for row in cursor.fetchall()]
        return f"Available tables: {', '.join(tables)}"
    except Exception as e:
        return f"Error listing tables: {e}"
    finally:
        conn.close()

if __name__ == "__main__":
    # Start the server using stdio transport (required for local agent discovery)
    mcp.run(transport="stdio")
