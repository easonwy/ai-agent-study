import logging
import sqlite3
from pathlib import Path

from fastmcp import FastMCP

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger("SupportMCP")

APP_DIR = Path(__file__).resolve().parent
DB_PATH = APP_DIR / "interprise_orders_v1.db"

mcp = FastMCP("EnterpriseSupport")


@mcp.tool()
def lookup_order_secure(order_id: str, customer_email: str) -> str:
    """
    Securely look up order details.
    Requires BOTH Order ID (e.g. ORD-101) AND the associated Customer Email.
    """
    logger.info(
        "[Tool] lookup_order_secure called for order_id=%s email=%s",
        order_id,
        customer_email,
    )
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM orders WHERE id=? AND customer_email=?",
            (order_id, customer_email),
        )
        row = cursor.fetchone()
        if row:
            logger.info("[Tool] lookup_order_secure success for order_id=%s", order_id)
            return f"Order Found: {row[2]} is currently {row[1]}. Last updated: {row[4]}."

    logger.warning("[Tool] lookup_order_secure verification failed for order_id=%s", order_id)
    return "ERROR: Verification failed. Order ID and Email do not match our records."


if __name__ == "__main__":
    logger.info("[Bootstrap] Starting EnterpriseSupport MCP server with DB at %s", DB_PATH)
    mcp.run(transport="stdio", show_banner=False)
