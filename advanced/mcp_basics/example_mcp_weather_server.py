# example_mcp_weather_server.py
from fastmcp import FastMCP

# Create an MCP server named "Weather Service"
mcp = FastMCP("WeatherService")

@mcp.tool()
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    # In a real app, you'd call a weather API here
    return f"The weather in {city} is currently 23°C and Sunny."

if __name__ == "__main__":
    mcp.run(transport="stdio")
