#!/usr/bin/env python3
import asyncio
import json
from mcp.server.fastmcp import FastMCP
from mcp.server.models import InitializationOptions
import httpx

# Create the MCP server
mcp = FastMCP("Weather Service")

@mcp.tool()
async def get_weather(location: str, meal_time: str = "lunch") -> dict:
    """Get weather data for food recommendations"""
    # You can integrate with real weather APIs here
    # For now, using mock data similar to your current implementation
    import random
    
    temps = [28, 30, 32, 34]
    conditions = ["sunny", "cloudy", "hot", "humid"]
    temp = random.choice(temps)
    condition = random.choice(conditions)
    
    if temp > 30:
        suggestion = f"Hot weather! Perfect for cooling {meal_time} like cendol, ais kacang, or cold laksa."
    elif temp < 25:
        suggestion = f"Cooler weather great for hot {meal_time} like bak kut teh, steamboat, or curry noodles."
    else:
        suggestion = f"Perfect temperature for any {meal_time} - explore outdoor food courts!"
    
    return {
        "location": location,
        "temperature": temp,
        "condition": condition,
        "suggestion": suggestion
    }

if __name__ == "__main__":
    mcp.run()
