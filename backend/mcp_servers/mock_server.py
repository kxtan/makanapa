#!/usr/bin/env python3
"""Mock MCP server for testing when other services aren't available"""
import asyncio
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Mock Data Server")

@mcp.tool()
async def get_weather(location: str, meal_time: str = "lunch") -> dict:
    """Mock weather data for testing"""
    return {
        "location": location,
        "temperature": 30,
        "condition": "sunny",
        "suggestion": f"Perfect weather for {meal_time}!"
    }

@mcp.tool()
async def get_local_events(location: str) -> list:
    """Mock events data for testing"""
    return [
        {
            "name": "Test Food Festival",
            "date": "2025-09-20",
            "location": location,
            "type": "food_festival",
            "impact": "Great time to try festival food!"
        }
    ]

if __name__ == "__main__":
    mcp.run()
