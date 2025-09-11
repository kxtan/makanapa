#!/usr/bin/env python3
import asyncio
from mcp.server.fastmcp import FastMCP
from datetime import datetime, timedelta

mcp = FastMCP("Local Events Service")

@mcp.tool()
async def get_local_events(location: str) -> list:
    """Get local events that might affect food recommendations"""
    # Mock events data - replace with real event API integration
    sample_events = [
        {
            "name": "Penang Food Festival",
            "date": "2025-09-15",
            "location": "Georgetown",
            "type": "food_festival",
            "impact": "Expect crowds at popular spots, try festival stalls!"
        },
        {
            "name": "Malaysia Day Celebrations",
            "date": "2025-09-16", 
            "location": "KL",
            "type": "national_holiday",
            "impact": "Many shops closed, street food and malls still open"
        }
    ]
    
    return [e for e in sample_events if location.lower() in e["location"].lower()]

@mcp.tool()
async def get_restaurant_busy_times(location: str, day_of_week: str) -> dict:
    """Get expected busy times for restaurants in area"""
    # Mock busy time data
    busy_times = {
        "breakfast": "8:00-10:00 AM",
        "lunch": "12:00-2:00 PM", 
        "dinner": "6:00-8:00 PM"
    }
    
    return {
        "location": location,
        "day": day_of_week,
        "busy_times": busy_times,
        "recommendation": "Consider arriving 30 minutes before or after peak times"
    }

if __name__ == "__main__":
    mcp.run()
