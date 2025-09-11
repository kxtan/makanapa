import streamlit as st
import time
import json
import re
import asyncio
from typing import Optional, Dict
from datetime import datetime, timezone as dt_timezone
from zoneinfo import ZoneInfo

# Optional dependencies for auto geolocation + reverse geocoding
try:
    from geopy.geocoders import Nominatim # pip install geopy
except ImportError:
    Nominatim = None

try:
    from streamlit_js_eval import streamlit_js_eval # pip install streamlit_js_eval
except ImportError:
    streamlit_js_eval = None

# MCP dependencies
try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    import httpx
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False

# Import your backend classes and functions from your project
from test import RestaurantReviewVectorizer
from llm_enhancer import get_llm_enhancer

# Initialize backend objects only once
@st.cache_resource
def get_vectorizer():
    v = RestaurantReviewVectorizer(
        model_name="all-MiniLM-L6-v2",
        persist_directory="./restaurant_chroma_db"
    )
    v.create_or_get_collection("restaurant_reviews")
    return v

@st.cache_resource
def get_enhancer():
    return get_llm_enhancer()

vectorizer = get_vectorizer()
llm_enhancer = get_enhancer()

# System prompts
SYSTEM_PROMPT = """You are MakanApa, a friendly Malaysian food discovery assistant. Your job is to help users figure out what to eat based on their current situation, mood, location, and preferences.

Guidelines:
- Be conversational and warm, like a local friend giving food advice
- Ask follow-up questions to understand their context (time of day, mood, location, group size, dietary restrictions)
- Suggest specific dishes and restaurants when possible
- Use the search results to give personalized recommendations
- Keep responses concise but helpful
- Use Malaysian food terminology naturally (makan, tapau, shiok, etc.)
- If they seem indecisive, offer 2-3 concrete options

Start conversations by asking what they're in the mood for or their current situation."""

CONTEXT_ANALYZER_PROMPT = """
Analyze the user's message and current context to determine if we have enough information for good food recommendations.

Current context: {context}
User message: "{user_input}"
Conversation history: {history}

Required context for good recommendations:
- Location (where are they eating?)
- Meal type (breakfast/lunch/dinner/snack?)
- Dietary restrictions (vegetarian/halal/allergies?)
- Group size (solo/couple/family/group?)
- Budget preference (cheap eats/mid-range/fine dining?)
- Mood/craving (spicy/comfort/light/specific cuisine?)
- Timezone and local_time for time-aware suggestions

Rules:
1. If this is the first message or we're missing 3+ key pieces of context, ask ONE specific follow-up question
2. If we have enough context (3+ pieces), proceed with recommendations
3. Always be conversational and friendly
4. Use Malaysian terms naturally

Response format (JSON only):
{{
    "needs_more_context": true/false,
    "follow_up_question": "What's your question?",
    "missing_info": ["location", "meal_type"],
    "ready_for_recommendation": true/false,
    "confidence_level": "low/medium/high"
}}
"""

CONTEXT_EXTRACTOR_PROMPT = """
Extract food recommendation context from this message: "{user_input}"
Current context: {current_context}

Look for mentions of:
- location: (Penang, Georgetown, KL, specific areas)
- meal_type: (breakfast, lunch, dinner, snack, supper)
- dietary_restrictions: (vegetarian, halal, no pork, allergies)
- group_size: (solo, couple, family, friends, group)
- budget: (cheap, budget, affordable, mid-range, expensive, fancy)
- mood: (spicy, light, comfort, healthy, adventurous, quick)
- cuisine_preference: (Chinese, Malay, Indian, Western, fusion, local)
- timezone: (IANA like Asia/Kuala_Lumpur)
- local_time: (YYYY-MM-DD HH:MM)

Return ONLY the extracted values as JSON. Use null for missing information.
Example: {{"location": "Penang", "meal_type": "lunch", "mood": "spicy", "dietary_restrictions": null}}
"""

# ---------------------------- MCP Manager ----------------------------

class MCPManager:
    def __init__(self):
        self.servers = {}
        self.config = self.load_mcp_config()

    def load_mcp_config(self):
        """Load MCP server configuration from file or return default config."""
        try:
            with open('mcp_config.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            # Default configuration pointing to separate server files
            return {
                "mcpServers": {
                    "weather": {
                        "command": "python",
                        "args": ["mcp_servers/weather_server.py"],
                        "name": "Weather Service",
                        "description": "Get weather data for better food recommendations"
                    },
                    "events": {
                        "command": "python",
                        "args": ["mcp_servers/events_server.py"],
                        "name": "Local Events",
                        "description": "Find local events and festivals"
                    },
                    "mock_server": {
                        "command": "python",
                        "args": ["mcp_servers/mock_server.py"],
                        "name": "Mock Data Server",
                        "description": "Mock weather and events for testing"
                    }
                }
            }

    async def connect_to_servers(self):
        """Connect to all configured MCP servers."""
        if not MCP_AVAILABLE:
            st.warning("MCP packages not installed. Run: pip install mcp fastmcp httpx")
            return

        for name, config in self.config.get("mcpServers", {}).items():
            try:
                # All servers are now STDIO-based
                client = await self.connect_stdio_server(config)
                self.servers[name] = {
                    "client": client,
                    "config": config,
                    "tools": await self.get_server_tools(client)
                }
                st.success(f"Connected to MCP server '{name}'")
                
            except Exception as e:
                st.warning(f"Could not connect to MCP server '{name}': {e}")

    async def connect_stdio_server(self, config: dict):
        """Connect to STDIO-based MCP server."""
        server_params = StdioServerParameters(
            command=config["command"],
            args=config["args"]
        )
        
        stdio_transport = await stdio_client(server_params)
        read, write = stdio_transport
        
        session = ClientSession(read, write)
        await session.initialize()
        return session

    async def get_server_tools(self, client):
        """Get available tools from server."""
        try:
            response = await client.list_tools()
            return {tool.name: tool for tool in response.tools}
        except Exception:
            return {}

    async def call_tool(self, server_name: str, tool_name: str, arguments: dict):
        """Call a specific tool on a specific server."""
        if server_name not in self.servers:
            return {"error": f"Server {server_name} not connected"}
            
        client = self.servers[server_name]["client"]
        try:
            result = await client.call_tool(tool_name, arguments)
            return result.content[0].text if result.content else {"error": "No content returned"}
        except Exception as e:
            return {"error": str(e)}

# Mock MCP functions for fallback when servers aren't available
def get_mock_weather_data(location: str, meal_time: str = "lunch") -> dict:
    """Mock weather data for testing when MCP servers aren't available."""
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

def get_mock_events_data(location: str) -> list:
    """Mock events data for testing when MCP servers aren't available."""
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

# ---------------------------- Utilities ----------------------------

def run_async(coro):
    """Helper function to run async code in Streamlit's synchronous context."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)

def clean_json_response(response_text: str) -> str:
    """Extract JSON from fenced code or return raw text if not fenced."""
    if not response_text or not response_text.strip():
        return "{}"  # Return empty JSON object for empty responses
    
    # Try to extract from code fence first
    m = re.search(r"``````", response_text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    
    # If no code fence, check if it's already valid JSON
    try:
        json.loads(response_text.strip())
        return response_text.strip()
    except json.JSONDecodeError:
        # If not valid JSON, wrap in a basic structure
        return '{"error": "Invalid JSON response"}'

def detect_timezone_and_local_time():
    """Read the browser's IANA timezone and compute local time."""
    try:
        tz_name = getattr(st.context, "timezone", None) # e.g., "Asia/Kuala_Lumpur"
        if not tz_name:
            return None, None

        now_utc = datetime.now(dt_timezone.utc)
        try:
            tz_obj = ZoneInfo(tz_name)
        except Exception:
            return tz_name, None

        now_local = now_utc.astimezone(tz_obj)
        return tz_name, now_local
    except Exception:
        return None, None

def infer_meal_type(dt: Optional[datetime]) -> Optional[str]:
    if not dt:
        return None
    h = dt.hour
    if 5 <= h < 11: return "breakfast"
    if 11 <= h < 15: return "lunch"
    if 15 <= h < 18: return "tea/snack"
    if 18 <= h < 22: return "dinner"
    return "supper"

def get_browser_coords():
    """Request browser geolocation via JS and return {'lat','lon','accuracy'} if available."""
    if streamlit_js_eval is None:
        return None

    js = """
    async () => {
        if (!('geolocation' in navigator)) return { error: 'unsupported' };
        try {
            return await new Promise((resolve) => {
                navigator.geolocation.getCurrentPosition(
                    (pos) => resolve({
                        lat: pos.coords.latitude,
                        lon: pos.coords.longitude,
                        accuracy: pos.coords.accuracy
                    }),
                    (err) => resolve({ error: err.message }),
                    { enableHighAccuracy: true, timeout: 10000, maximumAge: 300000 }
                );
            });
        } catch(e) {
            return { error: String(e) };
        }
    }
    """
    
    res = streamlit_js_eval(js_expressions=js, key="get_geo_on_load")
    if isinstance(res, dict) and "lat" in res:
        return {"lat": res["lat"], "lon": res["lon"], "accuracy": res.get("accuracy")}
    return None

@st.cache_data(ttl=3600, show_spinner=False)
def reverse_to_city_town(lat: float, lon: float) -> Dict[str, Optional[str]]:
    """Reverse geocode lat/lon to city/town using Nominatim (OpenStreetMap)."""
    if Nominatim is None:
        return {}

    geolocator = Nominatim(user_agent="makanapa-app/1.0 (contact@example.com)")
    loc = geolocator.reverse((lat, lon), language="en", zoom=14, addressdetails=True, exactly_one=True)
    if not loc:
        return {}

    addr = (loc.raw or {}).get("address", {})
    city = addr.get("city") or addr.get("town") or addr.get("village") or addr.get("municipality")
    suburb = addr.get("suburb") or addr.get("neighbourhood")
    state = addr.get("state")
    country = addr.get("country")

    return {"city": city, "suburb": suburb, "state": state, "country": country, "addr": addr}

# Context-aware rating weight function
def get_dynamic_rating_weight(context: Dict) -> float:
    """Determine how much to weight ratings based on user context"""
    # Higher rating weight for special occasions
    if context.get("mood") in ["celebration", "date", "business", "special occasion"]:
        return 0.5
    # Lower rating weight for budget-conscious users
    if context.get("budget") in ["cheap", "budget"]:
        return 0.2
    # Higher weight for tourists/first-time visitors
    if "first time" in str(context.get("user_notes", "")).lower():
        return 0.4
    # Higher weight for family dining
    if context.get("group_size") in ["family", "group"]:
        return 0.35
    # Default moderate weight
    return 0.3

# Hybrid scoring function
def apply_hybrid_scoring(search_results, rating_weight: float = 0.3, n_results: int = 5):
    """Apply hybrid scoring to search results using existing vectorizer output"""
    enhanced_results = []
    
    for result in search_results:
        semantic_score = result.get('similarity_score', 0)
        rating = result.get('rating', 0)
        
        # Normalize rating to 0-1 scale (assuming 5-star system)
        normalized_rating = rating / 5.0 if rating > 0 else 0
        
        # Hybrid score: combine semantic similarity and rating
        hybrid_score = (1 - rating_weight) * semantic_score + rating_weight * normalized_rating
        
        # Add hybrid score to result
        result_copy = result.copy()
        result_copy['hybrid_score'] = hybrid_score
        enhanced_results.append(result_copy)
    
    # Sort by hybrid score and return top results
    enhanced_results.sort(key=lambda x: x['hybrid_score'], reverse=True)
    return enhanced_results[:n_results]

# Quality-first search function
def apply_quality_first_search(user_input, context, min_rating: float = 4.0):
    """Two-stage search: first try high-rated restaurants, then broaden if needed"""
    context_str = ", ".join([f"{k}: {v}" for k, v in context.items() if v])
    enhanced_query = f"{user_input}. Context: {context_str}"
    
    try:
        # Stage 1: Try to get high-rated results by getting more results and filtering
        initial_results = vectorizer.search(
            query=enhanced_query,
            n_results=15,  # Get more results to filter
            include_distances=True
        )
        
        # Filter for high-rated restaurants
        high_rated_results = [r for r in initial_results if r.get('rating', 0) >= min_rating]
        
        if len(high_rated_results) >= 5:
            return high_rated_results[:5]
        
        # Stage 2: If not enough high-rated results, use lower threshold
        fallback_results = [r for r in initial_results if r.get('rating', 0) >= 3.5]
        
        # Combine and deduplicate by restaurant name
        seen_restaurants = {r['restaurant'] for r in high_rated_results}
        for result in fallback_results:
            if result['restaurant'] not in seen_restaurants and len(high_rated_results) < 5:
                high_rated_results.append(result)
                seen_restaurants.add(result['restaurant'])
        
        return high_rated_results[:5]
        
    except Exception as e:
        st.error(f"Quality-first search error: {e}")
        # Fallback to regular search
        return vectorizer.search(query=enhanced_query, n_results=5, include_distances=True)

def extract_context_from_message(user_input, current_context):
    """Extract context clues from user input using LLM."""
    try:
        extraction_prompt = CONTEXT_EXTRACTOR_PROMPT.format(
            user_input=user_input,
            current_context=current_context
        )
        
        result = llm_enhancer.enhance_search_results(extraction_prompt, [])
        response_text = result.get("enhanced_summary", "{}")
        
        if not response_text.strip():
            return current_context
            
        cleaned_response = clean_json_response(response_text)
        extracted = json.loads(cleaned_response)
        
        # Validate extracted data
        if not isinstance(extracted, dict):
            return current_context
            
        updated_context = current_context.copy()
        for key, value in extracted.items():
            if value and key in updated_context:
                updated_context[key] = value
                
        return updated_context
        
    except json.JSONDecodeError as e:
        st.warning(f"JSON parsing error: {e}")
        return current_context
    except Exception as e:
        st.error(f"Context extraction error: {e}")
        return current_context

def analyze_context_completeness(user_input, context, chat_history):
    """Analyze if we need more context before making recommendations."""
    try:
        analysis_prompt = CONTEXT_ANALYZER_PROMPT.format(
            context=context,
            user_input=user_input,
            history=chat_history[-5:] if len(chat_history) > 5 else chat_history
        )
        
        result = llm_enhancer.enhance_search_results(analysis_prompt, [])
        response_text = result.get("enhanced_summary", "{}")
        
        if not response_text.strip():
            # Return default analysis
            return {
                "needs_more_context": False,
                "follow_up_question": "Let me suggest something based on what you've told me!",
                "ready_for_recommendation": True,
                "confidence_level": "medium"
            }
            
        cleaned_response = clean_json_response(response_text)
        analysis = json.loads(cleaned_response)
        
        # Validate required fields
        required_fields = ["needs_more_context", "ready_for_recommendation"]
        for field in required_fields:
            if field not in analysis:
                analysis[field] = False
                
        return analysis
        
    except json.JSONDecodeError as e:
        st.warning(f"JSON parsing error in analysis: {e}")
        return {
            "needs_more_context": False,
            "follow_up_question": "Let me suggest something based on what you've told me!",
            "ready_for_recommendation": True
        }
    except Exception as e:
        st.error(f"Context analysis error: {e}")
        return {
            "needs_more_context": False,
            "ready_for_recommendation": True
        }

# MCP initialization
async def initialize_mcp():
    """Initialize MCP connections."""
    if not st.session_state.mcp_connected and MCP_AVAILABLE:
        mcp_manager = MCPManager()
        await mcp_manager.connect_to_servers()
        st.session_state.mcp_manager = mcp_manager
        st.session_state.mcp_connected = True
        return mcp_manager
    return st.session_state.get('mcp_manager')

# Enhanced food recommendation function with MCP support
async def get_food_recommendation_with_mcp(user_input, context, chat_history):
    """Get food recommendations using context and MCP-enhanced data."""
    enhanced_context = context.copy()
    
    # Try to get MCP manager, fallback to mock data if not available
    mcp_manager = st.session_state.get('mcp_manager')
    
    if mcp_manager and mcp_manager.servers:
        # Get weather data from MCP server
        if context.get("location"):
            try:
                weather_result = await mcp_manager.call_tool(
                    "weather",
                    "get_weather",
                    {
                        "location": context["location"],
                        "meal_time": context.get("meal_type", "lunch")
                    }
                )
                
                if weather_result and "error" not in str(weather_result):
                    # Parse JSON response if it's a string
                    if isinstance(weather_result, str):
                        weather_result = json.loads(weather_result)
                    enhanced_context["weather"] = weather_result
                else:
                    # Fallback to mock data
                    enhanced_context["weather"] = get_mock_weather_data(
                        context["location"],
                        context.get("meal_type", "lunch")
                    )
            except Exception as e:
                st.warning(f"Weather service error: {e}")
                enhanced_context["weather"] = get_mock_weather_data(
                    context["location"],
                    context.get("meal_type", "lunch")
                )
        
        # Get local events from MCP server
        if context.get("location"):
            try:
                events_result = await mcp_manager.call_tool(
                    "events",
                    "get_local_events",
                    {"location": context["location"]}
                )
                
                if events_result and "error" not in str(events_result):
                    # Parse JSON response if it's a string
                    if isinstance(events_result, str):
                        events_result = json.loads(events_result)
                    enhanced_context["events"] = events_result
                else:
                    # Fallback to mock data
                    enhanced_context["events"] = get_mock_events_data(context["location"])
            except Exception as e:
                st.warning(f"Events service error: {e}")
                enhanced_context["events"] = get_mock_events_data(context["location"])
    else:
        # Use mock data when MCP is not available
        if context.get("location"):
            enhanced_context["weather"] = get_mock_weather_data(
                context["location"],
                context.get("meal_type", "lunch")
            )
            enhanced_context["events"] = get_mock_events_data(context["location"])

    # Use existing search logic with enhanced context
    try:
        context_str = ", ".join([f"{k}: {v}" for k, v in enhanced_context.items() if v])
        enhanced_query = f"{user_input}. Context: {context_str}"

        # Get search configuration from session state
        search_config = st.session_state.get('search_config', {"method": "hybrid", "rating_weight": 0.3})

        # Apply different search strategies based on configuration
        if search_config["method"] == "quality":
            search_results = apply_quality_first_search(
                user_input,
                enhanced_context,
                min_rating=search_config.get("min_rating", 4.0)
            )
        elif search_config["method"] == "semantic":
            search_results = vectorizer.search(
                query=enhanced_query,
                n_results=5,
                include_distances=True
            )
        else:
            # Hybrid search (default)
            initial_results = vectorizer.search(
                query=enhanced_query,
                n_results=15,
                include_distances=True
            )
            rating_weight = search_config.get("rating_weight", get_dynamic_rating_weight(enhanced_context))
            search_results = apply_hybrid_scoring(initial_results, rating_weight, n_results=5)

        # Enhanced recommendation prompt with MCP data
        recommendation_prompt = f"""
User context: {enhanced_context}
User request: {user_input}
Chat history: {chat_history[-3:] if len(chat_history) > 3 else chat_history}
Restaurant search results: {search_results}

{SYSTEM_PROMPT}

Based on the context (including weather and local events), restaurant data,
and user preferences, provide specific, actionable food recommendations.
Consider how weather and events might affect the dining experience.
Include restaurant names, dish suggestions, and why they match the user's needs.
Be conversational and use Malaysian terms naturally.
"""

        enhanced_response = llm_enhancer.enhance_search_results(
            recommendation_prompt,
            search_results
        )
        
        return enhanced_response.get("enhanced_summary", "Let me think of something good for you!")

    except Exception as e:
        return f"Aiya! Having some trouble with my recommendations. Error: {str(e)}"

# Debug functions
def debug_search_issues():
    """Debug function to check vector database status"""
    try:
        # Check if collection exists and has data
        collection = vectorizer.collection
        count = collection.count()
        st.write(f"Collection has {count} documents")
        
        if count == 0:
            st.error("Vector database is empty! You need to load restaurant data first.")
            return False
            
        # Test a simple search
        test_results = vectorizer.search("food", n_results=3)
        st.write(f"Test search returned {len(test_results)} results")
        
        return True
        
    except Exception as e:
        st.error(f"Database error: {e}")
        return False

def test_llm_enhancer():
    """Test if LLM enhancer is working"""
    try:
        test_prompt = "Say hello in JSON format: {\"message\": \"hello\"}"
        result = llm_enhancer.enhance_search_results(test_prompt, [])
        st.write("LLM Test Result:", result)
        return True
    except Exception as e:
        st.error(f"LLM Enhancer Error: {e}")
        return False

# ---------------------------- UI ----------------------------

st.title("🍜 MakanApa")
st.markdown("**Your friendly food discovery companion with enhanced context awareness - Let's figure out what to eat!**")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hey there! I'm MakanApa, your enhanced food discovery buddy 😊 I can now consider weather, events, and more context to give you better recommendations! What's your makan situation right now?"}
    ]

if "user_context" not in st.session_state:
    st.session_state.user_context = {
        "location": None,
        "meal_type": None,
        "dietary_restrictions": None,
        "group_size": None,
        "budget": None,
        "mood": None,
        "cuisine_preference": None,
        "timezone": None,
        "local_time": None,
        "coords": None,
        "location_label": None,
    }

if "did_reverse_geo" not in st.session_state:
    st.session_state.did_reverse_geo = False

# Initialize search config
if "search_config" not in st.session_state:
    st.session_state.search_config = {"method": "hybrid", "rating_weight": 0.3}

# Initialize MCP state
if "mcp_manager" not in st.session_state:
    st.session_state.mcp_manager = None
    st.session_state.mcp_connected = False

# Auto timezone + local time + meal type
tz_name, now_local = detect_timezone_and_local_time()
if tz_name:
    st.session_state.user_context["timezone"] = tz_name
if now_local:
    st.session_state.user_context["local_time"] = now_local.strftime("%Y-%m-%d %H:%M")
    if not st.session_state.user_context.get("meal_type"):
        st.session_state.user_context["meal_type"] = infer_meal_type(now_local)

# Auto geolocation + reverse geocoding to city/town
if not st.session_state.did_reverse_geo:
    coords = get_browser_coords()
    if coords:
        st.session_state.user_context["coords"] = coords
        place = reverse_to_city_town(coords["lat"], coords["lon"])
        if place.get("city"):
            st.session_state.user_context["location"] = place["city"]
            label = None
            if place.get("city") and place.get("state"):
                label = f"{place['city']}, {place['state']}"
            else:
                label = place.get("city") or place.get("state") or None
            st.session_state.user_context["location_label"] = label
        st.session_state.did_reverse_geo = True

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Sidebar with context display and quick starters
with st.sidebar:
    st.markdown("### 🎯 What I Know About You")
    context_items = []
    for key, value in st.session_state.user_context.items():
        if value:
            title = key.replace("_", " ").title()
            if key == "coords" and isinstance(value, dict):
                display = f"lat: {value.get('lat'):.4f}, lon: {value.get('lon'):.4f}"
                context_items.append(f"**{title}**: {display}")
            else:
                context_items.append(f"**{title}**: {value}")

    if context_items:
        for item in context_items:
            st.markdown(f"- {item}")
    else:
        st.markdown("*Tell me more about your makan situation!*")

    # Reset context button
    if st.button("🔄 Reset My Info"):
        base = st.session_state.user_context
        st.session_state.user_context = {k: None for k in base}
        st.session_state.did_reverse_geo = False
        st.rerun()

    st.markdown("---")

    # MCP Servers Status
    st.markdown("### 🔌 MCP Servers")
    if st.session_state.mcp_connected and st.session_state.mcp_manager:
        st.success("✅ MCP Connected")
        for name, server_info in st.session_state.mcp_manager.servers.items():
            tools_count = len(server_info.get("tools", {}))
            st.markdown(f"**{name}**: {tools_count} tools")
            
            # Show available tools in expander
            if tools_count > 0:
                with st.expander(f"Show {name} tools"):
                    tools = server_info.get("tools", {})
                    for tool_name, tool_info in tools.items():
                        st.caption(f"- **{tool_name}**: {getattr(tool_info, 'description', 'No description')}")
    else:
        if MCP_AVAILABLE:
            if st.button("🔌 Connect to MCP Servers"):
                with st.spinner("Connecting to MCP servers..."):
                    try:
                        run_async(initialize_mcp())
                        st.success("MCP servers connected!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to connect: {e}")
            st.info("MCP servers not connected (using mock data)")
        else:
            st.warning("MCP not available")
            st.caption("Install with: `pip install mcp fastmcp httpx`")
            st.info("Using mock weather/events data")

    st.markdown("---")

    # Debug Mode
    debug_mode = st.checkbox("🐛 Debug Mode")

    if debug_mode:
        st.markdown("### Debug Info")
        st.json(st.session_state.user_context)
        
        if st.session_state.messages:
            last_msg = st.session_state.messages[-1]
            st.write("Last message:", last_msg)
            
        if st.button("🔍 Debug Database"):
            debug_search_issues()
            
        if st.button("Test LLM"):
            test_llm_enhancer()

    # Search method configuration
    st.markdown("### ⚙️ Search Settings")
    search_method = st.selectbox(
        "Search Method:",
        ["Smart Hybrid", "Quality First", "Pure Semantic"],
        help="Smart Hybrid balances relevance and ratings. Quality First prioritizes highly-rated places. Pure Semantic focuses only on text similarity."
    )

    if search_method == "Smart Hybrid":
        rating_importance = st.slider(
            "Rating Importance",
            0.0, 1.0,
            st.session_state.search_config.get("rating_weight", 0.3),
            0.1,
            help="Higher values prioritize highly-rated restaurants"
        )
        
        st.session_state.search_config = {"method": "hybrid", "rating_weight": rating_importance}
        st.caption(f"Current weight: {rating_importance:.1f} (0=semantic only, 1=rating only)")
        
    elif search_method == "Quality First":
        min_rating = st.slider(
            "Minimum Rating",
            3.0, 5.0,
            st.session_state.search_config.get("min_rating", 4.0),
            0.1,
            help="Minimum rating threshold for initial search"
        )
        
        st.session_state.search_config = {"method": "quality", "min_rating": min_rating}
        st.caption(f"Will prioritize restaurants with {min_rating:.1f}+ stars")
        
    else:
        st.session_state.search_config = {"method": "semantic"}
        st.caption("Pure semantic search - only text similarity matters")

    st.markdown("---")

    # Quick Starters
    st.markdown("### 💡 Quick Starters")
    starter_prompts = [
        "I'm craving something spicy for lunch",
        "What's good for breakfast in Penang?",
        "Suggest something light for dinner",
        "Best local food for first-time visitors",
        "I want comfort food for a rainy day",
        "Quick snack near Georgetown"
    ]

    for prompt in starter_prompts:
        if st.button(prompt, key=f"starter_{prompt}"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.rerun()

    st.markdown("---")

    # Tools section
    st.markdown("### 🛠️ Tools")
    if st.button("Clear Chat History"):
        st.session_state.messages = [
            {"role": "assistant", "content": "Fresh start! What are you looking to makan today? 🍽️"}
        ]
        st.rerun()

    if st.button("Show DB Stats"):
        stats = vectorizer.get_database_stats()
        if isinstance(stats, dict) and "error" in stats:
            st.error(stats["error"])
        else:
            st.json(stats)

    # Show current search method
    current_method = st.session_state.search_config["method"]
    if current_method == "hybrid":
        weight = st.session_state.search_config.get("rating_weight", 0.3)
        st.caption(f"🔍 Using Hybrid Search (Rating weight: {weight:.1f})")
    elif current_method == "quality":
        min_rating = st.session_state.search_config.get("min_rating", 4.0)
        st.caption(f"🔍 Using Quality-First Search (Min: {min_rating:.1f}⭐)")
    else:
        st.caption("🔍 Using Pure Semantic Search")

    # Small hint if optional deps missing
    if streamlit_js_eval is None:
        st.caption("💡 Tip: `pip install streamlit_js_eval` for auto location detection")
    if Nominatim is None:
        st.caption("💡 Tip: `pip install geopy` for reverse geocoding")

# Main Chat Input Handler
if user_input := st.chat_input("Tell me about your makan situation..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.user_context = extract_context_from_message(
        user_input, st.session_state.user_context
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Checking weather, events, and makan options..."):
            try:
                analysis = analyze_context_completeness(
                    user_input,
                    st.session_state.user_context,
                    st.session_state.messages
                )

                if analysis.get("needs_more_context", False):
                    response_text = analysis.get(
                        "follow_up_question",
                        "Tell me a bit more about what you're looking for!"
                    )
                else:
                    # Use MCP-enhanced recommendation
                    response_text = run_async(get_food_recommendation_with_mcp(
                        user_input,
                        st.session_state.user_context,
                        st.session_state.messages
                    ))

                # Stream response
                response_placeholder = st.empty()
                full_response = ""
                for chunk in response_text.split():
                    full_response += chunk + " "
                    response_placeholder.markdown(full_response + "▌")
                    time.sleep(0.02)

                response_placeholder.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})

            except Exception as e:
                error_response = f"Aiya! Something went wrong with my food brain 🤯 Can you try asking me again? Error: {str(e)}"
                st.error(error_response)
                st.session_state.messages.append({"role": "assistant", "content": error_response})
