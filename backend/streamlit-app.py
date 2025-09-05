import streamlit as st
import time
import json
import re
from typing import Optional, Dict
from datetime import datetime, timezone as dt_timezone
from zoneinfo import ZoneInfo

# Optional dependencies for auto geolocation + reverse geocoding
try:
    from geopy.geocoders import Nominatim  # pip install geopy
except ImportError:
    Nominatim = None

try:
    from streamlit_js_eval import streamlit_js_eval  # pip install streamlit_js_eval
except ImportError:
    streamlit_js_eval = None

# Import your backend classes and functions from your project
from test import RestaurantReviewVectorizer
from llm_enhancer import get_llm_enhancer

# Initialize backend objects only once
@st.cache_resource
def get_vectorizer():
    v = RestaurantReviewVectorizer(
        model_name="all-MiniLM-L6-v2",
        persist_directory="./backend/restaurant_chroma_db"
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
{
  "needs_more_context": true/false,
  "follow_up_question": "What's your question?",
  "missing_info": ["location", "meal_type"],
  "ready_for_recommendation": true/false,
  "confidence_level": "low/medium/high"
}
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
Example: {"location": "Penang", "meal_type": "lunch", "mood": "spicy", "dietary_restrictions": null}
"""

# ---------------------------- Utilities ----------------------------

def clean_json_response(response_text: str) -> str:
    """Extract JSON from fenced code or return raw text if not fenced."""
    m = re.search(r"``````", response_text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return response_text.strip()

def detect_timezone_and_local_time():
    """Read the browser's IANA timezone and compute local time."""
    try:
        tz_name = getattr(st.context, "timezone", None)  # e.g., "Asia/Kuala_Lumpur"
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
    """
    Request browser geolocation via JS and return {'lat','lon','accuracy'} if available.
    Requires streamlit_js_eval and runs in secure contexts (HTTPS).
    """
    if streamlit_js_eval is None:
        return None
    js = """
    async () => {
      if (!('geolocation' in navigator)) return { error: 'unsupported' };
      try {
        // This may prompt the user for permission depending on prior state
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
    """
    Reverse geocode lat/lon to city/town using Nominatim (OpenStreetMap).
    Returns dict with city, suburb, state, country.
    """
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

def extract_context_from_message(user_input, current_context):
    """Extract context clues from user input using LLM."""
    try:
        extraction_prompt = CONTEXT_EXTRACTOR_PROMPT.format(
            user_input=user_input,
            current_context=current_context
        )
        result = llm_enhancer.enhance_search_results(extraction_prompt, [])
        response_text = result.get("enhanced_summary", "{}")
        cleaned_response = clean_json_response(response_text)
        extracted = json.loads(cleaned_response)
        updated_context = current_context.copy()
        for key, value in extracted.items():
            if value and key in updated_context:
                updated_context[key] = value
        return updated_context
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
        cleaned_response = clean_json_response(response_text)
        analysis = json.loads(cleaned_response)
        return analysis
    except Exception as e:
        st.error(f"Context analysis error: {e}")
        return {
            "needs_more_context": False,
            "follow_up_question": "Let me suggest something based on what you've told me!",
            "ready_for_recommendation": True
        }

def get_food_recommendation(user_input, context, chat_history):
    """Get food recommendations using context and search results."""
    try:
        context_str = ", ".join([f"{k}: {v}" for k, v in context.items() if v])
        enhanced_query = f"{user_input}. Context: {context_str}"
        search_results = vectorizer.search(
            query=enhanced_query,
            n_results=5,
            include_distances=True
        )
        recommendation_prompt = f"""
        User context: {context}
        User request: {user_input}
        Chat history: {chat_history[-3:] if len(chat_history) > 3 else chat_history}
        Restaurant  {search_results}

        {SYSTEM_PROMPT}

        Based on the context and restaurant data, provide specific, actionable food recommendations.
        Include restaurant names, dish suggestions, and why they match the user's needs.
        Be conversational and use Malaysian terms naturally.
        """
        enhanced_response = llm_enhancer.enhance_search_results(recommendation_prompt, search_results)
        return enhanced_response.get("enhanced_summary", "Let me think of something good for you!")
    except Exception as e:
        return f"Aiya! Having some trouble with my recommendations. Error: {str(e)}"

# ---------------------------- UI ----------------------------

st.title("🍜 MakanApa")
st.markdown("**Your friendly food discovery companion - Let's figure out what to eat!**")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hey there! I'm MakanApa, your food discovery buddy 😊 What's your makan situation right now? Are you feeling hungry, looking for something specific, or just browsing for ideas?"}
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
            st.markdown(f"-  {item}")
    else:
        st.markdown("*Tell me more about your makan situation!*")

    # Reset context button
    if st.button("🔄 Reset My Info"):
        base = st.session_state.user_context
        st.session_state.user_context = {k: None for k in base}
        st.session_state.did_reverse_geo = False
        st.rerun()

    st.markdown("---")
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

    # Small hint if optional deps missing
    if streamlit_js_eval is None:
        st.caption("Tip: pip install streamlit_js_eval to enable auto location detection.")
    if Nominatim is None:
        st.caption("Tip: pip install geopy to enable reverse geocoding to city/town.")

# Chat input handling
if user_input := st.chat_input("Tell me about your makan situation..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.user_context = extract_context_from_message(
        user_input, st.session_state.user_context
    )
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking about your makan options..."):
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
                    response_text = get_food_recommendation(
                        user_input,
                        st.session_state.user_context,
                        st.session_state.messages
                    )

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
