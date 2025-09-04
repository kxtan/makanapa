import os
import json
import requests
from typing import List, Dict, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class LLMEnhancer:
    def __init__(self):
        """Initialize the LLM enhancer with OpenRouter configuration"""
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        self.model_name = os.getenv("OPENROUTER_MODEL", "openai/gpt-3.5-turbo")
        
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is required")
        
        # Initialize OpenRouter configuration
        self.llm_available = True
        print(f"✅ LLM Enhancer initialized with model: {self.model_name}")
        
        print(f"✅ LLM Enhancer initialized with model: {self.model_name}")

    def enhance_search_results(self, query: str, search_results: List[Dict]) -> Dict:
        """
        Enhance semantic search results using LLM analysis
        
        Args:
            query: The original search query
            search_results: List of search results from vector database
            
        Returns:
            Dictionary containing enhanced results and LLM analysis
        """
        if not search_results:
            return {
                "enhanced_summary": "No relevant results found for your search query.",
                "original_results": search_results
            }
        
        # Try to use OpenRouter API directly
        try:
            formatted_results = self._format_results_for_llm(search_results)
            
            # Call OpenRouter API directly
            enhanced_analysis = self._call_openrouter_api(query, formatted_results)
            
            return {
                "enhanced_summary": enhanced_analysis,
                "original_results": search_results,
                "llm_model": self.model_name
            }
            
        except Exception as e:
            print(f"❌ Error in LLM enhancement: {str(e)}")
            return self._fallback_enhancement(query, search_results, str(e))

    def _format_results_for_llm(self, results: List[Dict]) -> str:
        """Format search results into a readable string for LLM context"""
        formatted = []
        
        for i, result in enumerate(results, 1):
            restaurant = result.get('restaurant', 'Unknown Restaurant')
            location = result.get('location', 'Unknown Location')
            rating = result.get('rating', 0)
            similarity = result.get('similarity_score', 0)
            text = result.get('text', '')[:200]  # Truncate long text
            
            formatted.append(
                f"Result {i}: {restaurant} ({location}) - Rating: {rating}/5 - "
                f"Similarity: {similarity:.3f}\n"
                f"Review Excerpt: \"{text}...\"\n"
            )
        
        return "\n".join(formatted)

    def _call_openrouter_api(self, query: str, formatted_results: str) -> str:
        """Call OpenRouter API directly for LLM enhancement"""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            prompt = f"""You are a restaurant recommendation assistant. Your task is to analyze semantic search results and provide enhanced, contextual recommendations.

Guidelines:
1. Analyze the search results and identify patterns or common themes
2. Provide a concise summary of the best recommendations
3. Highlight key features like cuisine type, atmosphere, price range, and specialties
4. If results are from different locations, mention this and provide location-specific insights
5. Be honest about limitations - if results are sparse or not very relevant, say so
6. Keep responses conversational and helpful
7. Focus on the most relevant information from the search results

Search Query: {query}

Search Results:
{formatted_results}

Please analyze these search results and provide enhanced recommendations:"""
            
            payload = {
                "model": self.model_name,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3,
                "max_tokens": 1000
            }
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            response.raise_for_status()
            result = response.json()
            
            return result["choices"][0]["message"]["content"]
            
        except Exception as e:
            print(f"❌ OpenRouter API call failed: {e}")
            raise

    def _fallback_enhancement(self, query: str, results: List[Dict], error_msg: Optional[str] = None) -> Dict:
        """Fallback enhancement when LLM is not available"""
        enhancement = self.simple_enhancement(query, results)
        result = {
            "enhanced_summary": enhancement,
            "original_results": results,
            "llm_model": "simple_fallback"
        }
        if error_msg:
            result["error"] = error_msg
        return result

    def simple_enhancement(self, query: str, results: List[Dict]) -> str:
        """
        Simple enhancement method for quick testing
        """
        if not results:
            return "No results found for your search."
        
        # Just return a basic enhancement for now
        top_result = results[0]
        restaurant = top_result.get('restaurant', 'a restaurant')
        location = top_result.get('location', '')
        rating = top_result.get('rating', 0)
        
        return f"Based on your search for '{query}', I found {restaurant} in {location} with a {rating}/5 rating that seems most relevant. The reviews suggest this might be a good match for your preferences."


# Singleton instance for easy access
llm_enhancer = None

def get_llm_enhancer():
    """Get or create the LLM enhancer singleton"""
    global llm_enhancer
    if llm_enhancer is None:
        try:
            llm_enhancer = LLMEnhancer()
        except Exception as e:
            print(f"⚠️  LLM Enhancer initialization failed: {e}")
            print("⚠️  Falling back to simple enhancement mode")
            llm_enhancer = SimpleEnhancer()
    return llm_enhancer


class SimpleEnhancer:
    """Fallback enhancer when LLM is not available"""
    def enhance_search_results(self, query: str, search_results: List[Dict]) -> Dict:
        return {
            "enhanced_summary": self.simple_enhancement(query, search_results),
            "original_results": search_results,
            "llm_model": "simple_fallback"
        }
    
    def simple_enhancement(self, query: str, results: List[Dict]) -> str:
        if not results:
            return "No results found for your search query."
        
        restaurants = []
        locations = set()
        
        for result in results[:3]:  # Top 3 results
            restaurant = result.get('restaurant', 'Unknown')
            location = result.get('location', '')
            rating = result.get('rating', 0)
            
            restaurants.append(f"{restaurant} (Rating: {rating}/5)")
            if location:
                locations.add(location)
        
        location_str = ", ".join(locations) if locations else "various locations"
        
        return (f"Based on your search for '{query}', I found several options including "
                f"{', '.join(restaurants)} in {location_str}. These appear to be the most "
                f"relevant matches from our database.")


if __name__ == "__main__":
    # Test the enhancer
    enhancer = get_llm_enhancer()
    
    # Test with mock data
    test_results = [
        {
            "restaurant": "Italian Bistro",
            "location": "Kuching",
            "rating": 4.5,
            "similarity_score": 0.85,
            "text": "Amazing pasta and cozy atmosphere. The service was excellent and the wine selection was perfect."
        },
        {
            "restaurant": "Pizza Palace",
            "location": "Kuching", 
            "rating": 4.2,
            "similarity_score": 0.78,
            "text": "Great pizza and friendly staff. The restaurant has a warm, family-friendly vibe."
        }
    ]
    
    enhanced = enhancer.enhance_search_results("cozy Italian restaurant", test_results)
    print("Enhanced Result:")
    print(enhanced["enhanced_summary"])