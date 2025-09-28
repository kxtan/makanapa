from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import uvicorn

# Import the vectorizer from test.py
from test import RestaurantReviewVectorizer
from llm_enhancer import get_llm_enhancer

# Initialize FastAPI app
app = FastAPI(
    title="MakanApa API",
    description="Food discovery API for restaurant recommendations",
    version="1.0.0"
)

# Add CORS middleware to allow requests from frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class SearchRequest(BaseModel):
    query: str
    n_results: Optional[int] = 5
    filters: Optional[Dict] = None

class SearchResult(BaseModel):
    text: str
    restaurant: str
    location: str
    rating: float
    similarity_score: Optional[float] = None
    metadata: Dict

class SearchResponse(BaseModel):
    results: List[SearchResult]
    query: str
    total_results: int
    enhanced_summary: Optional[str] = None
    llm_model: Optional[str] = None

# Initialize vectorizer (similar to Streamlit app)
vectorizer = RestaurantReviewVectorizer(
    model_name="all-MiniLM-L6-v2",
    persist_directory="./restaurant_chroma_db"
)
vectorizer.create_or_get_collection("restaurant_reviews")

# Initialize LLM enhancer
llm_enhancer = get_llm_enhancer()

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "MakanApa API is running",
        "version": "1.0.0",
        "endpoints": {
            "search": "/search",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "database_ready": True}

@app.post("/search", response_model=SearchResponse)
async def search_restaurants(request: SearchRequest):
    """
    Search for restaurants based on semantic similarity
    
    Args:
        query: Search query text
        n_results: Number of results to return (default: 5)
        filters: Optional metadata filters (e.g., {"location": "Kuching"})
    """
    try:
        # Perform semantic search
        results = vectorizer.search(
            query=request.query,
            n_results=request.n_results,
            filters=request.filters,
            include_distances=True
        )
        
        # Format results for response
        formatted_results = []
        for result in results:
            formatted_results.append(SearchResult(
                text=result['text'],
                restaurant=result['restaurant'],
                location=result['location'],
                rating=result['rating'],
                similarity_score=result.get('similarity_score'),
                metadata=result['metadata']
            ))
        
        # Enhance results with LLM
        enhanced_results = llm_enhancer.enhance_search_results(
            request.query, results
        )
        
        return SearchResponse(
            results=formatted_results,
            query=request.query,
            total_results=len(formatted_results),
            enhanced_summary=enhanced_results["enhanced_summary"],
            llm_model=enhanced_results.get("llm_model", "unknown")
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")

@app.get("/search")
async def search_get(query: str, n_results: int = 5):
    """
    GET endpoint for search (convenience method)
    """
    request = SearchRequest(query=query, n_results=n_results)
    return await search_restaurants(request)

@app.get("/stats")
async def get_database_stats():
    """Get database statistics"""
    try:
        stats = vectorizer.get_database_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)