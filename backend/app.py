from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, List
import pandas as pd
from test import RestaurantReviewVectorizer  # Assuming your file is named test.py

app = FastAPI(
    title="Restaurant Review Semantic Search API",
    description="API to interact with Chroma vector DB using semantic search",
    version="1.0.0"
)

# Initialize the vectorizer and load collection
vectorizer = RestaurantReviewVectorizer(
    model_name="all-MiniLM-L6-v2",
    persist_directory="./restaurant_chroma_db"
)
vectorizer.create_or_get_collection("restaurant_reviews")

class SearchRequest(BaseModel):
    query: str
    n_results: int = 5
    filters: Optional[Dict] = None

@app.post("/search", summary="Semantic search in reviews")
def search_reviews(request: SearchRequest):
    try:
        results = vectorizer.search(
            query=request.query,
            n_results=request.n_results,
            filters=request.filters,
            include_distances=True
        )
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats", summary="Database statistics")
def db_stats():
    stats = vectorizer.get_database_stats()
    if "error" in stats:
        raise HTTPException(status_code=500, detail=stats["error"])
    return stats
