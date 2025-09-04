"""FastAPI application for restaurant review vector search."""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import pandas as pd
import io
from datetime import datetime
from typing import List, Optional, Dict, Any

from ..services.processor import RestaurantProcessor
from ..models.schemas import (
    SearchRequest, SearchResponse, SearchResult,
    ProcessRequest, ProcessingStats, DatabaseStats,
    HealthResponse
)
from ..config.settings import config
from ..utils.logging import logger
from ..utils.exceptions import VectorizerError


# Global processor instance
processor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global processor
    try:
        logger.info("Starting Restaurant Review Vector Search API")
        processor = RestaurantProcessor()
        logger.info("API initialized successfully")
        yield
    except Exception as e:
        logger.error(f"Failed to initialize API: {str(e)}")
        raise
    finally:
        logger.info("API shutdown complete")


app = FastAPI(
    title=config.api_title,
    version=config.api_version,
    description="Semantic search API for restaurant reviews using vector embeddings",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Restaurant Review Vector Search API",
        "version": config.api_version,
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        db_connected = processor.health_check() if processor else False
        total_docs = processor.get_database_stats().total_chunks if db_connected else 0

        return HealthResponse(
            status="healthy" if db_connected else "unhealthy",
            timestamp=datetime.now(),
            database_connected=db_connected,
            total_documents=total_docs
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@app.post("/search", response_model=SearchResponse)
async def search_reviews(search_request: SearchRequest):
    """
    Semantic search endpoint for restaurant reviews.

    Args:
        search_request: Search parameters

    Returns:
        Search results with similarity scores
    """
    if not processor:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        start_time = time.time()

        results = processor.search_reviews(
            query=search_request.query,
            n_results=search_request.n_results,
            filters=search_request.filters,
            include_distances=search_request.include_distances
        )

        search_time = time.time() - start_time

        return SearchResponse(
            query=search_request.query,
            results=results,
            total_results=len(results),
            search_time=search_time
        )

    except VectorizerError as e:
        logger.error(f"Search request failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected search error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/process", response_model=ProcessingStats)
async def process_restaurant_data(
    file: UploadFile = File(...),
    text_column: str = "Review",
    chunk_method: str = "sentence",
    batch_size: int = 100
):
    """
    Process and store restaurant data from CSV file.

    Args:
        file: CSV file containing restaurant data
        text_column: Column name containing review text
        chunk_method: Method for chunking text
        batch_size: Batch size for processing

    Returns:
        Processing statistics
    """
    if not processor:
        raise HTTPException(status_code=503, detail="Service not initialized")

    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV")

    try:
        # Read CSV file
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))

        # Create processing request
        request = ProcessRequest(
            text_column=text_column,
            chunk_method=chunk_method,
            batch_size=batch_size
        )

        # Process the data
        stats = processor.process_dataframe(df, request)

        logger.info(f"Successfully processed {len(df)} restaurants")
        return stats

    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="CSV file is empty")
    except pd.errors.ParserError:
        raise HTTPException(status_code=400, detail="Invalid CSV format")
    except VectorizerError as e:
        logger.error(f"Processing request failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected processing error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/stats", response_model=DatabaseStats)
async def get_database_stats():
    """Get database statistics."""
    if not processor:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        return processor.get_database_stats()
    except VectorizerError as e:
        logger.error(f"Failed to get stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/locations", response_model=List[str])
async def get_locations():
    """Get list of available locations."""
    if not processor:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        stats = processor.get_database_stats()
        return stats.locations
    except VectorizerError as e:
        logger.error(f"Failed to get locations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search/location/{location}")
async def search_by_location(
    location: str,
    query: str,
    n_results: int = 5
):
    """
    Search reviews filtered by location.

    Args:
        location: Location to filter by
        query: Search query
        n_results: Number of results to return

    Returns:
        Search results for the specified location
    """
    if not processor:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        results = processor.search_reviews(
            query=query,
            n_results=n_results,
            filters={"location": location},
            include_distances=True
        )

        return {
            "location": location,
            "query": query,
            "results": results,
            "total_results": len(results)
        }

    except VectorizerError as e:
        logger.error(f"Location search failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


# Entry point for running the API
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "restaurant_vectorizer.api.main:app",
        host=config.api_host,
        port=config.api_port,
        reload=True
    )