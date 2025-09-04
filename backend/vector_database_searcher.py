import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
import numpy as np
import time

class VectorDatabaseSearcher:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", persist_directory: str = "./restaurant_chroma_db"):
        """
        Initialize the vector database searcher with Chroma vector database

        Args:
            model_name: Embedding model name    
            persist_directory: Directory for Chroma database persistence
        """
        self.model = SentenceTransformer(model_name)
        self.persist_directory = persist_directory

        # Initialize Chroma client with persistence
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = None

        print(f"Initialized vector database searcher with model: {model_name}")
        print(f"Chroma database location: {persist_directory}")

    def load_collection(self, collection_name: str = "restaurant_reviews"):
        """Load an existing collection for searching"""
        try:
            self.collection = self.client.get_collection(collection_name)
            print(f"✅ Loaded collection: {collection_name}")
            print(f"   Collection contains {self.collection.count()} documents")
            return self.collection
        except Exception as e:
            print(f"❌ Error loading collection {collection_name}: {str(e)}")
            raise ValueError(f"Collection {collection_name} not found. Please create it first.")

    def search(self, query: str, 
               n_results: int = 5, 
               filters: Optional[Dict] = None,
               include_distances: bool = True) -> List[Dict]:
        """
        Semantic search in the restaurant review database

        Args:
            query: Search query text
            n_results: Number of results to return
            filters: Metadata filters (e.g., {'location': 'Kuching'})
            include_distances: Whether to include similarity scores
        """
        if self.collection is None:
            raise ValueError("Collection not loaded. Call load_collection() first.")

        start_time = time.time()

        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=filters,
                include=['documents', 'metadatas', 'distances'] if include_distances else ['documents', 'metadatas']
            )

            search_time = time.time() - start_time

            # Format results - handle cases where no results are found
            formatted_results = []
            if results['documents'] and results['documents'][0]:
                for i in range(len(results['documents'][0])):
                    result = {
                        'text': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'restaurant': results['metadatas'][0][i].get('restaurant', 'Unknown'),
                        'location': results['metadatas'][0][i].get('location', 'Unknown'),
                        'rating': results['metadatas'][0][i].get('rating', 0)
                    }

                    if include_distances and results['distances'] and results['distances'][0]:
                        # Convert distance to similarity score (closer to 1 = more similar)
                        similarity = 1 - results['distances'][0][i]
                        result['similarity_score'] = similarity

                    formatted_results.append(result)

            print(f"🔍 Search completed in {search_time:.4f} seconds")
            return formatted_results

        except Exception as e:
            print(f"❌ Search error: {str(e)}")
            return []

    def get_database_stats(self) -> Dict:
        """Get statistics about the database"""
        if self.collection is None:
            return {'error': 'Collection not loaded'}

        try:
            count = self.collection.count()

            # Get sample metadata to analyze
            sample = self.collection.get(limit=100, include=['metadatas'])

            locations = set()
            restaurants = set()
            ratings = []

            if sample['metadatas']:
                for metadata in sample['metadatas']:
                    if 'location' in metadata:
                        locations.add(metadata['location'])
                    if 'restaurant' in metadata:
                        restaurants.add(metadata['restaurant'])
                    if 'rating' in metadata:
                        ratings.append(metadata['rating'])

            return {
                'total_chunks': count,
                'unique_locations': len(locations),
                'unique_restaurants': len(restaurants),
                'locations': list(locations),
                'avg_rating': np.mean(ratings) if ratings else 0,
                'collection_name': self.collection.name
            }
        except Exception as e:
            return {'error': str(e)}