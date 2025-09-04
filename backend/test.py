
# Restaurant Review Vectorizer with Chroma Vector Database - FINAL CORRECTED VERSION
# ==================================================================================
# Fixed all regex issues and improved error handling

import pandas as pd
import numpy as np
import re
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple, Optional
import uuid
import time

class RestaurantReviewVectorizer:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", persist_directory: str = "./restaurant_chroma_db"):
        """
        Initialize the vectorizer with Chroma vector database

        Args:
            model_name: Embedding model name    
            persist_directory: Directory for Chroma database persistence
        """
        self.model = SentenceTransformer(model_name)
        self.persist_directory = persist_directory

        # Initialize Chroma client with persistence
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = None

        print(f"Initialized vectorizer with model: {model_name}")
        print(f"Chroma database will be stored in: {persist_directory}")

    def create_or_get_collection(self, collection_name: str = "restaurant_reviews"):
        """Create or retrieve existing collection"""
        try:
            self.collection = self.client.get_collection(collection_name)
            print(f"âœ… Loaded existing collection: {collection_name}")
            print(f"   Collection contains {self.collection.count()} documents")
        except Exception:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={
                    "hnsw:space": "cosine",  # Use cosine similarity for semantic search
                    "hnsw:construction_ef": 200,  # Higher quality index
                    "hnsw:M": 16  # Better connectivity
                }
            )
            print(f"âœ… Created new collection: {collection_name}")

        return self.collection

    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess review text - FIXED ALL REGEX PATTERNS"""
        if not isinstance(text, str):
            return ""

        # Remove HTML tags if any
        text = re.sub(r'<[^>]+>', '', text)

        # Fix common encoding issues
        text = text.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')

        # Remove excessive whitespace and newlines - CORRECTED
        text = re.sub(r'\s+', ' ', text)

        # Remove special characters but keep punctuation for context - CORRECTED
        #text = re.sub(r'[^\w\s.,!?()\-'""]', ' ', text)

        # Remove very short words (less than 2 characters) except common ones
        words = text.split()
        words = [w for w in words if len(w) >= 2 or w.lower() in ['i', 'a']]

        return ' '.join(words).strip()

    def chunk_reviews(self, reviews: List[str], method: str = "sentence", 
                     max_chunk_size: int = 800, overlap: int = 100) -> List[Dict]:
        """
        Chunk restaurant reviews using different strategies

        Args:
            reviews: List of review texts
            method: Chunking method ('sentence', 'paragraph', 'fixed', 'none')
            max_chunk_size: Maximum characters per chunk
            overlap: Character overlap between chunks
        """
        all_chunks = []

        for idx, review in enumerate(reviews):
            if not review or len(review.strip()) < 20:
                continue

            if method == "sentence":
                chunks = self._sentence_chunking(review, max_chunk_size)
            elif method == "paragraph":
                chunks = review.split('\n\n')
            elif method == "fixed":
                chunks = self._fixed_chunking(review, max_chunk_size, overlap)
            else:
                chunks = [review]  # No chunking

            for chunk_idx, chunk in enumerate(chunks):
                if len(chunk.strip()) > 20:  # Skip very short chunks
                    all_chunks.append({
                        'text': chunk.strip(),
                        'review_id': idx,
                        'chunk_id': chunk_idx,
                        'chunk_method': method
                    })

        return all_chunks

    def _sentence_chunking(self, text: str, max_size: int) -> List[str]:
        """Split text by sentences, combining until max_size"""
        sentences = re.split(r'[.!?]+', text)
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            if len(current_chunk + sentence) < max_size:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def _fixed_chunking(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Create fixed-size chunks with overlap"""
        words = text.split()
        chunks = []

        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)

            if i + chunk_size >= len(words):
                break

        return chunks

    def process_restaurant_data(self, df: pd.DataFrame, 
                              text_column: str = 'Review',
                              chunk_method: str = "sentence",
                              batch_size: int = 100) -> Dict:
        """
        Main method to process restaurant review data and store in Chroma

        Args:
            df: DataFrame with restaurant data
            text_column: Column name containing review text
            chunk_method: Method for chunking text
            batch_size: Batch size for processing
        """
        if self.collection is None:
            raise ValueError("Collection not initialized. Call create_or_get_collection() first.")

        print(f"\nðŸ”„ Processing {len(df)} restaurants...")
        start_time = time.time()

        all_documents = []
        all_metadatas = []
        all_ids = []

        for idx, row in df.iterrows():
            # Preprocess the review text
            clean_text = self.preprocess_text(str(row[text_column]))

            if not clean_text:
                continue

            # Create base metadata for this review
            base_metadata = {
                'restaurant': str(row.get('Restaurant', '')),
                'location': str(row.get('Location', '')),
                'rating': float(row.get('Rating', 0)),
                'count': int(row.get('Count', 0)),
                'weighted_rating': float(row.get('weightedRating', 0)),
                'original_review_id': int(idx)
            }

            # Chunk the review
            chunks = self.chunk_reviews([clean_text], method=chunk_method)

            # Process each chunk
            for i, chunk_info in enumerate(chunks):
                chunk_metadata = base_metadata.copy()
                chunk_metadata.update({
                    'chunk_id': i,
                    'chunk_length': len(chunk_info['text']),
                    'chunk_method': chunk_method,
                    'processing_timestamp': int(time.time())
                })

                # Create unique ID for each chunk
                chunk_id = f"restaurant_{idx}_chunk_{i}_{uuid.uuid4().hex[:8]}"

                all_documents.append(chunk_info['text'])
                all_metadatas.append(chunk_metadata)
                all_ids.append(chunk_id)

        # Add documents to Chroma in batches
        print(f"ðŸ“¥ Adding {len(all_documents)} chunks to Chroma database...")

        for i in range(0, len(all_documents), batch_size):
            batch_docs = all_documents[i:i + batch_size]
            batch_metadata = all_metadatas[i:i + batch_size]
            batch_ids = all_ids[i:i + batch_size]

            try:
                self.collection.add(
                    documents=batch_docs,
                    metadatas=batch_metadata,
                    ids=batch_ids
                )
                print(f"  âœ… Processed batch {i//batch_size + 1}/{(len(all_documents)-1)//batch_size + 1}")
            except Exception as e:
                print(f"  âŒ Error processing batch {i//batch_size + 1}: {str(e)}")
                continue

        processing_time = time.time() - start_time
        total_chunks = self.collection.count()

        print(f"\nâœ… Processing complete!")
        print(f"   Total chunks in database: {total_chunks}")
        print(f"   Processing time: {processing_time:.2f} seconds")
        print(f"   Database location: {self.persist_directory}")

        return {
            'total_chunks': total_chunks,
            'processing_time': processing_time,
            'collection_name': self.collection.name
        }

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
            raise ValueError("Collection not initialized. Call create_or_get_collection() first.")

        start_time = time.time()

        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=filters,
                include=['documents', 'metadatas', 'distances'] if include_distances else ['documents', 'metadatas']
            )

            search_time = time.time() - start_time

            # Format results
            formatted_results = []
            for i in range(len(results['documents'][0])):
                result = {
                    'text': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'restaurant': results['metadatas'][0][i].get('restaurant', 'Unknown'),
                    'location': results['metadatas'][0][i].get('location', 'Unknown'),
                    'rating': results['metadatas'][0][i].get('rating', 0)
                }

                if include_distances:
                    # Convert distance to similarity score (closer to 1 = more similar)
                    similarity = 1 - results['distances'][0][i]
                    result['similarity_score'] = similarity

                formatted_results.append(result)

            print(f"ðŸ” Search completed in {search_time:.4f} seconds")
            return formatted_results

        except Exception as e:
            print(f"âŒ Search error: {str(e)}")
            return []

    def get_database_stats(self) -> Dict:
        """Get statistics about the database"""
        if self.collection is None:
            return {'error': 'Collection not initialized'}

        try:
            count = self.collection.count()

            # Get sample metadata to analyze
            sample = self.collection.get(limit=100, include=['metadatas'])

            locations = set()
            restaurants = set()
            ratings = []

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

    def delete_collection(self, collection_name: str):
        """Delete a collection (use with caution)"""
        try:
            self.client.delete_collection(collection_name)
            print(f"âœ… Deleted collection: {collection_name}")
        except Exception as e:
            print(f"âŒ Error deleting collection: {str(e)}")


# Usage Example and Main Function
def main():
    """
    Example usage of the Restaurant Review Vectorizer with Chroma
    """
    print("ðŸš€ Restaurant Review Vectorizer with Chroma Database")
    print("=" * 60)

    # Initialize vectorizer with Chroma backend
    vectorizer = RestaurantReviewVectorizer(
        model_name="all-MiniLM-L6-v2",  # Fast, good for general purpose
        persist_directory="./restaurant_chroma_db"
    )

    # Create or get collection
    collection = vectorizer.create_or_get_collection("restaurant_reviews")

    try:
        # Load your restaurant data
        df = pd.read_csv('data/combined_restaurant_reviews.csv')
        print(f"\nðŸ“Š Loaded {len(df)} restaurants from CSV")

        # Process and store in Chroma
        processing_stats = vectorizer.process_restaurant_data(
            df, 
            text_column='Review',
            chunk_method='sentence',  # Best for restaurant reviews
            batch_size=50
        )

        # Get database statistics
        print("\nðŸ“ˆ Database Statistics:")
        stats = vectorizer.get_database_stats()
        for key, value in stats.items():
            print(f"   {key}: {value}")

        # Example searches
        print("\nðŸ” Example Searches:")
        print("-" * 30)

        # Search 1: General cuisine preference
        print("\n1. Search: 'great pasta and cozy atmosphere'")
        results = vectorizer.search("great pasta and cozy atmosphere", n_results=3)
        for i, result in enumerate(results, 1):
            print(f"   {i}. {result['restaurant']} ({result['location']})")
            print(f"      Score: {result.get('similarity_score', 0):.3f}")
            print(f"      Text: {result['text'][:100]}...")

        # Search 2: Location-filtered search
        print("\n2. Filtered Search: 'good coffee' in Kuching")
        kuching_results = vectorizer.search(
            "good coffee", 
            n_results=2,
            filters={"location": "Kuching"}
        )
        for i, result in enumerate(kuching_results, 1):
            print(f"   {i}. {result['restaurant']}")
            print(f"      Score: {result.get('similarity_score', 0):.3f}")
            print(f"      Text: {result['text'][:100]}...")

        # Search 3: Price-conscious dining
        print("\n3. Search: 'affordable family restaurant'")
        family_results = vectorizer.search("affordable family restaurant", n_results=3)
        for i, result in enumerate(family_results, 1):
            print(f"   {i}. {result['restaurant']} (Rating: {result['rating']:.1f})")
            print(f"      Score: {result.get('similarity_score', 0):.3f}")
            print(f"      Text: {result['text'][:100]}...")

        print(f"\nâœ… All operations completed successfully!")
        print(f"   Database persisted at: {vectorizer.persist_directory}")

    except FileNotFoundError:
        print("âŒ CSV file not found. Please ensure 'combined_restaurant_reviews.csv' exists.")
    except Exception as e:
        print(f"âŒ An error occurred: {str(e)}")


if __name__ == "__main__":
    main()