import pandas as pd
import re
import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
import uuid
import time

class VectorDatabaseCreator:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", persist_directory: str = "./restaurant_chroma_db"):
        """
        Initialize the vector database creator with Chroma vector database

        Args:
            model_name: Embedding model name    
            persist_directory: Directory for Chroma database persistence
        """
        self.model = SentenceTransformer(model_name)
        self.persist_directory = persist_directory

        # Initialize Chroma client with persistence
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = None

        print(f"Initialized vector database creator with model: {model_name}")
        print(f"Chroma database will be stored in: {persist_directory}")

    def create_or_get_collection(self, collection_name: str = "restaurant_reviews"):
        """Create or retrieve existing collection"""
        try:
            self.collection = self.client.get_collection(collection_name)
            print(f"✅ Loaded existing collection: {collection_name}")
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
            print(f"✅ Created new collection: {collection_name}")

        return self.collection

    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess review text"""
        if not isinstance(text, str):
            return ""

        # Remove HTML tags if any
        text = re.sub(r'<[^>]+>', '', text)

        # Fix common encoding issues
        text = text.replace('&', '&').replace('<', '<').replace('>', '>')

        # Remove excessive whitespace and newlines
        text = re.sub(r'\s+', ' ', text)

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

        print(f"\n🔍 Processing {len(df)} restaurants...")
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
                'original_review_id': int(str(idx))
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
        print(f"📥 Adding {len(all_documents)} chunks to Chroma database...")

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
                print(f"  ✅ Processed batch {i//batch_size + 1}/{(len(all_documents)-1)//batch_size + 1}")
            except Exception as e:
                print(f"  ❌ Error processing batch {i//batch_size + 1}: {str(e)}")
                continue

        processing_time = time.time() - start_time
        total_chunks = self.collection.count()

        print(f"\n✅ Processing complete!")
        print(f"   Total chunks in database: {total_chunks}")
        print(f"   Processing time: {processing_time:.2f} seconds")
        print(f"   Database location: {self.persist_directory}")

        return {
            'total_chunks': total_chunks,
            'processing_time': processing_time,
            'collection_name': self.collection.name
        }

    def delete_collection(self, collection_name: str):
        """Delete a collection (use with caution)"""
        try:
            self.client.delete_collection(collection_name)
            print(f"✅ Deleted collection: {collection_name}")
        except Exception as e:
            print(f"❌ Error deleting collection: {str(e)}")