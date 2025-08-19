#!/usr/bin/env python3
"""
ChromaDB Integration for Netflix Agent Vector Memory
"""

import chromadb
from chromadb.utils import embedding_functions
import logging
import hashlib
from datetime import datetime
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class ChromaDBVectorMemory:
    """Enhanced ChromaDB vector memory system for Netflix Agent."""
    
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.persist_directory = persist_directory
        self.client = None
        self.conversations_collection = None
        self.enabled = False
        
        # Initialize ChromaDB
        self._initialize_chromadb()
    
    def _initialize_chromadb(self):
        """Initialize ChromaDB client and collections."""
        try:
            # Create persistent client
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            
            # Create or get conversations collection
            self.conversations_collection = self.client.get_or_create_collection(
                name="netflix_conversations",
                metadata={"hnsw:space": "cosine", "description": "Netflix agent conversation history"}
            )
            
            # Create or get context collection for better semantic search
            self.context_collection = self.client.get_or_create_collection(
                name="netflix_context",
                metadata={"hnsw:space": "cosine", "description": "Netflix content context"}
            )
            
            self.enabled = True
            logger.info("✅ ChromaDB vector memory initialized successfully")
            logger.info(f"📁 Storage path: {self.persist_directory}")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize ChromaDB: {e}")
            self.enabled = False
    
    def get_embedding(self, text: str) -> List[float]:
        """Generate embeddings for text using hash-based method (fallback)."""
        try:
            # Use hash-based embedding for consistency
            hash_obj = hashlib.md5(text.encode('utf-8'))
            hash_bytes = hash_obj.digest()
            
            # Create 384-dimensional embedding from hash
            embedding = []
            for i in range(24):  # 24 * 16 = 384 dimensions
                byte_idx = i % 16
                embedding.extend([
                    float(hash_bytes[byte_idx]) / 255.0,
                    float(hash_bytes[byte_idx] ^ 0xFF) / 255.0,
                    float(hash_bytes[byte_idx] << 1 & 0xFF) / 255.0,
                    float(hash_bytes[byte_idx] >> 1) / 255.0,
                    float(hash_bytes[byte_idx] & 0x0F) / 255.0,
                    float(hash_bytes[byte_idx] & 0xF0) / 255.0,
                    float(hash_bytes[byte_idx] + i) % 256 / 255.0,
                    float(hash_bytes[byte_idx] - i) % 256 / 255.0,
                    float(hash_bytes[byte_idx] * 2) % 256 / 255.0,
                    float(hash_bytes[byte_idx] // 2) / 255.0,
                    float((hash_bytes[byte_idx] + 128) % 256) / 255.0,
                    float(hash_bytes[byte_idx] ^ 0x55) / 255.0,
                    float(hash_bytes[byte_idx] ^ 0xAA) / 255.0,
                    float((hash_bytes[byte_idx] * 3) % 256) / 255.0,
                    float((hash_bytes[byte_idx] + 42) % 256) / 255.0,
                    float((hash_bytes[byte_idx] * 7 + i) % 256) / 255.0
                ])
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            # Return zero vector as fallback
            return [0.0] * 384
    
    def add_conversation(self, query: str, response: str, metadata: Optional[Dict] = None) -> bool:
        """Add a conversation to the vector database."""
        if not self.enabled or not self.conversations_collection:
            return False
        
        try:
            # Create conversation document
            conversation_text = f"Query: {query}\nResponse: {response}"
            embedding = self.get_embedding(conversation_text)
            
            # Prepare metadata
            meta = {
                "query": query,
                "response": response[:500],  # Truncate for storage
                "timestamp": datetime.now().isoformat(),
                "type": "conversation",
                **(metadata or {})
            }
            
            # Generate unique ID
            conversation_id = f"conv_{datetime.now().timestamp()}_{hash(query) % 10000}"
            
            # Add to ChromaDB
            self.conversations_collection.add(
                embeddings=[embedding],
                documents=[conversation_text],
                metadatas=[meta],
                ids=[conversation_id]
            )
            
            logger.info(f"✅ Added conversation to vector database: {query[:50]}...")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to add conversation to vector database: {e}")
            return False
    
    def add_context(self, content: str, content_type: str, metadata: Optional[Dict] = None) -> bool:
        """Add Netflix content context to the vector database."""
        if not self.enabled or not self.context_collection:
            return False
        
        try:
            embedding = self.get_embedding(content)
            
            # Prepare metadata
            meta = {
                "content_type": content_type,
                "timestamp": datetime.now().isoformat(),
                "type": "context",
                **(metadata or {})
            }
            
            # Generate unique ID
            context_id = f"ctx_{datetime.now().timestamp()}_{hash(content) % 10000}"
            
            # Add to context collection
            self.context_collection.add(
                embeddings=[embedding],
                documents=[content],
                metadatas=[meta],
                ids=[context_id]
            )
            
            logger.info(f"✅ Added context to vector database: {content_type}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to add context to vector database: {e}")
            return False
    
    def get_similar_conversations(self, query: str, n_results: int = 5) -> List[Dict]:
        """Find similar conversations using vector similarity."""
        if not self.enabled or not self.conversations_collection:
            return []
        
        try:
            query_embedding = self.get_embedding(query)
            
            results = self.conversations_collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
            
            similar_conversations = []
            if results['metadatas'] and results['metadatas'][0]:
                for i, metadata in enumerate(results['metadatas'][0]):
                    similar_conversations.append({
                        'query': metadata.get('query', ''),
                        'response': metadata.get('response', ''),
                        'timestamp': metadata.get('timestamp', ''),
                        'similarity': 1 - results['distances'][0][i] if results['distances'] else 0.0
                    })
            
            return similar_conversations
            
        except Exception as e:
            logger.error(f"❌ Failed to query similar conversations: {e}")
            return []
    
    def get_context(self, query: str, n_results: int = 3) -> str:
        """Get contextual information from vector database."""
        if not self.enabled:
            return "Vector memory not available."
        
        try:
            # Get similar conversations
            similar_conv = self.get_similar_conversations(query, n_results)
            
            if not similar_conv:
                return "No relevant vector context found."
            
            # Format context
            context_parts = []
            for conv in similar_conv:
                context_parts.append(
                    f"Previous Query: {conv['query']}\n"
                    f"Previous Response: {conv['response'][:200]}...\n"
                    f"Similarity: {conv['similarity']:.3f}"
                )
            
            return "\n\n".join(context_parts)
            
        except Exception as e:
            logger.error(f"❌ Failed to get context: {e}")
            return "Error retrieving vector context."
    
    def search_content(self, query: str, n_results: int = 5) -> List[Dict]:
        """Search Netflix content context."""
        if not self.enabled or not self.context_collection:
            return []
        
        try:
            query_embedding = self.get_embedding(query)
            
            results = self.context_collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
            
            content_results = []
            if results['metadatas'] and results['metadatas'][0]:
                for i, metadata in enumerate(results['metadatas'][0]):
                    content_results.append({
                        'content': results['documents'][0][i] if results['documents'] else '',
                        'content_type': metadata.get('content_type', ''),
                        'similarity': 1 - results['distances'][0][i] if results['distances'] else 0.0
                    })
            
            return content_results
            
        except Exception as e:
            logger.error(f"❌ Failed to search content: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector database statistics."""
        if not self.enabled:
            return {"enabled": False, "reason": "ChromaDB not available"}
        
        try:
            conv_count = self.conversations_collection.count() if self.conversations_collection else 0
            ctx_count = self.context_collection.count() if self.context_collection else 0
            
            return {
                "enabled": True,
                "conversations_count": conv_count,
                "context_count": ctx_count,
                "total_vectors": conv_count + ctx_count,
                "storage_path": self.persist_directory,
                "database_type": "ChromaDB",
                "collections": ["netflix_conversations", "netflix_context"]
            }
        except Exception as e:
            return {"enabled": False, "error": str(e)}
    
    def clear_all(self) -> bool:
        """Clear all data from vector database."""
        if not self.enabled:
            return False
        
        try:
            if self.conversations_collection:
                self.conversations_collection.delete(where={})
            if self.context_collection:
                self.context_collection.delete(where={})
            
            logger.info("✅ Cleared all vector database data")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to clear vector database: {e}")
            return False

# Example usage
if __name__ == "__main__":
    # Initialize vector memory
    vector_memory = ChromaDBVectorMemory()
    
    # Test adding conversations
    vector_memory.add_conversation(
        "show movies from 2020",
        "Here are some movies from 2020: The Trial of the Chicago 7, Mank, The Midnight Sky...",
        {"genre": "drama", "year": "2020"}
    )
    
    vector_memory.add_conversation(
        "find comedy movies",
        "Here are some comedy movies: The Good Place, Brooklyn Nine-Nine, The Office...",
        {"genre": "comedy"}
    )
    
    # Test similarity search
    similar = vector_memory.get_similar_conversations("movies from 2020", 3)
    print("Similar conversations:", similar)
    
    # Get stats
    stats = vector_memory.get_stats()
    print("Vector memory stats:", stats)
