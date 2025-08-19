#!/usr/bin/env python3
"""
Netflix Web Agent - Browser Interface
"""

import os
from flask import Flask, render_template_string, request, jsonify
from flask_cors import CORS
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from snowflake_connector import SnowflakeConnector
from config import GROQ_API_KEY
import logging
import re
from datetime import datetime
import uuid
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize Groq LLM
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama3-8b-8192"
)

# Simple Memory System
class HybridMemory:
    def __init__(self):
        self.conversations_file = "web_conversations_memory.json"
        self.conversations = self.load_conversations()
        
        # Initialize vector database for semantic search
        try:
            import chromadb
            from sentence_transformers import SentenceTransformer
            import numpy as np
            
            self.vector_db_enabled = True
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Initialize ChromaDB
            self.chroma_client = chromadb.Client()
            self.collection_name = "netflix_conversations_v3"
            
            # Create or get collection
            try:
                self.collection = self.chroma_client.get_collection(self.collection_name)
            except:
                self.collection = self.chroma_client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "Netflix agent conversation embeddings"}
                )
            
            print("✅ Vector database (ChromaDB) initialized successfully!")
            
        except Exception as e:
            print(f"⚠️ Vector database initialization failed: {e}")
            print("🔄 Falling back to JSON-only memory system")
            self.vector_db_enabled = False
            self.embedding_model = None
            self.collection = None
    
    def load_conversations(self):
        """Load conversations from JSON file."""
        try:
            if os.path.exists(self.conversations_file):
                with open(self.conversations_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.error(f"Failed to load conversations: {e}")
            return {}
    
    def save_conversations(self):
        """Save conversations to JSON file."""
        try:
            with open(self.conversations_file, 'w', encoding='utf-8') as f:
                json.dump(self.conversations, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save conversations: {e}")
    
    def add_conversation(self, session_id: str, query: str, response: str, metadata: dict = None):
        """Add conversation to both JSON and vector database."""
        # Add to JSON memory (reliable fallback)
        if session_id not in self.conversations:
            self.conversations[session_id] = {
                "created_at": datetime.now().isoformat(),
                "conversations": [],
                "user_preferences": {},
                "search_patterns": []
            }
        
        conversation_entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "response": response,
            "metadata": metadata or {},
            "conversation_id": f"{session_id}_{len(self.conversations[session_id]['conversations'])}"
        }
        
        self.conversations[session_id]["conversations"].append(conversation_entry)
        
        # Update user preferences
        self._update_user_preferences(session_id, query, metadata)
        
        # Save to JSON
        self.save_conversations()
        
        # Add to vector database for semantic search
        if self.vector_db_enabled:
            try:
                self._add_to_vector_db(conversation_entry, session_id)
            except Exception as e:
                logger.warning(f"Vector DB storage failed, using JSON fallback: {e}")
        
        return True
    
    def _add_to_vector_db(self, conversation_entry: dict, session_id: str):
        """Add conversation to vector database for semantic search."""
        try:
            # Create semantic embedding from query + response
            conversation_text = f"{conversation_entry['query']} {conversation_entry['response']}"
            
            # Truncate text to avoid token limits
            conversation_text = conversation_text[:1000]
            
            # Generate embedding
            embedding = self.embedding_model.encode(conversation_text).tolist()
            
            # Add to ChromaDB
            self.collection.add(
                embeddings=[embedding],
                documents=[conversation_text],
                metadatas=[{
                    "session_id": session_id,
                    "conversation_id": conversation_entry["conversation_id"],
                    "query": conversation_entry["query"],
                    "response": conversation_entry["response"][:200],
                    "timestamp": conversation_entry["timestamp"],
                    "query_type": conversation_entry["metadata"].get("query_type", ""),
                    "content_type": conversation_entry["metadata"].get("content_type", ""),
                    "results_count": conversation_entry["metadata"].get("results_count", 0)
                }],
                ids=[conversation_entry["conversation_id"]]
            )
            
        except Exception as e:
            logger.error(f"Failed to add to vector DB: {e}")
            raise e
    
    def _update_user_preferences(self, session_id: str, query: str, metadata: dict):
        """Update user preferences based on their queries."""
        if "user_preferences" not in self.conversations[session_id]:
            self.conversations[session_id]["user_preferences"] = {
                "favorite_genres": {},
                "favorite_years": {},
                "content_type_preference": {},
                "search_patterns": []
            }
        
        prefs = self.conversations[session_id]["user_preferences"]
        
        # Track content type preferences
        if metadata.get("content_type"):
            content_type = metadata["content_type"]
            prefs["content_type_preference"][content_type] = \
                prefs["content_type_preference"].get(content_type, 0) + 1
        
        # Track search patterns
        prefs["search_patterns"].append({
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "results_count": metadata.get("results_count", 0)
        })
        
        # Keep only recent patterns
        if len(prefs["search_patterns"]) > 20:
            prefs["search_patterns"] = prefs["search_patterns"][-20:]
    
    def get_similar_conversations(self, query: str, n_results: int = 5):
        """Get similar conversations using semantic search with fallback to keyword search."""
        if self.vector_db_enabled:
            try:
                # Try semantic search first
                semantic_results = self._semantic_search(query, n_results)
                if semantic_results:
                    return semantic_results
            except Exception as e:
                logger.warning(f"Semantic search failed, falling back to keyword search: {e}")
        
        # Fallback to keyword search
        return self._keyword_search(query, n_results)
    
    def _semantic_search(self, query: str, n_results: int = 5):
        """Search conversations using semantic similarity."""
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode(query).tolist()
            
            # Search in vector database
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=["documents", "metadatas", "distances"]
            )
            
            # Format results
            similar = []
            for i, metadata in enumerate(results['metadatas'][0]):
                similar.append({
                    'query': metadata.get('query', '')[:100],
                    'response': metadata.get('response', '')[:200],
                    'similarity': round(1 - results['distances'][0][i], 3),  # Convert distance to similarity
                    'timestamp': metadata.get('timestamp', ''),
                    'query_type': metadata.get('query_type', ''),
                    'content_type': metadata.get('content_type', '')
                })
            
            return similar
            
        except Exception as e:
            logger.error(f"Semantic search error: {e}")
            return []
    
    def _keyword_search(self, query: str, n_results: int = 5):
        """Fallback keyword-based search."""
        similar = []
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        for session_id, session_data in self.conversations.items():
            for conv in session_data.get("conversations", []):
                # Simple keyword matching
                conv_text = (conv.get("query", "") + " " + conv.get("response", "")).lower()
                conv_words = set(conv_text.split())
                
                # Calculate similarity based on word overlap
                intersection = query_words.intersection(conv_words)
                if intersection:
                    similarity = len(intersection) / len(query_words.union(conv_words))
                    
                    similar.append({
                        'query': conv.get("query", "")[:100],
                        'response': conv.get("response", "")[:200],
                        'similarity': round(similarity, 3),
                        'timestamp': conv.get("timestamp", ""),
                        'query_type': conv.get("metadata", {}).get("query_type", ""),
                        'content_type': conv.get("metadata", {}).get("content_type", "")
                    })
        
        # Sort by similarity and return top results
        similar.sort(key=lambda x: x['similarity'], reverse=True)
        return similar[:n_results]
    
    def get_context_for_query(self, session_id: str, current_query: str) -> str:
        """Get relevant context from previous conversations for the current query."""
        if session_id not in self.conversations:
            return ""
        
        # Get semantically similar conversations
        similar_conversations = self.get_similar_conversations(current_query, n_results=3)
        
        # Filter to user's own conversations
        user_conversations = [
            conv for conv in similar_conversations 
            if any(
                conv['query'] == c.get('query', '') 
                for c in self.conversations[session_id].get("conversations", [])
            )
        ]
        
        if not user_conversations:
            return ""
        
        context_parts = []
        for conv in user_conversations[:2]:  # Limit to 2 most relevant
            context_parts.append(f"Previous: {conv['query']} → {conv['response'][:80]}...")
        
        return "\n".join(context_parts)
    
    def get_user_preferences(self, session_id: str) -> dict:
        """Get user preferences and patterns."""
        if session_id not in self.conversations:
            return {}
        
        prefs = self.conversations[session_id].get("user_preferences", {})
        
        # Analyze preferences
        favorite_content_type = max(
            prefs.get("content_type_preference", {}).items(),
            key=lambda x: x[1],
            default=("Mixed", 0)
        )
        
        return {
            "favorite_content_type": favorite_content_type[0],
            "total_searches": len(prefs.get("search_patterns", [])),
            "recent_searches": prefs.get("search_patterns", [])[-5:],
            "preferences_available": bool(prefs.get("content_type_preference"))
        }
    
    def get_stats(self):
        """Get memory statistics."""
        total_convs = sum(len(session.get("conversations", [])) for session in self.conversations.values())
        
        return {
            "enabled": True,
            "total_sessions": len(self.conversations),
            "total_conversations": total_convs,
            "storage_type": "hybrid_json_vector",
            "vector_db_enabled": self.vector_db_enabled,
            "file_path": self.conversations_file
        }
    
    def clear_vector_db(self):
        """Clear all data from vector database."""
        if self.vector_db_enabled and self.collection:
            try:
                self.chroma_client.delete_collection(self.collection_name)
                self.collection = self.chroma_client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "Netflix agent conversation embeddings"}
                )
                return True
            except Exception as e:
                logger.error(f"Failed to clear vector DB: {e}")
                return False
        return False

# Initialize memory
memory = HybridMemory()

def classify_query(query: str) -> dict:
    """Classify the type of query the user is asking."""
    query_lower = query.lower()
    
    # Check for director queries first
    director_keywords = ["directed by", "director", "by"]
    if any(keyword in query_lower for keyword in director_keywords):
        director_name = None
        if "directed by" in query_lower:
            director_name = query_lower.split("directed by")[-1].strip()
        elif "by" in query_lower and any(word in query_lower for word in ["director", "directed"]):
            director_name = query_lower.split("by")[-1].strip()
        
        if director_name:
            return {
                "type": "filter_by_director",
                "director": director_name,
                "content_type": "Movie" if "movie" in query_lower else None
            }
    
    # Check for genre queries
    genres = ["action", "comedy", "drama", "sci-fi", "horror", "romance", "thriller"]
    found_genres = [genre for genre in genres if genre in query_lower]
    
    # Check for year queries
    year = None
    if any(word in query_lower for word in ["2020", "2021", "2022", "2023", "2024"]):
        year_match = re.search(r'\b(202[0-4])\b', query_lower)
        if year_match:
            year = year_match.group(1)
    
    # Check content type
    content_type = None
    if "movie" in query_lower:
        content_type = "Movie"
    elif "show" in query_lower or "series" in query_lower or "tv" in query_lower:
        content_type = "TV Show"
    
    # Return appropriate type
    if found_genres and year:
        return {
            "type": "filter_by_genre_and_year",
            "genres": found_genres,
            "year": year,
            "content_type": content_type
        }
    elif found_genres:
        return {
            "type": "filter_by_genre", 
            "genres": found_genres,
            "content_type": content_type
        }
    elif year:
        return {"type": "filter_by_year", "year": year, "content_type": content_type}
    
    return {"type": "general_search", "content_type": content_type}

def retrieve_data(query_info: dict):
    """Retrieve data from Snowflake based on query type."""
    try:
        connector = SnowflakeConnector()
        query_type = query_info["type"]
        
        if query_type == "filter_by_year":
            year = query_info["year"]
            content_type = query_info.get("content_type")
            
            content_condition = ""
            if content_type:
                content_condition = f" AND type = '{content_type}'"
            
            sql = f"""
            SELECT title, type, listed_in, release_year, country, duration, description
            FROM netflix_titles 
            WHERE release_year = {year}{content_condition}
            LIMIT 15
            """
        elif query_type == "filter_by_genre":
            genres = query_info["genres"]
            content_type = query_info.get("content_type")
            
            genre_condition = " OR ".join([f"LOWER(listed_in) LIKE '%{genre}%'" for genre in genres])
            
            content_condition = ""
            if content_type:
                content_condition = f" AND type = '{content_type}'"
            
            sql = f"""
            SELECT title, type, listed_in, release_year, country, duration, description
            FROM netflix_titles 
            WHERE ({genre_condition}){content_condition}
            LIMIT 15
            """
        elif query_type == "filter_by_genre_and_year":
            genres = query_info["genres"]
            year = query_info["year"]
            content_type = query_info.get("content_type")
            
            genre_condition = " OR ".join([f"LOWER(listed_in) LIKE '%{genre}%'" for genre in genres])
            
            content_condition = ""
            if content_type:
                content_condition = f" AND type = '{content_type}'"
            
            sql = f"""
            SELECT title, type, listed_in, release_year, country, duration, description
            FROM netflix_titles 
            WHERE release_year = {year} AND ({genre_condition}){content_condition}
            LIMIT 15
            """
        elif query_type == "filter_by_director":
            director = query_info["director"]
            content_type = query_info.get("content_type")
            
            content_condition = ""
            if content_type:
                content_condition = f" AND type = '{content_type}'"
            
            sql = f"""
            SELECT title, type, listed_in, release_year, country, duration, description, director
            FROM netflix_titles 
            WHERE LOWER(director) LIKE '%{director.lower()}%'{content_condition}
            LIMIT 15
            """
        else:
            sql = """
            SELECT title, type, listed_in, release_year, country, duration, description
            FROM netflix_titles 
            LIMIT 10
            """
        
        result = connector.execute_query(sql)
        return result
        
    except Exception as e:
        logger.error(f"Data retrieval error: {e}")
        raise e

def generate_response(query: str, data, session_id: str = None) -> str:
    """Generate a response using the LLM with enhanced context."""
    try:
        if data is None or (hasattr(data, 'empty') and data.empty) or (hasattr(data, '__len__') and len(data) == 0):
            return "I couldn't find any Netflix content matching your query. Please try different search terms or be more specific."
        
        # Convert data to string for LLM
        data_str = data.to_string(index=False)
        
        # Get context from previous conversations if session_id is provided
        context_info = ""
        user_preferences = ""
        
        if session_id:
            # Get relevant context
            context = memory.get_context_for_query(session_id, query)
            if context:
                context_info = f"\n\nPrevious relevant conversations:\n{context}"
            
            # Get user preferences
            prefs = memory.get_user_preferences(session_id)
            if prefs.get("preferences_available"):
                user_preferences = f"\n\nBased on your preferences, you seem to enjoy {prefs.get('favorite_content_type', 'various types of')} content."
        
        # Create enhanced prompt with context
        prompt = f"""
        You are a helpful Netflix content assistant. The user asked: "{query}"
        
        {context_info}
        {user_preferences}
        
        Here is the data I found:
        {data_str}
        
        Please provide a helpful, informative response about the Netflix content.
        {f"Consider the user's previous interests and preferences based on: {context}" if context_info else ""}
        Focus on the specific type of content the user requested (movies, TV shows, or both).
        Include relevant details like titles, genres, years, and descriptions.
        Be conversational and engaging. Keep the response concise but informative.
        If the user asked for movies specifically, focus on movies. If they asked for TV shows, focus on TV shows.
        """
        
        # Get LLM response
        messages = [HumanMessage(content=prompt)]
        response = llm.invoke(messages)
        
        return response.content
        
    except Exception as e:
        logger.error(f"Response generation error: {e}")
        return f"I encountered an error while generating a response: {str(e)}"

def run_netflix_agent(query: str, session_id: str) -> dict:
    """Run the Netflix agent with the given query."""
    try:
        print(f"🎯 Agent: Starting with query: {query}")
        
        # Classify query
        query_info = classify_query(query)
        print(f"🎯 Agent: Query classified as: {query_info}")
        
        # Retrieve data
        print("🎯 Agent: Retrieving data from Snowflake...")
        data = retrieve_data(query_info)
        print(f"🎯 Agent: Retrieved {len(data) if data is not None else 0} records")
        
        # Generate response
        print("🎯 Agent: Generating LLM response...")
        response = generate_response(query, data, session_id)
        print(f"🎯 Agent: Response generated: {response[:100]}...")
        
        # Store in memory
        try:
            metadata = {
                "query_type": query_info['type'],
                "content_type": query_info.get('content_type'),
                "results_count": len(data)
            }
            
            memory.add_conversation(session_id, query, response, metadata)
            print("🎯 Agent: Conversation saved to memory")
                
        except Exception as mem_error:
            print(f"🎯 Agent: Memory storage error: {mem_error}")
            logger.error(f"Memory storage error: {mem_error}")
        
        result = {
            "success": True,
            "query_type": query_info['type'],
            "results_count": len(data),
            "response": response,
            "raw_data": data.to_dict('records') if hasattr(data, 'to_dict') else []
        }
        
        print(f"🎯 Agent: Returning result: {result}")
        return result
        
    except Exception as e:
        print(f"🎯 Agent: Execution failed: {e}")
        logger.error(f"Agent execution failed: {e}")
        return {
            "success": False,
            "error": str(e)
        }

# HTML Template
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🎬 Netflix Web Agent</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .header p {
            font-size: 1.2em;
            opacity: 0.9;
        }
        
        .main-content {
            padding: 30px;
        }
        
        .search-section {
            background: #f8f9fa;
            padding: 25px;
            border-radius: 15px;
            margin-bottom: 30px;
        }
        
        .search-box {
            display: flex;
            gap: 15px;
            margin-bottom: 20px;
        }
        
        .search-input {
            flex: 1;
            padding: 15px;
            border: 2px solid #e9ecef;
            border-radius: 10px;
            font-size: 16px;
            transition: border-color 0.3s;
        }
        
        .search-input:focus {
            outline: none;
            border-color: #e74c3c;
        }
        
        .search-btn {
            padding: 15px 30px;
            background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
            color: white;
            border: none;
            border-radius: 10px;
            font-size: 16px;
            cursor: pointer;
            transition: transform 0.2s;
        }
        
        .search-btn:hover {
            transform: translateY(-2px);
        }
        
        .search-btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }
        
        .examples {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }
        
        .example-btn {
            padding: 10px 15px;
            background: #e9ecef;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            transition: background 0.3s;
            font-size: 14px;
        }
        
        .example-btn:hover {
            background: #dee2e6;
        }
        
        .response-section {
            background: white;
            border: 2px solid #e9ecef;
            border-radius: 15px;
            padding: 25px;
            margin-bottom: 30px;
            min-height: 200px;
        }
        
        .response-content {
            white-space: pre-wrap;
            line-height: 1.6;
            font-size: 16px;
        }
        
        .loading {
            text-align: center;
            color: #6c757d;
            font-style: italic;
        }
        
        .error {
            color: #dc3545;
            background: #f8d7da;
            padding: 15px;
            border-radius: 8px;
            border: 1px solid #f5c6cb;
        }
        
        .stats-section {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 15px;
            margin-bottom: 20px;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 20px;
        }
        
        .stat-item {
            text-align: center;
            padding: 15px;
            background: white;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        .stat-number {
            font-size: 2em;
            font-weight: bold;
            color: #e74c3c;
        }
        
        .stat-label {
            color: #6c757d;
            margin-top: 5px;
        }
        
        .memory-commands {
            display: flex;
            gap: 15px;
            margin-top: 20px;
            flex-wrap: wrap;
        }
        
        .memory-btn {
            padding: 10px 20px;
            background: #6c757d;
            color: white;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            transition: background 0.3s;
        }
        
        .memory-btn:hover {
            background: #5a6268;
        }
        
        .footer {
            text-align: center;
            padding: 20px;
            color: #6c757d;
            border-top: 1px solid #e9ecef;
        }
        
        @media (max-width: 768px) {
            .search-box {
                flex-direction: column;
            }
            
            .examples {
                grid-template-columns: 1fr;
            }
            
            .memory-commands {
                flex-direction: column;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎬 Netflix Web Agent</h1>
            <p>Your AI-powered Netflix content discovery assistant with Hybrid Memory</p>
        </div>
        
        <div class="main-content">
            <div class="search-section">
                <h2>🔍 Search Netflix Content</h2>
                <div class="search-box">
                    <input type="text" id="queryInput" class="search-input" 
                           placeholder="e.g., show movies from 2021, action movies, movies directed by Christopher Nolan..."
                           onkeypress="if(event.key==='Enter') searchNetflix()">
                    <button onclick="searchNetflix()" id="searchBtn" class="search-btn">
                        🔍 Search
                    </button>
                    <button onclick="testFunction()" style="background: #28a745; padding: 15px 20px; color: white; border: none; border-radius: 10px; cursor: pointer;">
                        🧪 Test
                    </button>
                </div>
                
                <div class="examples">
                    <button class="example-btn" onclick="setQuery('show movies from 2021')">2021 Movies</button>
                    <button class="example-btn" onclick="setQuery('show movies from 2020')">2020 Movies</button>
                    <button class="example-btn" onclick="setQuery('action movies')">Action Movies</button>
                    <button class="example-btn" onclick="setQuery('comedy shows')">Comedy Shows</button>
                    <button class="example-btn" onclick="setQuery('horror movies')">Horror Movies</button>
                    <button class="example-btn" onclick="setQuery('movies directed by Christopher Nolan')">Nolan Movies</button>
                </div>
            </div>
            
            <div class="response-section" id="responseSection">
                <div class="response-content" id="responseContent">
                    Welcome! Ask me anything about Netflix content. I can help you find movies and TV shows by year, genre, director, and more.
                    <br><br>
                    <strong>🚀 New Feature:</strong> Hybrid Memory System with semantic search and user preferences!
                </div>
            </div>
            
            <div class="stats-section">
                <h3>📊 Hybrid Memory Statistics</h3>
                <div class="stats-grid" id="statsGrid">
                    <div class="stat-item">
                        <div class="stat-number" id="totalSessions">0</div>
                        <div class="stat-label">Sessions</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-number" id="totalConversations">0</div>
                        <div class="stat-label">Conversations</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-number" id="storageType">Hybrid</div>
                        <div class="stat-label">Storage Type</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-number" id="vectorDBStatus">🔄</div>
                        <div class="stat-label">Vector DB</div>
                    </div>
                </div>
                
                <div class="memory-commands">
                    <button class="memory-btn" onclick="getMemoryStats()">🔄 Refresh Stats</button>
                    <button class="memory-btn" onclick="clearMemory()">🗑️ Clear Memory</button>
                    <button class="memory-btn" onclick="getVectorDBStatus()">🧠 Vector DB Status</button>
                    <button class="memory-btn" onclick="getUserPreferences()">👤 My Preferences</button>
                </div>
            </div>
        </div>
        
        <div class="footer">
            <p>Powered by Groq LLM + Snowflake Database | Hybrid Memory System (JSON + Vector DB)</p>
        </div>
    </div>

    <script>
        let currentSession = generateSessionId();
        
        // Debug: Test if script is loading
        console.log('Netflix Web Agent script loaded successfully!');
        console.log('Current session:', currentSession);
        
        function generateSessionId() {
            return 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
        }
        
        function setQuery(query) {
            console.log('Setting query:', query);
            document.getElementById('queryInput').value = query;
        }
        
        function testFunction() {
            console.log('Test function called!');
            alert('JavaScript is working! searchNetflix function exists: ' + (typeof searchNetflix === 'function'));
        }
        
        async function searchNetflix() {
            console.log('searchNetflix function called!');
            const query = document.getElementById('queryInput').value.trim();
            console.log('Query:', query);
            
            if (!query) {
                alert('Please enter a search query');
                return;
            }
            
            const searchBtn = document.getElementById('searchBtn');
            const responseContent = document.getElementById('responseContent');
            
            // Disable button and show loading
            searchBtn.disabled = true;
            searchBtn.textContent = '⏳ Searching...';
            responseContent.innerHTML = '<div class="loading">🔍 Searching Netflix database...<br>🤖 Generating AI response...<br>💾 Saving to memory...</div>';
            
            try {
                console.log('Sending request to /search...');
                const response = await fetch('/search', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        query: query,
                        session_id: currentSession
                    })
                });
                
                console.log('Response received:', response);
                const data = await response.json();
                console.log('Response data:', data);
                
                if (data.success) {
                    responseContent.innerHTML = '📊 Query Type: ' + data.query_type + '<br>✅ Found: ' + data.results_count + ' results<br><br>🤖 Response:<br><br>' + data.response;
                } else {
                    responseContent.innerHTML = '<div class="error">❌ Error: ' + data.error + '</div>';
                }
                
                // Refresh stats
                getMemoryStats();
                
            } catch (error) {
                console.error('Search error:', error);
                responseContent.innerHTML = '<div class="error">❌ Network Error: ' + error.message + '</div>';
            } finally {
                // Re-enable button
                searchBtn.disabled = false;
                searchBtn.textContent = '🔍 Search';
            }
        }
        
        async function getMemoryStats() {
            try {
                const response = await fetch('/stats');
                const data = await response.json();
                
                document.getElementById('totalSessions').textContent = data.total_sessions;
                document.getElementById('totalConversations').textContent = data.total_conversations;
                document.getElementById('storageType').textContent = data.storage_type;
                
                // Update Vector DB status
                const vectorDBStatus = document.getElementById('vectorDBStatus');
                if (data.vector_db_enabled) {
                    vectorDBStatus.textContent = '✅';
                    vectorDBStatus.title = 'Vector DB Enabled';
                } else {
                    vectorDBStatus.textContent = '❌';
                    vectorDBStatus.title = 'Vector DB Disabled';
                }
                
            } catch (error) {
                console.error('Failed to get stats:', error);
            }
        }
        
        async function clearMemory() {
            if (confirm('Are you sure you want to clear all memory? This cannot be undone.')) {
                try {
                    const response = await fetch('/clear_memory', { method: 'POST' });
                    const data = await response.json();
                    
                    if (data.success) {
                        alert('Memory cleared successfully!');
                        getMemoryStats();
                        document.getElementById('responseContent').innerHTML = 'Memory cleared. Ready for new conversations!';
                    } else {
                        alert('Failed to clear memory: ' + data.error);
                    }
                } catch (error) {
                    alert('Error clearing memory: ' + error.message);
                }
            }
        }
        
        async function getVectorDBStatus() {
            try {
                const response = await fetch('/vector_db_status');
                const data = await response.json();
                alert('Vector DB Status:\\nEnabled: ' + data.vector_db_enabled + '\\nStorage Type: ' + data.storage_type + '\\nTotal Conversations: ' + data.total_conversations);
            } catch (error) {
                alert('Error getting Vector DB Status: ' + error.message);
            }
        }
        
        async function getUserPreferences() {
            try {
                const response = await fetch('/user_preferences/' + currentSession);
                const data = await response.json();
                let prefsText = 'User Preferences:\\n';
                if (data.preferences_available) {
                    prefsText += 'Favorite Content Type: ' + data.favorite_content_type + '\\n';
                    prefsText += 'Total Searches: ' + data.total_searches + '\\n';
                    prefsText += 'Recent Searches:\\n';
                    data.recent_searches.forEach(search => {
                        prefsText += '  - ' + search.query + ' (Results: ' + search.results_count + ', Timestamp: ' + search.timestamp + ')\\n';
                    });
                } else {
                    prefsText += 'No user preferences available for this session.';
                }
                alert(prefsText);
            } catch (error) {
                alert('Error getting user preferences: ' + error.message);
            }
        }
        
        // Load initial stats
        getMemoryStats();
        
        // Auto-focus search input
        document.getElementById('queryInput').focus();
    </script>
</body>
</html>
'''

@app.route('/')
def home():
    """Home page with the Netflix agent interface."""
    return render_template_string(HTML_TEMPLATE)

@app.route('/search', methods=['POST'])
def search():
    """Handle Netflix content search requests."""
    try:
        print(f"🔍 Search request received")
        data = request.get_json()
        print(f"📥 Request data: {data}")
        
        query = data.get('query', '').strip()
        session_id = data.get('session_id', str(uuid.uuid4()))
        
        print(f"🔍 Query: {query}")
        print(f"🆔 Session ID: {session_id}")
        
        if not query:
            print("❌ No query provided")
            return jsonify({"success": False, "error": "Query is required"})
        
        print("🚀 Starting Netflix agent...")
        # Run the Netflix agent
        result = run_netflix_agent(query, session_id)
        print(f"✅ Agent result: {result}")
        
        return jsonify(result)
        
    except Exception as e:
        print(f"❌ Search error: {e}")
        logger.error(f"Search error: {e}")
        return jsonify({"success": False, "error": str(e)})

@app.route('/stats')
def get_stats():
    """Get memory statistics."""
    try:
        stats = memory.get_stats()
        return jsonify(stats)
    except Exception as e:
        logger.error(f"Stats error: {e}")
        return jsonify({"error": str(e)})

@app.route('/clear_memory', methods=['POST'])
def clear_memory():
    """Clear all memory."""
    try:
        memory.conversations = {}
        memory.save_conversations()
        
        # Also clear vector database if enabled
        if memory.vector_db_enabled:
            memory.clear_vector_db()
        
        return jsonify({"success": True, "message": "Memory cleared successfully"})
    except Exception as e:
        logger.error(f"Clear memory error: {e}")
        return jsonify({"success": False, "error": str(e)})

@app.route('/user_preferences/<session_id>')
def get_user_preferences(session_id):
    """Get user preferences and patterns."""
    try:
        prefs = memory.get_user_preferences(session_id)
        return jsonify(prefs)
    except Exception as e:
        logger.error(f"User preferences error: {e}")
        return jsonify({"error": str(e)})

@app.route('/similar_conversations', methods=['POST'])
def get_similar_conversations():
    """Get similar conversations for a query."""
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        n_results = data.get('n_results', 5)
        
        if not query:
            return jsonify({"success": False, "error": "Query is required"})
        
        similar = memory.get_similar_conversations(query, n_results)
        return jsonify({
            "success": True,
            "similar_conversations": similar,
            "vector_db_enabled": memory.vector_db_enabled
        })
        
    except Exception as e:
        logger.error(f"Similar conversations error: {e}")
        return jsonify({"success": False, "error": str(e)})

@app.route('/vector_db_status')
def get_vector_db_status():
    """Get vector database status."""
    try:
        return jsonify({
            "vector_db_enabled": memory.vector_db_enabled,
            "storage_type": memory.get_stats()["storage_type"],
            "total_conversations": memory.get_stats()["total_conversations"]
        })
    except Exception as e:
        logger.error(f"Vector DB status error: {e}")
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    print("🎬 Starting Netflix Web Agent...")
    print("🌐 Open your browser and go to: http://localhost:5000")
    print("🚀 Press Ctrl+C to stop the server")
    
    app.run(host='0.0.0.0', port=5000, debug=True)
