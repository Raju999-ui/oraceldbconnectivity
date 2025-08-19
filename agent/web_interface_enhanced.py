#!/usr/bin/env python3
"""
Enhanced Web Interface for Netflix Agent with ChromaDB Vector Memory
"""

from flask import Flask, render_template_string, request, jsonify
from flask_cors import CORS
import pandas as pd
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from snowflake_connector import SnowflakeConnector
from config import GROQ_API_KEY
import logging
import re
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Groq LLM
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama3-8b-8192"
)

# Enhanced memory system with ChromaDB vector support
try:
    from chromadb_integration import ChromaDBVectorMemory
    vector_memory = ChromaDBVectorMemory()
    CHROMADB_AVAILABLE = True
    logger.info("✅ ChromaDB vector memory loaded successfully")
except ImportError as e:
    vector_memory = None
    CHROMADB_AVAILABLE = False
    logger.warning(f"⚠️ ChromaDB not available: {e}")
except Exception as e:
    vector_memory = None
    CHROMADB_AVAILABLE = False
    logger.error(f"❌ ChromaDB initialization failed: {e}")

class HybridMemory:
    def __init__(self):
        self.conversations = []
        self.vector_memory = vector_memory
    
    def add_conversation(self, query: str, response: str, metadata: dict = None):
        """Add conversation to both simple and vector memory."""
        # Add to simple memory
        self.conversations.append({
            'query': query,
            'response': response,
            'timestamp': datetime.now().isoformat()
        })
        
        # Add to vector memory if available
        if self.vector_memory and self.vector_memory.enabled:
            try:
                self.vector_memory.add_conversation(query, response, metadata)
            except Exception as e:
                logger.warning(f"Vector memory storage failed: {e}")
    
    def get_similar_conversations(self, query: str, n_results: int = 5):
        """Get similar conversations from vector memory."""
        if self.vector_memory and self.vector_memory.enabled:
            try:
                return self.vector_memory.get_similar_conversations(query, n_results)
            except Exception as e:
                logger.warning(f"Vector similarity search failed: {e}")
        return []
    
    def get_context(self, query: str, n_results: int = 3):
        """Get contextual information from vector memory."""
        if self.vector_memory and self.vector_memory.enabled:
            try:
                return self.vector_memory.get_context(query, n_results)
            except Exception as e:
                logger.warning(f"Vector context retrieval failed: {e}")
        return "Vector memory not available."
    
    def get_stats(self) -> dict:
        """Get memory statistics."""
        simple_stats = {
            "total_conversations": len(self.conversations),
            "memory_type": "hybrid"
        }
        
        if self.vector_memory and self.vector_memory.enabled:
            try:
                vector_stats = self.vector_memory.get_stats()
                simple_stats.update({
                    "vector_memory": vector_stats,
                    "chromadb_available": True
                })
            except Exception as e:
                logger.warning(f"Vector stats retrieval failed: {e}")
                simple_stats.update({
                    "vector_memory": {"enabled": False, "error": str(e)},
                    "chromadb_available": CHROMADB_AVAILABLE
                })
        else:
            simple_stats.update({
                "vector_memory": {"enabled": False},
                "chromadb_available": CHROMADB_AVAILABLE
            })
        
        return simple_stats

# Initialize hybrid memory
memory = HybridMemory()

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

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
        year = re.search(r'\b(202[0-4])\b', query).group(1)
    
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

def retrieve_data(query_info: dict) -> pd.DataFrame:
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
            WHERE ({genre_condition}) AND release_year = {year}{content_condition}
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

def generate_response(query: str, data: pd.DataFrame) -> str:
    """Generate a response using the LLM."""
    try:
        if data is None or (hasattr(data, 'empty') and data.empty) or (hasattr(data, '__len__') and len(data) == 0):
            return "I couldn't find any Netflix content matching your query. Please try different search terms or be more specific."
        
        # Convert data to string for LLM
        data_str = data.to_string(index=False)
        
        # Create prompt
        prompt = f"""
        You are a helpful Netflix content assistant. The user asked: "{query}"
        
        Here is the data I found:
        {data_str}
        
        Please provide a helpful, informative response about the Netflix content. 
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

def run_netflix_agent(query: str) -> str:
    """Run the Netflix agent with the given query."""
    try:
        print(f"🔍 Processing: {query}")
        
        # Classify query
        query_info = classify_query(query)
        print(f"📊 Query type: {query_info['type']}")
        
        # Retrieve data
        print("🗄️  Querying Snowflake database...")
        data = retrieve_data(query_info)
        print(f"✅ Found {len(data)} results")
        
        # Generate response
        print("🤖 Generating response...")
        response = generate_response(query, data)
        print(f"✅ Response generated successfully, length: {len(response)}")
        
        # Store in memory
        try:
            print("💾 Storing in memory...")
            metadata = {
                "query_type": query_info['type'],
                "content_type": query_info.get('content_type'),
                "results_count": len(data)
            }
            memory.add_conversation(query, response, metadata)
            print("✅ Memory storage completed")
        except Exception as mem_error:
            print(f"⚠️  Memory storage failed: {mem_error}")
        
        print(f"🎯 Returning response to web interface")
        return response
        
    except Exception as e:
        print(f"❌ Agent execution failed: {e}")
        logger.error(f"Agent execution failed: {e}")
        return f"I encountered an error while processing your request: {str(e)}"

# HTML template for the web interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Netflix Agent - Enhanced Web Interface</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .container {
            background: white;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        h1 {
            color: #e50914;
            text-align: center;
            margin-bottom: 30px;
        }
        .query-form {
            margin-bottom: 30px;
        }
        input[type="text"] {
            width: 70%;
            padding: 12px;
            border: 2px solid #ddd;
            border-radius: 8px;
            font-size: 16px;
        }
        button {
            width: 25%;
            padding: 12px;
            background: #e50914;
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            cursor: pointer;
            margin-left: 10px;
        }
        button:hover {
            background: #b2070f;
        }
        .response {
            background: #f8f9fa;
            border-left: 4px solid #e50914;
            padding: 20px;
            border-radius: 8px;
            white-space: pre-wrap;
            font-size: 14px;
            line-height: 1.6;
        }
        .loading {
            text-align: center;
            color: #666;
            font-style: italic;
        }
        .examples {
            background: #f0f0f0;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
        }
        .examples h3 {
            margin-top: 0;
            color: #333;
        }
        .examples ul {
            margin: 0;
            padding-left: 20px;
        }
        .examples li {
            margin: 5px 0;
            color: #666;
        }
        .vector-commands {
            background: #e8f4fd;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            border-left: 4px solid #007bff;
        }
        .vector-commands h3 {
            margin-top: 0;
            color: #007bff;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎬 Netflix Data Agent (Enhanced)</h1>
        
        <div class="examples">
            <h3>💡 Try these examples:</h3>
            <ul>
                <li>show movies from 2020</li>
                <li>find comedy movies</li>
                <li>action movies from 2023</li>
                <li>drama shows</li>
                <li>documentaries</li>
            </ul>
        </div>
        
        <div class="vector-commands">
            <h3>🧠 Vector Memory Commands:</h3>
            <ul>
                <li>memory stats - View memory statistics</li>
                <li>similar conversations - Find similar past queries</li>
                <li>vector context [query] - Get contextual information</li>
            </ul>
        </div>
        
        <div class="query-form">
            <form id="queryForm">
                <input type="text" id="queryInput" placeholder="Ask about Netflix movies and shows..." required>
                <button type="submit">🔍 Search</button>
            </form>
        </div>
        
        <div id="response"></div>
    </div>

    <script>
        document.getElementById('queryForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const query = document.getElementById('queryInput').value;
            const responseDiv = document.getElementById('response');
            
            // Show loading
            responseDiv.innerHTML = '<div class="loading">🤖 Processing your request...</div>';
            
            try {
                const response = await fetch('/query', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({query: query})
                });
                
                // Handle non-OK responses gracefully
                if (!response.ok) {
                    const rawText = await response.text();
                    try {
                        const maybeJson = JSON.parse(rawText);
                        const msg = maybeJson && (maybeJson.error || maybeJson.message || JSON.stringify(maybeJson));
                        responseDiv.innerHTML = '<div class="response" style="color: red;">❌ Request failed (' + response.status + '): ' + msg + '</div>';
                    } catch (_) {
                        responseDiv.innerHTML = '<div class="response" style="color: red;">❌ Request failed (' + response.status + '): ' + rawText + '</div>';
                    }
                    return;
                }

                // Try to parse JSON; show helpful message if invalid
                let data;
                try {
                    data = await response.json();
                } catch (parseErr) {
                    responseDiv.innerHTML = '<div class="response" style="color: red;">❌ Invalid JSON response from server.</div>';
                    return;
                }

                if (data && data.success) {
                    responseDiv.innerHTML = '<div class="response">' + data.response + '</div>';
                } else if (data && data.error) {
                    responseDiv.innerHTML = '<div class="response" style="color: red;">❌ Error: ' + data.error + '</div>';
                } else {
                    responseDiv.innerHTML = '<div class="response" style="color: red;">❌ Unexpected response format.</div>';
                }
            } catch (error) {
                responseDiv.innerHTML = '<div class="response" style="color: red;">❌ Network error: ' + error.message + '</div>';
            }
        });
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/query', methods=['POST'])
def query():
    try:
        print(f"🔍 Received query request")
        data = request.get_json()
        query = data.get('query', '').strip()
        
        if not query:
            print(f"❌ Empty query received")
            return jsonify({'success': False, 'error': 'Please provide a query'})
        
        print(f"📝 Processing query: {query}")
        
        # Handle special memory commands
        if query.lower() == 'memory stats':
            try:
                print("📊 Getting memory stats...")
                stats = memory.get_stats()
                print("✅ Memory stats retrieved successfully")
                
                response = f"""
📊 Memory Statistics:
-------------------
Memory Type: {stats.get('memory_type', 'Unknown')}
Total Conversations: {stats.get('total_conversations', 0)}
ChromaDB Available: {stats.get('chromadb_available', False)}
"""
                
                if stats.get('vector_memory', {}).get('enabled'):
                    vector_stats = stats['vector_memory']
                    response += f"""
Vector Memory: ✅ Enabled
- Conversations: {vector_stats.get('conversations_count', 0)}
- Context: {vector_stats.get('context_count', 0)}
- Total Vectors: {vector_stats.get('total_vectors', 0)}
- Storage: {vector_stats.get('storage_path', 'Unknown')}
"""
                else:
                    response += "Vector Memory: ❌ Disabled"
                
                return jsonify({
                    'success': True,
                    'response': response,
                    'query': query
                })
            except Exception as mem_error:
                print(f"❌ Memory stats error: {mem_error}")
                return jsonify({
                    'success': False,
                    'error': f"Failed to get memory stats: {str(mem_error)}"
                })
        
        elif query.lower() == 'similar conversations':
            try:
                print("🔍 Getting similar conversations...")
                similar = memory.get_similar_conversations("Netflix movies", 5)
                print(f"✅ Found {len(similar)} similar conversations")
                
                response = "🔍 Recent Similar Conversations:\n" + "-" * 40 + "\n"
                
                if similar:
                    for i, conv in enumerate(similar, 1):
                        response += f"{i}. Query: {conv['query'][:50]}...\n"
                        response += f"   Response: {conv['response'][:100]}...\n"
                        response += f"   Similarity: {conv.get('similarity', 'N/A')}\n\n"
                else:
                    response += "No similar conversations found."
                
                return jsonify({
                    'success': True,
                    'response': response,
                    'query': query
                })
            except Exception as e:
                print(f"❌ Similar conversations error: {e}")
                return jsonify({
                    'success': False,
                    'error': f"Failed to get similar conversations: {str(e)}"
                })
        
        elif query.lower().startswith('vector context '):
            try:
                search_query = query[15:]  # Remove 'vector context ' prefix
                print(f"🧠 Getting vector context for: {search_query}")
                context = memory.get_context(search_query)
                print("✅ Vector context retrieved")
                
                response = f"🧠 Vector Context for '{search_query}':\n" + "-" * 50 + "\n" + context
                
                return jsonify({
                    'success': True,
                    'response': response,
                    'query': query
                })
            except Exception as e:
                print(f"❌ Vector context error: {e}")
                return jsonify({
                    'success': False,
                    'error': f"Failed to get vector context: {str(e)}"
                })
        
        # Run the Netflix agent for regular queries
        print(f"🤖 Running Netflix agent for query: {query}")
        try:
            response = run_netflix_agent(query)
            print(f"✅ Agent response generated, length: {len(response)}")
            
            return jsonify({
                'success': True,
                'response': response,
                'query': query
            })
        except Exception as agent_error:
            print(f"❌ Agent execution error: {agent_error}")
            return jsonify({
                'success': False,
                'error': f"Agent execution failed: {str(agent_error)}"
            })
        
    except Exception as e:
        print(f"❌ Error in query endpoint: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'message': 'Netflix Agent Enhanced Web Interface is running'})

@app.route('/test')
def test():
    return jsonify({'success': True, 'message': 'Test endpoint working!'})

if __name__ == '__main__':
    print("🌐 Starting Netflix Agent Enhanced Web Interface...")
    print("📱 Open your browser and go to: http://localhost:5000")
    print("🔍 You can now use your Netflix agent with ChromaDB vector memory!")
    print("=" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
