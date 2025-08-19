# 🎬 Netflix Web Agent - Complete Documentation

## 📋 Table of Contents
1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Key Components](#key-components)
4. [How It Works](#how-it-works)
5. [Memory System](#memory-system)
6. [API Endpoints](#api-endpoints)
7. [Installation & Setup](#installation--setup)
8. [Usage Examples](#usage-examples)
9. [Technical Details](#technical-details)
10. [Troubleshooting](#troubleshooting)

---

## 🚀 Overview

The **Netflix Web Agent** is an AI-powered content discovery system that combines:
- **Groq LLM** for intelligent responses
- **Snowflake Database** for Netflix content data
- **Hybrid Memory System** (JSON + ChromaDB) for context awareness
- **Flask Web Interface** for user interaction

### ✨ Key Features
- 🔍 **Smart Query Classification** - Automatically detects query types
- 🧠 **Hybrid Memory** - Combines reliable JSON storage with semantic search
- 🎯 **Context-Aware Responses** - Remembers user preferences and past conversations
- 🌐 **Modern Web Interface** - Responsive design with real-time updates
- 📊 **User Preference Tracking** - Learns from user behavior

---

## 🏗️ System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Browser  │    │   Flask Server  │    │  Snowflake DB  │
│                 │◄──►│                 │◄──►│                 │
│  User Interface│    │  API Endpoints  │    │ Netflix Titles  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │  Hybrid Memory  │
                       │                 │
                       │  JSON + ChromaDB│
                       └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │   Groq LLM     │
                       │                 │
                       │ Response Gen.   │
                       └─────────────────┘
```

---

## 🔧 Key Components

### 1. **Flask Web Server** (`app.py`)
- **Purpose**: Handles HTTP requests and serves the web interface
- **Features**: CORS enabled, debug mode, auto-reload
- **Port**: 5000 (configurable)

### 2. **Hybrid Memory System** (`HybridMemory` class)
- **JSON Storage**: Reliable, persistent conversation history
- **ChromaDB**: Vector database for semantic similarity search
- **Fallback Mechanism**: Automatically switches to JSON-only if vector DB fails

### 3. **Query Classifier** (`classify_query` function)
- **Year Detection**: Automatically finds years (2020-2024) in queries
- **Genre Recognition**: Identifies action, comedy, drama, horror, etc.
- **Director Queries**: Detects "directed by" or "director" keywords
- **Content Type**: Distinguishes between movies and TV shows

### 4. **Data Retrieval** (`retrieve_data` function)
- **Dynamic SQL Generation**: Builds queries based on classification
- **Snowflake Integration**: Connects to your Netflix database
- **Result Limiting**: Returns top 15 results for performance

### 5. **LLM Response Generator** (`generate_response` function)
- **Context Injection**: Includes relevant past conversations
- **User Preferences**: Personalizes responses based on history
- **Groq Integration**: Uses llama3-8b-8192 model for responses

---

## 🔄 How It Works

### **Step-by-Step Process:**

1. **User Input** → User types query in web interface
2. **Query Classification** → System analyzes query type automatically
3. **Data Retrieval** → Snowflake database queried based on classification
4. **Context Gathering** → Relevant past conversations retrieved from memory
5. **LLM Processing** → Groq generates intelligent response with context
6. **Memory Storage** → Conversation saved to both JSON and ChromaDB
7. **Response Display** → Formatted result shown to user

### **Query Classification Examples:**

| User Query | Detected Type | SQL Generated |
|------------|---------------|---------------|
| "show movies from 2021" | `filter_by_year` | `WHERE release_year = 2021` |
| "action movies" | `filter_by_genre` | `WHERE LOWER(listed_in) LIKE '%action%'` |
| "movies directed by Christopher Nolan" | `filter_by_director` | `WHERE LOWER(director) LIKE '%christopher nolan%'` |
| "horror movies from 2020" | `filter_by_genre_and_year` | `WHERE release_year = 2020 AND LOWER(listed_in) LIKE '%horror%'` |

---

## 🧠 Memory System

### **Hybrid Approach Benefits:**

#### **JSON Storage (Primary)**
- ✅ **Reliable**: File-based, never fails
- ✅ **Fast**: Direct file I/O operations
- ✅ **Persistent**: Survives server restarts
- ✅ **Simple**: Easy to debug and maintain

#### **ChromaDB Vector Storage (Secondary)**
- 🚀 **Semantic Search**: Find similar conversations using AI embeddings
- 🎯 **Context Awareness**: Better understanding of user intent
- 🔍 **Similarity Matching**: Advanced search beyond keywords
- 📈 **Scalable**: Handles large conversation databases

### **Memory Operations:**

```python
# Adding conversation
memory.add_conversation(session_id, query, response, metadata)

# Retrieving similar conversations
similar = memory.get_similar_conversations(query, n_results=5)

# Getting user preferences
prefs = memory.get_user_preferences(session_id)

# Context for LLM
context = memory.get_context_for_query(session_id, current_query)
```

---

## 🌐 API Endpoints

### **Core Endpoints:**

| Endpoint | Method | Purpose | Response |
|----------|--------|---------|----------|
| `/` | GET | Main web interface | HTML page |
| `/search` | POST | Process Netflix queries | JSON result |
| `/stats` | GET | Memory statistics | JSON stats |
| `/clear_memory` | POST | Clear all memory | Success message |
| `/user_preferences/<session_id>` | GET | User preferences | JSON preferences |
| `/vector_db_status` | GET | ChromaDB status | JSON status |
| `/similar_conversations` | POST | Find similar chats | JSON results |

### **Search Request Format:**
```json
{
  "query": "show movies from 2021",
  "session_id": "session_1234567890_abc123"
}
```

### **Search Response Format:**
```json
{
  "success": true,
  "query_type": "filter_by_year",
  "results_count": 15,
  "response": "Here are the movies from 2021...",
  "raw_data": [...]
}
```

---

## 🛠️ Installation & Setup

### **Prerequisites:**
- Python 3.8+
- Snowflake account with Netflix data
- Groq API key

### **Installation Steps:**

1. **Clone Repository:**
```bash
git clone <your-repo-url>
cd netflix-web-agent
```

2. **Install Dependencies:**
```bash
pip install -r requirements.txt
```

3. **Configure Environment:**
```bash
# Create config.py with your credentials
GROQ_API_KEY = "your_groq_api_key"
SNOWFLAKE_CONFIG = {
    "account": "your_account",
    "user": "your_username",
    "password": "your_password",
    "warehouse": "your_warehouse",
    "database": "your_database",
    "schema": "your_schema"
}
```

4. **Run the Application:**
```bash
python netflix_web_agent.py
```

5. **Access Web Interface:**
```
http://localhost:5000
```

---

## 📖 Usage Examples

### **Basic Search:**
1. Open browser to `http://localhost:5000`
2. Type: "show me action movies from 2021"
3. Click "🔍 Search" button
4. View results with AI-generated descriptions

### **Memory Features:**
1. **Check Stats**: Click "🔄 Refresh Stats" to see memory usage
2. **View Preferences**: Click "👤 My Preferences" to see your patterns
3. **Vector DB Status**: Click "🧠 Vector DB Status" to check ChromaDB
4. **Clear Memory**: Click "🗑️ Clear Memory" to reset (with confirmation)

### **Example Queries:**
- "What comedy shows are available?"
- "Show me horror movies from 2020"
- "Movies directed by Christopher Nolan"
- "Action movies with good ratings"
- "TV series from 2022"

---

## 🔬 Technical Details

### **Memory Storage Structure:**
```json
{
  "session_id_123": {
    "created_at": "2024-01-01T10:00:00",
    "conversations": [
      {
        "timestamp": "2024-01-01T10:00:00",
        "query": "show movies from 2021",
        "response": "Here are the movies...",
        "metadata": {
          "query_type": "filter_by_year",
          "content_type": "Movie",
          "results_count": 15
        },
        "conversation_id": "session_123_0"
      }
    ],
    "user_preferences": {
      "favorite_genres": {},
      "content_type_preference": {"Movie": 1},
      "search_patterns": [...]
    }
  }
}
```

### **Vector Database Schema:**
- **Collection**: `netflix_conversations_v3`
- **Embeddings**: 384-dimensional vectors (all-MiniLM-L6-v2)
- **Metadata**: Query, response, timestamp, query_type, content_type
- **Search**: Semantic similarity with fallback to keyword search

### **Performance Optimizations:**
- **Result Limiting**: Max 15 results per query
- **Text Truncation**: Embeddings limited to 1000 characters
- **Caching**: JSON file caching for fast access
- **Async Operations**: Non-blocking memory operations

---

## 🚨 Troubleshooting

### **Common Issues & Solutions:**

#### **1. ChromaDB Initialization Failed**
```
⚠️ Vector database initialization failed: [error]
🔄 Falling back to JSON-only memory system
```
**Solution**: System automatically falls back to JSON storage. Check ChromaDB installation.

#### **2. JavaScript Functions Not Defined**
```
Uncaught ReferenceError: searchNetflix is not defined
```
**Solution**: Refresh browser page. Check browser console for syntax errors.

#### **3. Snowflake Connection Error**
```
Failed to connect to Snowflake: [error]
```
**Solution**: Verify credentials in `config.py` and network connectivity.

#### **4. Memory Storage Issues**
```
Memory storage error: [error]
```
**Solution**: Check file permissions for `web_conversations_memory.json`.

### **Debug Mode:**
- **Flask Debug**: Enabled by default
- **Console Logging**: Detailed logs in terminal
- **Browser Console**: JavaScript debugging information
- **Network Tab**: Monitor API requests/responses

### **Performance Monitoring:**
- **Memory Usage**: Check stats endpoint
- **Response Times**: Monitor search endpoint performance
- **Vector DB Status**: Verify ChromaDB health
- **Session Count**: Track active user sessions

---

## 🔮 Future Enhancements

### **Planned Features:**
- 📱 **Mobile App**: Native iOS/Android applications
- 🎨 **Advanced UI**: Dark mode, custom themes
- 📊 **Analytics Dashboard**: User behavior insights
- 🔗 **API Integration**: Third-party service connections
- 🌍 **Multi-language Support**: Internationalization
- 🔐 **User Authentication**: Login system with profiles

### **Scalability Improvements:**
- **Redis Caching**: Faster memory access
- **Load Balancing**: Multiple server instances
- **Database Optimization**: Query performance tuning
- **CDN Integration**: Global content delivery

---

## 📞 Support & Contributing

### **Getting Help:**
- **Documentation**: This file and inline code comments
- **Code Comments**: Detailed explanations in source code
- **Error Logs**: Check terminal and browser console
- **GitHub Issues**: Report bugs and feature requests

### **Contributing:**
1. Fork the repository
2. Create feature branch
3. Add comprehensive comments
4. Test thoroughly
5. Submit pull request

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- **Groq**: For providing the LLM API
- **ChromaDB**: For vector database capabilities
- **Snowflake**: For data warehouse infrastructure
- **Flask**: For the web framework
- **Sentence Transformers**: For text embedding models

---

*Last Updated: January 2024*
*Version: 1.0.0*
*Author: Your Name*
