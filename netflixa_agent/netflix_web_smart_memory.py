#!/usr/bin/env python3
"""
Netflix Web Agent - Smart Memory Version
"""

import streamlit as st
from snowflake_connector import SnowflakeConnector
from netflix_agent import classify_query, extract_search_terms, retrieve_data, generate_response, AgentState
import uuid
import logging
import re

# Configure logging
logging.basicConfig(level=logging.ERROR)

def analyze_memory_query(query: str, chat_history: list) -> str:
    """Analyze memory-related queries and provide smart answers."""
    if not chat_history:
        return "No previous conversation history found. This is your first query!"
    
    query_lower = query.lower()
    
    # Extract years mentioned in previous queries
    years_mentioned = []
    for item in chat_history:
        # Look for year patterns in queries
        year_matches = re.findall(r'\b(20\d{2})\b', item['query'])
        years_mentioned.extend(year_matches)
    
    # Extract content types mentioned
    content_types = []
    for item in chat_history:
        if 'movie' in item['query'].lower():
            content_types.append('movies')
        if 'tv show' in item['query'].lower() or 'tv' in item['query'].lower():
            content_types.append('TV shows')
    
    # Extract genres mentioned
    genres_mentioned = []
    for item in chat_history:
        if 'action' in item['query'].lower():
            genres_mentioned.append('action')
        if 'comedy' in item['query'].lower():
            genres_mentioned.append('comedy')
        if 'drama' in item['query'].lower():
            genres_mentioned.append('drama')
        if 'horror' in item['query'].lower():
            genres_mentioned.append('horror')
        if 'romantic' in item['query'].lower() or 'romance' in item['query'].lower():
            genres_mentioned.append('romance')
        if 'thriller' in item['query'].lower():
            genres_mentioned.append('thriller')
    
    # Check for statistics queries
    stats_queries = [item for item in chat_history if 'statistics' in item['query'].lower() or 'stats' in item['query'].lower()]
    
    # Smart answers based on query type
    if any(word in query_lower for word in ["year", "years"]):
        if years_mentioned:
            unique_years = list(set(years_mentioned))
            if len(unique_years) == 1:
                return f"🎬 **You asked about content from {unique_years[0]}**\n\nIn your previous queries, you specifically asked about content from **{unique_years[0]}**."
            else:
                return f"🎬 **You asked about multiple years: {', '.join(unique_years)}**\n\nIn your conversation, you've asked about content from these years: {', '.join(unique_years)}."
        else:
            return "❓ **No specific years found in your previous queries.**\n\nYou haven't asked about content from specific years yet."
    
    elif any(word in query_lower for word in ["movie", "movies"]):
        if 'movies' in content_types:
            return "🎬 **Yes, you asked about movies!**\n\nIn your previous queries, you specifically asked about movies."
        else:
            return "❓ **No movie queries found.**\n\nYou haven't specifically asked about movies in your previous queries."
    
    elif any(word in query_lower for word in ["tv", "show", "shows"]):
        if 'TV shows' in content_types:
            return "📺 **Yes, you asked about TV shows!**\n\nIn your previous queries, you specifically asked about TV shows."
        else:
            return "❓ **No TV show queries found.**\n\nYou haven't specifically asked about TV shows in your previous queries."
    
    elif any(word in query_lower for word in ["genre", "action", "comedy", "drama", "horror", "romance", "thriller"]):
        if genres_mentioned:
            unique_genres = list(set(genres_mentioned))
            return f"🎭 **You asked about these genres: {', '.join(unique_genres)}**\n\nIn your previous queries, you've asked about: {', '.join(unique_genres)} content."
        else:
            return "❓ **No specific genres found.**\n\nYou haven't asked about specific genres in your previous queries."
    
    elif any(word in query_lower for word in ["statistics", "stats", "overview"]):
        if stats_queries:
            return "📊 **Yes, you asked about statistics!**\n\nYou've asked for database statistics and overview information."
        else:
            return "❓ **No statistics queries found.**\n\nYou haven't asked about statistics in your previous queries."
    
    elif "did i" in query_lower or "have i" in query_lower:
        # Handle "did I ask about..." questions
        if "tv" in query_lower or "show" in query_lower:
            if 'TV shows' in content_types:
                return "📺 **Yes, you did ask about TV shows!**\n\nIn your previous queries, you specifically asked about TV shows."
            else:
                return "❓ **No, you haven't asked about TV shows yet.**\n\nYou haven't specifically asked about TV shows in your previous queries."
        
        elif "movie" in query_lower:
            if 'movies' in content_types:
                return "🎬 **Yes, you did ask about movies!**\n\nIn your previous queries, you specifically asked about movies."
            else:
                return "❓ **No, you haven't asked about movies yet.**\n\nYou haven't specifically asked about movies in your previous queries."
        
        elif "statistics" in query_lower or "stats" in query_lower:
            if stats_queries:
                return "📊 **Yes, you did ask about statistics!**\n\nYou've asked for database statistics and overview information."
            else:
                return "❓ **No, you haven't asked about statistics yet.**\n\nYou haven't asked about statistics in your previous queries."
        
        elif "genre" in query_lower or any(genre in query_lower for genre in ["action", "comedy", "drama"]):
            if genres_mentioned:
                unique_genres = list(set(genres_mentioned))
                return f"🎭 **Yes, you did ask about genres!**\n\nYou asked about these genres: {', '.join(unique_genres)}"
            else:
                return "❓ **No, you haven't asked about specific genres yet.**\n\nYou haven't asked about specific genres in your previous queries."
    
    else:
        # Default: show recent conversation summary
        recent_queries = chat_history[-3:]  # Last 3 queries
        summary = "📝 **Recent Conversation Summary:**\n\n"
        for i, item in enumerate(recent_queries, 1):
            summary += f"**Query {len(chat_history) - len(recent_queries) + i}:** {item['query']}\n"
        return summary

def run_netflix_query_smart(query: str, session_id: str = None) -> str:
    """Run a Netflix query with smart memory handling."""
    try:
        if not session_id:
            session_id = str(uuid.uuid4())
        
        # Check if this is a memory-related query - IMPROVED DETECTION
        query_lower = query.lower()
        memory_keywords = [
            "previous", "history", "memory", "remember", "before", "last", 
            "which", "what did", "did i", "have i", "asked about", "discussed",
            "talked about", "mentioned", "searched for", "looked for"
        ]
        
        if any(keyword in query_lower for keyword in memory_keywords):
            # Handle memory queries with smart analysis
            if 'chat_history' in st.session_state and st.session_state.chat_history:
                return analyze_memory_query(query, st.session_state.chat_history)
            else:
                return "No previous conversation history found. This is your first query!"
        
        # Regular query processing
        state = {
            "session_id": session_id,
            "messages": [],
            "query": query,
            "query_type": "",
            "snowflake_data": None,
            "response": "",
            "error": "",
            "conversation_context": "",
            "previous_actions": [],
            "sql_info": {}
        }
        
        state = classify_query(state)
        state = extract_search_terms(state)
        state = retrieve_data(state)
        state = generate_response(state)
        
        return state.get("response", "No response generated")
        
    except Exception as e:
        return f"❌ **Error:** {str(e)}"

# Page config
st.set_page_config(page_title="Netflix Agent - Smart Memory", page_icon="🎬", layout="wide")

# Title
st.title("🎬 Netflix Data Agent - Smart Memory")
st.write("Ask me anything about Netflix titles! I remember and understand our conversation.")

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Main content
col1, col2 = st.columns([3, 1])

with col1:
    # Query input
    query = st.text_input("Enter your query:", placeholder="e.g., show movies from 2021")
    
    # Process button
    if st.button("🚀 Search", type="primary", use_container_width=True):
        if query.strip():
            with st.spinner("Processing..."):
                response = run_netflix_query_smart(query.strip(), st.session_state.session_id)
                
                # Add to history
                st.session_state.chat_history.append({
                    "query": query.strip(),
                    "response": response,
                    "timestamp": str(uuid.uuid4())[:8]
                })
            
            st.success("Done!")

with col2:
    st.header("💡 Smart Memory Examples")
    examples = [
        "show movies from 2021",
        "show TV shows from 2020",
        "show me action movies",
        "show me statistics",
        "which year you displayed movies?",
        "what genres did I ask about?",
        "did I ask about TV shows?",
        "what years did we discuss?"
    ]
    
    for example in examples:
        if st.button(example, key=f"ex_{example}"):
            # Use JavaScript to set the input value
            st.markdown(f"""
            <script>
                document.querySelector('input[placeholder*="show movies"]').value = "{example}";
            </script>
            """, unsafe_allow_html=True)

# Display results
if st.session_state.chat_history:
    st.header("💬 Conversation History")
    
    for i, item in enumerate(reversed(st.session_state.chat_history), 1):
        with st.expander(f"Query {len(st.session_state.chat_history) - i + 1}: {item['query']}", expanded=True):
            st.markdown(f"**Query:** {item['query']}")
            st.markdown("**Response:**")
            st.markdown(item['response'])
            st.caption(f"Session: {item['timestamp']}")

# Sidebar
with st.sidebar:
    st.header("📋 Actions")
    
    # Clear history
    if st.button("🗑️ Clear History"):
        st.session_state.chat_history = []
        st.rerun()
    
    # Session info
    st.markdown("---")
    st.write(f"**Session:** {st.session_state.session_id[:8]}...")
    st.write(f"**Queries:** {len(st.session_state.chat_history)}")
    
    # Smart memory info
    if st.session_state.chat_history:
        st.markdown("---")
        st.header("🧠 Memory Analysis")
        
        # Extract years
        years = []
        for item in st.session_state.chat_history:
            year_matches = re.findall(r'\b(20\d{2})\b', item['query'])
            years.extend(year_matches)
        
        if years:
            unique_years = list(set(years))
            st.write(f"**Years discussed:** {', '.join(unique_years)}")
        
        # Extract content types
        content_types = set()
        for item in st.session_state.chat_history:
            if 'movie' in item['query'].lower():
                content_types.add('Movies')
            if 'tv' in item['query'].lower():
                content_types.add('TV Shows')
        
        if content_types:
            st.write(f"**Content types:** {', '.join(content_types)}")

# Instructions
st.markdown("---")
st.markdown("""
**💡 Smart Memory Features:**
- Ask "which year you displayed movies?" for specific answers
- Ask "what genres did I ask about?" for genre analysis
- Ask "did I ask about TV shows?" for content type checking
- Ask "what years did we discuss?" for year summary
- The agent now understands context and provides intelligent answers!
""")
