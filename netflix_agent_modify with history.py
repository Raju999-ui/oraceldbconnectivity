from typing import Dict, Any, List, TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
import json
import pandas as pd
from snowflake_connector import SnowflakeConnector
from config import GROQ_API_KEY
import logging
import pickle
import os
from datetime import datetime
import uuid
import pickle
import os
from datetime import datetime
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Groq LLM
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama3-8b-8192"
)

# Memory storage class
class ConversationMemory:
    def __init__(self, storage_file="conversation_memory.pkl"):
        self.storage_file = storage_file
        self.conversations = self.load_memory()
    
    def load_memory(self):
        """Load conversation memory from file."""
        try:
            if os.path.exists(self.storage_file):
                with open(self.storage_file, 'rb') as f:
                    return pickle.load(f)
            return {}
        except Exception as e:
            logger.error(f"Failed to load memory: {e}")
            return {}
    
    def save_memory(self):
        """Save conversation memory to file."""
        try:
            with open(self.storage_file, 'wb') as f:
                pickle.dump(self.conversations, f)
        except Exception as e:
            logger.error(f"Failed to save memory: {e}")
    
    def get_conversation(self, session_id: str):
        """Get conversation history for a session."""
        return self.conversations.get(session_id, {
            'messages': [],
            'actions': [],
            'queries': [],
            'created_at': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat()
        })
    
    def add_message(self, session_id: str, role: str, content: str):
        """Add a message to conversation history."""
        if session_id not in self.conversations:
            self.conversations[session_id] = {
                'messages': [],
                'actions': [],
                'queries': [],
                'created_at': datetime.now().isoformat(),
                'last_updated': datetime.now().isoformat()
            }
        
        self.conversations[session_id]['messages'].append({
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat()
        })
        self.conversations[session_id]['last_updated'] = datetime.now().isoformat()
        self.save_memory()
    
    def add_action(self, session_id: str, action: str, details: Dict[str, Any]):
        """Add an action to the conversation log."""
        if session_id not in self.conversations:
            self.conversations[session_id] = {
                'messages': [],
                'actions': [],
                'queries': [],
                'created_at': datetime.now().isoformat(),
                'last_updated': datetime.now().isoformat()
            }
        
        self.conversations[session_id]['actions'].append({
            'action': action,
            'details': details,
            'timestamp': datetime.now().isoformat()
        })
        self.conversations[session_id]['last_updated'] = datetime.now().isoformat()
        self.save_memory()
    
    def add_query(self, session_id: str, query: str, response: str):
        """Add a query-response pair to history."""
        if session_id not in self.conversations:
            self.conversations[session_id] = {
                'messages': [],
                'actions': [],
                'queries': [],
                'created_at': datetime.now().isoformat(),
                'last_updated': datetime.now().isoformat()
            }
        
        self.conversations[session_id]['queries'].append({
            'query': query,
            'response': response,
            'timestamp': datetime.now().isoformat()
        })
        self.conversations[session_id]['last_updated'] = datetime.now().isoformat()
        self.save_memory()
    
    def get_context_summary(self, session_id: str, max_messages: int = 5) -> str:
        """Get a summary of recent conversation context."""
        if session_id not in self.conversations:
            return "No previous conversation context."
        
        conv = self.conversations[session_id]
        recent_messages = conv['messages'][-max_messages:] if conv['messages'] else []
        recent_queries = conv['queries'][-3:] if conv['queries'] else []
        
        context = "Recent conversation context:\n"
        
        if recent_messages:
            context += "Recent messages:\n"
            for msg in recent_messages:
                role = "User" if msg['role'] == 'human' else "Assistant"
                context += f"- {role}: {msg['content'][:100]}...\n"
        
        if recent_queries:
            context += "Recent queries:\n"
            for q in recent_queries:
                context += f"- Q: {q['query'][:80]}...\n"
        
        return context

# Initialize global memory
memory = ConversationMemory()

# State definition
class AgentState(TypedDict):
    session_id: Annotated[str, "Unique session identifier"]
    messages: Annotated[List, "The messages in the conversation"]
    query: Annotated[str, "The user's query about Netflix data"]
    query_type: Annotated[str, "The type of query (search, filter, statistics, etc.)"]
    snowflake_data: Annotated[Any, "Data retrieved from Snowflake"]
    response: Annotated[str, "The final response to the user"]
    error: Annotated[str, "Any error that occurred"]
    conversation_context: Annotated[str, "Previous conversation context"]
    previous_actions: Annotated[List, "Previous actions taken in this session"]

def get_conversation_context(state: AgentState) -> AgentState:
    """Retrieve conversation context from memory."""
    session_id = state["session_id"]
    context = memory.get_context_summary(session_id)
    previous_actions = memory.conversations.get(session_id, {}).get('actions', [])
    
    state["conversation_context"] = context
    state["previous_actions"] = previous_actions
    
    # Add context to messages for LLM processing
    if context and context != "No previous conversation context.":
        context_message = f"Previous conversation context:\n{context}"
        state["messages"].append(HumanMessage(content=context_message))
    
    return state

def classify_query(state: AgentState) -> AgentState:
    """Classify the type of query the user is asking."""
    query = state["query"].lower()
    
    # Define query patterns
    if any(word in query for word in ["directed", "director", "by"]):
        state["query_type"] = "director_search"
    elif any(word in query for word in ["search", "find", "look for", "show me"]):
        state["query_type"] = "search"
    elif any(word in query for word in ["movie", "tv show", "series"]):
        state["query_type"] = "filter_by_type"
    elif any(word in query for word in ["country", "nation"]):
        state["query_type"] = "filter_by_country"
    elif any(word in query for word in ["year", "released", "2020", "2021", "2022", "2023", "2024"]):
        state["query_type"] = "filter_by_year"
    elif any(word in query for word in ["rating", "rated", "pg", "r", "tv-ma"]):
        state["query_type"] = "filter_by_rating"
    elif any(word in query for word in ["statistics", "stats", "count", "how many", "summary"]):
        state["query_type"] = "statistics"
    elif any(word in query for word in ["sample", "example", "few", "some"]):
        state["query_type"] = "sample_data"
    elif any(word in query for word in ["memory", "remember", "previous", "history", "what did you"]):
        state["query_type"] = "memory_query"
    else:
        state["query_type"] = "general_search"
    
    # Log the classification action
    memory.add_action(state["session_id"], "query_classification", {
        "query": state["query"],
        "classified_as": state["query_type"]
    })
    
    return state

def extract_search_terms(state: AgentState) -> AgentState:
    """Extract search terms from the user query."""
    query = state["query"]
    
    if state["query_type"] == "director_search":
        # Extract director name and content type for director queries
        import re
        
        # Look for patterns like "movies by [director]" or "[director] directed"
        director_patterns = [
            r'movies?\s+by\s+([a-zA-Z\s]+)',
            r'([a-zA-Z\s]+)\s+directed',
            r'which\s+movies?\s+([a-zA-Z\s]+)\s+directed',
            r'([a-zA-Z\s]+)\s+movies?'
        ]
        
        director_name = None
        content_type = "Movie"  # Default to movies for director queries
        
        for pattern in director_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                director_name = match.group(1).strip()
                break
        
        if director_name:
            state["director_name"] = director_name
            state["content_type"] = content_type
        else:
            # Fallback to LLM extraction
            prompt = f"""
            Extract the director name from this query about Netflix titles: "{query}"
            Return only the director's name.
            """
            try:
                response = llm.invoke([HumanMessage(content=prompt)])
                director_name = response.content.strip()
                state["director_name"] = director_name
                state["content_type"] = "Movie"
            except Exception as e:
                logger.error(f"Failed to extract director name: {e}")
                state["director_name"] = query
                state["content_type"] = "Movie"
    elif state["query_type"] == "memory_query":
        # Handle memory-related queries
        state["search_terms"] = "memory_query"
    else:
        # Use LLM to extract search terms for other query types
        prompt = f"""
        Extract the main search terms from this query about Netflix titles: "{query}"
        Return only the key terms, separated by commas.
        """
        
        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            search_terms = response.content.strip()
            state["search_terms"] = search_terms
        except Exception as e:
            logger.error(f"Failed to extract search terms: {e}")
            state["search_terms"] = query
    
    # Log the extraction action
    memory.add_action(state["session_id"], "search_term_extraction", {
        "query": query,
        "extracted_terms": state.get("search_terms", "N/A"),
        "director_name": state.get("director_name", "N/A")
    })
    
    return state

def retrieve_data(state: AgentState) -> AgentState:
    """Retrieve data from Snowflake based on the query type."""
    try:
        if state["query_type"] == "memory_query":
            # Handle memory queries without database access
            session_id = state["session_id"]
            conv = memory.conversations.get(session_id, {})
            
            if "memory" in state["query"].lower() or "remember" in state["query"].lower():
                # Show conversation memory
                memory_data = {
                    'total_queries': len(conv.get('queries', [])),
                    'total_actions': len(conv.get('actions', [])),
                    'session_duration': _calculate_session_duration(conv),
                    'recent_queries': conv.get('queries', [])[-5:],
                    'recent_actions': conv.get('actions', [])[-5:]
                }
                state["snowflake_data"] = memory_data
            else:
                # Show conversation history
                state["snowflake_data"] = conv
            return state
        
        with SnowflakeConnector() as snowflake:
            query_type = state["query_type"]
            logger.info(f"Processing query type: {query_type}")
            
            if query_type == "director_search":
                director_name = state.get("director_name", "")
                content_type = state.get("content_type", "Movie")
                logger.info(f"Director search for: '{director_name}', content type: {content_type}")
                
                if director_name:
                    # Try to get movies by director first
                    logger.info(f"Attempting to find movies by director: {director_name}")
                    data = snowflake.get_titles_by_director_and_type(director_name, content_type, limit=15)
                    logger.info(f"Director+type search returned {len(data)} results")
                    
                    # If no results, try broader director search but still filter by type
                    if data.empty:
                        logger.info("No results from director+type search, trying broader director search")
                        # Try fuzzy matching for director names
                        all_directors_data = snowflake.get_titles_by_director(director_name, limit=50)
                        logger.info(f"Broader director search returned {len(all_directors_data)} results")
                        
                        if not all_directors_data.empty:
                            # Filter by content type if we have director data
                            logger.info("Filtering results by content type")
                            data = all_directors_data[all_directors_data['TYPE'].str.lower() == content_type.lower()]
                            logger.info(f"After filtering by type: {len(data)} results")
                            if len(data) > 15:
                                data = data.head(15)
                        else:
                            # If still no results, get movies of the requested type as fallback
                            logger.info("No director results found, falling back to movies by type")
                            data = snowflake.get_titles_by_type(content_type, limit=15)
                            state["fallback_message"] = f"I couldn't find any movies directed by '{director_name}', but here are some movies available on Netflix."
                else:
                    # If no director name extracted, get movies as fallback
                    logger.info("No director name extracted, falling back to movies by type")
                    data = snowflake.get_titles_by_type("Movie", limit=15)
                    state["fallback_message"] = "I couldn't identify a specific director, but here are some movies available on Netflix."
                
                logger.info(f"Final data for director search: {len(data)} results, all movies: {all(data['TYPE'].str.lower() == 'movie') if not data.empty else 'N/A'}")
            
            elif query_type == "search":
                search_terms = state.get("search_terms", state["query"])
                data = snowflake.search_titles(search_terms, limit=15)
            
            elif query_type == "filter_by_type":
                if "movie" in state["query"].lower():
                    data = snowflake.get_titles_by_type("Movie", limit=15)
                elif "tv show" in state["query"].lower() or "series" in state["query"].lower():
                    data = snowflake.get_titles_by_type("TV Show", limit=15)
                else:
                    data = snowflake.get_sample_data(15)
            
            elif query_type == "filter_by_country":
                # Extract country from query
                query = state["query"].lower()
                if "india" in query:
                    data = snowflake.get_titles_by_country("India", limit=15)
                elif "usa" in query or "united states" in query:
                    data = snowflake.get_titles_by_country("United States", limit=15)
                elif "uk" in query or "united kingdom" in query:
                    data = snowflake.get_titles_by_country("United Kingdom", limit=15)
                else:
                    data = snowflake.get_sample_data(15)
            
            elif query_type == "filter_by_year":
                # Extract year from query
                import re
                year_match = re.search(r'\b(19|20)\d{2}\b', state["query"])
                if year_match:
                    year = int(year_match.group())
                    data = snowflake.get_titles_by_year(year, limit=15)
                else:
                    data = snowflake.get_sample_data(15)
            
            elif query_type == "filter_by_rating":
                # Extract rating from query
                query = state["query"].lower()
                ratings = ["g", "pg", "pg-13", "r", "nc-17", "tv-y", "tv-y7", "tv-g", "tv-pg", "tv-14", "tv-ma"]
                rating = next((r for r in ratings if r in query), "PG-13")
                data = snowflake.get_titles_by_rating(rating, limit=15)
            
            elif query_type == "statistics":
                data = snowflake.get_statistics()
            
            elif query_type == "sample_data":
                data = snowflake.get_sample_data(10)
            
            else:
                data = snowflake.get_sample_data(10)
            
            state["snowflake_data"] = data
            
            # Log the data retrieval action
            memory.add_action(state["session_id"], "data_retrieval", {
                "query_type": query_type,
                "data_count": len(data) if hasattr(data, '__len__') else "N/A",
                "success": True
            })
            
    except Exception as e:
        logger.error(f"Failed to retrieve data: {e}")
        state["error"] = f"Failed to retrieve data from Snowflake: {str(e)}"
        state["snowflake_data"] = None
        
        # Log the error action
        memory.add_action(state["session_id"], "data_retrieval_error", {
            "query_type": state.get("query_type", "unknown"),
            "error": str(e)
        })
    
    return state

def generate_response(state: AgentState) -> AgentState:
    """Generate a natural language response based on the retrieved data."""
    try:
        if state.get("error"):
            state["response"] = f"I encountered an error: {state['error']}"
            return state
        
        data = state["snowflake_data"]
        query = state["query"]
        
        if state["query_type"] == "memory_query":
            # Handle memory-related responses
            if isinstance(data, dict):
                if 'total_queries' in data:  # Memory summary
                    response = f"📚 **Conversation Memory Summary**\n\n"
                    response += f"• **Total Queries**: {data['total_queries']}\n"
                    response += f"• **Total Actions**: {data['total_actions']}\n"
                    response += f"• **Session Duration**: {data['session_duration']}\n\n"
                    
                    if data['recent_queries']:
                        response += "**Recent Queries:**\n"
                        for i, q in enumerate(data['recent_queries'], 1):
                            response += f"{i}. {q['query'][:60]}...\n"
                    
                    if data['recent_actions']:
                        response += "\n**Recent Actions:**\n"
                        for i, a in enumerate(data['recent_actions'], 1):
                            response += f"{i}. {a['action']}: {str(a['details'])[:50]}...\n"
                else:  # Full conversation history
                    response = f"📖 **Full Conversation History**\n\n"
                    
                    if data.get('queries'):
                        response += "**All Queries and Responses:**\n"
                        for i, q in enumerate(data['queries'], 1):
                            response += f"\n**Query {i}**: {q['query']}\n"
                            response += f"**Response**: {q['response'][:200]}...\n"
                            response += f"**Time**: {q['timestamp']}\n"
                    else:
                        response += "No previous conversation history found."
                
                state["response"] = response
                return state
        
        if isinstance(data, pd.DataFrame):
            if data.empty:
                if state["query_type"] == "director_search":
                    director_name = state.get("director_name", "this director")
                    state["response"] = f"I couldn't find any movies directed by {director_name} in the Netflix database. This could be because:\n\n1. The director's name might be spelled differently in our database\n2. The movies might not be available on Netflix\n3. The director might not have any movies in our current dataset\n\nTry searching with a different spelling or check if the director has other works available."
                else:
                    state["response"] = f"I couldn't find any Netflix titles matching your query: '{query}'. Please try different search terms or criteria."
                return state
            
            # Check if we have a fallback message for director searches
            fallback_message = state.get("fallback_message", "")
            if fallback_message:
                response = fallback_message + "\n\n"
            else:
                response = ""
            
            # Format the data for display
            if len(data) > 10:
                display_data = data.head(10)
                response += f"I found {len(data)} Netflix titles matching your query. Here are the first 10:\n\n"
            else:
                display_data = data
                response += f"I found {len(data)} Netflix titles matching your query:\n\n"
            
            # Create a formatted response
            for idx, row in display_data.iterrows():
                title = row.get('TITLE', row.get('title', 'Unknown Title'))
                content_type = row.get('TYPE', row.get('type', 'Unknown Type'))
                year = row.get('RELEASE_YEAR', row.get('release_year', 'Unknown Year'))
                country = row.get('COUNTRY', row.get('country', 'Unknown Country'))
                rating = row.get('RATING', row.get('rating', 'Unknown Rating'))
                
                response += f"• {title} ({content_type}, {year}, {country}, Rated: {rating})\n"
            
            if len(data) > 10:
                response += f"\n... and {len(data) - 10} more titles."
            
            state["response"] = response
        
        elif isinstance(data, dict) and state["query_type"] == "statistics":
            response = "Here are the statistics for Netflix titles:\n\n"
            
            if data.get('total_titles'):
                response += f"Total Titles: {data['total_titles']}\n\n"
            
            if data.get('by_type'):
                response += "By Content Type:\n"
                for item in data['by_type']:
                    response += f"• {item.get('TYPE', item.get('type', 'Unknown'))}: {item.get('count', 'Unknown')}\n"
                response += "\n"
            
            if data.get('by_rating'):
                response += "By Rating:\n"
                for item in data['by_rating']:
                    response += f"• {item.get('RATING', item.get('rating', 'Unknown'))}: {item.get('count', 'Unknown')}\n"
                response += "\n"
            
            if data.get('by_year'):
                response += "By Release Year (Top 10):\n"
                for item in data['by_year']:
                    response += f"• {item.get('RELEASE_YEAR', item.get('release_year', 'Unknown'))}: {item.get('count', 'Unknown')}\n"
            
            state["response"] = response
        
        else:
            response = f"I retrieved some data for your query: '{query}', but I'm not sure how to format it properly."
            state["response"] = response
        
        # Log the response generation action
        memory.add_action(state["session_id"], "response_generation", {
            "query": query,
            "response_length": len(state["response"]),
            "query_type": state["query_type"]
        })
        
    except Exception as e:
        logger.error(f"Failed to generate response: {e}")
        state["response"] = f"I encountered an error while processing your request: {str(e)}"
        
        # Log the error action
        memory.add_action(state["session_id"], "response_generation_error", {
            "query": query,
            "error": str(e)
        })
    
    return state

def should_continue(state: AgentState) -> str:
    """Determine if the conversation should continue or end."""
    if state.get("error") or state.get("response"):
        return END
    return "continue"

# Create the graph
def create_netflix_agent():
    """Create the Netflix data retrieval agent graph."""
    
    # Create the graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("get_conversation_context", get_conversation_context)
    workflow.add_node("classify_query", classify_query)
    workflow.add_node("extract_search_terms", extract_search_terms)
    workflow.add_node("retrieve_data", retrieve_data)
    workflow.add_node("generate_response", generate_response)
    
    # Add edges
    workflow.add_edge("get_conversation_context", "classify_query")
    workflow.add_edge("classify_query", "extract_search_terms")
    workflow.add_edge("extract_search_terms", "retrieve_data")
    workflow.add_edge("retrieve_data", "generate_response")
    workflow.add_edge("generate_response", END)
    
    # Set entry point
    workflow.set_entry_point("get_conversation_context")
    
    # Compile the graph
    app = workflow.compile()
    
    return app

# Main function to run the agent
def run_netflix_agent(query: str, session_id: str = None) -> str:
    """Run the Netflix agent with a user query."""
    try:
        # Generate session ID if not provided
        if not session_id:
            session_id = str(uuid.uuid4())
        
        # Create the agent
        agent = create_netflix_agent()
        
        # Initialize state
        initial_state = {
            "session_id": session_id,
            "messages": [],
            "query": query,
            "query_type": "",
            "snowflake_data": None,
            "response": "",
            "error": "",
            "conversation_context": "",
            "previous_actions": []
        }
        
        # Run the agent
        result = agent.invoke(initial_state)
        
        # Store the conversation in memory
        memory.add_message(session_id, "human", query)
        memory.add_message(session_id, "assistant", result["response"])
        memory.add_query(session_id, query, result["response"])
        
        return result["response"], session_id
        
    except Exception as e:
        logger.error(f"Agent execution failed: {e}")
        error_msg = f"I encountered an error while processing your request: {str(e)}"
        
        # Store error in memory
        if session_id:
            memory.add_message(session_id, "assistant", error_msg)
            memory.add_action(session_id, "execution_error", {"error": str(e)})
        
        return error_msg, session_id

def _calculate_session_duration(conv):
    """Calculate session duration from conversation data."""
    try:
        if not conv.get('created_at'):
            return "Unknown"
        
        created = datetime.fromisoformat(conv['created_at'])
        last_updated = datetime.fromisoformat(conv.get('last_updated', conv['created_at']))
        duration = last_updated - created
        
        if duration.days > 0:
            return f"{duration.days} days, {duration.seconds // 3600} hours"
        elif duration.seconds > 3600:
            return f"{duration.seconds // 3600} hours, {(duration.seconds % 3600) // 60} minutes"
        else:
            return f"{duration.seconds // 60} minutes"
    except:
        return "Unknown"

if __name__ == "__main__":
    print("🎬 Netflix Data Retrieval Agent with Memory")
    print("=" * 60)
    print("Connected to Snowflake Database: JNXKWEC-OX72808")
    print("Now with conversation memory and action tracking!")
    print("Ask me anything about Netflix titles!")
    print("Special commands:")
    print("- 'memory' or 'remember' - Show conversation memory")
    print("- 'history' - Show full conversation history")
    print("- 'quit' or 'exit' to stop")
    print("=" * 60)
    
    # Generate a session ID for this conversation
    current_session = str(uuid.uuid4())
    print(f"Session ID: {current_session[:8]}...")
    
    while True:
        try:
            # Get user input
            user_query = input("\n🔍 Enter your query: ").strip()
            
            # Check if user wants to quit
            if user_query.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye! Thanks for using Netflix Data Retrieval Agent!")
                print(f"Your conversation has been saved with session ID: {current_session[:8]}...")
                break
            
            # Skip empty queries
            if not user_query:
                print("❌ Please enter a valid query.")
                continue
            
            print(f"\n📝 Processing: {user_query}")
            print("-" * 50)
            
            # Run the agent with user query
            response, session_id = run_netflix_agent(user_query, current_session)
            print(f"🤖 Response: {response}")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye! Thanks for using Netflix Data Retrieval Agent!")
            print(f"Your conversation has been saved with session ID: {current_session[:8]}...")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Please try again with a different query.")
