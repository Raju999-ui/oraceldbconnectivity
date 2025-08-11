from typing import Dict, Any, List, TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
import json
import pandas as pd
from snowflake_connector import SnowflakeConnector
from config import GROQ_API_KEY
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Groq LLM
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama3-8b-8192"
)

# State definition
class AgentState(TypedDict):
    messages: Annotated[List, "The messages in the conversation"]
    query: Annotated[str, "The user's query about Netflix data"]
    query_type: Annotated[str, "The type of query (search, filter, statistics, etc.)"]
    snowflake_data: Annotated[Any, "Data retrieved from Snowflake"]
    response: Annotated[str, "The final response to the user"]
    error: Annotated[str, "Any error that occurred"]

def classify_query(state: AgentState) -> AgentState:
    """Classify the type of query the user is asking."""
    query = state["query"].lower()
    
    # Define query patterns
    if any(word in query for word in ["search", "find", "look for", "show me"]):
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
    else:
        state["query_type"] = "general_search"
    
    return state

def extract_search_terms(state: AgentState) -> AgentState:
    """Extract search terms from the user query."""
    query = state["query"]
    
    # Use LLM to extract search terms
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
    
    return state

def retrieve_data(state: AgentState) -> AgentState:
    """Retrieve data from Snowflake based on the query type."""
    try:
        with SnowflakeConnector() as snowflake:
            query_type = state["query_type"]
            
            if query_type == "search":
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
            
    except Exception as e:
        logger.error(f"Failed to retrieve data: {e}")
        state["error"] = f"Failed to retrieve data from Snowflake: {str(e)}"
        state["snowflake_data"] = None
    
    return state

def generate_response(state: AgentState) -> AgentState:
    """Generate a natural language response based on the retrieved data."""
    try:
        if state.get("error"):
            state["response"] = f"I encountered an error: {state['error']}"
            return state
        
        data = state["snowflake_data"]
        query = state["query"]
        
        if isinstance(data, pd.DataFrame):
            if data.empty:
                response = f"I couldn't find any Netflix titles matching your query: '{query}'. Please try different search terms or criteria."
            else:
                # Format the data for display
                if len(data) > 10:
                    display_data = data.head(10)
                    response = f"I found {len(data)} Netflix titles matching your query. Here are the first 10:\n\n"
                else:
                    display_data = data
                    response = f"I found {len(data)} Netflix titles matching your query:\n\n"
                
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
        
        else:
            response = f"I retrieved some data for your query: '{query}', but I'm not sure how to format it properly."
        
        state["response"] = response
        
    except Exception as e:
        logger.error(f"Failed to generate response: {e}")
        state["response"] = f"I encountered an error while processing your request: {str(e)}"
    
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
    workflow.add_node("classify_query", classify_query)
    workflow.add_node("extract_search_terms", extract_search_terms)
    workflow.add_node("retrieve_data", retrieve_data)
    workflow.add_node("generate_response", generate_response)
    
    # Add edges
    workflow.add_edge("classify_query", "extract_search_terms")
    workflow.add_edge("extract_search_terms", "retrieve_data")
    workflow.add_edge("retrieve_data", "generate_response")
    workflow.add_edge("generate_response", END)
    
    # Set entry point
    workflow.set_entry_point("classify_query")
    
    # Compile the graph
    app = workflow.compile()
    
    return app

# Main function to run the agent
def run_netflix_agent(query: str) -> str:
    """Run the Netflix agent with a user query."""
    try:
        # Create the agent
        agent = create_netflix_agent()
        
        # Initialize state
        initial_state = {
            "messages": [],
            "query": query,
            "query_type": "",
            "snowflake_data": None,
            "response": "",
            "error": ""
        }
        
        # Run the agent
        result = agent.invoke(initial_state)
        
        return result["response"]
        
    except Exception as e:
        logger.error(f"Agent execution failed: {e}")
        return f"I encountered an error while processing your request: {str(e)}"

if __name__ == "__main__":
    print("🎬 Netflix Data Retrieval Agent")
    print("=" * 50)
    print("Connected to Snowflake Database: JNXKWEC-OX72808")
    print("Ask me anything about Netflix titles!")
    print("Type 'quit' or 'exit' to stop")
    print("=" * 50)
    
    while True:
        try:
            # Get user input
            user_query = input("\n🔍 Enter your query: ").strip()
            
            # Check if user wants to quit
            if user_query.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye! Thanks for using Netflix Data Retrieval Agent!")
                break
            
            # Skip empty queries
            if not user_query:
                print("❌ Please enter a valid query.")
                continue
            
            print(f"\n📝 Processing: {user_query}")
            print("-" * 50)
            
            # Run the agent with user query
            response = run_netflix_agent(user_query)
            print(f"🤖 Response: {response}")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye! Thanks for using Netflix Data Retrieval Agent!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Please try again with a different query.")
