import os
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.tools import BaseTool
import json

# ------------------ SETUP ------------------
# Credentials (Note: In production, use environment variables for security)
GROQ_API_KEY = "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"
SNOWFLAKE_USER = "RAJU"
SNOWFLAKE_PASSWORD = "RAJUsridevi1234"
SNOWFLAKE_ACCOUNT = "JNXKWEC-OX72808"
SNOWFLAKE_WAREHOUSE = "COMPUTE_WH"
SNOWFLAKE_DATABASE = "DB1"
SNOWFLAKE_SCHEMA = "PRODUCT_SCHEMA"

# Snowflake connection URI - using proper format
snowflake_uri = (
    f"snowflake://{SNOWFLAKE_USER}:{SNOWFLAKE_PASSWORD}"
    f"@{SNOWFLAKE_ACCOUNT}/"
    f"{SNOWFLAKE_DATABASE}/{SNOWFLAKE_SCHEMA}?"
    f"warehouse={SNOWFLAKE_WAREHOUSE}"
)

# Connect to Snowflake
db = SQLDatabase.from_uri(snowflake_uri)

# LLM (Groq API)
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model="llama3-70b-8192"  # You can switch to smaller/faster models
)

# Memory for multi-turn conversation
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Toolkit for SQL interaction
toolkit = SQLDatabaseToolkit(db=db, llm=llm)

# Create SQL Agent
agent_executor = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    memory=memory,
    verbose=True
)

# ------------------ LANGGRAPH STATE DEFINITION ------------------
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    query: str
    response: str
    error: str
    retry_count: int

# ------------------ LANGGRAPH NODES ------------------
def process_query(state: AgentState) -> AgentState:
    """Process the user query and generate response"""
    try:
        query = state["query"]
        print(f"\n🔍 Processing query: {query}")
        
        # Use the existing agent executor
        response = agent_executor.run(query)
        
        return {
            **state,
            "response": response,
            "error": "",
            "retry_count": 0
        }
    except Exception as e:
        error_msg = str(e)
        print(f"\n❌ Error occurred: {error_msg}")
        return {
            **state,
            "error": error_msg,
            "response": "",
            "retry_count": state.get("retry_count", 0) + 1
        }

def retry_query(state: AgentState) -> AgentState:
    """Retry the query with a more explicit formulation"""
    try:
        query = state["query"]
        retry_count = state.get("retry_count", 0)
        
        if retry_count >= 2:
            return {
                **state,
                "response": "Maximum retry attempts reached. Please rephrase your question.",
                "error": "Max retries exceeded"
            }
        
        print(f"🔄 Attempting to reframe the query and retry... (Attempt {retry_count + 1})")
        
        # Create a more explicit query for the LLM
        retry_query_text = f"Rephrase the following to be valid SQL for Snowflake database: {query}"
        
        # Use the LLM to rephrase the query
        fixed_query = llm.predict(retry_query_text)
        print(f"🔄 Rephrased query: {fixed_query}")
        
        # Try to execute the fixed query
        response = agent_executor.run(fixed_query)
        
        return {
            **state,
            "response": f"✅ Fixed Answer: {response}",
            "error": "",
            "retry_count": retry_count + 1
        }
        
    except Exception as e:
        return {
            **state,
            "error": f"Retry failed: {str(e)}",
            "response": "",
            "retry_count": state.get("retry_count", 0) + 1
        }



def format_response(state: AgentState) -> AgentState:
    """Format the final response for display"""
    if state.get("response"):
        print(f"\n📊 Answer: {state['response']}")
    elif state.get("error"):
        print(f"\n⚠️ Final Error: {state['error']}")
    
    return state

# ------------------ LANGGRAPH GRAPH CONSTRUCTION ------------------
def create_agent_graph():
    """Create the LangGraph agent workflow"""
    
    # Create the graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("process_query", process_query)
    workflow.add_node("retry_query", retry_query)
    workflow.add_node("format_response", format_response)
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "process_query",
        lambda state: "retry" if state.get("error") and state.get("retry_count", 0) < 2 else "format_response"
    )
    
    workflow.add_conditional_edges(
        "retry_query",
        lambda state: "retry" if state.get("error") and state.get("retry_count", 0) < 2 else "format_response"
    )
    
    # Add final edge
    workflow.add_edge("format_response", END)
    
    # Set entry point
    workflow.set_entry_point("process_query")
    
    # Compile the graph
    return workflow.compile()

# ------------------ MAIN EXECUTION ------------------
def main():
    """Main execution loop"""
    print("🚀 Snowflake Netflix LangGraph Agent is ready! (type 'exit' to quit)\n")
    
    # Create the agent graph
    agent_graph = create_agent_graph()
    
    while True:
        try:
            # Get user input
            query = input("Ask me about Netflix titles: ")
            
            if query.lower() == "exit":
                print("👋 Goodbye!")
                break
            
            if not query.strip():
                print("Please enter a valid query.")
                continue
            
            # Initialize state
            initial_state = {
                "messages": [HumanMessage(content=query)],
                "query": query,
                "response": "",
                "error": "",
                "retry_count": 0
            }
            
            # Execute the graph
            print("\n🔄 Processing your request...")
            result = agent_graph.invoke(initial_state)
            
            # Add response to messages for memory
            if result.get("response"):
                result["messages"].append(AIMessage(content=result["response"]))
            
            print("\n" + "="*50)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n💥 Unexpected error: {e}")
            print("Please try again.")

if __name__ == "__main__":
    main()
