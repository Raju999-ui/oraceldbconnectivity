import os
import google.generativeai as genai
from langgraph.graph import END, StateGraph
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_core.runnables import Runnable
import snowflake.connector
from pydantic import BaseModel

class AgentState(BaseModel):
    question: str
    sql: str = ""
    result: object = None

# 1. SET YOUR KEYS
os.environ["GOOGLE_API_KEY"] = "AIzaSyADx8ynZpyhbn-5-f_sBz0-G0xYfNIsKD0"

# 2. LLM - Gemini
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

# 3. Snowflake Connection
def connect_to_snowflake():
    return snowflake.connector.connect(
        user='RAJU',
        password='RAJUsridevi1234',
        account='jnxkwec-ox72808',  # <-- Only the account identifier, not the URL
        warehouse='COMPUTE_WH',
        database='DB1',
        schema='PRODUCT_SCHEMA'
    )

# 4. Run SQL and return results from products table
def run_query(sql_query):
    try:
        con = connect_to_snowflake()
        cursor = con.cursor()
        cursor.execute(sql_query)
        rows = cursor.fetchall()
        column_names = [desc[0] for desc in cursor.description]
        result = [dict(zip(column_names, row)) for row in rows]
        cursor.close()
        con.close()
        return result
    except Exception as e:
        return {"error": str(e)}

# 5. Agent Node: Convert user input to SQL and run it
def agent_node(state):
    question = state.question

    # Ask Gemini to convert natural language to SQL
    prompt = f"""
    You are a SQL expert. Convert this user question into a valid Snowflake SQL SELECT query
    using the 'products' table which has the following columns:
    product_id, product_name, product_use, product_seller.

    - Do NOT use backticks or quotes around table or column names.
    - Only give a valid SELECT SQL query. No explanation.

    Question: {question}
    """
    response = llm.invoke([HumanMessage(content=prompt)])
    sql_query = response.content.strip()

    # Run query
    result = run_query(sql_query)

    return {
        "question": question,
        "sql": sql_query,
        "result": result
    }

# 6. Define Graph
workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)
workflow.set_entry_point("agent")
workflow.set_finish_point("agent")

app = workflow.compile()

# 7. Run Agent Loop
if __name__ == "__main__":
    print("🧠 LangGraph Agent for Snowflake (products table)")
    print("Ask me anything about the products table. Type 'exit' to stop.\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        response = app.invoke({"question": user_input})

        print(f"\n📝 SQL Used:\n{response['sql']}")
        print(f"\n📦 Results:\n{response['result']}\n")
