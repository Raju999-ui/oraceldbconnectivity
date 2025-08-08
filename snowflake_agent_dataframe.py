import os
import pandas as pd
import google.generativeai as genai
from langgraph.graph import StateGraph
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
import snowflake.connector
from typing import TypedDict, Optional

# 1. API Key for Gemini
os.environ["GOOGLE_API_KEY"] = "AIzaSyADx8ynZpyhbn-5-f_sBz0-G0xYfNIsKD0"

# 2. Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

# 3. Snowflake Connection
def connect_to_snowflake():
    return snowflake.connector.connect(
        user="RAJU",  # your Snowflake username
        password="RAJUsridevi1234",  # your Snowflake password
        account="jnxkwec-ox72808",  # e.g., abcd-xy12345.ap-southeast-1
        warehouse="COMPUTE_WH",
        database="DB1",
        schema="PRODUCT_SCHEMA"
    )

# 4. Execute SQL query and return DataFrame
def run_query(sql_query):
    try:
        con = connect_to_snowflake()
        cursor = con.cursor()
        cursor.execute(sql_query)
        rows = cursor.fetchall()
        col_names = [desc[0] for desc in cursor.description]
        df = pd.DataFrame(rows, columns=col_names)  # Store in DataFrame
        cursor.close()
        con.close()
        return df
    except Exception as e:
        return pd.DataFrame({"error": [str(e)]})

# 5. LangGraph Agent Node
def agent_node(state):
    question = state["question"]

    # Ask Gemini to create a SQL query for the products table
    prompt = f"""
    You are a SQL expert. Convert this question into a SELECT SQL query
    for the 'products' table with columns:
    product_id, product_name, product_use, product_seller.

    Only output the SQL query. Do NOT include explanations, markdown, or code fences.
    Do NOT use backticks or any formatting. Only output valid Snowflake SQL.

    Question: {question}
    """
    sql_response = llm.invoke([HumanMessage(content=prompt)])
    sql_query = sql_response.content.strip()
    # Remove code fences and backticks if present
    sql_query = sql_query.replace("```sql", "").replace("```", "").replace("`", "").strip()

    print("DEBUG: Generated SQL:", sql_query)  # Add this line

    # Run query and get DataFrame
    df_result = run_query(sql_query)

    return {
        "question": question,
        "sql": sql_query,
        "result": df_result
    }

class AgentState(TypedDict):
    question: str
    sql: Optional[str]
    result: object

# 6. Create LangGraph workflow
workflow = StateGraph(state_schema=AgentState)
workflow.add_node("agent", agent_node)
workflow.set_entry_point("agent")
workflow.set_finish_point("agent")
app = workflow.compile()

# 7. Main loop
if __name__ == "__main__":
    print("🧠 Snowflake DataFrame Agent")
    print("Type your question (or 'exit' to quit):\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        response = app.invoke({"question": user_input})
        print("\n📝 SQL Used:")
        print(response["sql"])
        print("\n📦 Results:")
        print(response["result"].to_string(index=False))
        print()
