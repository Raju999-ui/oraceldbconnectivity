import os
from dotenv import load_dotenv
from langchain_core.runnables import Runnable
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from typing import TypedDict

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Use Groq LLM
llm = ChatGroq(temperature=0, model="llama3-70b-8192", groq_api_key=groq_api_key)

# In-memory storage
family_memory = {}

# Define state schema
class GraphState(TypedDict):
    input: str
    next: str
    output: str

# Router
def router(state: GraphState) -> GraphState:
    user_input = state["input"]
    if user_input.lower() in family_memory:
        return {"next": "recall", "input": user_input, "output": ""}
    else:
        return {"next": "learn", "input": user_input, "output": ""}

# Learner
def learn(state: GraphState) -> GraphState:
    user_input = state["input"]
    print("🤖: I don't know that. Can you please teach me?")
    user_response = input("You: ").strip()
    family_memory[user_input.lower()] = user_response
    return {"input": user_input, "next": "", "output": f"Thank you! I learned that {user_input} is {user_response}"}

# Recall
def recall(state: GraphState) -> GraphState:
    user_input = state["input"]
    result = family_memory.get(user_input.lower(), "Sorry, I still don't know that.")
    return {"input": user_input, "next": "", "output": result}

# Build LangGraph with schema
workflow = StateGraph(GraphState)

workflow.add_node("router", router)
workflow.add_node("learn", learn)
workflow.add_node("recall", recall)

workflow.add_conditional_edges(
    "router",
    lambda x: x["next"],
    {
        "learn": "learn",
        "recall": "recall"
    }
)

workflow.add_edge("learn", END)
workflow.add_edge("recall", END)

workflow.set_entry_point("router")
app = workflow.compile()

# === Chat Loop ===
print("🤖 Family Tree Memory Agent\nType 'exit' to quit.\n")

while True:
    user_question = input("You: ").strip()
    if user_question.lower() == "exit":
        break

    result = app.invoke({"input": user_question, "next": "", "output": ""})
    print(f"🤖: {result['output']}")
