import os
from typing import TypedDict, Optional

from langgraph.graph import StateGraph, END
from groq import Groq
from dotenv import load_dotenv


load_dotenv()


class GraphState(TypedDict, total=False):
    prompt: str
    code: str
    error: str


def _create_groq_client() -> Optional[Groq]:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None
    return Groq(api_key=api_key)


groq_client = _create_groq_client()


def coding_node(state: GraphState) -> GraphState:
    prompt = state.get("prompt", "")
    if not prompt:
        return {"error": "No prompt provided"}

    if groq_client is None:
        return {
            "error": "GROQ_API_KEY is not set. Please set it in your environment to enable LLM calls.",
        }

    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You are a helpful coding assistant. Return only code unless asked otherwise."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=500,
    )

    completion = response.choices[0].message.content or ""
    return {"code": completion}


workflow = StateGraph(GraphState)
workflow.add_node("coding_node", coding_node)
workflow.set_entry_point("coding_node")
workflow.add_edge("coding_node", END)
graph = workflow.compile()


if __name__ == "__main__":
    initial_state: GraphState = {
        "prompt": "Write a Python function to check if a number is prime.",
    }
    result = graph.invoke(initial_state)
    print("=== Generated Code ===")
    print(result.get("code", result.get("error", "No output")))
