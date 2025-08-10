import os
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from openai import OpenAI
from dotenv import load_dotenv
from google.colab import userdata

# load_dotenv() # Remove load_dotenv as we will use Colab Secrets

api_key = userdata.get("GROQ_API_KEY") # Get API key from Colab Secrets
if not api_key:
    raise ValueError("Please set the GROQ_API_KEY in Colab Secrets")

client = OpenAI(api_key=api_key)

# Define the state for the reporting graph
class ReportState(TypedDict):
    report_query: str
    report: Annotated[str, "Should be a string"]
    error: Annotated[str, "Should be a string"]

# Define the reporting node as a function
def reporting_node(state: ReportState) -> ReportState:
    report_query = state.get("report_query", "")
    error = None

    if not report_query:
        error = "No report query provided"
        return {"error": error, "report": ""}

    prompt = f"""
    You are a data reporting assistant. Based on the query below, provide a clear and concise summary report answer:
    Query: {report_query}
    """

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert data reporting assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=300,
        )

        report_content = response.choices[0].message.content.strip()
        return {"report": report_content, "error": None}

    except Exception as e:
        error = f"An error occurred: {str(e)}"
        return {"error": error, "report": ""}

# Create a StateGraph for reporting
report_workflow = StateGraph(ReportState)

# Add the reporting node
report_workflow.add_node("generate_report", reporting_node)

# Set the entry point
report_workflow.set_entry_point("generate_report")

# Add a conditional edge to handle errors
report_workflow.add_conditional_edges(
    "generate_report", # The node to add the edge from
    lambda state: "end_with_error" if state.get("error") else "end_with_success", # Condition to determine which edge to follow
    {
        "end_with_error": END, # If error, go to END
        "end_with_success": END # If no error, go to END
    }
)

# Compile the graph
report_app = report_workflow.compile()

# Example usage
initial_report_state = {"report_query": "Summarize the sales performance for last quarter.", "report": ""}

# Run the graph
report_result = report_app.invoke(initial_report_state)

# Print the final state
print("Report Summary:\n", report_result.get("report", "No report generated"))
if report_result.get("error"):
    print("Error:", report_result.get("error"))