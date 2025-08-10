import os
import json
from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, END
from groq import Groq # Import Groq client
from dotenv import load_dotenv
from google.colab import userdata # Import userdata

# load_dotenv()  # Remove load_dotenv as we will use Colab Secrets

api_key = userdata.get("GROQ_API_KEY")  # Get API key from Colab Secrets
if not api_key:
    raise ValueError("Please set the GROQ_API_KEY in Colab Secrets")

client = Groq(api_key=api_key) # Initialize Groq client

# Define the state
class ScheduleState(TypedDict):
    user_input: str
    appointments: List[dict]
    error: Annotated[str, "Should be a string"]

# Define the scheduler node as a function
def scheduler_node(state: ScheduleState) -> ScheduleState:
    user_input = state.get("user_input", "")
    appointments = state.get("appointments", [])
    error = None

    if not user_input:
        error = "No scheduling input provided"
        return {"error": error, "appointments": appointments}

    prompt = f"""
    Parse the following command and respond *only* in JSON format with keys "event" and "datetime" in ISO format.
    Command: {user_input}
    Example response: {{"event": "Team meeting", "datetime": "2025-08-11T15:00:00"}}
    """

    try:
        response = client.chat.completions.create(
            model="llama3-8b-8192", # Use a Groq model
            messages=[
                {"role": "system", "content": "You are an expert scheduling assistant that *only* outputs JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=150,
            response_format={"type": "json_object"}
        )

        result_text = response.choices[0].message.content.strip()
        parsed = json.loads(result_text)
        event = parsed.get("event")
        event_time = parsed.get("datetime")

        if not event or not event_time:
            error = "Failed to parse scheduling details"
            return {"error": error, "appointments": appointments}

        appointments.append({"event": event, "datetime": event_time})
        confirmation = f"Scheduled '{event}' at {event_time}"
        # In langgraph, nodes return the updates to the state.
        # We are returning the updated appointments list and a confirmation message
        # which we will store in the state for later use or printing.
        return {"appointments": appointments, "confirmation": confirmation, "error": None}

    except Exception as e:
        error = f"An error occurred: {str(e)}"
        return {"error": error, "appointments": appointments}

# Create a StateGraph
workflow = StateGraph(ScheduleState)

# Add the scheduler node
workflow.add_node("schedule", scheduler_node)

# Set the entry point
workflow.set_entry_point("schedule")

# Add a conditional edge to handle errors
# If 'error' is not None, we end the graph with the error state.
# Otherwise, we end the graph with the updated state.
workflow.add_conditional_edges(
    "schedule", # The node to add the edge from
    lambda state: "end_with_error" if state.get("error") else "end_with_success", # Condition to determine which edge to follow
    {
        "end_with_error": END, # If error, go to END
        "end_with_success": END # If no error, go to END
    }
)


# Compile the graph
app = workflow.compile()

# Initial state
initial_state = {"user_input": "Schedule a team meeting tomorrow at 3 PM", "appointments": []}

# Run the graph
result = app.invoke(initial_state)

# Print the final state
print(result)