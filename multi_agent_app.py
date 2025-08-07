# multi_agent_app.py

from config import GEMINI_API_KEY
import google.generativeai as genai
from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional
import gradio as gr

# ✅ Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

# ✅ Define LangGraph state
class AppState(TypedDict):
    news: Optional[str]
    match: Optional[str]
    jobs: Optional[str]
    report: Optional[str]

# ✅ Agent 1: News Agent
def news_agent(state: AppState) -> AppState:
    prompt = "Give me 3 trending news headlines in India with short summaries."
    response = model.generate_content(prompt)
    return {"news": response.text}

# ✅ Agent 2: Match Agent
def match_agent(state: AppState) -> AppState:
    prompt = "Provide the latest cricket or football match result with score and summary."
    response = model.generate_content(prompt)
    return {"match": response.text}

# ✅ Agent 3: Job Agent
def job_agent(state: AppState) -> AppState:
    prompt = "List 3 remote developer jobs with title and 1-line description."
    response = model.generate_content(prompt)
    return {"jobs": response.text}

# ✅ Agent 4: Final Report Generator
def report_agent(state: AppState) -> AppState:
    prompt = f"""
Combine the information below into a structured professional report:

📰 News:
{state.get('news')}

⚽ Match:
{state.get('match')}

💼 Jobs:
{state.get('jobs')}

Format:
1. Today's News
2. Sports Summary
3. Remote Jobs
"""
    response = model.generate_content(prompt)
    return {"report": response.text}

# ✅ LangGraph
graph = StateGraph(AppState)
graph.add_node("news_agent", news_agent)
graph.add_node("match_agent", match_agent)
graph.add_node("job_agent", job_agent)
graph.add_node("report_agent", report_agent)

graph.set_entry_point("news_agent")
graph.add_edge("news_agent", "match_agent")
graph.add_edge("match_agent", "job_agent")
graph.add_edge("job_agent", "report_agent")
graph.add_edge("report_agent", END)

app = graph.compile()

# ✅ Gradio UI function
def run_all_agents():
    result = app.invoke({})
    return (
        result["news"],
        result["match"],
        result["jobs"],
        result["report"]
    )

# ✅ Build Gradio interface
with gr.Blocks(title="🧠 Multi-Agent Gemini System") as demo:
    gr.Markdown("## 🧠 Multi-Agent Reporting System")
    gr.Markdown("Click below to run agents and generate full report.")

    run_button = gr.Button("🚀 Run All Agents")

    news_box = gr.Textbox(label="📰 News Summary", lines=4)
    match_box = gr.Textbox(label="⚽ Match Info", lines=3)
    jobs_box = gr.Textbox(label="💼 Remote Job Listings", lines=4)
    report_box = gr.Textbox(label="📄 Final Report", lines=10)

    run_button.click(fn=run_all_agents, outputs=[news_box, match_box, jobs_box, report_box])

# ✅ Run the app
if __name__ == "__main__":
    demo.launch()
