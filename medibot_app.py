from config import GEMINI_API_KEY
import google.generativeai as genai
from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional
import gradio as gr

# Setup Gemini
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

# Define state
class HealthState(TypedDict):
    symptoms: Optional[str]
    diagnosis: Optional[str]
    medicine: Optional[str]
    tips: Optional[str]
    plan: Optional[str]
    report: Optional[str]

# --- AGENTS ---

def diagnosis_agent(state: HealthState) -> HealthState:
    prompt = f"Given the symptoms: {state['symptoms']}, what could be possible causes (keep it brief but informative)?"
    result = model.generate_content(prompt)
    return {"diagnosis": result.text}

def medicine_agent(state: HealthState) -> HealthState:
    prompt = f"Based on the symptoms '{state['symptoms']}', list over-the-counter or commonly prescribed medicines and usage."
    result = model.generate_content(prompt)
    return {"medicine": result.text}

def tips_agent(state: HealthState) -> HealthState:
    prompt = f"Suggest 5 general health or lifestyle tips to manage or prevent conditions related to: {state['symptoms']}"
    result = model.generate_content(prompt)
    return {"tips": result.text}

def planner_agent(state: HealthState) -> HealthState:
    prompt = "Build a simple weekly follow-up or self-care plan based on the user's symptoms and suggestions."
    result = model.generate_content(prompt)
    return {"plan": result.text}

def report_agent(state: HealthState) -> HealthState:
    prompt = f"""
Build a professional health summary report with the following:

Symptoms: {state['symptoms']}
Possible Diagnosis: {state['diagnosis']}
Recommended Medicine: {state['medicine']}
Health Tips: {state['tips']}
Weekly Plan: {state['plan']}

Make it clear, structured and use emojis and formatting.
"""
    result = model.generate_content(prompt)
    return {"report": result.text}

# --- LANGGRAPH ---
graph = StateGraph(HealthState)

graph.add_node("diagnosis_agent", diagnosis_agent)
graph.add_node("medicine_agent", medicine_agent)
graph.add_node("tips_agent", tips_agent)
graph.add_node("planner_agent", planner_agent)
graph.add_node("report_agent", report_agent)

graph.set_entry_point("diagnosis_agent")
graph.add_edge("diagnosis_agent", "medicine_agent")
graph.add_edge("medicine_agent", "tips_agent")
graph.add_edge("tips_agent", "planner_agent")
graph.add_edge("planner_agent", "report_agent")
graph.add_edge("report_agent", END)

app = graph.compile()

# --- GRADIO UI ---
def run_medibot(symptom_input):
    state = app.invoke({"symptoms": symptom_input})
    return (
        state["diagnosis"],
        state["medicine"],
        state["tips"],
        state["plan"],
        state["report"]
    )

with gr.Blocks(title="MediBot AI") as demo:
    gr.Markdown("# 🏥 MediBot AI — Your Medical Assistant Agent")
    symptoms = gr.Textbox(label="🧾 Enter your symptoms (e.g., fever, cough, fatigue)", lines=3)

    run_btn = gr.Button("🧠 Diagnose")
    diag = gr.Textbox(label="🔍 Diagnosis")
    meds = gr.Textbox(label="💊 Medicines")
    tips = gr.Textbox(label="🧠 Health Tips")
    plan = gr.Textbox(label="📅 Weekly Plan")
    report = gr.Textbox(label="📄 Final Medical Report", lines=10)

    run_btn.click(fn=run_medibot, inputs=symptoms, outputs=[diag, meds, tips, plan, report])

# Run app
if __name__ == "__main__":
    demo.launch(share=True)
