import warnings
from langchain_groq import ChatGroq
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated
from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver()

warnings.filterwarnings("ignore", message="Core Pydantic V1 functionality")

generator_llm = ChatGroq(model="openai/gpt-oss-20b")
reflector_llm = ChatGroq(model="openai/gpt-oss-20b")

MAX_ITERATIONS = 3

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    task: str
    iteration: int
    approved: bool

def generate(state: State):
    system = SystemMessage(content=(
        "You are a writer. Produce or revise a draft based on the task and, "
        "if present, the critique in the conversation. Output only the draft text."
    ))
    response = generator_llm.invoke([system] + state["messages"])
    print(f"\n--- Draft (iteration {state['iteration'] + 1}) ---\n{response.content}\n")
    return {
        "messages": [response],
        "iteration": state["iteration"] + 1,
    }

def reflect(state: State):
    last_draft = state["messages"][-1].content
    system = SystemMessage(content=(
        "You are a strict editor. Critique the draft below against the original task. "
        "Be specific about concrete flaws (clarity, accuracy, structure, tone). "
        "If it's genuinely good, respond with exactly: APPROVED. "
        "Otherwise, list the flaws — do not rewrite the draft yourself."
    ))
    critique_prompt = HumanMessage(content=f"Task: {state['task']}\n\nDraft:\n{last_draft}")
    response = reflector_llm.invoke([system, critique_prompt])
    print(f"--- Critique ---\n{response.content}\n")

    approved = response.content.strip().startswith("APPROVED")
    return {
        "messages": [response],
        "approved": approved,
    }

def should_continue(state: State):
    if state["approved"] or state["iteration"] >= MAX_ITERATIONS:
        return END
    return "generate"

graph = StateGraph(State)
graph.add_node("generate", generate)
graph.add_node("reflect", reflect)

graph.add_edge(START, "generate")
graph.add_edge("generate", "reflect")
graph.add_conditional_edges("reflect", should_continue, {"generate": "generate", END: END})

workflow = graph.compile(checkpointer=checkpointer)
# every invoke needs a thread_id in its config now
config = {"configurable": {"thread_id": "task-1"}}

task = "Write a 3-sentence bio of Ada Lovelace for a general audience."
result = workflow.invoke({
    "messages": [HumanMessage(content=task)],
    "task": task,
    "iteration": 0,
    "approved": False,},
    config=config,)

print("=== FINAL DRAFT ===")
print(result["messages"][-2].content if not result["approved"] else result["messages"][-2].content)