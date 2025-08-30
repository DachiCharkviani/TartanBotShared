import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from langchain_core.messages import HumanMessage
from agents.agents import GRAPH  # must expose GRAPH in agents/agents.py

def run(query: str, session_id: str = "smoke-test-1"):
    msgs = [HumanMessage(content=query)]
    config = {"configurable": {"thread_id": session_id}}
    state = GRAPH.invoke({"messages": msgs}, config=config)
    
    print(state["messages"][-1].content)
    
    for m in state["messages"]:
        role = getattr(m, "type", getattr(m, "role", "message"))
        content = getattr(m, "content", str(m))
        print(f"- {role}: {content[:120]}")

if __name__ == "__main__":
    run("what are the minor requirements of Biological Sciences")