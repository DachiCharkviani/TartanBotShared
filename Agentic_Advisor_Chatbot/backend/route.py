from fastapi import APIRouter

from langchain_core.messages import HumanMessage
from pydantic import BaseModel
from typing import Dict, List
from collections import defaultdict

from backend.settings import API_HOST, API_PORT, CORS_ORIGINS
from agents.agents import GRAPH

SESSION_MESSAGES: Dict[str, List] = defaultdict(list)
'''app = FastAPI(title="Agentic RAG API")


app.add_middleware(CORSMiddleware,
                   allow_origins=CORS_ORIGINS,
                   allow_credentials=True,
                   allow_methods=["*"],
                   allow_headers=["*"],)'''

router = APIRouter()

class ChatRequest(BaseModel):
    session_id: str
    message: str

@router.post("/chat")

def chat_endpoint(req: ChatRequest):
    
    msgs = SESSION_MESSAGES[req.session_id][:]
    messages =[HumanMessage(content=req.message)]
    print(messages)
    
    # Get the final response
    final_response = None
    try:
        config = {"configurable": {"thread_id": req.session_id}} # Example config
        final_state = GRAPH.invoke({"messages":messages}, config=config)
        final_response = final_state['messages'][-1].content
        if final_state.get('error'):
            print(f"Error encountered: {final_state.get('error')}")
    except Exception as e:
        print(f"ERROR running agent graph for query ': {e}")
    return {"reply":final_response} if final_response else {"reply":"I apologize, but I couldn't generate a response."}