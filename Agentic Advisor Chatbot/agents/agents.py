
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from typing import Literal, Sequence, Annotated, Dict
from typing_extensions import TypedDict
from langchain_core.output_parsers import StrOutputParser
#from langchain_groq import ChatGroq
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, SystemMessage, BaseMessage
from langgraph.graph.message import add_messages
from langgraph.graph import END, StateGraph, START, MessagesState
#from langgraph.prebuilt import ToolNode, tools_condition
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
#from IPython.display import Image, display, Markdown
#from langgraph.checkpoint.memory import MemorySaver
#from langchain.tools.retriever import create_retriever_tool
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
import re

from backend.settings import PROGRAMS,OPENAI_CHAT_MODEL, OPENAI_API_KEY, OPENAI_EMBEDDING_MODEL, PERSIST_DIR, FILE_PATH_ADDMAJORS, FILE_PATH_MINORS, FILE_PATH_COURSES, FILE_PATH_MAJORS


def get_embeddings():
    return OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL, api_key=OPENAI_API_KEY)

def get_vectorestore(courseloc):
    embeddings = get_embeddings()
    print(PERSIST_DIR+courseloc)
    print(embeddings)
    return Chroma(persist_directory= PERSIST_DIR+courseloc, embedding_function= embeddings)

def extract_course_code(query: str) -> str:
    match = re.search(r"(\d{2,3})[- ]?(\d{3})", query)
    if match:
        return f"{match.group(1)}-{match.group(2)}"
    return None

def get_llm():
    return ChatOpenAI(model="gpt-4o")

# Graph state
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    source: str
    department: str


class RouteQuery(BaseModel):
    department: str = Field(description=f"Most relevant programs from the list: {PROGRAMS }.")

def agent(state):
    """
    
    Classifies the user message for the target program using an LLM.
    """
    print("***********CALL AGENT**************")
    
    messages = state['messages']
    model = get_llm()
    # Prepare prompt for classification
    prompt_template = ChatPromptTemplate.from_messages([
        SystemMessage(
            content=f"""You are an expert at routing a user question to different vectorstores or web search.
                    Analyze the user's message to determine its the most relevant program's vectorstores.
                  The available programs are: {', '.join(PROGRAMS)}  
                The IS vectorstore contains documents related to Information Systems Concentration, advising documents, 
                concentration FAQs, requirements and sample curriculum.
                The BS vectorstore contains documents related to Biological Sciences Concentration, advising documents, 
                requirements and sample curriculum.
                The BA vectorstore contains documents related to Business Administration Concentration, advising documents, 
                requirements and sample curriculum.
                The CS vectorstore contains documents related to Computer Science Concentration, advising documents, 
                requirements and sample curriculum.
                The GENED vectorstore contains documents related to General Education Concentration, advising documents, 
                requirements and sample curriculum.
                The Courses vectorstore contains documents related to all courses offered by CMU. 
                It contains the course number (for ex: 67-373, or 67373 or 67 373), course name,
                prerequisites, offerings, long description maximum and minimum units etc.
                The Minors vectorstore contains documents related to minor course requirements such as minor name, advisor name, required courses, etc
                The AddMajors vectorstore contains documents related to additional major details of the course. 
                Use the corresponding vectorstore for questions on these topics. 
                Otherwise, use web-search.
                Respond ONLY with the structured JSON output format."""
        ),
        HumanMessage(content=f"User Query: {messages}")
    ])
    classifier_chain = prompt_template | model.with_structured_output(RouteQuery)
    try:
        result: RouteQuery = classifier_chain.invoke({}) # Pass empty dict as input seems required now
        print(f"  Classification Result: program='{result.department}'")
        return {
            "department": result.department
            }
    except Exception as e:
        print(f"  Error during classification: {e}")
        return {
            "department": "web_search",
            "error": f"Classification failed: {e}"
            }
    
def retrieve_context_node(state: AgentState) -> Dict[str, str]:
    """
    Retrieves relevant context from the vector store based on the query and department.
    """
    print("--- Retrieving Context ---")
    messages = state["messages"]
    department = state["department"]
    course_code = ""
   
    # Initialize embedding model and vector store access
    if department == "IS":
        vector_store = get_vectorestore(courseloc="IS/MajorData") 
    
    elif department == "CS":
        vector_store = get_vectorestore(courseloc="CS/MajorData")

    elif department == "BS":
        vector_store = get_vectorestore(courseloc="BS/MajorData")

    elif department == "BA":
        vector_store = get_vectorestore(courseloc="BA/MajorData")

    elif department == "GENED":
        vector_store = get_vectorestore(courseloc="GENED/MajorData")
    
    elif department == "Courses":
        vector_store = get_vectorestore(courseloc="Courses")
        course_code = extract_course_code(messages[0].content)
        print("course",course_code)
    
    elif department == "Minors":
        vector_store = get_vectorestore(courseloc="MinorData")
    elif department == "AddMajors":
        vector_store = get_vectorestore(courseloc="AddMajorData")
    elif department == "web-search":
        print("  Skipping retrieval: Department unknown or not applicable.")
        return {"context": "", "error": "Cannot retrieve context without a valid department."}
    
    if course_code:
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={
            "k":5,
                "filter": {
                    "source": course_code
                }
            }
            
        )
    else:
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={
                'k': 5, # Retrieve top 5 relevant docs
                
                }
        )

    try:
        print("retriever",vector_store)
        retrieved_docs = retriever.invoke(messages[0].content)
        print("retrieved docs",retrieved_docs)
        if retrieved_docs:
            context = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
            return {'messages':[context]}
        else:
            print("  No relevant documents found in vector store for this department.")
            return {"context": "", "error": "No relevant context found."}
    except Exception as e:
        print(f"  Error during context retrieval: {e}")
        return {"context": "", "error": f"Retrieval failed: {e}"}

from pydantic import BaseModel, Field    
def grade_documents(state) -> Literal['generate','rewrite']:
    """
    Determine whether the retrieved documents are relevant to the question.

    Args:
        state(messages): the current state

    Returns:
        str: A decision for whether the documents are relevant or not.
    """
    print("********************Check Relevance***************")

    #Data model
    class grade(BaseModel):
        """ Binary score for relevance check"""
        binary_score: str = Field(description="Relevance Score 'yes' or 'no'")

    model = get_llm()
    llm_with_tool = model.with_structured_output(grade)

    prompt = PromptTemplate(
        template="""
        You are a grader assessing relevance of a retrieved document to a user question.\n
        Here is the retrieved document : \n\n{context} \n\n
        Here is the user query: {question} \n
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.
        """,
        input_variables=['context','question'],
    )

    #chain
    chain = prompt | llm_with_tool

    messages = state['messages']
    last_message = messages[-1]
    print(messages)

    question = messages[0].content
    print(question)
    docs = last_message.content
    print("docs",docs)

    scored_result = chain.invoke({'question':question,'context':docs})
    score = scored_result.binary_score

    if score=="yes":
        print("Docs are relevent")
        return 'generate'
    else:
        print("Docs are not relevant")
        return 'rewrite'

def generate(state):

    """
    Generates a response using RAG based on the query and retrieved context.

    Args:
        state(messages): the current state
    Returns:
        dict: the updated message
    """

    print("***********Generate*************")
    messages = state['messages']
    question = messages[0].content
    last_message = messages[-1]

    docs = last_message.content

    prompt = PromptTemplate(
        template="""You are a helpful assistant for CMU Qatar students.
        Use the following context to answer the question about the corresponding program requirements, courses, concentrations, curriculum, or policies.

        Context: {context}

        Question: {question}
        Guidelines for your response:
        1. Base your answer STRICTLY on the provided context
        2. If the context contains specific course numbers, requirements, or policies, include them
        3. If the context is insufficient, acknowledge this and suggest where to find more information
        4. Keep the response focused on CMU Qatar program specifics
        5. Format your response in a clear, structured manner
        
        
        Answer:""",
        input_variables=["context", "question"]
    )

    model = get_llm()
    rag_chain = prompt | model | StrOutputParser()

    response = rag_chain.invoke({"context": docs, "question": question})
    return {"messages": [response]}

def rewrite(state:AgentState):
    """
    Transform the query to produce a better question.
    
    Args:
        state(messages): the current state

    Returns:
        dict: the updated state with re-phrased question

    """
    print("---TRANSFORM QUERY---")
    messages = state["messages"]
    question = messages[0].content

    message_content = f"""Look at the input and try to reason about the underlying semantic intent / meaning related to CMU Qatar Information Systems program.
                    Here is the initial question: {question}
                    Reformulate this as a clearer question about cmu courses, concentrations, curriculum, advising, or academic policies: """
    message = [HumanMessage(content=message_content)]

    model = get_llm()

    response = model.invoke(message)
    return {"messages": [response]}

def build_agent_graph() -> StateGraph:
    """Builds the LangGraph agent."""
    workflow=StateGraph(AgentState)
    #add nodes
    workflow.add_node("agent", agent)
    #retrieve = ToolNode([retriever_tool_major,retriever_tool_course])
    workflow.add_node("retrieve", retrieve_context_node)
    workflow.add_node("generate", generate)
    workflow.add_node("rewrite", rewrite)

    workflow.add_edge(START, "agent")
    workflow.add_edge("agent", "retrieve")
    workflow.add_conditional_edges("retrieve",grade_documents,)
    workflow.add_edge("generate", END)
    workflow.add_edge("rewrite","agent")

    #memory = MemorySaver()
    #graph = workflow.compile(checkpointer=memory)
    graph = workflow.compile()
    return graph

GRAPH = build_agent_graph()