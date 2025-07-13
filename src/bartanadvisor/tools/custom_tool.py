from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import List, Optional
import numpy as np
from typing import Dict, Any, Callable, List, Optional, Type, ClassVar
from pydantic import PrivateAttr, ConfigDict
from langchain.schema import Document
from bartanadvisor.faiss_store import search_advising, search_courses

class MyCustomToolInput(BaseModel):
    """Input schema for MyCustomTool."""

    argument: str = Field(..., description="Description of the argument.")

class MyCustomTool(BaseTool):
    name: str = "Name of my tool"
    description: str = (
        "Clear description for what this tool is useful for, your agent will need this information to use it."
    )
    args_schema: Type[BaseModel] = MyCustomToolInput

    def _run(self, argument: str) -> str:
        # Implementation goes here
        return "this is an example of a tool output, ignore it and move along."
    
# --- CrewAI Custom Tools ---

class AdvisingInput(BaseModel):
    question: str = Field(..., description="Keywords to search advising documents.")

# BA Advising Tool
tools: List[BaseTool] = []

class SearchAdvisingBATool(BaseTool):
    name: str = "search_advising_ba"
    description: str = "Search Business Administration advising documents."
    args_schema: Type[BaseModel] = AdvisingInput

    def _run(self, question: str) -> str:
        docs = search_advising("ba", question)
        return "\n\n".join(d.page_content for d in docs)

class SearchAdvisingISTool(BaseTool):
    name: str = "search_advising_is"
    description: str = "Search Information Systems advising documents."
    args_schema: Type[BaseModel] = AdvisingInput

    def _run(self, question: str) -> str:
        docs = search_advising("is", question)
        return "\n\n".join(d.page_content for d in docs)

class SearchAdvisingCSTool(BaseTool):
    name: str = "search_advising_cs"
    description: str = "Search Computer Science advising documents."
    args_schema: Type[BaseModel] = AdvisingInput

    def _run(self, question: str) -> str:
        docs = search_advising("cs", question)
        return "\n\n".join(d.page_content for d in docs)

class SearchAdvisingBioTool(BaseTool):
    name: str = "search_advising_bio"
    description: str = "Search Biological Sciences advising documents."
    args_schema: Type[BaseModel] = AdvisingInput

    def _run(self, question: str) -> str:
        docs = search_advising("bio", question)
        return "\n\n".join(d.page_content for d in docs)

# Courses Tool
class SearchCoursesInput(BaseModel):
    query: str = Field(..., description="Keywords to search the courses index.")

class SearchCoursesTool(BaseTool):
    name: str = "search_courses"
    description: str = "Search the course catalog."
    args_schema: Type[BaseModel] = SearchCoursesInput

    def _run(self, query: str) -> str:
        docs = search_courses(query)
        return "\n\n".join(d.page_content for d in docs)

# Combined Search Tool
class CombinedSearchInput(BaseModel):
    domains: List[str] = Field(
        ..., description="Domains to search: 'courses', 'ba', 'is', 'cs', 'bio'."
    )
    query: str = Field(..., description="Keywords to search across selected domains.")
    k: int = Field(5, description="Results per domain.")

class CombinedSearchTool(BaseTool):
    name: str = "search_kb"
    description: str = "Search across multiple knowledge bases (courses + advising)."
    args_schema: Type[BaseModel] = CombinedSearchInput

    def _run(self, domains: List[str], query: str, k: int = 5) -> str:
        results = []
        for dom in domains:
            if dom == "courses":
                docs = search_courses(query, k)
            else:
                docs = search_advising(dom, query, k)
            header = f"--- Results from {dom} ---"
            body = "\n\n".join(d.page_content for d in docs)
            results.append(f"{header}\n{body}")
        return "\n\n".join(results)

# Register tools
tools = [
    SearchAdvisingBATool(),
    SearchAdvisingISTool(),
    SearchAdvisingCSTool(),
    SearchAdvisingBioTool(),
    SearchCoursesTool(),
    CombinedSearchTool(),
]