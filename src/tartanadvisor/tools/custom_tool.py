from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import List, Optional
import numpy as np
from typing import Dict, Any, Callable, List, Optional, Type, ClassVar
from pydantic import PrivateAttr, ConfigDict
from langchain.schema import Document
from tartanadvisor.faiss_store import search_advising, search_courses

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









# from tartanadvisor.utils.faiss_builder import build_or_load_faiss


# class VectorSearchInput(BaseModel):
#     store_name: str
#     query: str
#     k: Optional[int] = 5

# class VectorSearchTool(BaseTool):
#     # suppress Pydantic decorator checks
#     model_config = ConfigDict(check_decorators=False)

#     name: ClassVar[str]        = "vector_search"
#     description: str = (
#         "Perform semantic search over a named FAISS store. "
#         "Args: store_name, query, k."
#     )
#     args_schema: Type[BaseModel] = VectorSearchInput

#     # private attributes
#     _retrievers: Dict[str, Any] = PrivateAttr(default_factory=dict)

#     def __init__(
#         self,
#         stores_config: Dict[str, Dict[str, Any]],
#     ):
#         super().__init__()
#         # for each named store, build or load:
#         for name, params in stores_config.items():
#             self._retrievers[name] = build_or_load_faiss(
#                 data_paths=params["data_paths"],
#                 persist_path=params.get("persist_path"),
#                 re_init=params.get("re_init", False),
#                 chunk_size=params.get("chunk_size", 1000),
#                 chunk_overlap=params.get("chunk_overlap", 100),
#             ).as_retriever()

#     def _run(self, store_name: str, query: str, k: int = 5) -> List[dict]:
#         retriever = self._retrievers.get(store_name)
#         if retriever is None:
#             return [{"error": f"Unknown store '{store_name}'"}]
#         docs: List[Document] = retriever.get_relevant_documents(query)
#         return [
#             {"text": d.page_content, "metadata": d.metadata}
#             for d in docs[:k]
#         ]

#     async def _arun(self, *args, **kwargs):
#         return self._run(*args, **kwargs)








# # class VectorSearchInput(BaseModel):
# #     store_name: str
# #     query: str = Field(..., description="The query string to search the vector database.")
# #     k: Optional[int] = 5

# # class VectorSearchTool(BaseTool):
# #     # ← disable decorator validation here
# #     model_config = ConfigDict(check_decorators=False)

# #     name: ClassVar[str]        = "single_vector_search"
# #     description: str              = (
# #         "Use this to perform semantic search over a named FAISS store. "
# #         "Arguments: store_name (str), query (str), k (int, default=5)."
# #     )
# #     args_schema: Type[BaseModel] = VectorSearchInput

# #     # declare your private attributes so Pydantic ignores them
# #     _db: Any          = PrivateAttr()
# #     _retriever: Any   = PrivateAttr()

# #     def __init__(
# #         self,
# #         data_paths: List[str],
# #         chunk_size: int = 1000,
# #         chunk_overlap: int = 100,
# #         re_init: bool = False,
# #         persist_path: Optional[str] = None,
# #     ):
# #         super().__init__()
# #         db = build_or_load_faiss(
# #             data_paths=data_paths,
# #             persist_path=persist_path,
# #             re_init=re_init,
# #             chunk_size=chunk_size,
# #             chunk_overlap=chunk_overlap,
# #         )
# #         self._db = db
# #         self._retriever = db.as_retriever()

# #     def _run(self, store_name: str, query: str, k: int = 5) -> List[dict]:
# #         retriever = self._retriever if store_name == self.name else None
# #         if retriever is None:
# #             return [{"error": f"Unknown store '{store_name}'"}]
# #         docs = retriever.get_relevant_documents(query)
# #         return [{"text": d.page_content, "metadata": d.metadata} for d in docs[:k]]

# #     async def _arun(self, *args, **kwargs):
# #         return self._run(*args, **kwargs)






# # class VectorSearchInput(BaseModel):
# #     store_name: str
# #     query: str
# #     k: Optional[int]

# # class VectorSearchTool(BaseTool):
# #     name: str                     = "vector_search"
# #     description: str              = (
# #         "Use this to perform semantic search over a named FAISS store. "
# #         "Arguments: store_name (str), query (str), k (int, default=5)."
# #     )
# #     args_schema: Type[BaseModel]  = VectorSearchInput

# #     _faiss_stores: Dict[str, Any] = PrivateAttr()
# #     _embed_fn: Callable           = PrivateAttr()

# #     def __init__(self, faiss_stores: dict, embed_fn):
# #         super().__init__()
# #         self._faiss_stores = faiss_stores
# #         self._embed_fn = embed_fn

# #     def _run(self, store_name: str, query: str, k: int = 5) -> List[dict]:
# #         # 1) Embed
# #         q_vec = self._embed_fn([query])  # np.ndarray (1×dim)
# #         # 2) Lookup store
# #         store = self._faiss_stores.get(store_name)
# #         if store is None:
# #             return [{"error": f"Unknown store '{store_name}'"}]
# #         # 3) Search
# #         distances, indices = store.query(q_vec, k)
# #         results = []
# #         for dist, idx in zip(distances[0], indices[0]):
# #             doc = store.get_document_by_id(int(idx))
# #             results.append({
# #                 "id": idx,
# #                 "score": float(dist),
# #                 "text": doc.get("text", "")
# #             })
# #         return results

# #     async def _arun(self, *args, **kwargs):
# #         # If your embedding or FAISS queries are I/O‐bound you could asyncify here.
# #         return self._run(*args, **kwargs)
    












# # from typing import List, Optional, Type
# # from pydantic import BaseModel, PrivateAttr
# # from crewei.tools import BaseTool
# # from langchain.embeddings import OpenAIEmbeddings
# # from langchain.vectorstores import FAISS
# # from langchain.schema import Document

# # class VectorSearchInput(BaseModel):
# #     store_name: str
# #     query: str
# #     k: Optional[int] = 5

# # class VectorSearchTool(BaseTool):
# #     name        = "vector_search"
# #     description = (
# #         "Retrieve text chunks from a named FAISS store. "
# #         "Args: store_name (str), query (str), k (int)."
# #     )
# #     args_schema = VectorSearchInput

# #     _retrievers: dict = PrivateAttr()
# #     _embeddings = PrivateAttr()

# #     def __init__(self, persist_paths: dict):
# #         """
# #         persist_paths: { store_name: path_to_faiss_index_dir }
# #         """
# #         super().__init__()
# #         self._embeddings = OpenAIEmbeddings()
# #         self._retrievers = {}
# #         for name, path in persist_paths.items():
# #             # load the local FAISS index + its documents
# #             self._retrievers[name] = FAISS.load_local(
# #                 path,
# #                 self._embeddings,
# #                 # needed if you saved via `from_documents`
# #                 allow_dangerous_deserialization=True,
# #             ).as_retriever(search_type="similarity", search_kwargs={"k": 5})

# #     def _run(self, store_name: str, query: str, k: int = 5) -> List[dict]:
# #         retriever = self._retrievers.get(store_name)
# #         if not retriever:
# #             return [{"error": f"Unknown store '{store_name}'"}]

# #         docs = retriever.get_relevant_documents(query)
# #         results = []
# #         for d in docs[:k]:
# #             results.append({
# #                 "id":      getattr(d, "metadata", {}).get("source", ""),
# #                 "score":   None,  # you can’t extract FAISS score here easily
# #                 "text":    d.page_content,
# #             })
# #         return results

# #     async def _arun(self, *args, **kwargs):
# #         return self._run(*args, **kwargs)
# # 