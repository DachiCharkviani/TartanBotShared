import os
import json
from pathlib import Path
from typing import List, Dict
from pydantic import BaseModel, Field
from crewai.tools import BaseTool
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Determine project-root-relative paths
BASE_DIR = Path(__file__).parent  # src/tartanadvisor
CONFIG_DIR = BASE_DIR / "config" / "data"
INDEX_DIR = BASE_DIR.parent / "faiss_indexes"
INDEX_DIR.mkdir(exist_ok=True)

# Configuration paths
CONFIG = {
    "advising": {
        "ba": CONFIG_DIR / "advising" / "biz",
        "is": CONFIG_DIR / "advising" / "is",
        "cs": CONFIG_DIR / "advising" / "cs",
        "bio": CONFIG_DIR / "advising" / "bio",
    },
    "courses": CONFIG_DIR / "courses",
}

# Embedding model initialization
embeddings = OpenAIEmbeddings()


def _load_documents_from_path(path: Path) -> List[Document]:
    docs: List[Document] = []
    if not path.exists():
        raise FileNotFoundError(f"Config path not found: {path}")
    for full in path.rglob("*.*"):
        ext = full.suffix.lower()
        if ext == ".txt":
            loader = TextLoader(str(full), encoding="utf-8")
            docs.extend(loader.load())
        elif ext == ".json":
            with full.open(encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                for record in data:
                    docs.append(Document(page_content=json.dumps(record), metadata={"source": str(full)}))
            else:
                docs.append(Document(page_content=json.dumps(data), metadata={"source": str(full)}))
    return docs


# def build_faiss_index(name: str, docs: List[Document]) -> FAISS:
#     index_path = INDEX_DIR / f"{name}.faiss"
#     if (index_path.with_suffix('.index')).exists():
#         return FAISS.load_local(str(index_path), embeddings)
#     if not docs:
#         raise ValueError(f"No documents found for index '{name}'. Check your config path.")
    
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=1000,
#         chunk_overlap=200
#     )
#     docs = text_splitter.split_documents(docs)

#     vs = FAISS.from_documents(docs, embeddings)
#     vs.save_local(str(index_path))
#     return vs

def build_faiss_index(name: str, docs: List[Document]) -> FAISS:
    index_folder = INDEX_DIR / f"{name}.faiss"
    if index_folder.exists():
        return FAISS.load_local(
            str(index_folder),
            embeddings,
            allow_dangerous_deserialization=True
        )

    if not docs:
        raise ValueError(f"No documents found for index '{name}'.")
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(docs)
    vs = FAISS.from_documents(docs, embeddings)
    vs.save_local(str(index_folder))
    return vs

def setup_all_indexes() -> Dict[str, FAISS]:
    stores: Dict[str, FAISS] = {}
    # Advising domains
    for domain, p in CONFIG['advising'].items():
        docs = _load_documents_from_path(p)
        stores[f"advising_{domain}"] = build_faiss_index(f"advising_{domain}", docs)

    # Courses
    course_docs = _load_documents_from_path(CONFIG['courses'])
    stores['courses'] = build_faiss_index('courses', course_docs)
    return stores

# Initialize stores
STORES = setup_all_indexes()

# Low-level search functions

def search_advising(domain: str, query: str, k: int = 5) -> List[Document]:
    key = f"advising_{domain}"
    if key not in STORES:
        raise ValueError(f"Unknown domain: {domain}")
    return STORES[key].similarity_search(query, k=k)


def search_courses(query: str, k: int = 5) -> List[Document]:
    return STORES['courses'].similarity_search(query, k=k)
