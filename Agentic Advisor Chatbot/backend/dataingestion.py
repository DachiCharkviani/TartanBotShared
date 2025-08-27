from langchain_community.document_loaders import JSONLoader
from langchain_text_splitters import RecursiveJsonSplitter
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from backend.settings import OPENAI_API_KEY, OPENAI_EMBEDDING_MODEL, PERSIST_DIR, FILE_PATH_ADDMAJORS, FILE_PATH_MINORS, FILE_PATH_COURSES, FILE_PATH_MAJORS

import glob

import json
from pathlib import Path
from tqdm import tqdm
#sys.path.append(os.path.dirname(os.path.abspath('..')))

def get_embeddings():
    return OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL, api_key=OPENAI_API_KEY)

def load_json_files():
    major_files = ['Biological Sciences','Business Administration','Computer Science','GenEd', 'Information Systems']
    json_major_files = {}
    for f in major_files:
        json_major_files[f] = glob.glob(os.path.join(FILE_PATH_MAJORS+f+'/'+'*.json'))
        print(FILE_PATH_MAJORS+f+'/'+'*.json')
        
    json_Courses_files = glob.glob(os.path.join(FILE_PATH_COURSES,'*.json'))
    json_minor_files = glob.glob(os.path.join(FILE_PATH_MINORS,'*.json'))
    json_addmajor_files = glob.glob(os.path.join(FILE_PATH_ADDMAJORS,'*.json'))

    return json_major_files,json_Courses_files,json_minor_files,json_addmajor_files

def major_metadata_func(department, category):
    def metadata_func(record: dict, metadata: dict) -> dict:
        if "source" in metadata:
            metadata["source"] = Path(metadata["source"]).stem
        metadata["department"] = department  
        metadata["category"] = category
        return metadata
    return metadata_func

def metadata_func_Courses(record: dict, metadata: dict) -> dict:
    if "source" in metadata:
        metadata["source"] = Path(metadata["source"]).stem
    metadata["departmentid"] = Path(metadata["source"]).stem.split('-')[0]  
    metadata['category'] = 'Courses'
    return metadata

def metadata_func_Minors(record: dict, metadata: dict) -> dict:
    if "source" in metadata:
        metadata["source"] = Path(metadata["source"]).stem
    metadata["department"] = 'Minors' 
    metadata['category'] = 'Minors'
    return metadata

def metadata_func_addMajors(record: dict, metadata: dict) -> dict:
    if "source" in metadata:
        metadata["source"] = Path(metadata["source"]).stem
    metadata["department"] = 'AddMajors' 
    metadata['category'] = 'AddMajors'
    return metadata

def major_file_chunk(json_major_files):
    Major_docs = {}
    i =0
    programs = ['BS','BA','CS','GENED','IS']
    for file,v in json_major_files.items():
        li = []
        for val in v:
            loader = JSONLoader(file_path=val,
                                jq_schema=".[]",
                                text_content=False,
                                metadata_func=major_metadata_func(programs[i],programs[i]+" Major"))
            raw_docs = loader.load()
            li.extend(raw_docs)
        Major_docs[programs[i]] = li
        i = i + 1
    return Major_docs

def course_file_chunk(json_Courses_files):
    Courses_docs = []
    for file in json_Courses_files:
        loader = JSONLoader(file_path=file,
                                jq_schema="""
                                        def clean:
                                            
                                            if type == "object" then
                                                with_entries(select(.value != null
                                                    and (( ( (.value | type) != "object" and (.value | type) != "array") and .value != "" )
                                                    or ( ((.value | type) == "object" or (.value | type) == "array") and ((.value | clean | length) > 0) ))))
                                            elif type == "array" then
                                                map(clean) | map(select(. != null and . != "" and ((type != "object" and type != "array") or (length > 0))))
                                            else
                                                .
                                            end;
                                        clean
                                        """,
                                text_content=False,
                                metadata_func=metadata_func_Courses)
        raw_docs = loader.load()
        Courses_docs.extend(raw_docs)
    return Courses_docs

def minor_file_chunk(json_minor_files):
    Minor_docs = []

    for file in json_minor_files:
        loader = JSONLoader(file_path=file,
                            jq_schema=".[]",
                            text_content=False,
                            metadata_func=metadata_func_Minors)
        raw_docs = loader.load()
        Minor_docs.extend(raw_docs)
    return Minor_docs

def addmajor_file_chunk(json_addmajor_files):
    AddMajor_docs = []

    for file in json_addmajor_files:
        loader = JSONLoader(file_path=file,
                            jq_schema=".[]",
                            text_content=False,
                            metadata_func=metadata_func_addMajors)
        raw_docs = loader.load()
        AddMajor_docs.extend(raw_docs)
    return AddMajor_docs

def major_data_persist(Major_docs, jsonsplitter, embeddings):
    for k,docs in Major_docs.items():
        Major_json_chunks = []
        for doc in docs:
            if isinstance(doc.page_content, str):
                content = json.loads(doc.page_content)
            else:
                content = doc.page_content
                    
            chunks = jsonsplitter.split_text(content, convert_lists=True)
                
            for chunk in chunks:
                Major_json_chunks.append(Document(
                    page_content=chunk,
                    metadata=doc.metadata
                ))
        ISmajor_db=Chroma.from_documents(documents=Major_json_chunks,
                                        embedding=embeddings,
                                        persist_directory=PERSIST_DIR+k+"/MajorData")
        ISmajor_db.persist()

def minor_data_persist(Minor_docs,jsonsplitter):
    Minors_json_chunks = []
    for doc in Minor_docs:
        if isinstance(doc.page_content, str):
            content = json.loads(doc.page_content)
        else:
            content = doc.page_content
            
        chunks = jsonsplitter.split_text(content, convert_lists=True)
        
        for chunk in chunks:
            Minors_json_chunks.append(Document(
                page_content=chunk,
                metadata=doc.metadata
            ))
    return Minors_json_chunks

def addmajor_data_persist(AddMajor_docs, jsonsplitter):
    Addmajor_json_chunks = []
    for doc in AddMajor_docs:
        if isinstance(doc.page_content, str):
            content = json.loads(doc.page_content)
        else:
            content = doc.page_content
            
        chunks = jsonsplitter.split_text(content, convert_lists=True)
        
        for chunk in chunks:
            Addmajor_json_chunks.append(Document(
                page_content=chunk,
                metadata=doc.metadata
            ))
    return Addmajor_json_chunks

def store_documents_in_chrome(
    documents: list[Document],
    persist_dir: str = "",
    embedding_model=None,
    batch_size: int = 100,
):
    
    """
    Store a list of LangChain Document objects in ChromaDB with batching.

    Args:
        documents : List of documents to embed and store.
        persist_dir : Directory to persist ChromaDB.
        embedding_model: Embedding model .
        batch_size : Number of documents to process per batch.
    """

    embedding_model = get_embeddings()

    vectordb = Chroma(
        persist_directory=persist_dir,
        embedding_function=embedding_model
    )

    for i in tqdm(range(0, len(documents), batch_size), desc="Storing documents in Chroma"):
        batch = documents[i:i + batch_size]
        vectordb.add_documents(batch)

    # Persist to disk
    vectordb.persist()

    print(f"Stored {len(documents)} documents in '{persist_dir}'.")

    return vectordb

def courses_data_split(Courses_docs, jsonsplitter):
    Courses_json_chunks = []
    for doc in Courses_docs:
        if isinstance(doc.page_content, str):
            content = json.loads(doc.page_content)
        else:
            content = doc.page_content
            
        chunks = jsonsplitter.split_text(content, convert_lists=True)
        
        for chunk in chunks:
            Courses_json_chunks.append(Document(
                page_content=chunk,
                metadata=doc.metadata
            ))
    return Courses_json_chunks

def main():

    embeddings = get_embeddings()
    jsonsplitter = RecursiveJsonSplitter(max_chunk_size=1000)
    #load jsonfiles
    json_major_files,json_Courses_files,json_minor_files,json_addmajor_files = load_json_files()

    #json files into chunks and persist into Chroma
    Major_docs = major_file_chunk(json_major_files)
    major_data_persist(Major_docs,jsonsplitter,embeddings)

    Minor_docs = minor_file_chunk(json_minor_files)
    Minors_json_chunks = minor_data_persist(Minor_docs,jsonsplitter)
    Minors_db=Chroma.from_documents(documents=Minors_json_chunks,
                                        embedding=embeddings,
                                        persist_directory=PERSIST_DIR+"/MinorData")
    Minors_db.persist()

    AddMajor_docs = addmajor_file_chunk(json_addmajor_files)
    Addmajor_json_chunks = addmajor_data_persist(AddMajor_docs, jsonsplitter)
    addmajors_db=Chroma.from_documents(documents=Addmajor_json_chunks,
                                     embedding=embeddings,
                                     persist_directory=PERSIST_DIR+"/AddMajorData")
    addmajors_db.persist()

    Courses_docs = course_file_chunk(json_Courses_files)
    Courses_json_chunks = courses_data_split(Courses_docs, jsonsplitter)
    courses_db=store_documents_in_chrome(Courses_json_chunks,persist_dir=PERSIST_DIR+"Courses")

if __name__ == "__main__":
    main()  


