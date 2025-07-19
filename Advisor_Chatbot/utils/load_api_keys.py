from dotenv import load_dotenv
import os

load_dotenv(override=True)

def get_env_var(env_variable:str, required: bool=True):
    env_value = os.getenv(env_variable)
    if required and env_value is None:
        raise EnvironmentError(f"Required Environment variable {env_variable} not found.")
    return env_value

def load_api_tokens():
    return {
        "HF_TOKEN" : get_env_var("HF_TOKEN"),
        "FILE_PATH_MAJORS": get_env_var("FILE_PATH_MAJORS"),
        "PERSIST_DIR": get_env_var("PERSIST_DIR"),
        "OPENAI_API_KEY" : get_env_var("OPENAI_API_KEY"),
        "GROQ_API_KEY": get_env_var("GROQ_API_KEY"),
        "LANGCHAIN_API_KEY" : get_env_var("LANGCHAIN_API_KEY"),
        "LANGCHAIN_TRACING_V2" : get_env_var("LANGCHAIN_TRACING_V2"),
        "LANGCHAIN_PROJECT" : get_env_var("LANGCHAIN_PROJECT"),
        "LANGCHAIN_ENDPOINT" : get_env_var("LANGCHAIN_ENDPOINT"),
        "PERSIST_DIR_FAISS" : get_env_var("PERSIST_DIR_FAISS"),
        "FILE_PATH_MINORS": get_env_var("FILE_PATH_MINORS"),
        "FILE_PATH_ADDMAJORS": get_env_var("FILE_PATH_ADDMAJORS"),
        "FILE_PATH_COURSES": get_env_var("FILE_PATH_COURSES"),
    }

