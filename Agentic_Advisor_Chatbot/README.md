## 1. Requirements
- An OpenAI key 

## 2. Install venv
# from project root
python -m venv .chatenv

# activate venv
.chatenv\Scripts\activate

# install requirements.txt
pip install -r requirements.txt

# Ingest data into Chroma (if chromadb is not there)
python backend/dataingestion.py

# Run the FastAPI
python main.py
# serves at http://127.0.0.1:8000

# Run the frontend
streamlit run frontend/streamlit_app.py
