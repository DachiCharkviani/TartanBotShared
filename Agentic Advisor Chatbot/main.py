from fastapi import FastAPI
from backend.route import router
from fastapi.middleware.cors import CORSMiddleware
from backend.settings import API_HOST, API_PORT, CORS_ORIGINS

app = FastAPI(title="Advisor ChatBot")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if CORS_ORIGINS == ["*"] else CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=API_HOST, port=API_PORT, reload=True)