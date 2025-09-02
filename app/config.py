from langchain_ollama import OllamaEmbeddings
from pydantic_settings import BaseSettings


class DatabaseSettings(BaseSettings):
    dbname: str = "simple_llm"
    user: str = ""
    password: str = ""
    host: str = "localhost"
    port: int = 5432

    class Config:
        env_file = ".env"  # Specify the .env file to load variables from
        env_file_encoding = "utf-8"


def get_embeddings():
    # from langchain_huggingface.embeddings import HuggingFaceEmbeddings
    # return HuggingFaceEmbeddings(
    #     model_name="sentence-transformers/all-MiniLM-L6-v2",
    #     model_kwargs={"device": "cpu"},
    # )
    return OllamaEmbeddings(model="nomic-embed-text")


embedding = get_embeddings()

db_settings = DatabaseSettings()
