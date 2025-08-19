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


db_settings = DatabaseSettings()
