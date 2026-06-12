from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Slack
    slack_bot_token: str = ""
    slack_signing_secret: str = ""
    slack_my_user_id: str = ""  # 허용된 사용자 ID (보안)

    # Ollama
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "gemma4:12b"
    ollama_embed_model: str = "nomic-embed-text"

    # Database
    database_url: str = "sqlite+aiosqlite:///./assistant.db"

    # ChromaDB
    chroma_host: str = "localhost"
    chroma_port: int = 8002

    # Google
    google_credentials_json: str = ""
    expense_sheet_id: str = ""


settings = Settings()
