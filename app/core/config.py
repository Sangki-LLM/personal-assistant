from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Slack
    slack_bot_token: str = ""
    slack_signing_secret: str = ""
    slack_my_user_id: str = ""  # 허용된 사용자 ID (보안)

    # Gemini (기본 LLM — 할당량 초과 시 Ollama fallback)
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"

    # Ollama (fallback LLM)
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "gemma4:12b"
    ollama_embed_model: str = "nomic-embed-text"

    # Database
    database_url: str = "sqlite+aiosqlite:///./assistant.db"

    # ChromaDB
    chroma_host: str = "localhost"
    chroma_port: int = 8002

    # Google (파일 경로 우선, 없으면 JSON 문자열 직접 사용)
    google_credentials_path: str = "/app/credentials.json"
    google_credentials_json: str = ""
    google_calendar_id: str = "primary"  # 사용자 캘린더 ID (Gmail 주소 권장)
    expense_sheet_id: str = ""


settings = Settings()
