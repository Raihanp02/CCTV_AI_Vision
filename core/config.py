from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    CAMERA_MONITORING_SOURCES: list[str | int] = []
    CAMERA_MONITORING_FRAME_WIDTH: int = 1280
    CAMERA_MONITORING_FRAME_HEIGHT: int = 720

    class Config:
        env_file = ".env"

settings = Settings()