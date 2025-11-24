"""Configuration loader for the Azure OpenAI microservice."""
from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

DEFAULT_API_VERSION = "2025-01-01-preview"
DEFAULT_INTERVAL_SECONDS = 60  # default periodic call interval

@dataclass(frozen=True)
class ServiceConfig:
    base_endpoint: str
    api_key: str
    deployment: str
    api_version: str = DEFAULT_API_VERSION
    interval_seconds: int = DEFAULT_INTERVAL_SECONDS
    request_timeout: int = 30

    @staticmethod
    def load() -> "ServiceConfig":
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
        api_key = os.getenv("AZURE_OPENAI_API_KEY", "").strip()
        deployment = os.getenv("DEPLOYMENT_NAME", "").strip()
        api_version = os.getenv("API_VERSION", DEFAULT_API_VERSION).strip()
        interval_raw = os.getenv("SERVICE_INTERVAL_SECONDS", str(DEFAULT_INTERVAL_SECONDS)).strip()
        timeout_raw = os.getenv("SERVICE_REQUEST_TIMEOUT", "30").strip()

        if not endpoint:
            raise ValueError("AZURE_OPENAI_ENDPOINT is not set.")
        if not api_key:
            raise ValueError("AZURE_OPENAI_API_KEY is not set.")
        if not deployment:
            raise ValueError("DEPLOYMENT_NAME is not set.")

        # sanitize endpoint: keep host only
        if "/openai" in endpoint:
            endpoint = endpoint.split("/openai")[0].rstrip('/')
        else:
            endpoint = endpoint.rstrip('/')

        try:
            interval_seconds = int(interval_raw)
        except ValueError:
            interval_seconds = DEFAULT_INTERVAL_SECONDS
        try:
            request_timeout = int(timeout_raw)
        except ValueError:
            request_timeout = 30

        return ServiceConfig(
            base_endpoint=endpoint,
            api_key=api_key,
            deployment=deployment,
            api_version=api_version,
            interval_seconds=interval_seconds,
            request_timeout=request_timeout,
        )

