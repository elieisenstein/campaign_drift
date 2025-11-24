"""Azure Chat client using raw REST calls (requests)."""
from __future__ import annotations
from typing import Optional, Tuple, Dict, Any, List
import requests
from config import ServiceConfig

class AzureChatClient:
    def __init__(self, cfg: ServiceConfig):
        self.cfg = cfg
        self.url = f"{cfg.base_endpoint}/openai/deployments/{cfg.deployment}/chat/completions?api-version={cfg.api_version}"

    def chat(self,
             prompt: str,
             system_message: str = "You are a helpful assistant.",
             max_tokens: int = 256,
             temperature: float = 0.7,
             top_p: float = 1.0) -> Tuple[Optional[str], Dict[str, Any]]:
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ]
        headers = {"api-key": self.cfg.api_key, "Content-Type": "application/json"}
        body = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        resp = requests.post(self.url, headers=headers, json=body, timeout=self.cfg.request_timeout)
        if not resp.ok:
            raise RuntimeError(f"Azure OpenAI request failed {resp.status_code}: {resp.text}")
        data = resp.json()
        assistant_text: Optional[str] = None
        try:
            choices = data.get("choices", [])
            if choices:
                first = choices[0]
                message = first.get("message") or first.get("content") or {}
                if isinstance(message, dict):
                    assistant_text = message.get("content")
                elif isinstance(message, str):
                    assistant_text = message
        except Exception:
            pass
        return assistant_text, data

