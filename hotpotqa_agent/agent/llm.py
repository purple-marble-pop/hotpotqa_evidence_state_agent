import json
import os
import warnings
from json import JSONDecodeError
from typing import Any, Dict, Optional
from dotenv import load_dotenv
load_dotenv()

def ensure_valid_ssl_cert_env() -> None:
    """Repair invalid certificate env vars before httpx/OpenAI initializes."""
    ssl_cert_file = os.getenv("SSL_CERT_FILE")
    if not ssl_cert_file or os.path.exists(ssl_cert_file):
        return

    try:
        import certifi
    except ImportError:
        os.environ.pop("SSL_CERT_FILE", None)
        warnings.warn(
            f"SSL_CERT_FILE points to a missing file: {ssl_cert_file}. "
            "The variable was unset so Python can use its default certificates.",
            RuntimeWarning,
        )
        return

    certifi_path = certifi.where()
    os.environ["SSL_CERT_FILE"] = certifi_path
    warnings.warn(
        f"SSL_CERT_FILE points to a missing file: {ssl_cert_file}. "
        f"Using certifi certificate bundle instead: {certifi_path}",
        RuntimeWarning,
    )

class LLMClient:
    """OpenAI-compatible LLM client. Works with SiliconFlow or other compatible APIs."""
    def __init__(self, base_url: Optional[str]=None, api_key: Optional[str]=None, model: Optional[str]=None, temperature: Optional[float]=None, max_tokens: Optional[int]=None):
        self.base_url = base_url or os.getenv("LLM_BASE_URL")
        self.api_key = api_key or os.getenv("LLM_API_KEY")
        self.model = model or os.getenv("LLM_MODEL")
        self.temperature = float(temperature if temperature is not None else os.getenv("LLM_TEMPERATURE"))
        self.max_tokens = int(max_tokens if max_tokens is not None else os.getenv("LLM_MAX_TOKENS"))
        self.enabled = bool(self.api_key and self.base_url)
        if self.enabled:
            ensure_valid_ssl_cert_env()
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        else:
            self.client = None

    def chat(self, system: str, user: str) -> str:
        if not self.enabled:
            raise RuntimeError("LLM is not configured. Please set LLM_API_KEY and LLM_BASE_URL.")
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content or ""

    def chat_json(self, system: str, user: str) -> Dict[str, Any]:
        raw = self.chat(system, user)
        try:
            return extract_json(raw)
        except (ValueError, JSONDecodeError):
            repair_system = "You repair malformed JSON. Return ONLY one valid JSON object. Do not add explanations."
            repair_user = (
                "Convert this malformed model output into valid JSON while preserving the fields and values.\n\n"
                f"{raw}"
            )
            return extract_json(self.chat(repair_system, repair_user))

def extract_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:].strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON object found in: {text}")
    return json.loads(text[start:end+1])
