import hashlib
import json
import os
import pathlib
import sys
import time
from datetime import datetime, timezone

MAX_LLM_ATTEMPTS = 6
TRANSIENT_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504}
TRANSIENT_RETRY_SECONDS = 60
REQUEST_TIMEOUT_SECONDS = 180
RESPONSES_INSTRUCTIONS = (
    "You are a deterministic research assistant. Follow the user task exactly and return only the requested content."
)


def cache_key(purpose: str, model: str, prompt: str, source_hash: str) -> str:
    """Combine purpose, model, prompt, and input hash into a stable cache key."""
    payload = json.dumps(
        {"purpose": purpose, "model": model, "prompt": prompt, "source_hash": source_hash},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def response_sse_text(text: str) -> str:
    """Parse Responses SSE text returned by a relay endpoint."""
    stripped = text.lstrip()
    if not (stripped.startswith("event:") or stripped.startswith("data:")):
        return text
    deltas: list[str] = []
    done_texts: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line.removeprefix("data:").strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            return text
        if data.get("type") == "response.output_text.done" and isinstance(data.get("text"), str):
            done_texts.append(data["text"])
        elif data.get("type") == "response.output_text.delta" and isinstance(data.get("delta"), str):
            deltas.append(data["delta"])
    if done_texts:
        return "\n".join(done_texts)
    if deltas:
        return "".join(deltas)
    return text


def response_text(response) -> str:
    """Handle OpenAI SDK objects, direct string responses, and Responses SSE text."""
    if isinstance(response, str):
        return response_sse_text(response)
    text = getattr(response, "output_text", None)
    if isinstance(text, str):
        return text
    raise TypeError(f"unsupported LLM response type: {type(response).__name__}")


def _default_openai_call(model: str, prompt: str) -> str:
    """Make one formal LLM call using the OpenAI configuration in the current environment."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required when no cached LLM response exists")
    from openai import OpenAI

    client_kwargs = {"api_key": api_key}
    if os.environ.get("OPENAI_BASE_URL"):
        client_kwargs["base_url"] = os.environ["OPENAI_BASE_URL"]
    client = OpenAI(timeout=REQUEST_TIMEOUT_SECONDS, **client_kwargs)
    response = client.responses.create(
        model=model,
        instructions=RESPONSES_INSTRUCTIONS,
        input=[{"role": "user", "content": prompt}],
    )
    return response_text(response)


def call_with_retries(call_openai, model: str, prompt: str) -> str:
    """Retry only transient network or server errors with backoff; raise all other errors."""
    last_response = ""
    for attempt in range(MAX_LLM_ATTEMPTS):
        try:
            last_response = str(call_openai(model, prompt) or "").strip()
            if last_response:
                return last_response
        except Exception as exc:
            status_code = getattr(exc, "status_code", None)
            class_name = exc.__class__.__name__
            transient = status_code in TRANSIENT_STATUS_CODES or class_name in {
                "APIConnectionError",
                "APITimeoutError",
            }
            if not transient or attempt == MAX_LLM_ATTEMPTS - 1:
                raise
            print(
                f"transient LLM error purpose retry={attempt + 1}/{MAX_LLM_ATTEMPTS} "
                f"status={status_code} class={class_name} sleep={TRANSIENT_RETRY_SECONDS}s",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(TRANSIENT_RETRY_SECONDS)
    return last_response


class CachedLLMClient:
    def __init__(self, cache_dir: pathlib.Path, call_openai=None):
        """Initialize the file-backed LLM client."""
        self.cache_dir = pathlib.Path(cache_dir)
        self.call_openai = call_openai or _default_openai_call

    def complete(self, purpose: str, model: str, prompt: str, source_hash: str) -> str:
        """Read the cache first, call the real LLM only on a miss, and write the result back."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        key = cache_key(purpose, model, prompt, source_hash)
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            response = str(data.get("response") or "")
            if purpose == "agent_harness":
                return response
            if not response:
                raise RuntimeError(f"cached LLM response is empty: {path}")
            return response
        response = call_with_retries(self.call_openai, model, prompt)
        if not response and purpose != "agent_harness":
            raise RuntimeError(f"empty LLM response for {purpose}")
        data = {
            "cached": False,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "key": key,
            "model": model,
            "prompt": prompt,
            "purpose": purpose,
            "response": response,
            "source_hash": source_hash,
            "instructions": RESPONSES_INSTRUCTIONS,
            "input_format": "responses_user_message",
        }
        path.write_text(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
        return response
