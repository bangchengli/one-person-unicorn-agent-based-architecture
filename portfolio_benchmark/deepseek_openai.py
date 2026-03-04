"""
DeepSeek via OpenAI SDK (base_url) — supports both deepseek-chat and deepseek-reasoner.

Usage:
    from portfolio_benchmark.deepseek_openai import DeepSeekClient

    ds = DeepSeekClient()  # reads DEEPSEEK_API_KEY from env
    content, reasoning, usage = ds.complete(
        model="deepseek-reasoner",
        messages=[{"role": "user", "content": "Hello"}],
        temperature=0.2,
        max_tokens=800
    )

Returned:
    content: str (assistant message content)
    reasoning: str (assistant message reasoning_content, may be empty for deepseek-chat)
    usage: dict  (token usage if available)
"""
import os
from typing import List, Dict, Tuple

from openai import OpenAI


class DeepSeekClient:
    def __init__(self, api_key: str | None = None, base_url: str = "https://api.deepseek.com"):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY", "")
        if not self.api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is not set")
        self.client = OpenAI(api_key=self.api_key, base_url=base_url)

    def complete(self,
                 model: str,
                 messages: List[Dict[str, str]],
                 temperature: float = 0.2,
                 max_tokens: int = 1024) -> Tuple[str, str, Dict]:
        """Call DeepSeek chat completions API using OpenAI SDK style."""
        resp = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        # Extract text and reasoning_content safely
        content = ""
        reasoning = ""
        try:
            ch0 = resp.choices[0]
            msg = ch0.message
            content = getattr(msg, "content", "") or (msg.get("content") if isinstance(msg, dict) else "")
            reasoning = getattr(msg, "reasoning_content", "") or (msg.get("reasoning_content") if isinstance(msg, dict) else "")
        except Exception:
            pass

        # Extract usage if present
        usage = {}
        try:
            u = getattr(resp, "usage", None)
            if isinstance(u, dict):
                usage = u
            elif u:
                # SDK may expose attributes
                usage = {
                    "prompt_tokens": getattr(u, "prompt_tokens", None),
                    "completion_tokens": getattr(u, "completion_tokens", None),
                    "total_tokens": getattr(u, "total_tokens", None),
                }
        except Exception:
            pass

        return str(content or ""), str(reasoning or ""), usage or {}
