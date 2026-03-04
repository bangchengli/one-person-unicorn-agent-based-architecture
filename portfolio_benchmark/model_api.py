import os
import requests

class ModelAPI:
    """Unified interface for multiple providers: Anthropic (Claude), xAI (Grok), DeepSeek, LLaMA (via OpenRouter/TogetherAI)."""

    def __init__(self, provider: str, model: str):
        self.provider = provider.lower()
        self.model = model
        self.api_key = self._load_api_key()

    def _load_api_key(self):
        env_map = {
            "anthropic": "ANTHROPIC_API_KEY",
            "xai": "XAI_API_KEY",
            "deepseek": "DEEPSEEK_API_KEY",
            "llama": "LLAMA_API_KEY"
        }
        if self.provider not in env_map:
            raise ValueError(f"Unknown provider: {self.provider}")
        key = os.getenv(env_map[self.provider])
        if not key:
            raise ValueError(f"API key for {self.provider} not found. Please set {env_map[self.provider]}")
        return key

    def chat(self, prompt: str, max_tokens: int = 1024):
        if self.provider == "anthropic":
            return self._call_anthropic(prompt, max_tokens)
        elif self.provider == "xai":
            return self._call_xai(prompt, max_tokens)
        elif self.provider == "deepseek":
            return self._call_deepseek(prompt, max_tokens)
        elif self.provider == "llama":
            return self._call_llama(prompt, max_tokens)
        else:
            raise ValueError("Unsupported provider")

    def _call_anthropic(self, prompt, max_tokens):
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": self.api_key,
            "content-type": "application/json",
            "anthropic-version": "2023-06-01"
        }
        data = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}]
        }
        resp = requests.post(url, headers=headers, json=data, timeout=60)
        resp.raise_for_status()
        return resp.json()

    def _call_xai(self, prompt, max_tokens):
        url = "https://api.x.ai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens
        }
        resp = requests.post(url, headers=headers, json=data, timeout=60)
        resp.raise_for_status()
        return resp.json()

    def _call_deepseek(self, prompt, max_tokens):
        url = "https://api.deepseek.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens
        }
        resp = requests.post(url, headers=headers, json=data, timeout=60)
        resp.raise_for_status()
        return resp.json()

    def _call_llama(self, prompt, max_tokens):
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": self.model,  # e.g., "meta-llama/llama-4-maverick"
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens
        }
        resp = requests.post(url, headers=headers, json=data, timeout=60)
        resp.raise_for_status()
        return resp.json()
