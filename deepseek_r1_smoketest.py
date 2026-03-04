
import os
import sys
import json
import re
import time

from openai import OpenAI

MODEL_ID = "deepseek-reasoner"  # R1
BASE_URL = "https://api.deepseek.com"

PROMPT = (
    "You are an expert portfolio construction advisor.\n"
    "Task: Generate a diversified US equities portfolio with 20 tickers ONLY.\n"
    "Output: A pure JSON array. Each element is an object with the key \"name\" containing the ticker symbol.\n"
    "No explanations. Example: [{\"name\": \"AAPL\"}, {\"name\": \"MSFT\"}]"
)

STOPWORDS = {"AND","THE","FOR","WITH","FROM","THIS","THAT","YOU","YOUR","A","AN","IN","ON","BY","TO","AS","AT","OF"}

def extract_text_and_reasoning(resp) -> tuple[str, str]:
    """Return (content, reasoning_content) strings if present."""
    try:
        choice0 = resp.choices[0]
        msg = choice0.message
        content = getattr(msg, "content", None) or ""
        reasoning = getattr(msg, "reasoning_content", None) or ""
        # Some SDKs return dict-like objects
        if not content and isinstance(msg, dict):
            content = msg.get("content", "")
        if not reasoning and isinstance(msg, dict):
            reasoning = msg.get("reasoning_content", "")
        return str(content or ""), str(reasoning or "")
    except Exception:
        return ("", "")

def robust_json_parse(raw: str):
    raw = (raw or "").strip()
    if not raw:
        return []
    # Try straight JSON
    try:
        obj = json.loads(raw)
        if isinstance(obj, list):
            return [x for x in obj if isinstance(x, dict)]
        if isinstance(obj, dict):
            return [obj]
    except Exception:
        pass
    # Line by line
    items = []
    for line in raw.splitlines():
        s = line.strip().rstrip(",")
        if not s or s.startswith("//") or s.startswith("#"):
            continue
        try:
            items.append(json.loads(s))
        except Exception:
            continue
    if items:
        return items
    # Brace scan
    buf, level, objs = [], 0, []
    for ch in raw:
        if ch == "{":
            level += 1
        if level > 0:
            buf.append(ch)
        if ch == "}":
            level -= 1
            if level == 0 and buf:
                try:
                    objs.append(json.loads("".join(buf)))
                except Exception:
                    pass
                buf = []
    return objs

def extract_tickers_fallback(text: str, n_max: int = 25):
    cands = re.findall(r"\b[A-Z]{1,5}\b", text or "")
    out, seen = [], set()
    for t in cands:
        if t in STOPWORDS: 
            continue
        if not t.isalpha():
            continue
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
        if len(out) >= n_max:
            break
    return out

def main():
    api_key = os.getenv("DEEPSEEK_API_KEY", "")
    if not api_key:
        print("Please set DEEPSEEK_API_KEY environment variable.")
        sys.exit(1)

    client = OpenAI(api_key=api_key, base_url=BASE_URL)

    messages = [{"role": "user", "content": PROMPT}]

    t0 = time.perf_counter()
    resp = client.chat.completions.create(model=MODEL_ID, messages=messages, temperature=0.2, max_tokens=1000)
    latency1 = time.perf_counter() - t0

    content, reasoning = extract_text_and_reasoning(resp)
    print("=== Round 1 ===")
    print("Reasoning (truncated):", (reasoning[:400] + "...") if len(reasoning) > 400 else reasoning)
    print("Content   (truncated):", (content[:400] + "...") if len(content) > 400 else content)
    print(f"Latency: {latency1:.2f}s\n")

    # Parse JSON content, fallback to reasoning if needed
    parsed = robust_json_parse(content) or robust_json_parse(reasoning)
    tickers = [x.get("name","").strip().upper() for x in parsed if isinstance(x, dict) and x.get("name")]
    tickers = [t for t in tickers if t]

    if not tickers:
        # Last resort: extract from content+reasoning
        tickers = extract_tickers_fallback(content + "\n" + reasoning, n_max=25)

    print(f"Tickers found ({len(tickers)}): {', '.join(tickers)}")
    if not tickers:
        print("No tickers parsed. The model did not follow the JSON-only instruction.")
        sys.exit(3)

    # Optional: demonstrate second round continuity using answer as context
    messages.append({'role': 'assistant', 'content': content})
    messages.append({'role': 'user', 'content': "A quick check: return the number of tickers you produced as an integer only."})
    t1 = time.perf_counter()
    resp2 = client.chat.completions.create(model=MODEL_ID, messages=messages, temperature=0.2, max_tokens=50)
    latency2 = time.perf_counter() - t1
    content2, _ = extract_text_and_reasoning(resp2)
    print("\n=== Round 2 (Count confirmation) ===")
    print("Content:", content2.strip())
    print(f"Latency: {latency2:.2f}s")

    # Final machine-friendly summary
    summary = {
        "model": MODEL_ID,
        "latency_round1_sec": round(latency1, 3),
        "latency_round2_sec": round(latency2, 3),
        "n_tickers": len(tickers),
        "tickers": tickers
    }
    print("\n=== JSON summary ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
