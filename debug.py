import json
import autogen
from copy import deepcopy

CONFIG_PATH = r"D:\my-fin-project\OAI_CONFIG_LIST"
MODEL_NAME = "gpt-5"

def _mask(s: str, keep=4):
    if not s or not isinstance(s, str):
        return s
    if len(s) <= keep * 2:
        return "*" * len(s)
    return s[:keep] + "*" * (len(s) - keep * 2) + s[-keep:]

def extract_json(text: str):
    if not text:
        return None
    stack = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "{":
            if stack == 0:
                start = i
            stack += 1
        elif ch == "}":
            if stack > 0:
                stack -= 1
                if stack == 0 and start is not None:
                    candidate = text[start : i + 1]
                    try:
                        return json.loads(candidate)
                    except Exception:
                        return None
    return None

def is_json_termination(msg: dict) -> bool:
    content = (msg.get("content") or "").strip()
    return extract_json(content) is not None

def show_model_entries(config_path: str, model: str):
    all_cfg = autogen.config_list_from_json(config_path)
    hits = []
    for c in all_cfg:
        if str(c.get("model", "")).strip() == model:
            hits.append(c)

    print("\n=== Matching config entries (masked) ===")
    if not hits:
        print(f"No exact match found for model='{model}' in config.")
        return [], all_cfg

    for i, c in enumerate(hits):
        api_type = c.get("api_type")
        base_url = c.get("base_url") or c.get("api_base") or c.get("endpoint")
        key = c.get("api_key") or c.get("key")
        print(f"[{i}] api_type={api_type} base_url={base_url} api_key={_mask(key)}")
    return hits, all_cfg

def build_config_list(config_path: str, model: str):
    # 1) 精确匹配
    cfg = autogen.config_list_from_json(config_path, filter_dict={"model": [model]})
    if cfg:
        return cfg

    # 2) 找不到就给出明确错误提示（避免“以为在测 gpt-5 实际没在测”）
    raise ValueError(
        f"Model config not found for: {model}\n"
        f"Please add an entry with model='{model}' to OAI_CONFIG_LIST."
    )

def validate_entry(entry: dict):
    api_type = entry.get("api_type")
    base_url = entry.get("base_url") or entry.get("api_base") or entry.get("endpoint")
    api_key = entry.get("api_key") or entry.get("key")

    problems = []
    if not api_key:
        problems.append("Missing api_key")
    if not api_type:
        problems.append("Missing api_type (should usually be 'openai' for OpenAI)")
    # base_url 对 OpenAI 官方可以为空（走默认），所以这里只做提示不强制
    return problems, api_type, base_url, api_key

def run_debug(model: str, config_path: str):
    cfg_list = build_config_list(config_path, model)
    # 只取第一条做测试（多条可自己扩展循环）
    entry = deepcopy(cfg_list[0])

    problems, api_type, base_url, api_key = validate_entry(entry)
    print("\n=== Selected entry (masked) ===")
    print({
        "model": entry.get("model"),
        "api_type": api_type,
        "base_url": base_url,
        "api_key": _mask(api_key),
    })
    if problems:
        print("!!! CONFIG PROBLEMS:", problems)
        print("Fix the config first, otherwise the call will almost certainly fail.\n")

    assistant_name = f"Test_{model.replace('/', '_').replace('-', '_')}"
    assistant = autogen.AssistantAgent(
        name=assistant_name,
        llm_config={"config_list": [entry], "timeout": 120, "temperature": 0.0},
        system_message="Return only valid JSON. No markdown. No extra text.",
    )

    user = autogen.UserProxyAgent(
        name="User",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=1,
        is_termination_msg=is_json_termination,
        code_execution_config=False,
    )

    prompt = f"""
Return ONLY one valid JSON object with double quotes, no trailing commas:
{{
  "ok": true,
  "model": "{model}",
  "ping": "hello",
  "answer": "If you can read this, the model call succeeded."
}}
"""

    print("\n=== Sending request... ===")
    try:
        res = user.initiate_chat(assistant, message=prompt)
    except Exception as e:
        print("\n==============================")
        print("MODEL:", model)
        print("CALL FAILED:", repr(e))
        return

    history = getattr(res, "chat_history", []) or []
    last = None
    for m in reversed(history):
        if m.get("name") == assistant_name and m.get("content"):
            last = str(m["content"])
            break

    parsed = extract_json(last or "")

    print("\n==============================")
    print("MODEL:", model)
    print("LAST ASSISTANT MESSAGE:\n", last)
    print("PARSED JSON:", parsed)
    print("chat_res.cost:", getattr(res, "cost", None))

def main():
    # 先展示 config 里是否真的有 gpt-5 这条，并且 key 是否为空
    show_model_entries(CONFIG_PATH, MODEL_NAME)
    # 再真正调用一次
    run_debug(MODEL_NAME, CONFIG_PATH)

if __name__ == "__main__":
    main()
