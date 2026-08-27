# An AgentAI that can provide real-time financial news, positive/negative developments, and price prediction of the selected stock

import autogen
from finrobot.utils import get_current_date, register_keys_from_json
from finrobot.agents.workflow import SingleAssistant
from contextlib import redirect_stdout, redirect_stderr
import os
import openai
import json
import re

def run_forecasting(company="NVDA"):
    llm_config = {
        "config_list": autogen.config_list_from_json(
            "D:\my-fin-project\OAI_CONFIG_LIST",
            filter_dict={"model": ["gpt-4-0125-preview"]},
        ),
        "timeout": 120,
        "temperature": 0,
    }

    # Register FINNHUB API keys
    register_keys_from_json("D:\my-fin-project\config_api_keys")

    assistant = SingleAssistant(
        "Market_Analyst",
        llm_config,
        # set to "ALWAYS" if you want to chat instead of simply receiving the prediction
        human_input_mode="NEVER",
    )
    result = assistant.chat(
        (
            f"Use all the tools provided to retrieve information available for {company} as of {get_current_date()}. "
            f"Analyze the positive developments and potential concerns of {company} with 2-4 most important factors respectively and keep them concise. "
            "Most factors should be inferred from company related news. "
            f"Then make a rough prediction (e.g. up/down by xxx%) of the {company} stock price movement for next week. "
            "Provide a summary analysis to support your prediction.\n"
            "---\n\n"
            "Please **return ONLY** a JSON object with the following keys:\n"
            "1. \"news\": an **array** of the raw news headlines or summaries (from your news-tool call).\n"
            "2. \"positive_developments\": an array of your 2–4 bullet points.\n"
            "3. \"potential_concerns\": an array of your 2–4 bullet points.\n"
            "4. \"prediction\": the single string like \"up by xxx%\".\n"
            "5. \"summary\": the paragraph of summary analysis.\n"
        )
    )
    return result

result = run_forecasting("NVDA")

def run_and_log_forecasting(company="NVDA", log_path="D:/my-fin-project/outputs/full_run.log"):
    with open(log_path, "w", encoding="utf-8") as f, redirect_stdout(f), redirect_stderr(f):
        result = run_forecasting(company)
        print(result)

# Example usage:
run_and_log_forecasting("NVDA")

def extract_json_with_openai(log_path, output_path):

    # 1. Read API Key from environment variable
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if openai.api_key is None:
        raise ValueError("Please set the OPENAI_API_KEY environment variable first.")

    # 2. Read log content
    with open(log_path, "r", encoding="utf-8") as f:
        log_content = f.read()

    # Helper to extract first JSON object/array by scanning braces/brackets (robust vs regex)
    def extract_first_json_from_text(text):
        start_idx = None
        start_char = None
        for i, ch in enumerate(text):
            if ch in ("{", "["):
                start_idx = i
                start_char = ch
                break
        if start_idx is None:
            return None
        matching = "}" if start_char == "{" else "]"
        count = 0
        in_string = False
        escape = False
        for i in range(start_idx, len(text)):
            ch = text[i]
            if ch == '"' and not escape:
                in_string = not in_string
            if in_string and ch == "\\" and not escape:
                escape = True
            else:
                escape = False
            if not in_string:
                if ch == start_char:
                    count += 1
                elif ch == matching:
                    count -= 1
                    if count == 0:
                        return text[start_idx : i + 1]
        return None

    # 3. Ask the model to extract the first JSON (best-effort) using the new Responses API; fallback to local extractor if needed
    model_reply = None
    try:
        resp = openai.responses.create(
            model="gpt-4o",
            input=[
                {"role": "system", "content": "You are an assistant that extracts JSON from text."},
                {
                    "role": "user",
                    "content": (
                        "Extract only the first valid JSON object or array from the following text. Return only the JSON, no extra explanation.\n\n"
                        "LOG:\n"
                        "-----\n"
                        f"{log_content}\n"
                        "-----"
                    ),
                },
            ],
            temperature=0,
            max_output_tokens=2000,
        )
        # Prefer the convenient aggregated text if available
        model_reply = getattr(resp, "output_text", None)
        if not model_reply:
            # Fallback: assemble text pieces from structured output
            parts = []
            for out in getattr(resp, "output", []) or []:
                for c in out.get("content", []) or []:
                    if isinstance(c, dict) and "text" in c:
                        parts.append(c["text"])
                    elif isinstance(c, str):
                        parts.append(c)
            if parts:
                model_reply = "\n".join(parts).strip()
    except Exception:
        model_reply = None

    # 4. Prefer model reply but validate and fall back to local extractor
    json_text = None
    if model_reply:
        json_text = extract_first_json_from_text(model_reply)
        # if model returned plain JSON without extra noise, this will capture it.
    if not json_text:
        # Try extracting directly from the log
        json_text = extract_first_json_from_text(log_content)

    if not json_text:
        raise ValueError("No JSON object/array found in model reply or log content.")

    # 5. Validate JSON
    try:
        obj = json.loads(json_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Extracted text is not valid JSON: {e}")

    # 6. Write validated JSON to output file (prettified)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

    print(f"Extracted JSON saved to {output_path}")


extract_json_with_openai(
    "D:/my-fin-project/outputs/full_run.log",
    "D:/my-fin-project/outputs/forcasting_result.txt",
)