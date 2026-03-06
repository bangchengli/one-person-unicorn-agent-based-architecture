import os
import json
import time
import math
import numpy as np
import pandas as pd
import requests
import autogen
import traceback
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List

PRICING_REGISTRY = {
    "gpt-4o": {"input": 2.50, "cached_input": 1.25, "output": 10.00},
    "gpt-4o-2024-05-13": {"input": 5.00, "cached_input": None, "output": 15.00},
    "gpt-4-turbo": {"input": 10.00, "cached_input": None, "output": 30.00},
    "gpt-4": {"input": 30.00, "cached_input": None, "output": 60.00},
    "gpt-3.5-turbo": {"input": 0.50, "cached_input": None, "output": 1.50},
    "gpt-5.2": {"input": 1.75, "cached_input": 0.175, "output": 14.00},
    "claude-3-7-sonnet-20250219": {"input": 3.00, "cached_input": None, "output": 15.00},
}


def fetch_ohlcv_polygon(ticker, start, end, api_key=None, multiplier=1, timespan="day", adjusted=True):
    api_key = api_key or os.getenv("POLYGON_API_KEY")
    if not api_key:
        raise ValueError("Missing POLYGON_API_KEY environment variable.")

    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker.upper()}/range/{multiplier}/{timespan}/{start}/{end}"
    params = {
        "adjusted": "true" if adjusted else "false",
        "sort": "asc",
        "limit": 50000,
        "apiKey": api_key,
    }

    try:
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        raise ConnectionError(f"Polygon API request failed: {str(e)}")

    results = data.get("results", []) or []
    if not results:
        raise ValueError(f"Polygon API returned no data for {ticker} between {start} and {end}.")

    df = pd.DataFrame(results)
    df["date"] = pd.to_datetime(df["t"], unit="ms", utc=True).dt.tz_convert(None)
    df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
    df = df[["date", "open", "high", "low", "close", "volume"]].set_index("date").sort_index()
    return df


def get_real_market_news(ticker: str, target_date: str = None) -> str:
    api_key = os.getenv("POLYGON_API_KEY")
    if not api_key:
        return json.dumps({"error": "Missing POLYGON_API_KEY"})

    url = "https://api.polygon.io/v2/reference/news"
    params = {
        "ticker": ticker.upper(),
        "limit": 5,
        "sort": "published_utc",
        "order": "desc",
        "apiKey": api_key
    }

    if target_date:
        params["published_utc.lte"] = target_date

    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        return json.dumps({"error": f"Failed to fetch news: {str(e)}"})

    results = data.get("results", [])
    if not results:
        return json.dumps({"message": "No recent news found."})

    cleaned_news = []
    for item in results:
        cleaned_news.append({
            "datetime": item.get("published_utc"),
            "title": item.get("title"),
            "summary": item.get("description"),
            "source": item.get("author"),
            "url": item.get("article_url"),
            "keywords": (item.get("keywords", []) or [])[:3]
        })

    return json.dumps(cleaned_news)


def _rsi(series, period=14):
    if len(series) <= period:
        return pd.Series([float("nan")] * len(series), index=series.index)

    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_market_snapshot(prices):
    if prices is None or prices.empty or "close" not in prices.columns:
        raise ValueError("Cannot compute snapshot: prices DataFrame is empty or invalid.")

    close = prices["close"].astype(float)
    high = prices["high"].astype(float)
    low = prices["low"].astype(float)
    volume = prices["volume"].astype(float)
    rets = close.pct_change().dropna()

    def _ann_vol(x):
        return float(x.std(ddof=1) * math.sqrt(252)) if len(x) > 1 else None

    def _max_drawdown(equity):
        peak = equity.cummax()
        dd = (equity / peak) - 1.0
        return float(dd.min()) if len(dd) else None

    equity = (1.0 + rets).cumprod()
    last_close = float(close.iloc[-1])

    r_5d = float((1.0 + rets.tail(5)).prod() - 1.0) if len(rets) >= 5 else None
    r_20d = float((1.0 + rets.tail(20)).prod() - 1.0) if len(rets) >= 20 else None
    vol_ann = _ann_vol(rets)
    mdd = _max_drawdown(equity)

    rsi14 = float(_rsi(close, 14).iloc[-1]) if len(close) >= 15 else None
    ma20 = float(close.rolling(20).mean().iloc[-1]) if len(close) >= 20 else None
    ma50 = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else None

    high_20d = float(high.rolling(20).max().iloc[-1]) if len(high) >= 20 else None
    low_20d = float(low.rolling(20).min().iloc[-1]) if len(low) >= 20 else None

    if len(prices) >= 15:
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr14 = float(tr.rolling(14).mean().iloc[-1])
    else:
        atr14 = None

    if len(prices) >= 20:
        typical_price = (high + low + close) / 3
        tp_v = typical_price * volume
        vwap20 = float(tp_v.rolling(20).sum().iloc[-1] / volume.rolling(20).sum().iloc[-1])
    else:
        vwap20 = None

    trend = "unknown"
    if ma20 is not None and ma50 is not None:
        trend = "bullish" if ma20 > ma50 else "bearish"

    return {
        "available": True,
        "last_close": last_close,
        "high_20d": high_20d,
        "low_20d": low_20d,
        "atr14": atr14,
        "vwap20": vwap20,
        "return_5d": r_5d,
        "return_20d": r_20d,
        "ann_vol": vol_ann,
        "max_drawdown": mdd,
        "rsi14": rsi14,
        "ma20": ma20,
        "ma50": ma50,
        "trend_signal": trend,
        "n_obs": int(len(prices)),
        "start_date": str(prices.index.min().date()),
        "end_date": str(prices.index.max().date()),
    }


def extract_all_json_objects(text):
    results = []
    stack = []
    start_idx = -1
    for i, char in enumerate(text):
        if char == '{':
            if not stack:
                start_idx = i
            stack.append(char)
        elif char == '}':
            if stack:
                stack.pop()
                if not stack:
                    results.append(text[start_idx:i + 1])
    return results


def _sum_usage_bucket(bucket: dict, model_hint: Optional[str] = None) -> Dict[str, Any]:
    if not isinstance(bucket, dict) or not bucket:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_cost": None}

    total_cost = bucket.get("total_cost", None)
    model_keys = [k for k in bucket.keys() if k != "total_cost"]

    if model_hint:
        if model_hint in bucket and isinstance(bucket[model_hint], dict):
            d = bucket[model_hint] or {}
            return {
                "prompt_tokens": int(d.get("prompt_tokens", 0) or 0),
                "completion_tokens": int(d.get("completion_tokens", 0) or 0),
                "total_cost": float(d.get("cost", total_cost)) if (d.get("cost", total_cost) is not None) else None,
            }
        for k in model_keys:
            if (model_hint in k) or (k in model_hint):
                d = bucket.get(k, {}) or {}
                return {
                    "prompt_tokens": int(d.get("prompt_tokens", 0) or 0),
                    "completion_tokens": int(d.get("completion_tokens", 0) or 0),
                    "total_cost": float(d.get("cost", total_cost)) if (d.get("cost", total_cost) is not None) else None,
                }

    prompt = 0
    completion = 0
    model_cost_sum = 0.0
    cost_count = 0

    for k in model_keys:
        d = bucket.get(k, {}) or {}
        prompt += int(d.get("prompt_tokens", 0) or 0)
        completion += int(d.get("completion_tokens", 0) or 0)
        if d.get("cost") is not None:
            model_cost_sum += float(d["cost"])
            cost_count += 1

    resolved_cost = model_cost_sum if cost_count > 0 else total_cost
    resolved_cost = float(resolved_cost) if resolved_cost is not None else None

    return {"prompt_tokens": prompt, "completion_tokens": completion, "total_cost": resolved_cost}


def extract_autogen_usage(chat_res, model_hint: Optional[str] = None) -> Dict[str, Any]:
    cost = getattr(chat_res, "cost", None)
    if not isinstance(cost, dict) or not cost:
        return {
            "billed_prompt_tokens": 0,
            "billed_completion_tokens": 0,
            "cached_inference_prompt_tokens": 0,
            "cached_inference_completion_tokens": 0,
            "autogen_reported_cost": None,
        }

    incl = _sum_usage_bucket(cost.get("usage_including_cached_inference", {}) or {}, model_hint=model_hint)
    excl = _sum_usage_bucket(cost.get("usage_excluding_cached_inference", {}) or {}, model_hint=model_hint)

    cached_prompt = max(0, int(incl["prompt_tokens"]) - int(excl["prompt_tokens"]))
    cached_completion = max(0, int(incl["completion_tokens"]) - int(excl["completion_tokens"]))

    return {
        "billed_prompt_tokens": int(excl["prompt_tokens"]),
        "billed_completion_tokens": int(excl["completion_tokens"]),
        "cached_inference_prompt_tokens": int(cached_prompt),
        "cached_inference_completion_tokens": int(cached_completion),
        "autogen_reported_cost": excl["total_cost"],
    }


def _resolve_pricing(model_name: str) -> Optional[Dict[str, float]]:
    if model_name in PRICING_REGISTRY:
        return PRICING_REGISTRY[model_name]
    for key, val in PRICING_REGISTRY.items():
        if key in model_name or model_name in key:
            return val
    return None


def calculate_cost(model_name: str, input_tokens: int, output_tokens: int, cached_input_tokens: int = 0) -> float:
    pricing = _resolve_pricing(model_name)
    if not pricing:
        return 0.0

    in_price = pricing["input"]
    out_price = pricing["output"]
    cached_price = pricing.get("cached_input", None)

    billed_input_tokens = max(0, int(input_tokens) - int(cached_input_tokens))

    in_cost = (billed_input_tokens / 1_000_000) * in_price
    out_cost = (int(output_tokens) / 1_000_000) * out_price

    cached_cost = 0.0
    if cached_price is not None and cached_input_tokens > 0:
        cached_cost = (int(cached_input_tokens) / 1_000_000) * float(cached_price)

    return round(in_cost + cached_cost + out_cost, 6)


def _parse_price(val):
    try:
        return float(str(val).replace('$', '').replace(',', '').strip())
    except Exception:
        return None


def evaluate_quant_sr(ticker: str, report_json: dict, report_date_str: str, current_close: float) -> Optional[
    Dict[str, Any]]:
    risk_levels = report_json.get("risk_levels", {})
    supports = risk_levels.get("support", [])
    resistances = risk_levels.get("resistance", [])

    if not supports or not resistances:
        return None

    valid_supports = [p for p in (_parse_price(s) for s in supports) if p is not None]
    valid_resistances = [p for p in (_parse_price(r) for r in resistances) if p is not None]

    if not valid_supports or not valid_resistances:
        return None

    s1 = max(valid_supports)
    r1 = min(valid_resistances)

    report_date = datetime.strptime(report_date_str, "%Y-%m-%d")
    start_future = (report_date + timedelta(days=1)).strftime("%Y-%m-%d")
    end_future = (report_date + timedelta(days=15)).strftime("%Y-%m-%d")

    try:
        future_df = fetch_ohlcv_polygon(ticker, start=start_future, end=end_future)
    except Exception:
        return None

    if future_df.empty:
        return None

    future_df = future_df.head(5)

    if future_df.empty:
        return None

    actual_low = future_df['low'].min()
    actual_high = future_df['high'].max()

    is_support_breached = bool(actual_low < s1)
    is_resistance_breached = bool(actual_high > r1)

    predicted_drawdown_tolerance = (current_close - s1) / current_close
    actual_max_drawdown = (current_close - actual_low) / current_close
    mae_margin = predicted_drawdown_tolerance - actual_max_drawdown

    days_in_bound = sum((future_df['close'] >= s1) & (future_df['close'] <= r1))
    coverage_ratio = days_in_bound / len(future_df)

    return {
        "predicted_support": s1,
        "predicted_resistance": r1,
        "actual_1w_low": actual_low,
        "actual_1w_high": actual_high,
        "support_breached": is_support_breached,
        "resistance_breached": is_resistance_breached,
        "mae_margin_pct": round(mae_margin * 100, 2),
        "coverage_ratio": round(coverage_ratio, 4)
    }


def run_financial_analyst(ticker, start, end, oai_config_path, api_keys_path, model_name):
    current_step = "Initializing"

    try:
        current_step = "1. Fetching historical stock data via Polygon API"
        prices = fetch_ohlcv_polygon(ticker, start, end)

        current_step = "2. Computing market technical snapshot"
        snapshot = compute_market_snapshot(prices)
        snapshot = json.loads(json.dumps(snapshot, ensure_ascii=False).replace("NaN", "null"))

        current_step = "3. Loading Autogen model configuration (config_list)"
        config_list = autogen.config_list_from_json(
            oai_config_path,
            filter_dict={"model": [model_name]},
        )
        if not config_list:
            raise ValueError(
                f"Model config not found for: {model_name}. Please check if the config file {oai_config_path} contains this model.")

        llm_config = {
            "config_list": config_list,
            "timeout": 240,
            "cache_seed": None,  # <-- NEW: Set to None to completely disable caching
        }

        current_step = "4. Initializing Autogen agents"
        assistant = autogen.AssistantAgent(
            name="Market_Analyst",
            llm_config=llm_config,
            system_message="You are a professional financial analyst. Use the available tools to fetch REAL news, then correlate it with the technical snapshot to generate a JSON report."
        )

        user_proxy = autogen.UserProxyAgent(
            name="User_Executor",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=6,
            is_termination_msg=lambda x: "full_assessment" in (x.get("content") or ""),
            code_execution_config={"work_dir": "coding", "use_docker": False},
        )

        def fetch_news_for_sim(ticker_symbol: str) -> str:
            return get_real_market_news(ticker_symbol, target_date=end)

        autogen.register_function(
            fetch_news_for_sim,
            caller=assistant,
            executor=user_proxy,
            name="get_market_news",
            description="Fetch REAL recent financial news for a specific ticker from Polygon.io."
        )

        as_of = str(prices.index.max().date()) if not prices.empty else end

        prompt = f"""
        You are a trader-oriented financial analyst.
        Ticker: {ticker}
        Date: {as_of}

        Technical Snapshot:
        {json.dumps(snapshot, indent=2)}

        Task:
        1. Call 'get_market_news' to see what is driving the market currently.
        2. Analyze the correlation between the news sentiment and the technical trend.
        3. Output a SINGLE JSON object matching this schema exactly (No Markdown):
        {{
          "meta": {{ "ticker": "...", "as_of": "...", "model": "{model_name}" }},
          "market_snapshot": {{...}},
          "news_analysis": [ {{ "headline": "...", "sentiment": "...", "impact": "..." }} ],
          "forecast": {{ "horizon": "1w", "direction": "...", "reasoning": "..." }},
          "risk_levels": {{ "support": [...], "resistance": [...] }},
          "full_assessment": "..."
        }}
        """

        current_step = "5. Interacting with LLM"
        t0 = time.time()
        chat_res = user_proxy.initiate_chat(assistant, message=prompt)
        latency = time.time() - t0

        current_step = "6. Parsing Token usage and chat history"
        usage = extract_autogen_usage(chat_res, model_hint=model_name)
        total_input_tokens = int(usage["billed_prompt_tokens"])
        total_output_tokens = int(usage["billed_completion_tokens"])
        cached_inf_prompt = int(usage["cached_inference_prompt_tokens"])
        cached_inf_completion = int(usage["cached_inference_completion_tokens"])
        autogen_cost = usage["autogen_reported_cost"]

        # --- FALLBACK MECHANISM FOR UNTRACKED TOKENS ---
        if total_input_tokens == 0 and total_output_tokens == 0:
            fallback_in_chars = 0
            fallback_out_chars = 0

            for msg in getattr(chat_res, "chat_history", []):
                content = str(msg.get("content", ""))
                role = msg.get("role", "")
                if role == "assistant":
                    fallback_out_chars += len(content)
                else:
                    fallback_in_chars += len(content)

            if fallback_in_chars == 0:
                fallback_in_chars = len(prompt)

            # Rule of thumb: 1 token is roughly 4 characters
            total_input_tokens = max(1, fallback_in_chars // 4)
            total_output_tokens = max(1, fallback_out_chars // 4)
        # -----------------------------------------------

        found_json = None
        for msg in reversed(getattr(chat_res, "chat_history", []) or []):
            content = msg.get("content")
            if not content:
                continue
            candidates = extract_all_json_objects(str(content))
            for candidate in candidates:
                try:
                    obj = json.loads(candidate)
                    if isinstance(obj, dict) and "full_assessment" in obj:
                        found_json = obj
                        break
                except Exception:
                    continue
            if found_json:
                break

        if not found_json:
            raise ValueError(
                "Failed to find a valid JSON report in the LLM output (the model might not have output JSON as requested).")

        found_json["_ops"] = {
            "llm_latency_sec": float(latency),
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "cached_inference_prompt_tokens": cached_inf_prompt,
            "cached_inference_completion_tokens": cached_inf_completion,
            "autogen_reported_cost": float(autogen_cost) if autogen_cost is not None else None,
        }
        return found_json

    except Exception as e:
        raise RuntimeError(f"Crash occurred at step: [{current_step}] -> Specific error: {str(e)}") from e


def run_batch_experiment(ticker, start, end, models_to_test, oai_config_path, api_keys_path, output_dir):
    results = []
    os.makedirs(output_dir, exist_ok=True)

    print(f"Starting Batch Experiment for {ticker}...")
    print(f"{'Model':<28} | {'Status':<10} | {'Latency':<10} | {'Cost ($)':<12}")
    print("-" * 70)

    for model in models_to_test:
        row = {
            "Model": model,
            "Ticker": ticker,
            "Date": datetime.now().strftime("%Y-%m-%d"),
            "Status": "Failed",
            "Latency (s)": 0.0,
            "Input Tokens": 0,
            "Output Tokens": 0,
            "Est. Cost ($)": 0.0,
            "Direction": None,
            "Output Length": 0,
            "Support Breached": None,
            "Resistance Breached": None,
            "MAE Margin (%)": None,
            "Coverage Ratio": None
        }

        try:
            report = run_financial_analyst(
                ticker=ticker,
                start=start,
                end=end,
                oai_config_path=oai_config_path,
                api_keys_path=api_keys_path,
                model_name=model
            )

            ops = report.get("_ops", {}) or {}
            row["Status"] = "Success"
            row["Latency (s)"] = round(float(ops.get("llm_latency_sec", 0) or 0), 2)
            row["Input Tokens"] = int(ops.get("input_tokens", 0) or 0)
            row["Output Tokens"] = int(ops.get("output_tokens", 0) or 0)

            if ops.get("autogen_reported_cost") is not None:
                row["Est. Cost ($)"] = round(float(ops["autogen_reported_cost"]), 6)
            else:
                row["Est. Cost ($)"] = calculate_cost(model, row["Input Tokens"], row["Output Tokens"])

            forecast = report.get("forecast", {}) or {}
            row["Direction"] = forecast.get("direction", "N/A")
            row["Output Length"] = len(report.get("full_assessment", "") or "")

            current_close = report.get("market_snapshot", {}).get("last_close")
            if current_close:
                quant_metrics = evaluate_quant_sr(ticker, report, end, current_close)
                if quant_metrics:
                    row["Support Breached"] = quant_metrics["support_breached"]
                    row["Resistance Breached"] = quant_metrics["resistance_breached"]
                    row["MAE Margin (%)"] = quant_metrics["mae_margin_pct"]
                    row["Coverage Ratio"] = quant_metrics["coverage_ratio"]
                    report["_quant_eval"] = quant_metrics

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = os.path.join(output_dir, f"{ticker}_{model}_{ts}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)

            print(f"{model:<28} | Success    | {row['Latency (s)']:<10} | {row['Est. Cost ($)']:<12}")

        except Exception as e:
            print(f"{model:<28} | Failed     | N/A        | N/A")
            print("\n" + "=" * 50)
            print(f"🚨 Error Report: {model}")
            print("=" * 50)
            print(str(e))
            print("-" * 50)

            error_details = traceback.format_exc()
            print(error_details)
            print("=" * 50 + "\n")

            row["Error Info"] = str(e)

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            error_log_path = os.path.join(output_dir, f"ERROR_{ticker}_{model}_{ts}.log")
            with open(error_log_path, "w", encoding="utf-8") as ef:
                ef.write(error_details)

        results.append(row)

    return pd.DataFrame(results)


if __name__ == "__main__":
    TICKER = "NVDA"
    START = "2024-01-01"
    END = "2025-12-31"

    MODELS = [
        "claude-sonnet-4-20250514", "meta-llama/llama-4-maverick", "grok-4",
        # "gpt-4o", "gpt-4o-mini", "gpt-5-mini", "gpt-5-nano", "gpt-5", "o3", "o4-mini"
    ]
    CONFIG_PATH = r"C:\Users\bangc\one-person-unicorn-infra\OAI_CONFIG_LIST"
    KEYS_PATH = r"C:\Users\bangc\one-person-unicorn-infra\config_api_keys"
    OUT_DIR = r"C:\Users\bangc\one-person-unicorn-infra\data\analyst_outputs"

    df = run_batch_experiment(TICKER, START, END, MODELS, CONFIG_PATH, KEYS_PATH, OUT_DIR)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(OUT_DIR, f"final_comparison_{TICKER}_{ts}.csv")
    df.to_csv(csv_path, index=False)

    latex_cols = ["Model", "Status", "Latency (s)", "Est. Cost ($)", "Direction", "Support Breached", "MAE Margin (%)",
                  "Coverage Ratio"]
    if set(latex_cols).issubset(df.columns):
        tex_path = os.path.join(OUT_DIR, f"final_table_{TICKER}_{ts}.tex")
        with open(tex_path, "w", encoding="utf-8") as f:
            f.write(df[latex_cols].to_latex(index=False, float_format="%.4f"))

    print("\nExperiment Complete. Summary:")
    print(df[["Model", "Status", "Latency (s)", "Direction", "Support Breached", "MAE Margin (%)", "Coverage Ratio"]])