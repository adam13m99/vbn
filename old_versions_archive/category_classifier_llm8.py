"""
Product Category CLASSIFIER V8 - Testing Gemini 3 Flash Preview
- Classifies 500k+ products into Level 1 / Level 2 based on strict JSON rules.
- Input: tf_menu.csv (item_id, category_name, item_title, item_description)
- Output: tf_menu_labeled_v8.csv (Appends results incrementally)

VERSION 8 CHANGES:
- 🆕 Updated to gemini-3-flash-preview model
- 🔔 Telegram notifications for all errors (429, 504, etc)
- 📊 Rate watchdog: Auto-stops if processing is too slow
- 🚨 Smart quota handling for 429 errors
- All previous features: system_instruction, detailed logging, optimal settings

ERROR HANDLING:
- Critical (STOP APP): 403, GOAWAY, permission errors, slow processing
- Transient (RETRY): 504, 429, 5xx, timeouts, network issues
- Exponential backoff with jitter for all retries
- Telegram alerts on every error occurrence
"""

from __future__ import annotations

import os
import json
import time
import threading
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from collections import Counter
from datetime import datetime
import sys
import random

import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

import google.generativeai as genai

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


# ============================ CONFIGURATION ============================

INPUT_FILE = "tf_menu.csv"
OUTPUT_FILE = "tf_menu_labeled_v8.csv"
TAXONOMY_FILE = "category_definitions3.json"

# FIX: Set API key as environment variable (not as parameter)
os.environ["GOOGLE_API_KEY"] = os.getenv("GENAI_API_KEY", "AIzaSyBJU5btf-h1y6WIxnq8YRjWsTM--0otxHU")

CONFIG: Dict[str, Any] = {
    # --- Gemini ---
    "MODEL_NAME": "gemini-3-flash-preview",  # V8: Testing Gemini 3 Flash Preview
    "TEMPERATURE": 0.0,

    # --- Processing --- OPTIMIZED for 10 RPM / 1500 RPD limits
    "MAX_WORKERS": 5,   # allows parallelism while staying under limits
    "BATCH_SIZE": 25,   # maximize items per request (keep high!)
    "RATE_LIMIT_PER_SEC": 0.10,  # = 9 req/min (safely under 10 RPM limit)

    # Retries / Backoff
    "MAX_RETRIES": 6,
    "BACKOFF_BASE_SEC": 2.0,
    "BACKOFF_MAX_SEC": 120.0,
    "RETRY_JITTER_SEC": 1.0,

    # bounded in-flight futures
    "MAX_IN_FLIGHT_MULTIPLIER": 3,

    # --- Telegram ---
    "TELEGRAM_ENABLED": True,
    "TELEGRAM_BOT_TOKEN": os.getenv("TELEGRAM_BOT_TOKEN", "8205938582:AAG-fhOjW4tMPkNRpYU8J_Xg7vgMLisHCBU"),
    "TELEGRAM_CHAT_ID": os.getenv("TELEGRAM_CHAT_ID", "-5091693030"),
    "TELEGRAM_TIMEOUT_SEC": 15,
    "TELEGRAM_MAX_MSG_CHARS": 4000,

    # --- Reporting ---
    "REPORT_EVERY_SECONDS": 30,  # every minute

    # --- Per request timeout (best-effort) ---
    "REQUEST_TIMEOUT_SEC": 60,  # reduced - Gemini times out around 60s anyway

    # --- V8: Rate Watchdog ---
    "MIN_ITEMS_PER_MINUTE": 1,  # Stop if processing slower than this (expect ~225/min at 9 RPM)
    "WATCHDOG_CHECK_INTERVAL": 120,  # Check rate every 2 minutes
}

# Configure Gemini - FIX: Don't pass api_key parameter, use env var instead
genai.configure()


# ============================ GLOBAL SHUTDOWN FLAG ============================

class ShutdownFlag:
    """Global flag to signal immediate shutdown on critical errors."""
    def __init__(self):
        self._should_shutdown = False
        self._reason = ""
        self._lock = threading.Lock()

    def set(self, reason: str):
        with self._lock:
            if not self._should_shutdown:
                self._should_shutdown = True
                self._reason = reason

    def is_set(self) -> bool:
        with self._lock:
            return self._should_shutdown

    def reason(self) -> str:
        with self._lock:
            return self._reason

shutdown_flag = ShutdownFlag()


# ============================ RATE LIMITER ============================

class RateLimiter:
    """Thread-safe rate limiter"""
    def __init__(self, max_calls_per_second: float):
        self.max_calls_per_second = max_calls_per_second
        self.min_interval = 1.0 / max_calls_per_second if max_calls_per_second > 0 else 0.0
        self.last_call = 0.0
        self.lock = threading.Lock()

    def wait(self):
        if self.min_interval <= 0:
            return
        with self.lock:
            now = time.time()
            elapsed = now - self.last_call
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
            self.last_call = time.time()

rate_limiter = RateLimiter(CONFIG["RATE_LIMIT_PER_SEC"])


# ============================ TELEGRAM REPORTER ============================

class TelegramReporter:
    def __init__(self, enabled: bool, bot_token: str, chat_id: str, timeout_sec: int, max_chars: int):
        self.enabled = enabled and bool(bot_token) and bool(chat_id)
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.timeout_sec = timeout_sec
        self.max_chars = max_chars

    def _chunk_text(self, text: str) -> List[str]:
        text = (text or "").strip()
        if not text:
            return []
        if len(text) <= self.max_chars:
            return [text]
        chunks: List[str] = []
        start = 0
        while start < len(text):
            end = min(start + self.max_chars, len(text))
            nl = text.rfind("\n", start, end)
            if nl != -1 and nl > start + 200:
                end = nl
            part = text[start:end].strip()
            if part:
                chunks.append(part)
            start = end
        return chunks

    def send(self, text: str):
        if not self.enabled:
            return
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload_base = {"chat_id": self.chat_id, "disable_web_page_preview": True}

        for part in self._chunk_text(text):
            payload = dict(payload_base)
            payload["text"] = part
            for attempt in range(3):
                try:
                    r = requests.post(url, json=payload, timeout=self.timeout_sec)
                    if r.status_code == 200:
                        break
                    time.sleep(1.5 * (attempt + 1))
                except Exception:
                    time.sleep(1.5 * (attempt + 1))

telegram = TelegramReporter(
    enabled=CONFIG["TELEGRAM_ENABLED"],
    bot_token=CONFIG["TELEGRAM_BOT_TOKEN"],
    chat_id=CONFIG["TELEGRAM_CHAT_ID"],
    timeout_sec=CONFIG["TELEGRAM_TIMEOUT_SEC"],
    max_chars=CONFIG["TELEGRAM_MAX_MSG_CHARS"],
)


# ============================ COST TRACKING ============================

@dataclass
class CostTracker:
    """
    Token/cost tracker.
    - Uses response.usage_metadata when available; otherwise falls back to estimates.
    Pricing placeholders (edit to your real price):
      input_cost_per_1m, output_cost_per_1m
    """
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0
    input_cost_per_1m: float = 0.50
    output_cost_per_1m: float = 3.00
    lock: threading.Lock = field(default_factory=threading.Lock)

    def update(self, in_tokens: int, out_tokens: int):
        with self.lock:
            self.input_tokens += int(in_tokens)
            self.output_tokens += int(out_tokens)
            self.calls += 1

    def summary_str(self) -> str:
        with self.lock:
            in_cost = (self.input_tokens / 1_000_000) * self.input_cost_per_1m
            out_cost = (self.output_tokens / 1_000_000) * self.output_cost_per_1m
            total = in_cost + out_cost
            return (
                f"Calls: {self.calls} | "
                f"Tokens: {self.input_tokens:,} in / {self.output_tokens:,} out | "
                f"Cost: ${total:.4f} (in ${in_cost:.4f} + out ${out_cost:.4f})"
            )

cost_tracker = CostTracker()


# ============================ STATS TRACKING ============================

@dataclass
class StatsTracker:
    processed: int = 0
    success: int = 0
    unknown: int = 0
    error: int = 0
    level1_counts: Counter = field(default_factory=Counter)
    level12_counts: Counter = field(default_factory=Counter)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def update_from_rows(self, rows: List[Dict[str, Any]]):
        with self.lock:
            for r in rows:
                self.processed += 1
                l1 = str(r.get("level_1", "ERROR"))
                l2 = str(r.get("level_2", "ERROR"))

                if l1 == "ERROR" or l2 == "ERROR":
                    self.error += 1
                elif l1 == "UNKNOWN" or l2 == "UNKNOWN":
                    self.unknown += 1
                else:
                    self.success += 1

                self.level1_counts[l1] += 1
                self.level12_counts[(l1, l2)] += 1

    def snapshot(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "processed": self.processed,
                "success": self.success,
                "unknown": self.unknown,
                "error": self.error,
                "level1_counts": self.level1_counts.copy(),
                "level12_counts": self.level12_counts.copy(),
            }

stats = StatsTracker()


# ============================ V6: ERROR NOTIFIER ============================

class ErrorNotifier:
    """V6: Sends Telegram notifications for errors"""
    def __init__(self, telegram_reporter):
        self.telegram = telegram_reporter
        self.error_counts: Dict[str, int] = {}
        self.lock = threading.Lock()
        self.last_notification_time: Dict[str, float] = {}
        self.min_notification_interval = 60  # Don't spam - at most 1 notification per error type per minute

    def notify_error(self, error_type: str, error_msg: str, batch_info: str = ""):
        """Send Telegram notification for an error"""
        with self.lock:
            self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
            count = self.error_counts[error_type]

            # Rate limit notifications - don't spam for same error type
            now = time.time()
            last_time = self.last_notification_time.get(error_type, 0)
            if now - last_time < self.min_notification_interval:
                return  # Skip notification - too soon

            self.last_notification_time[error_type] = now

            # Prepare message
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            msg = (
                f"🚨 ERROR ALERT @ {ts}\n"
                f"Type: {error_type}\n"
                f"Occurrence: #{count}\n"
            )
            if batch_info:
                msg += f"Batch: {batch_info}\n"
            msg += f"Message: {error_msg[:500]}\n"
            msg += f"\nStats: {stats.processed:,} items processed so far"

            self.telegram.send(msg)

error_notifier = ErrorNotifier(telegram)


# ============================ V6: RATE WATCHDOG ============================

class RateWatchdog:
    """V6: Monitors processing rate and stops if too slow"""
    def __init__(self, min_items_per_minute: int):
        self.min_items_per_minute = min_items_per_minute
        self.start_time = time.time()
        self.last_check_time = time.time()
        self.last_check_processed = 0
        self.lock = threading.Lock()

    def check_rate(self, current_processed: int) -> tuple[bool, str]:
        """
        Returns (should_stop, reason)
        should_stop=True if processing is too slow
        """
        with self.lock:
            now = time.time()

            # Skip check if less than 2 minutes have passed since last check
            if now - self.last_check_time < 120:
                return (False, "")

            # Skip check if less than 5 minutes have passed since start
            if now - self.start_time < 300:
                return (False, "")

            # Calculate rate since last check
            time_elapsed_min = (now - self.last_check_time) / 60.0
            items_processed = current_processed - self.last_check_processed

            if time_elapsed_min > 0:
                rate = items_processed / time_elapsed_min

                # Update for next check
                self.last_check_time = now
                self.last_check_processed = current_processed

                # Check if rate is too slow
                if rate < self.min_items_per_minute:
                    reason = (
                        f"Processing rate too slow: {rate:.1f} items/min "
                        f"(minimum: {self.min_items_per_minute} items/min). "
                        f"Likely hitting quota limits."
                    )
                    return (True, reason)

            return (False, "")

rate_watchdog = RateWatchdog(CONFIG["MIN_ITEMS_PER_MINUTE"])


# ============================ HELPERS ============================

def _clean_text(x: Any) -> str:
    if x is None or pd.isna(x):
        return ""
    return str(x).replace('"', "").replace("\n", " ").strip()

def _pct(part: int, total: int) -> str:
    if total <= 0:
        return "0.00%"
    return f"{(100.0 * part / total):.2f}%"

def _strip_code_fences(text: str) -> str:
    t = (text or "").strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[-1]
        if t.endswith("```"):
            t = t[:-3]
    return t.strip()

def get_processed_ids(filepath: str) -> set:
    if not os.path.exists(filepath):
        return set()
    try:
        df = pd.read_csv(filepath, usecols=["item_id"])
        return set(df["item_id"].astype(str))
    except Exception:
        return set()

def batch_iter_from_df(df: pd.DataFrame, batch_size: int):
    batch: List[Dict[str, Any]] = []
    for row in df.itertuples(index=False):
        batch.append(row._asdict())
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch

def save_final_reports(level1_counts: Counter, level12_counts: Counter):
    pd.DataFrame([{"level_1": k, "count": v} for k, v in level1_counts.most_common()]).to_csv(
        "final_level1_counts.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame([{"level_1": k[0], "level_2": k[1], "count": v} for k, v in level12_counts.most_common()]).to_csv(
        "final_level2_counts.csv", index=False, encoding="utf-8-sig"
    )

def format_progress_message(snap: Dict[str, Any]) -> str:
    processed = snap["processed"]
    success = snap["success"]
    unknown = snap["unknown"]
    error = snap["error"]
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return (
        f"[tf_menu] Minute report @ {ts}\n"
        f"Processed: {processed:,}\n"
        f"✅ Success: {success:,} ({_pct(success, processed)})\n"
        f"❓ Unknown: {unknown:,} ({_pct(unknown, processed)})\n"
        f"❌ Error:   {error:,} ({_pct(error, processed)})\n\n"
        f"[COST]\n{cost_tracker.summary_str()}\n"
    )

def format_shutdown_message(reason: str) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    snap = stats.snapshot()
    return (
        f"[tf_menu] 🚨 SHUTDOWN TRIGGERED @ {ts}\n"
        f"Reason: {reason}\n\n"
        f"Progress so far:\n"
        f"Processed: {snap['processed']:,}\n"
        f"✅ Success: {snap['success']:,}\n"
        f"❓ Unknown: {snap['unknown']:,}\n"
        f"❌ Error:   {snap['error']:,}\n\n"
        f"[COST]\n{cost_tracker.summary_str()}\n"
        f"\nThe app has stopped to prevent further API calls."
    )

def format_final_message(snap: Dict[str, Any], top_n_l1: int = 40, top_n_l12: int = 60) -> str:
    processed = snap["processed"]
    success = snap["success"]
    unknown = snap["unknown"]
    error = snap["error"]
    level1_counts: Counter = snap["level1_counts"]
    level12_counts: Counter = snap["level12_counts"]

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines: List[str] = []
    lines.append(f"[tf_menu] FINAL report @ {ts}")
    lines.append(f"Processed: {processed:,}")
    lines.append(f"✅ Success: {success:,}")
    lines.append(f"❓ Unknown: {unknown:,}")
    lines.append(f"❌ Error:   {error:,}")
    lines.append("")
    lines.append("[COST]")
    lines.append(cost_tracker.summary_str())
    lines.append("")
    lines.append(f"Top Level 1 categories (top {top_n_l1}):")
    for k, v in level1_counts.most_common(top_n_l1):
        lines.append(f"- {k}: {v:,}")
    lines.append("")
    lines.append(f"Top Level 1 → Level 2 pairs (top {top_n_l12}):")
    for (l1, l2), v in level12_counts.most_common(top_n_l12):
        lines.append(f"- {l1} → {l2}: {v:,}")
    lines.append("")
    lines.append("Saved CSVs: final_level1_counts.csv, final_level2_counts.csv")
    return "\n".join(lines)


# ============================ ERROR DETECTION + BACKOFF ============================

def is_critical_gemini_error(err: str) -> bool:
    e = (err or "").lower()

    if "status: 403" in e or "received http2 header with status: 403" in e:
        return True

    if "goaway received" in e and "client_misbehavior" in e:
        return True

    if "grpc_status:14" in e and "http2_error:11" in e:
        return True

    auth_phrases = ["permission denied", "authentication", "invalid api key", "unauthorized", "forbidden"]
    if any(p in e for p in auth_phrases):
        return True

    return False

def is_transient_gemini_error(err: str) -> bool:
    e = (err or "").lower()

    if "429" in e:
        return True

    if "504" in e:
        return True

    if "deadline expired" in e or "deadline_exceeded" in e:
        return True

    if "stream cancelled" in e or "stream canceled" in e:
        return True

    if "rpc::cancelled" in e or "status = cancelled" in e:
        return True

    if "unavailable" in e:
        return True

    for code in ["500", "502", "503"]:
        if code in e:
            return True

    transient_phrases = [
        "connection reset", "connection aborted", "timed out", "timeout", "tls",
        "socket", "temporarily unavailable", "server closed", "broken pipe",
    ]
    if any(p in e for p in transient_phrases):
        return True

    return False

def compute_backoff_seconds(attempt: int) -> float:
    base = float(CONFIG["BACKOFF_BASE_SEC"])
    cap = float(CONFIG["BACKOFF_MAX_SEC"])
    jitter = float(CONFIG["RETRY_JITTER_SEC"])

    wait = base * (2 ** attempt)
    wait += random.random() * jitter

    if wait > cap:
        wait = cap + (random.random() * jitter)

    return max(0.0, wait)


# ============================ JSON PARSING (validator-style robust parser) ============================

def parse_json_strict(text: str) -> Any:
    """
    STRICT-ish parser that tries:
    1) direct json.loads
    2) markdown code blocks
    3) substring between first '{' and last '}' OR first '[' and last ']'
    Raises ValueError if cannot parse.
    """
    raw = (text or "").strip()

    # Remove common code fences first (safe)
    cleaned = _strip_code_fences(raw)

    # 1) direct parse
    try:
        return json.loads(cleaned)
    except Exception:
        pass

    # 2) try to extract from ```json ... ``` or ``` ... ```
    try:
        lower = raw.lower()
        if "```json" in lower:
            start = lower.find("```json") + 7
            end = raw.find("```", start)
            if end != -1:
                candidate = raw[start:end].strip()
                return json.loads(candidate)
        if "```" in raw:
            start = raw.find("```") + 3
            end = raw.find("```", start)
            if end != -1:
                candidate = raw[start:end].strip()
                return json.loads(candidate)
    except Exception:
        pass

    # 3) boundary substring
    first_brace = raw.find("{")
    last_brace = raw.rfind("}")
    first_bracket = raw.find("[")
    last_bracket = raw.rfind("]")

    candidates: List[str] = []
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        candidates.append(raw[first_brace:last_brace + 1])
    if first_bracket != -1 and last_bracket != -1 and last_bracket > first_bracket:
        candidates.append(raw[first_bracket:last_bracket + 1])

    for cand in candidates:
        try:
            return json.loads(cand)
        except Exception:
            continue

    raise ValueError("Failed to parse JSON from model response")


# ============================ CLASSIFIER ENGINE (validator-style Gemini usage) ============================

class ClassifierEngine:
    """
    V5 OPTIMIZED: Uses system_instruction approach (taxonomy NOT in prompt)
    - Puts taxonomy in system_instruction for efficiency
    - Uses response_mime_type="application/json" for cleaner JSON
    - Much faster than embedding taxonomy in every prompt!
    - Thread-local model instances for true parallelism
    """

    def __init__(self, taxonomy_path: str):
        with open(taxonomy_path, "r", encoding="utf-8") as f:
            taxonomy = json.load(f)

        self.taxonomy_str = json.dumps(taxonomy, ensure_ascii=False, separators=(",", ":"))

        # Build system instruction with taxonomy
        self.system_instruction = (
            "You are an expert AI Food Data Classifier.\n"
            "Assign ONE 'Level 1' and ONE 'Level 2' category from the TAXONOMY below to each product.\n\n"
            "TAXONOMY (STRICT RULES):\n"
            f"{self.taxonomy_str}\n\n"
            "RULES:\n"
            "1. Analyze 'title' and 'desc'. Use 'context_category' only as a hint for ambiguity.\n"
            "2. Strictly check 'explanation' (INCLUDES) and 'exclusions' (MUST NOT INCLUDE).\n"
            "3. Prioritize specificity.\n"
            "4. If NO category fits, return 'UNKNOWN'.\n\n"
            "OUTPUT: JSON Array of objects. Each must contain:\n"
            "[\n"
            '  { "id": "item_id", "level_1": "...", "level_2": "...", "reason": "short reason" }\n'
            "]\n"
            "Return ONLY valid JSON. No extra text.\n"
        )

        # Thread-local storage for model instances
        self._thread_local = threading.local()

    def _get_thread_model(self):
        """Get thread-local model instance for true parallel processing"""
        if getattr(self._thread_local, "model", None) is None:
            self._thread_local.model = genai.GenerativeModel(
                model_name=CONFIG["MODEL_NAME"],
                system_instruction=self.system_instruction,
                generation_config={
                    "temperature": CONFIG["TEMPERATURE"],
                    "response_mime_type": "application/json",
                },
            )
        return self._thread_local.model

    def _fallback_results(self, products_for_prompt: List[Dict[str, Any]], reason: str) -> List[Dict[str, Any]]:
        return [{"id": p["id"], "level_1": "ERROR", "level_2": "ERROR", "reason": reason} for p in products_for_prompt]

    def _validate_and_normalize(self, products_for_prompt: List[Dict[str, Any]], parsed: Any) -> List[Dict[str, Any]]:
        id_order = [str(p["id"]) for p in products_for_prompt]

        # allow wrapper dict format {"results":[...]}
        if isinstance(parsed, dict) and "results" in parsed and isinstance(parsed["results"], list):
            parsed = parsed["results"]

        if not isinstance(parsed, list):
            return self._fallback_results(products_for_prompt, "JSON_SHAPE_INVALID")

        result_map: Dict[str, Dict[str, Any]] = {}
        for r in parsed:
            if isinstance(r, dict) and "id" in r:
                result_map[str(r["id"])] = r

        out: List[Dict[str, Any]] = []
        for pid in id_order:
            r = result_map.get(pid)
            if not isinstance(r, dict):
                out.append({"id": pid, "level_1": "ERROR", "level_2": "ERROR", "reason": "MISSING_ID"})
                continue
            out.append({
                "id": pid,
                "level_1": str(r.get("level_1", "ERROR")),
                "level_2": str(r.get("level_2", "ERROR")),
                "reason": str(r.get("reason", "")),
            })
        return out

    def _build_prompt(self, products_for_prompt: List[Dict[str, Any]]) -> str:
        """
        V5 OPTIMIZED: Just return products JSON (taxonomy is in system_instruction!)
        """
        return json.dumps(products_for_prompt, ensure_ascii=False, separators=(",", ":"))

    def classify_batch(self, batch_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        - If shutdown_flag is set, do NOT call the API.
        - On critical errors, set shutdown flag and raise RuntimeError.
        - On transient errors, retry with exponential backoff.
        - Use per-request timeout best-effort.
        """
        if shutdown_flag.is_set():
            raise RuntimeError(f"Shutdown already triggered: {shutdown_flag.reason()}")

        products_for_prompt: List[Dict[str, Any]] = []
        for item in batch_data:
            products_for_prompt.append({
                "id": str(item.get("item_id", "")),
                "title": _clean_text(item.get("item_title")),
                "desc": _clean_text(item.get("item_description")),
                "context_category": _clean_text(item.get("category_name")),
            })

        prompt = self._build_prompt(products_for_prompt)
        last_err: Optional[str] = None

        batch_ids = [p["id"] for p in products_for_prompt]
        logging.info(f"Processing batch: {len(batch_ids)} items, IDs: {batch_ids[:3]}...")

        for attempt in range(int(CONFIG["MAX_RETRIES"])):
            if shutdown_flag.is_set():
                raise RuntimeError(f"Shutdown triggered: {shutdown_flag.reason()}")

            try:
                logging.info(f"API call attempt {attempt+1}/{CONFIG['MAX_RETRIES']} for batch {batch_ids[:3]}...")

                # rate limiting
                rate_limiter.wait()

                # V5: Use thread-local model (no lock needed!)
                model = self._get_thread_model()

                logging.info(f"Calling Gemini API with timeout={CONFIG['REQUEST_TIMEOUT_SEC']}s...")
                # Pass prompt as list for system_instruction approach
                try:
                    response = model.generate_content(
                        [prompt],
                        request_options={"timeout": int(CONFIG["REQUEST_TIMEOUT_SEC"])},
                    )
                except TypeError:
                    # Fallback if timeout not supported
                    logging.info(f"Timeout parameter not supported, calling without timeout...")
                    response = model.generate_content([prompt])

                logging.info(f"API call successful! Processing response...")

                txt = getattr(response, "text", "") or ""

                # token usage best-effort
                in_tokens = 0
                out_tokens = 0
                usage = getattr(response, "usage_metadata", None)
                if usage is not None:
                    in_tokens = int(getattr(usage, "prompt_token_count", 0) or 0)
                    out_tokens = int(getattr(usage, "candidates_token_count", 0) or 0)
                else:
                    # fallback estimates (V5: much smaller prompts - taxonomy in system_instruction)
                    in_tokens = max(1, len(prompt) // 3)
                    out_tokens = max(30, len(txt) // 4)

                cost_tracker.update(in_tokens, out_tokens)

                parsed = parse_json_strict(txt)
                normalized = self._validate_and_normalize(products_for_prompt, parsed)
                logging.info(f"✅ Batch completed successfully: {len(batch_ids)} items processed")
                return normalized

            except Exception as e:
                last_err = str(e)
                logging.error(f"❌ API call failed: {last_err[:200]}")

                # V6: Determine error type for notification
                error_type = "UNKNOWN_ERROR"
                if "429" in last_err:
                    error_type = "429_QUOTA_EXCEEDED"
                elif "504" in last_err:
                    error_type = "504_TIMEOUT"
                elif "403" in last_err:
                    error_type = "403_FORBIDDEN"
                elif "500" in last_err or "502" in last_err or "503" in last_err:
                    error_type = "5XX_SERVER_ERROR"

                # V6: Send Telegram notification
                error_notifier.notify_error(error_type, last_err, f"IDs: {batch_ids[:3]}")

                # critical stop
                if is_critical_gemini_error(last_err):
                    logging.critical(f"🚨 CRITICAL ERROR - Shutting down: {last_err}")
                    shutdown_flag.set(f"Critical Gemini error: {last_err}")
                    raise RuntimeError(shutdown_flag.reason())

                # transient retry with exponential backoff
                if is_transient_gemini_error(last_err):
                    wait_s = compute_backoff_seconds(attempt)
                    logging.warning(f"⚠️  [Retry] Transient error (attempt {attempt+1}/{CONFIG['MAX_RETRIES']}), sleeping {wait_s:.1f}s: {last_err[:200]}")
                    time.sleep(wait_s)
                    continue

                # non-critical retry also with backoff (conservative)
                wait_s = compute_backoff_seconds(attempt)
                logging.warning(f"⚠️  [Retry] Non-critical error (attempt {attempt+1}/{CONFIG['MAX_RETRIES']}), sleeping {wait_s:.1f}s: {last_err[:200]}")
                time.sleep(wait_s)

        logging.error(f"❌ Batch FAILED after {CONFIG['MAX_RETRIES']} attempts: {last_err[:200]}")
        return self._fallback_results(products_for_prompt, f"API_FAIL: {last_err or 'unknown'}")


# ============================ REPORTING THREAD (EVERY MINUTE) ============================

class MinuteReporter(threading.Thread):
    def __init__(self, stop_event: threading.Event, interval_sec: int):
        super().__init__(daemon=True)
        self.stop_event = stop_event
        self.interval_sec = max(10, int(interval_sec))

    def run(self):
        while not self.stop_event.is_set():
            slept = 0
            while slept < self.interval_sec and not self.stop_event.is_set():
                time.sleep(1)
                slept += 1
            if self.stop_event.is_set():
                break

            if shutdown_flag.is_set():
                telegram.send(format_shutdown_message(shutdown_flag.reason()))
                break

            # V6: Check rate watchdog
            current_processed = stats.snapshot()["processed"]
            should_stop, reason = rate_watchdog.check_rate(current_processed)
            if should_stop:
                logging.critical(f"🚨 RATE WATCHDOG TRIGGERED: {reason}")
                shutdown_flag.set(reason)
                telegram.send(format_shutdown_message(reason))
                break

            telegram.send(format_progress_message(stats.snapshot()))


# ============================ MAIN ============================

def main():
    print(f"=== AI PRODUCT CLASSIFIER ({CONFIG['MODEL_NAME']}) ===")
    print(f"Input file:  {INPUT_FILE}")
    print(f"Output file: {OUTPUT_FILE}")
    print(f"Taxonomy:    {TAXONOMY_FILE}")
    print(f"Workers:     {CONFIG['MAX_WORKERS']} | Batch size: {CONFIG['BATCH_SIZE']} | RPS: {CONFIG['RATE_LIMIT_PER_SEC']}")
    print(f"Retries:     {CONFIG['MAX_RETRIES']} | Backoff base: {CONFIG['BACKOFF_BASE_SEC']}s | Timeout: {CONFIG['REQUEST_TIMEOUT_SEC']}s")
    print("")

    print(f"Loading {INPUT_FILE}...")
    try:
        df = pd.read_csv(INPUT_FILE)
        df["item_id"] = df["item_id"].astype(str)
        total_rows = len(df)
        print(f"Total Products: {total_rows:,}")
    except Exception as e:
        print(f"Failed to read input file: {e}")
        telegram.send(f"[tf_menu] FAILED to read input file: {e}")
        return

    processed_ids = get_processed_ids(OUTPUT_FILE)
    print(f"Already Processed: {len(processed_ids):,} products")

    df_remaining = df[~df["item_id"].isin(processed_ids)]
    remaining = len(df_remaining)

    if remaining == 0:
        print("All products processed! Script finished.")
        telegram.send("[tf_menu] All products already processed. Nothing to do.")
        return

    print(f"Remaining to Process: {remaining:,}")

    if not os.path.exists(OUTPUT_FILE):
        pd.DataFrame(columns=[
            "item_id", "category_name", "item_title", "item_description",
            "level_1", "level_2", "reason"
        ]).to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    try:
        engine = ClassifierEngine(TAXONOMY_FILE)
    except Exception as e:
        print(f"Failed to load taxonomy or init model: {e}")
        telegram.send(f"[tf_menu] FAILED to load taxonomy or init model: {e}")
        return

    # Start minute reporter
    stop_event = threading.Event()
    reporter = MinuteReporter(stop_event=stop_event, interval_sec=CONFIG["REPORT_EVERY_SECONDS"])
    reporter.start()

    telegram.send(
        f"[tf_menu] Started V8 - Testing Gemini 3 Flash Preview.\n"
        f"Model: {CONFIG['MODEL_NAME']}\n"
        f"Remaining: {remaining:,}\n"
        f"Workers: {CONFIG['MAX_WORKERS']} | Batch: {CONFIG['BATCH_SIZE']} | RPS: {CONFIG['RATE_LIMIT_PER_SEC']}\n"
        f"Reporting: every {CONFIG['REPORT_EVERY_SECONDS']} seconds\n"
        f"Retries: {CONFIG['MAX_RETRIES']} | Timeout: {CONFIG['REQUEST_TIMEOUT_SEC']}s\n\n"
        f"🔔 V8 Features:\n"
        f"- NEW: gemini-3-flash-preview model\n"
        f"- Telegram error notifications (429, 504, etc)\n"
        f"- Rate watchdog: Stops if < {CONFIG['MIN_ITEMS_PER_MINUTE']} items/min\n"
        f"- system_instruction approach (8-10x faster!)\n"
        f"- Thread-local models for parallel processing\n\n"
        f"Critical errors (403/GOAWAY/slow rate) stop the app immediately.\n"
        f"Transient errors (429/504/5xx) are retried with exponential backoff.\n"
    )

    max_in_flight = max(1, int(CONFIG["MAX_WORKERS"]) * int(CONFIG["MAX_IN_FLIGHT_MULTIPLIER"]))
    batch_iter = batch_iter_from_df(df_remaining, int(CONFIG["BATCH_SIZE"]))

    def submit_next(executor, futures_map) -> bool:
        if shutdown_flag.is_set():
            return False
        try:
            batch = next(batch_iter)
        except StopIteration:
            return False
        fut = executor.submit(engine.classify_batch, batch)
        futures_map[fut] = batch
        return True

    total_batches = (remaining + int(CONFIG["BATCH_SIZE"]) - 1) // int(CONFIG["BATCH_SIZE"])
    logging.info(f"\n{'='*60}")
    logging.info(f"Starting processing: {remaining:,} items in {total_batches:,} batches")
    logging.info(f"Workers: {CONFIG['MAX_WORKERS']} | Batch size: {CONFIG['BATCH_SIZE']}")
    logging.info(f"{'='*60}\n")

    batches_completed = 0

    try:
        with ThreadPoolExecutor(max_workers=int(CONFIG["MAX_WORKERS"])) as executor:
            futures_map: Dict[Any, List[Dict[str, Any]]] = {}

            for _ in range(max_in_flight):
                if not submit_next(executor, futures_map):
                    break

            while futures_map:
                if shutdown_flag.is_set():
                    break

                for fut in as_completed(list(futures_map.keys())):
                    batch_input = futures_map.pop(fut)
                    batches_completed += 1

                    progress_pct = (batches_completed / total_batches) * 100
                    logging.info(f"📊 Progress: {batches_completed}/{total_batches} batches ({progress_pct:.1f}%)")

                    if shutdown_flag.is_set():
                        break

                    try:
                        results = fut.result()
                        result_map = {str(r.get("id")): r for r in results if isinstance(r, dict)}

                        processed_rows: List[Dict[str, Any]] = []
                        for input_row in batch_input:
                            input_id = str(input_row.get("item_id", ""))
                            api_res = result_map.get(input_id, {})

                            row_out = {
                                "item_id": input_id,
                                "category_name": input_row.get("category_name"),
                                "item_title": input_row.get("item_title"),
                                "item_description": input_row.get("item_description"),
                                "level_1": api_res.get("level_1", "ERROR"),
                                "level_2": api_res.get("level_2", "ERROR"),
                                "reason": api_res.get("reason", ""),
                            }
                            processed_rows.append(row_out)

                        pd.DataFrame(processed_rows).to_csv(
                            OUTPUT_FILE, mode="a", header=False, index=False, encoding="utf-8-sig"
                        )
                        stats.update_from_rows(processed_rows)

                    except RuntimeError as e:
                        logging.critical(f"🚨 RuntimeError in main loop: {e}")
                        shutdown_flag.set(str(e))
                        break

                    except Exception as e:
                        msg = f"Error in main loop batch handling: {e}"
                        logging.error(f"❌ {msg}")
                        telegram.send(f"[tf_menu] {msg}")

                    if shutdown_flag.is_set():
                        break

                    while len(futures_map) < max_in_flight:
                        if not submit_next(executor, futures_map):
                            break

                    break

            if shutdown_flag.is_set():
                print("\n🚨 SHUTDOWN TRIGGERED. Cancelling remaining in-flight requests...")
                telegram.send(format_shutdown_message(shutdown_flag.reason()))
                for f in list(futures_map.keys()):
                    f.cancel()

    finally:
        logging.info(f"\n{'='*60}")
        logging.info(f"Processing loop ended. Batches completed: {batches_completed}/{total_batches}")
        logging.info(f"{'='*60}\n")
        stop_event.set()

    if shutdown_flag.is_set():
        print("\n═══════════════════════════════════════════════════════════")
        print("🚨 APPLICATION STOPPED DUE TO CRITICAL GEMINI ERROR")
        print("Reason:", shutdown_flag.reason())
        print("Progress:", stats.snapshot())
        print("Cost:", cost_tracker.summary_str())
        print("═══════════════════════════════════════════════════════════")
        sys.exit(1)

    print("\nProcessing Complete.")
    snap = stats.snapshot()
    save_final_reports(snap["level1_counts"], snap["level12_counts"])
    telegram.send(format_final_message(snap))
    print(cost_tracker.summary_str())
    print("Saved final reports: final_level1_counts.csv, final_level2_counts.csv")


if __name__ == "__main__":
    main()
