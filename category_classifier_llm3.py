"""
Product Category CLASSIFIER (tf_menu.csv edition)
- Classifies 500k+ products into Level 1 / Level 2 based on strict JSON rules.
- Input: tf_menu.csv (item_id, category_name, item_title, item_description)
- Output: tf_menu_labeled.csv (Appends results incrementally)

FEATURES:
- Thread-local Gemini model instances (thread-safe)
- Rate limiter
- Resume by reading output file item_id
- Telegram minute reports (success/unknown/error shares + cost)
- Final report: counts per Level 1 and Level 1->Level 2

CRITICAL ERROR HANDLING (force stop app + stop API calls):
- 403 (http2 status 403)
- GOAWAY / client_misbehavior / grpc_status:14 + http2_error:11

TRANSIENT ERROR HANDLING (keep running with exponential backoff + per-request timeout):
- 504 Stream cancelled / RPC CANCELLED
- 504 Deadline expired before operation could complete
- 429 rate limit
- Other transient network/server issues

Per your request:
1) Exponential backoff (with jitter)
2) MAX_WORKERS lowered to 6
3) DO NOT reduce batch size
4) Add a per request timeout (best-effort; depends on installed google-generativeai version)
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
from tqdm import tqdm

import google.generativeai as genai


# ============================ CONFIGURATION ============================

INPUT_FILE = "tf_menu.csv"
OUTPUT_FILE = "tf_menu_labeled_v4_Bakoff.csv"
TAXONOMY_FILE = "category_definitions2.json"

CONFIG: Dict[str, Any] = {
    # --- Gemini ---
    "GEMINI_API_KEY": os.getenv("GENAI_API_KEY", "AIzaSyCS2tyzKaeqKmeYf7px31KMMv2LEvt85d4"),
    "MODEL_NAME": "gemini-3-flash-preview",
    "TEMPERATURE": 0.0,

    # --- Processing ---
    # (Per request) lower max workers to 6
    "MAX_WORKERS": 2,
    "BATCH_SIZE": 25,  # (Per request) do not reduce
    "RATE_LIMIT_PER_SEC": 10,

    # Retries / Backoff
    # With 504/CANCELLED/Deadline issues, 3 retries is often not enough.
    # Keeping this explicit and configurable.
    "MAX_RETRIES": 6,
    "BACKOFF_BASE_SEC": 2.0,        # exponential base wait multiplier
    "BACKOFF_MAX_SEC": 120.0,       # cap wait to avoid extremely long sleeps
    "RETRY_JITTER_SEC": 1.0,        # random jitter to avoid thundering herd

    # bounded in-flight futures to avoid creating tens of thousands of futures at once
    "MAX_IN_FLIGHT_MULTIPLIER": 3,

    # --- Telegram ---
    "TELEGRAM_ENABLED": True,
    "TELEGRAM_BOT_TOKEN": os.getenv("TELEGRAM_BOT_TOKEN", "8205938582:AAG-fhOjW4tMPkNRpYU8J_Xg7vgMLisHCBU"),
    "TELEGRAM_CHAT_ID": os.getenv("TELEGRAM_CHAT_ID", "-5091693030"),
    "TELEGRAM_TIMEOUT_SEC": 15,
    "TELEGRAM_MAX_MSG_CHARS": 4000,

    # --- Reporting ---
    "REPORT_EVERY_SECONDS": 60,  # every minute

    # --- Per request timeout (best-effort) ---
    # Some google-generativeai versions accept request_options={"timeout": ...}
    "REQUEST_TIMEOUT_SEC": 180,
}


# ============================ GLOBAL SHUTDOWN FLAG ============================

class ShutdownFlag:
    """
    Global flag to signal immediate shutdown on critical errors.
    Any worker thread can set this, and main loop will stop scheduling/cancel futures.
    """
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


# ============================ GLOBAL GEMINI CONFIG ============================

genai.configure(api_key=CONFIG["GEMINI_API_KEY"])


# ============================ RATE LIMITER ============================

class RateLimiter:
    def __init__(self, max_per_sec: float):
        self.interval = 1.0 / max_per_sec if max_per_sec > 0 else 0.0
        self.last_call = 0.0
        self.lock = threading.Lock()

    def wait(self):
        if self.interval <= 0:
            return
        with self.lock:
            now = time.time()
            diff = now - self.last_call
            if diff < self.interval:
                time.sleep(self.interval - diff)
            self.last_call = time.time()

limiter = RateLimiter(CONFIG["RATE_LIMIT_PER_SEC"])


# ============================ TELEGRAM REPORTER ============================

class TelegramReporter:
    def __init__(
        self,
        enabled: bool,
        bot_token: str,
        chat_id: str,
        timeout_sec: int,
        max_chars: int
    ):
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
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start = end
        return chunks

    def send(self, text: str):
        if not self.enabled:
            return
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload_base = {
            "chat_id": self.chat_id,
            "disable_web_page_preview": True,
        }

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
    Tracks token usage and cost.

    Pricing placeholders:
      - input:  $0.50 / 1M tokens
      - output: $3.00 / 1M tokens

    Note:
    - If response.usage_metadata exists, we use it (best).
    - Otherwise we fallback to rough estimates.
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


# ============================ HELPERS ============================

def _clean_text(x: Any) -> str:
    if x is None or pd.isna(x):
        return ""
    return str(x).replace('"', "").replace("\n", " ").strip()

def _strip_code_fences(text: str) -> str:
    t = (text or "").strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[-1]
        if t.endswith("```"):
            t = t[:-3]
    return t.strip()

def _pct(part: int, total: int) -> str:
    if total <= 0:
        return "0.00%"
    return f"{(100.0 * part / total):.2f}%"

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
    df_l1 = pd.DataFrame(
        [{"level_1": k, "count": v} for k, v in level1_counts.most_common()]
    )
    df_l1.to_csv("final_level1_counts.csv", index=False, encoding="utf-8-sig")

    df_l12 = pd.DataFrame(
        [{"level_1": k[0], "level_2": k[1], "count": v} for k, v in level12_counts.most_common()]
    )
    df_l12.to_csv("final_level2_counts.csv", index=False, encoding="utf-8-sig")

def format_progress_message(snap: Dict[str, Any]) -> str:
    processed = snap["processed"]
    success = snap["success"]
    unknown = snap["unknown"]
    error = snap["error"]

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    msg = (
        f"[tf_menu] Minute report @ {ts}\n"
        f"Processed: {processed:,}\n"
        f"✅ Success: {success:,} ({_pct(success, processed)})\n"
        f"❓ Unknown: {unknown:,} ({_pct(unknown, processed)})\n"
        f"❌ Error:   {error:,} ({_pct(error, processed)})\n"
        f"\n[COST]\n{cost_tracker.summary_str()}\n"
    )
    return msg

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


# ============================ CRITICAL + TRANSIENT ERROR DETECTION ============================

def is_critical_gemini_error(err: str) -> bool:
    """
    Critical = stop whole app immediately (your request).
    - 403 errors (permission / forbidden / http2 403)
    - GOAWAY / client_misbehavior / grpc_status:14 + http2_error:11
    """
    e = (err or "").lower()

    if "status: 403" in e or "received http2 header with status: 403" in e:
        return True

    if "goaway received" in e and "client_misbehavior" in e:
        return True

    if "grpc_status:14" in e and "http2_error:11" in e:
        return True

    auth_phrases = [
        "permission denied",
        "authentication",
        "invalid api key",
        "unauthorized",
        "forbidden",
    ]
    if any(p in e for p in auth_phrases):
        return True

    return False

def is_transient_gemini_error(err: str) -> bool:
    """
    Transient = retry with exponential backoff (your request).
    Specifically handle:
    - "504 Stream cancelled; ... status = CANCELLED: Stream cancelled"
    - "504 Deadline expired before operation could complete."
    Also treat common transient conditions:
    - 429
    - 500/502/503/504
    - UNAVAILABLE, CANCELLED, DEADLINE_EXCEEDED (grpc-ish)
    """
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

    # network-ish transient hints
    transient_phrases = [
        "connection reset",
        "connection aborted",
        "timed out",
        "timeout",
        "tls",
        "socket",
        "temporarily unavailable",
        "server closed",
        "broken pipe",
    ]
    if any(p in e for p in transient_phrases):
        return True

    return False

def compute_backoff_seconds(attempt: int) -> float:
    """
    Exponential backoff with jitter, capped.
    attempt is 0-based.
    """
    base = float(CONFIG["BACKOFF_BASE_SEC"])
    cap = float(CONFIG["BACKOFF_MAX_SEC"])
    jitter = float(CONFIG["RETRY_JITTER_SEC"])

    # exponential: base * 2^attempt
    wait = base * (2 ** attempt)

    # add jitter in [0, jitter]
    wait += random.random() * jitter

    # cap
    if wait > cap:
        wait = cap + (random.random() * jitter)

    return max(0.0, wait)


# ============================ CLASSIFIER ENGINE (THREAD-LOCAL MODEL) ============================

class ClassifierEngine:
    """
    Thread-safe approach:
    - Do NOT share a single GenerativeModel instance across threads.
    - Use thread-local storage so each thread has its own model instance.
    - On critical API errors, set shutdown flag and stop the whole app.
    - On transient errors (504 CANCELLED / deadline expired), retry with exponential backoff + timeout.
    """

    def __init__(self, taxonomy_path: str):
        with open(taxonomy_path, "r", encoding="utf-8") as f:
            taxonomy = json.load(f)

        self.taxonomy_str = json.dumps(taxonomy, ensure_ascii=False, separators=(",", ":"))

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

        self._thread_local = threading.local()

    def _get_thread_model(self):
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
        return [
            {"id": p["id"], "level_1": "ERROR", "level_2": "ERROR", "reason": reason}
            for p in products_for_prompt
        ]

    def _validate_and_normalize(
        self, products_for_prompt: List[Dict[str, Any]], parsed: Any
    ) -> List[Dict[str, Any]]:
        id_order = [str(p["id"]) for p in products_for_prompt]

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

    def classify_batch(self, batch_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        - If shutdown_flag is set, do NOT call the API.
        - On critical Gemini errors, set shutdown_flag and raise RuntimeError to stop everything.
        - On transient errors (504 stream cancelled / deadline expired), retry with exponential backoff.
        - Per-request timeout applied best-effort.
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

        user_payload = json.dumps(products_for_prompt, ensure_ascii=False, separators=(",", ":"))
        last_err: Optional[str] = None

        for attempt in range(int(CONFIG["MAX_RETRIES"])):
            if shutdown_flag.is_set():
                raise RuntimeError(f"Shutdown triggered: {shutdown_flag.reason()}")

            try:
                limiter.wait()
                model = self._get_thread_model()

                # Per-request timeout (best-effort)
                # If your installed google-generativeai doesn't support request_options, it will TypeError.
                try:
                    response = model.generate_content(
                        [user_payload],
                        request_options={"timeout": int(CONFIG["REQUEST_TIMEOUT_SEC"])},
                    )
                except TypeError:
                    response = model.generate_content([user_payload])

                text = getattr(response, "text", "") or ""

                # Token usage if available
                in_tokens = 0
                out_tokens = 0
                usage = getattr(response, "usage_metadata", None)
                if usage is not None:
                    in_tokens = int(getattr(usage, "prompt_token_count", 0) or 0)
                    out_tokens = int(getattr(usage, "candidates_token_count", 0) or 0)
                else:
                    # Fallback estimates
                    in_tokens = max(1, len(user_payload) // 3)
                    out_tokens = max(30, len(text) // 4)

                cost_tracker.update(in_tokens, out_tokens)

                cleaned = _strip_code_fences(text)
                parsed = json.loads(cleaned)
                normalized = self._validate_and_normalize(products_for_prompt, parsed)
                return normalized

            except Exception as e:
                last_err = str(e)

                # CRITICAL: stop the entire app
                if is_critical_gemini_error(last_err):
                    shutdown_flag.set(f"Critical Gemini error: {last_err}")
                    raise RuntimeError(shutdown_flag.reason())

                # TRANSIENT: exponential backoff
                if is_transient_gemini_error(last_err):
                    wait_s = compute_backoff_seconds(attempt)
                    # Optional: print a short line so you see it's retrying
                    # (not strictly required, but useful for operations)
                    print(f"[Retry] transient error (attempt {attempt+1}/{CONFIG['MAX_RETRIES']}), sleeping {wait_s:.1f}s: {last_err}")
                    time.sleep(wait_s)
                    continue

                # NON-TRANSIENT, NON-CRITICAL:
                # Still retry, but with exponential backoff as requested (more conservative).
                wait_s = compute_backoff_seconds(attempt)
                print(f"[Retry] non-critical error (attempt {attempt+1}/{CONFIG['MAX_RETRIES']}), sleeping {wait_s:.1f}s: {last_err}")
                time.sleep(wait_s)

        # If we exhausted retries, return ERROR for these items
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

            snap = stats.snapshot()
            telegram.send(format_progress_message(snap))


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
        print(f"Failed to load taxonomy: {e}")
        telegram.send(f"[tf_menu] FAILED to load taxonomy: {e}")
        return

    # Start minute reporter
    stop_event = threading.Event()
    reporter = MinuteReporter(stop_event=stop_event, interval_sec=CONFIG["REPORT_EVERY_SECONDS"])
    reporter.start()

    telegram.send(
        f"[tf_menu] Started.\n"
        f"Model: {CONFIG['MODEL_NAME']}\n"
        f"Remaining: {remaining:,}\n"
        f"Workers: {CONFIG['MAX_WORKERS']} | Batch: {CONFIG['BATCH_SIZE']} | RPS: {CONFIG['RATE_LIMIT_PER_SEC']}\n"
        f"Reporting: every {CONFIG['REPORT_EVERY_SECONDS']} seconds\n"
        f"Retries: {CONFIG['MAX_RETRIES']} | Timeout: {CONFIG['REQUEST_TIMEOUT_SEC']}s\n"
        f"\nCritical errors (403 / GOAWAY client_misbehavior) will stop the app immediately.\n"
        f"Transient 504/Cancelled/DeadlineExpired will be retried with exponential backoff.\n"
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
    print(f"\nProcessing {remaining:,} items in {total_batches:,} batches...")
    pbar = tqdm(total=total_batches, unit="batch")

    try:
        with ThreadPoolExecutor(max_workers=int(CONFIG["MAX_WORKERS"])) as executor:
            futures_map: Dict[Any, List[Dict[str, Any]]] = {}

            # Prime pipeline
            for _ in range(max_in_flight):
                if not submit_next(executor, futures_map):
                    break

            while futures_map:
                if shutdown_flag.is_set():
                    break

                for fut in as_completed(list(futures_map.keys())):
                    batch_input = futures_map.pop(fut)
                    pbar.update(1)

                    if shutdown_flag.is_set():
                        break

                    try:
                        results = fut.result()  # may raise RuntimeError if shutdown triggered
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

                        # Append to output file immediately
                        pd.DataFrame(processed_rows).to_csv(
                            OUTPUT_FILE, mode="a", header=False, index=False, encoding="utf-8-sig"
                        )

                        stats.update_from_rows(processed_rows)

                    except RuntimeError as e:
                        shutdown_flag.set(str(e))
                        break

                    except Exception as e:
                        msg = f"Error in main loop batch handling: {e}"
                        print(msg)
                        telegram.send(f"[tf_menu] {msg}")

                    if shutdown_flag.is_set():
                        break

                    # Refill pipeline
                    while len(futures_map) < max_in_flight:
                        if not submit_next(executor, futures_map):
                            break

                    # Break to refresh as_completed list
                    break

            # If shutdown triggered, cancel remaining futures
            if shutdown_flag.is_set():
                print("\n🚨 SHUTDOWN TRIGGERED. Cancelling remaining in-flight requests...")
                telegram.send(format_shutdown_message(shutdown_flag.reason()))
                for f in list(futures_map.keys()):
                    f.cancel()

    finally:
        pbar.close()
        stop_event.set()

    # Force stop if shutdown triggered
    if shutdown_flag.is_set():
        print("\n═══════════════════════════════════════════════════════════")
        print("🚨 APPLICATION STOPPED DUE TO CRITICAL GEMINI ERROR")
        print("Reason:", shutdown_flag.reason())
        print("Progress:", stats.snapshot())
        print("Cost:", cost_tracker.summary_str())
        print("═══════════════════════════════════════════════════════════")
        sys.exit(1)

    # Normal finalize
    print("\nProcessing Complete.")
    snap = stats.snapshot()
    save_final_reports(snap["level1_counts"], snap["level12_counts"])
    telegram.send(format_final_message(snap))
    print(cost_tracker.summary_str())
    print("Saved final reports: final_level1_counts.csv, final_level2_counts.csv")


if __name__ == "__main__":
    main()
