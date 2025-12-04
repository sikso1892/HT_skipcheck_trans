import asyncio
import os
import random
import sys
import dotenv

from datetime import datetime
from pathlib import Path
from time import time
from typing import Optional

import numpy as np
import pandas as pd
from openai import AsyncOpenAI
from tqdm import tqdm

from build_prompt import build_skipcheck_prompt, build_trans_prompt

dotenv.load_dotenv()

async_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

ASYNC_CONCURRENCY = os.getenv("ASYNC_CONCURRENCY", "5")
LOG_SEPARATOR = "-" * 107
NEEDS_TRANSLATION_KEYWORDS = ("noneApply", "typos", "nonsensical", "multiLang")


def _looks_like_html(s: str) -> bool:
    if not isinstance(s, str):
        return False
    head = s.lstrip()[:30].lower()
    return head.startswith("<!doctype html") or head.startswith("<html")


async def safe_chat(messages, model, max_attempts=6, base_sleep=1.0, timeout=60):
    if "openrouter" in str(async_client.base_url):
        model = "openai/" + model

    for attempt in range(1, max_attempts + 1):
        try:
            resp = await async_client.chat.completions.create(
                model=model,
                messages=messages,
                timeout=timeout,
            )
            choices = resp.choices or []
            if not choices or hasattr(choices[0], "message") is False:
                raise ValueError("malformed response")
            content = choices[0].message.content
            if isinstance(content, list):
                content = "".join(
                    getattr(part, "text", "")
                    if hasattr(part, "text")
                    else part.get("text", "")
                    if isinstance(part, dict)
                    else ""
                    for part in content
                )
            if not isinstance(content, str) or not content.strip():
                raise ValueError("empty content")
            return content.strip()
        except Exception as e:
            msg = str(e)
            if (
                _looks_like_html(msg)
                or "HTTP code 520" in msg
                or "5xx" in msg
                or "Timeout" in msg
            ):
                if attempt == max_attempts:
                    return None
                sleep_s = min(base_sleep * (2 ** (attempt - 1)), 30.0) + random.uniform(
                    0, 0.5
                )
                await asyncio.sleep(sleep_s)
                continue

            if attempt == max_attempts:
                return None
            sleep_s = min(base_sleep * (2 ** (attempt - 1)), 30.0) + random.uniform(
                0, 0.5
            )
            await asyncio.sleep(sleep_s)
            continue


async def skipcheck_gpt(sys, user) -> str:
    out = await safe_chat([sys, user], model="gpt-5.1", max_attempts=6, timeout=60)
    return out if out is not None else "__ERROR__"


async def trans_gpt(sys, user) -> str:
    out = await safe_chat([sys, user], model="gpt-5-mini", max_attempts=6, timeout=60)
    return out if out is not None else "__ERROR__"


def is_empty_translation(val: Optional[str]) -> bool:
    if isinstance(val, float):
        if np.isnan(val):
            return True
    if isinstance(val, str):
        if val.strip() == "":
            return True
    if val is None:
        return True
    return False


class Tee:
    def __init__(self, filename: str, mode: str = "a"):
        self.file = open(filename, mode, encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, message: str):
        self.stdout.write(message)
        self.stdout.flush()
        self.file.write(message)
        self.file.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def close(self):
        self.file.close()


async def _process_row(idx: int, df_all: pd.DataFrame, is_double: bool) -> dict:
    logs: list[str] = []
    updates: dict[str, str] = {}

    src_lang = df_all.at[idx, "Src language"]
    origin_text = df_all.at[idx, "Origin"]

    sys_prompt, user_prompt = build_skipcheck_prompt(src_lang, origin_text)
    start = time()
    resp1 = await skipcheck_gpt(sys_prompt, user_prompt)
    elapsed = time() - start

    if resp1 == "__ERROR__":
        logs.append(f"row {idx}: skipcheck_gpt failed; Origin: {origin_text}")
        return {"idx": idx, "success": False, "updates": updates, "logs": logs}

    updates["Check boxes"] = resp1
    logs.append(
        f"row {idx}: Origin: {origin_text}, Check boxes={resp1}, 처리시간={elapsed:.2f}초"
    )

    check_val = resp1 if isinstance(resp1, str) else str(resp1)
    needs_translation = any(kw in check_val for kw in NEEDS_TRANSLATION_KEYWORDS)

    if not needs_translation:
        return {"idx": idx, "success": True, "updates": updates, "logs": logs}

    if not is_double:
        tgt_lang = df_all.at[idx, "Tgt language"]
        sys_prompt_tr, user_prompt_tr = build_trans_prompt(tgt_lang, origin_text)
        tr_start = time()
        resp2 = await trans_gpt(sys_prompt_tr, user_prompt_tr)
        tr_elapsed = time() - tr_start

        if resp2 == "__ERROR__":
            logs.append(
                f"row {idx}: trans_gpt failed; Origin: {origin_text}, Check boxes={check_val}"
            )
            return {"idx": idx, "success": True, "updates": updates, "logs": logs}

        updates["Translation"] = resp2
        logs.append(
            f"row {idx}: Origin: {origin_text}, Check boxes={check_val}, "
            f"Translation={resp2}, 처리시간={tr_elapsed:.2f}초"
        )
        logs.append(LOG_SEPARATOR)
        return {"idx": idx, "success": True, "updates": updates, "logs": logs}

    sys_prompt1, user_prompt1 = build_trans_prompt("en_US", origin_text)
    tr1_start = time()
    resp_tr1 = await trans_gpt(sys_prompt1, user_prompt1)
    tr1_elapsed = time() - tr1_start

    if resp_tr1 == "__ERROR__":
        logs.append(
            f"row {idx}: trans_gpt(en_US) failed; Origin: {origin_text}, "
            f"Check boxes={check_val}"
        )
        return {"idx": idx, "success": True, "updates": updates, "logs": logs}

    updates["Translation1"] = resp_tr1

    sys_prompt2, user_prompt2 = build_trans_prompt("en_GB", origin_text)
    tr2_start = time()
    resp_tr2 = await trans_gpt(sys_prompt2, user_prompt2)
    tr2_elapsed = time() - tr2_start

    if resp_tr2 == "__ERROR__":
        logs.append(
            f"row {idx}: trans_gpt(en_GB) failed; Origin: {origin_text}, "
            f"Check boxes={check_val}, Translation1(en_US)={resp_tr1}"
        )
        return {"idx": idx, "success": True, "updates": updates, "logs": logs}

    updates["Translation2"] = resp_tr2
    logs.append(
        f"row {idx}: Origin: {origin_text}, Check boxes={check_val}, "
        f"Translation1(en_US)={resp_tr1}, Translation2(en_GB)={resp_tr2}, "
        f"처리시간1={tr1_elapsed:.2f}초, 처리시간2={tr2_elapsed:.2f}초"
    )
    logs.append(LOG_SEPARATOR)

    return {"idx": idx, "success": True, "updates": updates, "logs": logs}


async def process_excel(
    input_excel_path: str,
    output_excel_path: str,
    concurrency: int = 5,
):
    log_filename = f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    log_path = Path(output_excel_path).parent / log_filename
    tee = Tee(str(log_path))
    _old_stdout = sys.stdout
    sys.stdout = tee

    try:
        input_path = Path(input_excel_path)
        output_path = Path(output_excel_path)

        df_all = pd.read_excel(input_path)
        df_all.columns = [str(c).strip() for c in df_all.columns]

        first_tgt = None
        if "Tgt language" in df_all.columns and not df_all.empty:
            first_idx = df_all.index[0]
            first_tgt = str(df_all.at[first_idx, "Tgt language"])

        is_double = input_path.name.startswith("HT test") and (first_tgt == "en_US")

        df_all["Check boxes"] = df_all["Check boxes"].astype("object")

        if not is_double:
            df_all["Translation"] = df_all["Translation"].astype("object")

            required_cols = [
                "status",
                "Translation",
                "Src language",
                "Origin",
                "Check boxes",
                "Tgt language",
            ]
            for col in required_cols:
                if col not in df_all.columns:
                    raise ValueError(f"필수 컬럼 '{col}' 이(가) 엑셀에 없습니다.")

            cond_status = df_all["status"] == "In progress OR No participation"
            cond_translation_empty = df_all["Translation"].apply(is_empty_translation)
            target_mask = cond_status & cond_translation_empty

        else:
            df_all["Translation1"] = df_all["Translation1"].astype("object")
            df_all["Translation2"] = df_all["Translation2"].astype("object")

            required_cols = [
                "status",
                "Translation1",
                "Translation2",
                "Src language",
                "Origin",
                "Check boxes",
                "Tgt language",
            ]
            for col in required_cols:
                if col not in df_all.columns:
                    raise ValueError(f"필수 컬럼 '{col}' 이(가) 엑셀에 없습니다.")

            cond_status = df_all["status"] == "In progress OR No participation"
            cond_tr1_empty = df_all["Translation1"].apply(is_empty_translation)
            cond_tr2_empty = df_all["Translation2"].apply(is_empty_translation)
            cond_translation_empty = cond_tr1_empty & cond_tr2_empty
            target_mask = cond_status & cond_translation_empty

        target_indices = df_all.index[target_mask].tolist()

        success_cnt = 0
        base_stem, ext = output_path.stem, output_path.suffix
        out_dir = output_path.parent
        if target_indices:
            concurrency = max(1, concurrency)
            semaphore = asyncio.Semaphore(concurrency)

            async def worker(idx: int) -> dict:
                async with semaphore:
                    return await _process_row(idx, df_all, is_double)

            tasks = [asyncio.create_task(worker(idx)) for idx in target_indices]
            with tqdm(total=len(tasks)) as progress:
                for coro in asyncio.as_completed(tasks):
                    result = await coro
                    progress.update(1)
                    idx = result["idx"]

                    for column, value in result["updates"].items():
                        df_all.at[idx, column] = value

                    for line in result["logs"]:
                        print(line)

                    if result["success"]:
                        success_cnt += 1
                        if success_cnt % 30 == 0:
                            chk_path = out_dir / f"{base_stem}_chk{success_cnt}{ext}"
                            df_all.to_excel(chk_path, index=False)
                            print(f"[checkpoint] saved: {chk_path}")
        else:
            print("No rows require processing.")

        df_all.to_excel(output_path, index=False)
        print(f"[final] saved: {output_path}")

    finally:
        sys.stdout = _old_stdout
        tee.close()


if __name__ == "__main__":
    INPUT_PATH = "data/NAC_500_samples.xlsx"
    OUTPUT_PATH = "data/results/NAC_500_samples_results.xlsx"

    print(
        f"Computing with ASYNC_CONCURRENCY: {ASYNC_CONCURRENCY}",
    )

    asyncio.run(
        process_excel(input_excel_path=INPUT_PATH, output_excel_path=OUTPUT_PATH)
    )
