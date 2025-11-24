import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
from build_prompt import build_skipcheck_prompt, build_trans_prompt
import os
from time import time
import openai
import sys
from datetime import datetime
import random
import time as _t

openai.api_key = os.getenv("OPENAI_API_KEY")


def _looks_like_html(s: str) -> bool:
    if not isinstance(s, str):
        return False
    head = s.lstrip()[:30].lower()
    return head.startswith("<!doctype html") or head.startswith("<html")


def safe_chat(messages, model, max_attempts=6, base_sleep=1.0, timeout=60):
    for attempt in range(1, max_attempts + 1):
        try:
            resp = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                timeout=timeout,
            )
            choices = resp.get("choices") or []
            if not choices or "message" not in choices[0]:
                raise ValueError("malformed response")
            content = choices[0]["message"].get("content", "")
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
                _t.sleep(sleep_s)
                continue

            if attempt == max_attempts:
                return None
            sleep_s = min(base_sleep * (2 ** (attempt - 1)), 30.0) + random.uniform(
                0, 0.5
            )
            _t.sleep(sleep_s)
            continue


def skipcheck_gpt(sys, user) -> str:
    out = safe_chat([sys, user], model="gpt-5", max_attempts=6, timeout=60)
    return out if out is not None else "__ERROR__"


def trans_gpt(sys, user) -> str:
    out = safe_chat([sys, user], model="gpt-5-mini", max_attempts=6, timeout=60)
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


def process_excel(
    input_excel_path: str,
    output_excel_path: str,
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

        for idx in target_indices:
            src_lang = df_all.at[idx, "Src language"]
            origin_text = df_all.at[idx, "Origin"]

            sys_prompt, user_prompt = build_skipcheck_prompt(src_lang, origin_text)
            s = time()
            resp1 = skipcheck_gpt(sys_prompt, user_prompt)
            e = time()

            if resp1 == "__ERROR__":
                print(f"row {idx}: skipcheck_gpt failed; Origin: {origin_text}")
                continue

            df_all.at[idx, "Check boxes"] = resp1
            success_cnt += 1
            print(
                f"row {idx}: Origin: {origin_text}, "
                f"Check boxes={resp1}, 처리시간={e - s:.2f}초"
            )

            if success_cnt % 30 == 0:
                chk_path = out_dir / f"{base_stem}_chk{success_cnt}{ext}"
                df_all.to_excel(chk_path, index=False)
                print(f"[checkpoint] saved: {chk_path}")

            check_val = resp1 if isinstance(resp1, str) else str(resp1)
            needs_translation = any(
                kw in check_val for kw in ["noneApply", "typos", "nonsensical", "multiLang"]
            )
            if not needs_translation:
                continue
            if not is_double:
                tgt_lang = df_all.at[idx, "Tgt language"]
                sys_prompt_tr, user_prompt_tr = build_trans_prompt(tgt_lang, origin_text)
                ss = time()
                resp2 = trans_gpt(sys_prompt_tr, user_prompt_tr)
                ee = time()

                if resp2 == "__ERROR__":
                    print(
                        f"row {idx}: trans_gpt failed; "
                        f"Origin: {origin_text}, Check boxes={check_val}"
                    )
                    continue

                df_all.at[idx, "Translation"] = resp2
                print(
                    f"row {idx}: Origin: {origin_text}, "
                    f"Check boxes={check_val}, Translation={resp2}, "
                    f"처리시간={ee - ss:.2f}초"
                )
                print(
                    "-----------------------------------------------------------------------------------------------------------"
                )

            else:
                sys_prompt1, user_prompt1 = build_trans_prompt("en_US", origin_text)
                ss1 = time()
                resp_tr1 = trans_gpt(sys_prompt1, user_prompt1)
                ee1 = time()

                if resp_tr1 == "__ERROR__":
                    print(
                        f"row {idx}: trans_gpt(en_US) failed; "
                        f"Origin: {origin_text}, Check boxes={check_val}"
                    )
                    continue

                df_all.at[idx, "Translation1"] = resp_tr1

                sys_prompt2, user_prompt2 = build_trans_prompt("en_GB", origin_text)
                ss2 = time()
                resp_tr2 = trans_gpt(sys_prompt2, user_prompt2)
                ee2 = time()

                if resp_tr2 == "__ERROR__":  
                    print(
                        f"row {idx}: trans_gpt(en_GB) failed; "
                        f"Origin: {origin_text}, Check boxes={check_val}, "
                        f"Translation1(en_US)={resp_tr1}"
                    )
                    continue

                df_all.at[idx, "Translation2"] = resp_tr2

                print(
                    f"row {idx}: Origin: {origin_text}, "
                    f"Check boxes={check_val}, "
                    f"Translation1(en_US)={resp_tr1}, "
                    f"Translation2(en_GB)={resp_tr2}, "
                    f"처리시간1={ee1 - ss1:.2f}초, 처리시간2={ee2 - ss2:.2f}초"
                )
                print(
                    "-----------------------------------------------------------------------------------------------------------"
                )

        df_all.to_excel(output_path, index=False)
        print(f"[final] saved: {output_path}")

    finally:
        sys.stdout = _old_stdout
        tee.close()


if __name__ == "__main__":
    INPUT_PATH = "/mnt/c/Users/Flitto/Documents/NAC/HT/data/HT test_NAC_5169_ja_JP-en_US_HT_34815_251112_120835_LLM 요청 파일.xlsx"
    OUTPUT_PATH = "/mnt/c/Users/Flitto/Documents/NAC/HT/data/test_NAC_5169_ja_JP-en_US_HT_34815_251112_120835_LLM 요청 파일.xlsx"
    
    process_excel(input_excel_path=INPUT_PATH, output_excel_path=OUTPUT_PATH)
