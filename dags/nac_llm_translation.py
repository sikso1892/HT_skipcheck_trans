import asyncio
import json
import os
import io
import typing
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import logging
import dotenv
import numpy as np
import pandas as pd
import pendulum
from airflow import DAG
from airflow.exceptions import AirflowException
from airflow.sdk import task, get_current_context, ObjectStoragePath
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from openai import AsyncOpenAI

from batch.build_prompt import build_skipcheck_prompt, build_trans_prompt

if typing.TYPE_CHECKING:
    from airflow.sdk import TaskInstanceState

dotenv.load_dotenv()

logger = logging.getLogger(__name__)

BATCH_ENDPOINT = "/v1/chat/completions"
BATCH_LIMIT = 5000
NEEDS_TRANSLATION_KEYWORDS = ("noneApply", "typos", "nonsensical", "multiLang")

dag = DAG(
    dag_id="nac_llm_translation_batch",
    schedule=None,
    start_date=pendulum.datetime(2024, 1, 1, tz="Asia/Seoul"),
    catchup=False,
    tags=["nac", "llm", "batch"],
    params={
        "input_path": "file:///mnt/data/home/moonsik/projects/HT_skipcheck_trans/data/nac_llm_translation_batch/external_requets/NAC_500_samples.xlsx",
        "batch_size": BATCH_LIMIT,
    },
)

ARTIFACT_BASE = ObjectStoragePath(
    f"file:///mnt/data/home/moonsik/projects/HT_skipcheck_trans/data/{dag.dag_id}/dags-artifacts/"
)


def as_optional_str(val: Any) -> str | None:
    if val is None:
        return None
    if isinstance(val, float) and np.isnan(val):
        return None
    return str(val)


def is_empty_translation(val: Any) -> bool:
    if isinstance(val, float):
        if np.isnan(val):
            return True
    if isinstance(val, str):
        if val.strip() == "":
            return True
    return val is None


def chunked(
    data: Sequence[Dict[str, Any]], chunk_size: int
) -> Iterable[Tuple[int, List[Dict[str, Any]]]]:
    for start in range(0, len(data), chunk_size):
        yield start, list(data[start : start + chunk_size])


def resolve_model_name(model: str, base_url: str | None) -> str:
    if base_url and "openrouter" in base_url.lower():
        if not model.startswith("openai/"):
            return f"openai/{model}"
    return model


def _artifact_base_for_ti(ti: "TaskInstanceState") -> ObjectStoragePath:
    map_index = getattr(ti, "map_index", -1)
    map_index_str = "0" if map_index is None or map_index < 0 else str(map_index)
    
    artifact_base: "ObjectStoragePath" = (
        ARTIFACT_BASE / ti.run_id / ti.task_id / map_index_str
    )
    artifact_base.mkdir(exist_ok=True, parents=True)
    return artifact_base


def _save_text_artifact(filename: str, content: str) -> str:
    try:
        ctx = get_current_context()
        ti: "TaskInstanceState" = ctx["ti"]
        base = _artifact_base_for_ti(ti)
    except Exception:
        base = ARTIFACT_BASE / "no_context"

    path = base / filename
    path.parent.mkdir(exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(content)
    return str(path)


def _save_json_artifact_for_ti(
    ti: "TaskInstanceState",
    filename: str,
    data: Dict[str, Any],
) -> str:
    base = _artifact_base_for_ti(ti)
    path = base / filename
    path.parent.mkdir(exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return str(path)


def _load_json_artifact_for_ti(
    ti: "TaskInstanceState",
    filename: str,
) -> Dict[str, Any] | None:
    base = _artifact_base_for_ti(ti)
    path = base / filename
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except Exception as e:
        logger.warning(
            "Failed to load JSON artifact %s for task %s.%s (run_id=%s): %s",
            filename,
            getattr(ti, "dag_id", "?"),
            getattr(ti, "task_id", "?"),
            getattr(ti, "run_id", "?"),
            e,
        )
        return None


def _load_existing_batch_id_from_artifact(
    ti: "TaskInstanceState",
    kind: str,
) -> str | None:
    """아티팩트에서 batch_id 로드 (kind: 'skip' | 'trans')."""
    info = _load_json_artifact_for_ti(ti, f"{kind}_batch_info.json")
    if not isinstance(info, dict):
        return None
    batch_id = info.get("batch_id")
    if isinstance(batch_id, str) and batch_id:
        return batch_id
    return None


def _serialize_for_json(obj: Any) -> Any:
    
    try:
        if hasattr(obj, "model_dump_json"):
            return json.loads(obj.model_dump_json())
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
    except Exception:
        pass

    try:
        json.dumps(obj)
        return obj
    except TypeError:
        # 어쩔 수 없으면 repr 문자열로 보존
        return {"repr": repr(obj)}


async def submit_batch_requests(
    client: AsyncOpenAI,
    requests: Sequence[Dict[str, Any]],
    completion_window: str,
) -> Dict[str, Any]:
    body = "\n".join(json.dumps(item, ensure_ascii=False) for item in requests)
    fp = io.BytesIO(body.encode("utf-8"))

    upload = await client.files.create(file=("batch.jsonl", fp), purpose="batch")

    batch = await client.batches.create(
        input_file_id=upload.id,
        endpoint=BATCH_ENDPOINT,
        completion_window=completion_window,
    )

    return {
        "file_id": upload.id,
        "batch_id": batch.id,
        "status": batch.status,
    }


async def _download_batch_output(
    client: AsyncOpenAI,
    file_id: str,
    prefix: str,
) -> Tuple[str, List[Dict[str, Any]]]:
    resp = await client.files.content(file_id)
    if isinstance(resp, (bytes, bytearray)):
        data = resp
    elif hasattr(resp, "read"):
        maybe_coroutine = resp.read()
        if asyncio.iscoroutine(maybe_coroutine):
            data = await maybe_coroutine
        else:
            data = maybe_coroutine
    else:
        # 예상치 못한 타입인 경우 문자열로 변환
        data = str(resp).encode("utf-8")
    text = data.decode("utf-8")
    filename = f"{prefix}_{file_id}.jsonl"
    artifact_path = _save_text_artifact(filename, text)
    records = [json.loads(line) for line in text.splitlines() if line.strip()]
    return artifact_path, records


def _extract_skipcheck_value(rec: Dict[str, Any]) -> str | None:
    body = rec.get("response", {}).get("body", {})
    choices = body.get("choices") or []
    if not choices:
        return None
    msg = choices[0].get("message", {})
    content = msg.get("content")
    if isinstance(content, list):
        parts: List[str] = []
        for part in content:
            if isinstance(part, dict) and "text" in part:
                parts.append(str(part["text"]))
        content = "".join(parts)
    if isinstance(content, str):
        content = content.strip()
        return content or None
    return None


def _extract_translation_value(rec: Dict[str, Any]) -> str | None:
    return _extract_skipcheck_value(rec)


async def wait_all_batches(
    batches: List[Dict[str, Any]],
    base_url: str | None,
    poll_interval: int = 10,
) -> List[Dict[str, Any]]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY 가 설정되어야 batch 상태를 조회할 수 있습니다.")

    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    state_map: Dict[str, Dict[str, Any]] = {b["batch_id"]: dict(b) for b in batches}

    while True:
        all_done = True
        for batch_id, info in state_map.items():
            if info.get("final_status") in {"completed", "failed", "cancelled"}:
                continue
            batch = await client.batches.retrieve(batch_id)
            state = batch.status
            if state in {"completed", "failed", "cancelled"}:
                info["final_status"] = state
                info["output_file_id"] = getattr(batch, "output_file_id", None)
                info["error_file_id"] = getattr(batch, "error_file_id", None)
            else:
                all_done = False

        if all_done:
            await client.close()
            return list(state_map.values())

        await asyncio.sleep(poll_interval)


def determine_targets(df: pd.DataFrame) -> Tuple[List[int], bool]:
    input_name = getattr(df, "_input_name", "")
    first_tgt = None
    if "Tgt language" in df.columns and not df.empty:
        first_idx = df.index[0]
        first_tgt = str(df.at[first_idx, "Tgt language"])

    is_double = input_name.startswith("HT test") and first_tgt == "en_US"

    required_common = [
        "status",
        "Src language",
        "Origin",
        "Check boxes",
        "Tgt language",
    ]
    for col in required_common:
        if col not in df.columns:
            raise ValueError(f"필수 컬럼 '{col}' 이(가) 엑셀에 없습니다.")

    df["Check boxes"] = df["Check boxes"].astype("object")

    if not is_double:
        if "Translation" not in df.columns:
            raise ValueError("필수 컬럼 'Translation' 이(가) 엑셀에 없습니다.")
        df["Translation"] = df["Translation"].astype("object")
        cond_translation_empty = df["Translation"].apply(is_empty_translation)
        target_mask = cond_translation_empty
    else:
        for col in ("Translation1", "Translation2"):
            if col not in df.columns:
                raise ValueError(f"필수 컬럼 '{col}' 이(가) 엑셀에 없습니다.")
        df["Translation1"] = df["Translation1"].astype("object")
        df["Translation2"] = df["Translation2"].astype("object")
        cond_tr1_empty = df["Translation1"].apply(is_empty_translation)
        cond_tr2_empty = df["Translation2"].apply(is_empty_translation)
        target_mask = cond_tr1_empty & cond_tr2_empty

    cond_status = df["status"].str.contains("In progress OR No participation")
    targets = df.index[(cond_status & target_mask)].tolist()
    return targets, is_double


def needs_translation_from_check(check_val: Any) -> bool:
    if not isinstance(check_val, str):
        check_val = "" if check_val is None else str(check_val)
    return any(kw in check_val for kw in NEEDS_TRANSLATION_KEYWORDS)


def determine_translation_targets(df: pd.DataFrame) -> Tuple[List[int], bool]:
    base_targets, is_double = determine_targets(df)
    cond_needs_tr = df["Check boxes"].apply(needs_translation_from_check)
    mask = df.index.isin(base_targets) & cond_needs_tr
    targets = df.index[mask].tolist()
    return targets, is_double


def build_skipcheck_requests(
    df: pd.DataFrame,
    target_indices: Sequence[int],
    model: str,
    prefix: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    requests: List[Dict[str, Any]] = []
    metadata: List[Dict[str, Any]] = []

    for counter, idx in enumerate(target_indices, start=1):
        src_lang = str(df.at[idx, "Src language"])
        origin_raw = df.at[idx, "Origin"]
        origin_text = "" if pd.isna(origin_raw) else str(origin_raw)
        tgt_lang = (
            as_optional_str(df.at[idx, "Tgt language"])
            if "Tgt language" in df.columns
            else None
        )

        sys_prompt, user_prompt = build_skipcheck_prompt(src_lang, origin_text)
        custom_id = f"{prefix}:{idx}"

        requests.append(
            {
                "custom_id": custom_id,
                "method": "POST",
                "url": BATCH_ENDPOINT,
                "body": {
                    "model": model,
                    "messages": [sys_prompt, user_prompt],
                },
            }
        )

        metadata.append(
            {
                "custom_id": custom_id,
                "row_index": int(idx),
                "src_language": src_lang,
                "tgt_language": tgt_lang,
                "origin": origin_text,
                "seq": counter,
            }
        )

    return requests, metadata


def build_translate_requests(
    df: pd.DataFrame,
    target_indices: Sequence[int],
    model: str,
    prefix: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    requests: List[Dict[str, Any]] = []
    metadata: List[Dict[str, Any]] = []

    for counter, idx in enumerate(target_indices, start=1):
        src_lang = str(df.at[idx, "Src language"])
        tgt_lang = str(df.at[idx, "Tgt language"])
        origin_raw = df.at[idx, "Origin"]
        origin_text = "" if pd.isna(origin_raw) else str(origin_raw)

        sys_prompt, user_prompt = build_trans_prompt(tgt_lang, origin_text)
        custom_id = f"{prefix}:{idx}"

        requests.append(
            {
                "custom_id": custom_id,
                "method": "POST",
                "url": BATCH_ENDPOINT,
                "body": {
                    "model": model,
                    "messages": [sys_prompt, user_prompt],
                },
            }
        )

        metadata.append(
            {
                "custom_id": custom_id,
                "row_index": int(idx),
                "src_language": src_lang,
                "tgt_language": tgt_lang,
                "origin": origin_text,
                "seq": counter,
            }
        )

    return requests, metadata


def load_dataframe(path: Any) -> pd.DataFrame:
    if isinstance(path, ObjectStoragePath):
        name = path.name
        suffix = Path(name).suffix.lower()
        if suffix in {".xlsx", ".xls"}:
            with path.open("rb") as f:
                df = pd.read_excel(f)
        elif suffix == ".csv":
            with path.open("rb") as f:
                df = pd.read_csv(f)
        elif suffix in {".tsv", ".txt"}:
            with path.open("rb") as f:
                df = pd.read_csv(f, sep="\t")
        else:
            raise ValueError(f"지원하지 않는 파일 형식입니다: {path}")
        df._input_name = name  # type: ignore[attr-defined]
    else:
        p = Path(str(path))
        suffix = p.suffix.lower()
        if suffix in {".xlsx", ".xls"}:
            df = pd.read_excel(p)
        elif suffix == ".csv":
            df = pd.read_csv(p)
        elif suffix in {".tsv", ".txt"}:
            df = pd.read_csv(p, sep="\t")
        else:
            raise ValueError(f"지원하지 않는 파일 형식입니다: {p}")
        df._input_name = p.name  # type: ignore[attr-defined]

    df.columns = [str(c).strip() for c in df.columns]
    return df


@task
def create_skipcheck_batches() -> List[Dict[str, Any]]:
    ctx = get_current_context()
    params: Dict[str, Any] = ctx.get("params", {}) if ctx else {}
    ti = ctx["ti"] if ctx else None

    input_path_param = params.get("input_path", "")
    if isinstance(input_path_param, ObjectStoragePath):
        input_path: Any = input_path_param
    else:
        s = str(input_path_param)
        if "://" in s:
            url = s
        else:
            url = Path(s).absolute().as_uri()
        input_path = ObjectStoragePath(url)

    prefix = os.getenv("BATCH_PREFIX", "skipcheck")
    model = os.getenv("SKIPCHECK_MODEL", "gpt-5.1")
    base_url = os.getenv("OPENAI_BASE_URL")

    raw_batch_size = params.get("batch_size")
    if raw_batch_size is None:
        env_batch_size = os.getenv("BATCH_MAX_REQUESTS")
        if env_batch_size is not None:
            raw_batch_size = int(env_batch_size)
        else:
            raw_batch_size = BATCH_LIMIT
    else:
        raw_batch_size = int(raw_batch_size)

    df = load_dataframe(input_path)
    target_indices, is_double = determine_targets(df)

    if not target_indices:
        return []

    effective_model = resolve_model_name(model, base_url)
    requests, metadata = build_skipcheck_requests(
        df,
        target_indices,
        effective_model,
        prefix,
    )

    chunk_size = min(int(raw_batch_size), BATCH_LIMIT)
    summary: List[Dict[str, Any]] = []

    for chunk_idx, (start, chunk) in enumerate(
        chunked(requests, chunk_size), start=1
    ):
        chunk_meta = metadata[start : start + len(chunk)]

        jsonl_body = "\n".join(
            json.dumps(item, ensure_ascii=False) for item in chunk
        )

        payload: Dict[str, Any] = {
            "requests": chunk,
            "jsonl": jsonl_body,
            "metadata": chunk_meta,
            "is_double": is_double,
        }

        if ti is not None:
            base = _artifact_base_for_ti(ti)
        else:
            base = ARTIFACT_BASE / "no_context"

        payload_path = base / f"skip_payload_chunk{chunk_idx:03d}.json"

        payload_path.parent.mkdir(exist_ok=True)
        with payload_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)

        info: Dict[str, Any] = {
            "chunk_index": chunk_idx,
            "size": len(chunk),
            "payload": payload_path,
        }

        summary.append(info)

    return summary


@task(retries=240, retry_delay=pendulum.duration(seconds=60))
def process_chunk(chunk: Dict[str, Any]) -> Dict[str, Any]:
    base_url = os.getenv("OPENAI_BASE_URL")
    api_key = os.getenv("OPENAI_API_KEY")
    completion_window = "24h"
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY 가 설정되어야 batch 를 호출할 수 있습니다.")

    ctx = get_current_context()
    ti: "TaskInstanceState" = ctx["ti"]

    chunk_index = int(chunk.get("chunk_index", -1))

    payload_info = chunk.get("payload")

    map_index = getattr(ti, "map_index", -1)

    # dag-artifacts 에 저장된 batch_id 만 사용 (XCom 은 사용하지 않음)
    existing_skip_batch_id = _load_existing_batch_id_from_artifact(ti, "skip")
    existing_trans_batch_id = _load_existing_batch_id_from_artifact(ti, "trans")

    if isinstance(payload_info, ObjectStoragePath):
        with payload_info.open() as f:
            payload = json.load(f)
        producer_task_id = "create_skipcheck_batches"
        producer_map_index = getattr(ti, "map_index", -1)
        xcom_key = None
    else:
        payload_info = payload_info or {}
        payload_format = payload_info.get("format")

        if payload_format == "s3_json":
            bucket = payload_info.get("bucket")
            key = payload_info.get("key")
            if not bucket or not key:
                raise ValueError("S3 payload 정보에는 bucket 과 key 가 필요합니다.")
            s3_hook = S3Hook(aws_conn_id="aws_default")
            payload_str = s3_hook.read_key(key=key, bucket_name=bucket)
            payload = json.loads(payload_str)
            producer_task_id = "create_skipcheck_batches"
            producer_map_index = -1
            xcom_key = None
        elif payload_format == "dict":
            raise ValueError(
                "payload_format 'dict' 는 XCom 기반이므로, 현재 DAG 설정(ObjectStorage-only)에서는 지원하지 않습니다."
            )
        else:
            raise ValueError(f"지원하지 않는 payload format 입니다: {payload_format}")

    requests = payload.get("requests") or []
    jsonl_body = payload.get("jsonl")
    metadata = payload.get("metadata") or []
    is_double = bool(payload.get("is_double", False))

    logger.info(
        "process_chunk start: dag_id=%s task_id=%s chunk_index=%s xcom_key=%s "
        "producer_task_id=%s producer_map_index=%s request_count=%s",
        ti.dag_id,
        ti.task_id,
        chunk_index,
        xcom_key,
        producer_task_id,
        producer_map_index,
        len(requests),
    )

    async def _inner() -> Dict[str, Any]:
        client = AsyncOpenAI(api_key=api_key, base_url=base_url)

        skip_input_path = None
        if isinstance(jsonl_body, str):
            skip_input_path = _save_text_artifact(
                f"skip_input_chunk{chunk_index:03d}.jsonl", jsonl_body
            )

        logger.info(
            "process_chunk[%s]: submitting skipcheck batch (requests=%s, completion_window=%s)",
            chunk_index,
            len(requests),
            completion_window,
        )

        if existing_skip_batch_id:
            batch_id = existing_skip_batch_id
            logger.info(
                "process_chunk[%s]: reusing existing skip batch_id=%s",
                chunk_index,
                batch_id,
            )
        else:
            submit_res = await submit_batch_requests(
                client,
                requests,
                completion_window,
            )
            batch_id = submit_res["batch_id"]
            # batch 생성 결과 및 batch_id 를 dag-artifacts 에 저장
            _save_json_artifact_for_ti(
                ti,
                "skip_batch_info.json",
                {
                    "chunk_index": chunk_index,
                    "batch_id": batch_id,
                    "submit_response": submit_res,
                },
            )

        b = await client.batches.retrieve(batch_id)
        # batch 조회 결과도 dag-artifacts 에 저장
        _save_json_artifact_for_ti(
            ti,
            "skip_batch_last_status.json",
            {
                "chunk_index": chunk_index,
                "batch_id": batch_id,
                "status": getattr(b, "status", None),
                "output_file_id": getattr(b, "output_file_id", None),
                "error_file_id": getattr(b, "error_file_id", None),
                "raw": _serialize_for_json(b),
            },
        )
        state = b.status
        if state not in {"completed", "failed", "cancelled"}:
            logger.info(
                "process_chunk[%s]: skipcheck batch still in state=%s, scheduling retry",
                chunk_index,
                state,
            )
            await client.close()
            raise AirflowException(
                f"Skip batch {batch_id} not completed yet (state={state})"
            )
        output_file_id = getattr(b, "output_file_id", None)

        logger.info(
            "process_chunk[%s]: skipcheck batch finished with state=%s output_file_id=%s",
            chunk_index,
            state,
            output_file_id,
        )

        row_updates: Dict[int, Dict[str, Any]] = {}

        skip_output_path = None
        trans_input_path = None
        trans_output_path = None
        updates_chunk_path = None
        
        translate_requests: List[Dict[str, Any]] = []
        translate_meta: List[Dict[str, Any]] = []

        if output_file_id:
            skip_output_path, records = await _download_batch_output(
                client, output_file_id, f"skip_output_chunk{chunk_index:03d}"
            )
            meta_list: List[Dict[str, Any]] = metadata
            meta_by_custom = {
                m["custom_id"]: m for m in meta_list if "custom_id" in m
            }

            for rec in records:
                custom_id = rec.get("custom_id")
                if not custom_id or custom_id not in meta_by_custom:
                    continue
                meta = meta_by_custom[custom_id]
                row_idx = int(meta["row_index"])
                check_val = _extract_skipcheck_value(rec)
                if check_val is None:
                    continue

                upd = row_updates.setdefault(row_idx, {})
                upd["Check boxes"] = check_val

                if any(kw in check_val for kw in NEEDS_TRANSLATION_KEYWORDS):
                    tgt_lang = str(meta.get("tgt_language") or meta.get("Tgt language"))
                    origin_text = str(meta.get("origin", ""))

                    if is_double:
                        # 두 번 번역하는 경우: Translation1, Translation2
                        sys_p1, user_p1 = build_trans_prompt(tgt_lang, origin_text)
                        tr_custom_id1 = f"translate1:{row_idx}"
                        translate_requests.append(
                            {
                                "custom_id": tr_custom_id1,
                                "method": "POST",
                                "url": BATCH_ENDPOINT,
                                "body": {
                                    "model": os.getenv("TRANS_MODEL1", os.getenv("TRANS_MODEL", "gpt-5-mini")),
                                    "messages": [sys_p1, user_p1],
                                },
                            }
                        )
                        translate_meta.append(
                            {
                                "custom_id": tr_custom_id1,
                                "row_index": row_idx,
                                "slot": 1,
                                "tgt_language": tgt_lang,
                                "origin": origin_text,
                            }
                        )

                        sys_p2, user_p2 = build_trans_prompt(tgt_lang, origin_text)
                        tr_custom_id2 = f"translate2:{row_idx}"
                        translate_requests.append(
                            {
                                "custom_id": tr_custom_id2,
                                "method": "POST",
                                "url": BATCH_ENDPOINT,
                                "body": {
                                    "model": os.getenv("TRANS_MODEL2", os.getenv("TRANS_MODEL", "gpt-5-mini")),
                                    "messages": [sys_p2, user_p2],
                                },
                            }
                        )
                        translate_meta.append(
                            {
                                "custom_id": tr_custom_id2,
                                "row_index": row_idx,
                                "slot": 2,
                                "tgt_language": tgt_lang,
                                "origin": origin_text,
                            }
                        )
                    else:
                        # 일반 케이스: 번역 한 번만 실행
                        sys_p, user_p = build_trans_prompt(tgt_lang, origin_text)
                        tr_custom_id = f"translate:{row_idx}"

                        translate_requests.append(
                            {
                                "custom_id": tr_custom_id,
                                "method": "POST",
                                "url": BATCH_ENDPOINT,
                                "body": {
                                    "model": os.getenv("TRANS_MODEL", "gpt-5-mini"),
                                    "messages": [sys_p, user_p],
                                },
                            }
                        )
                        translate_meta.append(
                            {
                                "custom_id": tr_custom_id,
                                "row_index": row_idx,
                                "slot": 0,
                                "tgt_language": tgt_lang,
                                "origin": origin_text,
                            }
                        )

        if translate_requests:
            logger.info(
                "process_chunk[%s]: %s rows need translation, submitting translate batch",
                chunk_index,
                len(translate_requests),
            )

        if translate_requests:
            tr_jsonl_body = "\n".join(
                json.dumps(item, ensure_ascii=False) for item in translate_requests
            )
            trans_input_path = _save_text_artifact(
                f"trans_input_chunk{chunk_index:03d}.jsonl", tr_jsonl_body
            )

            if existing_trans_batch_id:
                tr_batch_id = existing_trans_batch_id
                logger.info(
                    "process_chunk[%s]: reusing existing translate batch_id=%s",
                    chunk_index,
                    tr_batch_id,
                )
            else:
                trans_submit = await submit_batch_requests(
                    client,
                    translate_requests,
                    completion_window,
                )
                tr_batch_id = trans_submit["batch_id"]
                # translate batch 생성 결과 및 batch_id 를 dag-artifacts 에 저장
                _save_json_artifact_for_ti(
                    ti,
                    "trans_batch_info.json",
                    {
                        "chunk_index": chunk_index,
                        "batch_id": tr_batch_id,
                        "submit_response": trans_submit,
                    },
                )

            b_tr = await client.batches.retrieve(tr_batch_id)
            # translate batch 조회 결과도 dag-artifacts 에 저장
            _save_json_artifact_for_ti(
                ti,
                "trans_batch_last_status.json",
                {
                    "chunk_index": chunk_index,
                    "batch_id": tr_batch_id,
                    "status": getattr(b_tr, "status", None),
                    "output_file_id": getattr(b_tr, "output_file_id", None),
                    "error_file_id": getattr(b_tr, "error_file_id", None),
                    "raw": _serialize_for_json(b_tr),
                },
            )
            tr_state = b_tr.status
            if tr_state not in {"completed", "failed", "cancelled"}:
                logger.info(
                    "process_chunk[%s]: translate batch still in state=%s, scheduling retry",
                    chunk_index,
                    tr_state,
                )
                await client.close()
                raise AirflowException(
                    f"Translate batch {tr_batch_id} not completed yet (state={tr_state})"
                )

            out_tr_id = getattr(b_tr, "output_file_id", None)

            logger.info(
                "process_chunk[%s]: translate batch finished with state=%s output_file_id=%s",
                chunk_index,
                tr_state,
                out_tr_id,
            )

            if out_tr_id:
                trans_output_path, tr_records = await _download_batch_output(
                    client, out_tr_id, f"trans_output_chunk{chunk_index:03d}"
                )
                tr_meta_by_custom = {
                    m["custom_id"]: m for m in translate_meta if "custom_id" in m
                }
                for rec in tr_records:
                    cid = rec.get("custom_id")
                    if not cid or cid not in tr_meta_by_custom:
                        continue
                    m = tr_meta_by_custom[cid]
                    row_idx = int(m["row_index"])
                    text = _extract_translation_value(rec)
                    if text is None:
                        continue
                    slot = m.get("slot", 0)
                    if is_double:
                        if slot == 1:
                            col = "Translation1"
                        elif slot == 2:
                            col = "Translation2"
                        else:
                            continue
                    else:
                        col = "Translation"

                    upd = row_updates.setdefault(row_idx, {})
                    upd[col] = text

        await client.close()

        logger.info(
            "process_chunk[%s]: finished, updated_rows=%s",
            chunk_index,
            len(row_updates),
        )

        if row_updates:
            updates_chunk_path = _save_json_artifact_for_ti(
                ti,
                f"updates_chunk{chunk_index:03d}.json",
                row_updates,
            )

        return {
            "chunk_index": chunk_index,
            "updated_rows_preview": sorted(list(row_updates.keys()))[:10],
            "skip_input_path": skip_input_path,
            "skip_output_path": skip_output_path,
            "trans_input_path": trans_input_path,
            "trans_output_path": trans_output_path,
            "updates_chunk_path": updates_chunk_path,
        }

    return asyncio.run(_inner())


@task
def build_final_excel_from_updates(
    chunk_updates: List[Dict[str, Any]],
) -> Dict[str, Any]:
    ctx = get_current_context()
    params: Dict[str, Any] = ctx.get("params", {}) if ctx else {}
    ti = ctx["ti"]

    input_path_param = params.get("input_path")
    if input_path_param is None:
        raise RuntimeError("DAG params 에 'input_path' 가 설정되어 있어야 합니다.")

    if isinstance(input_path_param, ObjectStoragePath):
        input_path: Any = input_path_param
    else:
        s = str(input_path_param)
        if "://" in s:
            url = s
        else:
            url = Path(s).absolute().as_uri()
        input_path = ObjectStoragePath(url)

    df = load_dataframe(input_path)

    updates_list: List[Dict[str, Any]] = []

    run_artifact_base = ARTIFACT_BASE / ti.run_id / "process_chunk"
    try:
        for subdir in run_artifact_base.iterdir():
            if not subdir.is_dir():
                continue
            for path in subdir.glob("updates_chunk*.json"):
                try:
                    with path.open("r", encoding="utf-8") as f:
                        data = json.load(f)
                    if isinstance(data, dict):
                        updates_list.append(data)
                except Exception as e:
                    logger.warning("Failed to load updates artifact %s: %s", path, e)
    except FileNotFoundError:
        pass

    for updates in updates_list:
        for idx_str, cols in updates.items():
            try:
                row_idx = int(idx_str)
            except Exception:
                continue
            if row_idx not in df.index:
                continue
            if isinstance(cols, dict):
                for col, val in cols.items():
                    df.at[row_idx, col] = val

    if isinstance(input_path, ObjectStoragePath):
        base_name = Path(input_path.name).stem
    else:
        base_name = Path(str(input_path)).stem

    final_name = base_name + "_results.xlsx"
    artifact_base = _artifact_base_for_ti(ti)
    final_path = artifact_base / final_name
    final_path.parent.mkdir(exist_ok=True, parents=True)

    with final_path.open("wb") as f:
        df.to_excel(f, index=False)


    return {
        "filename": final_name,
        "format": "xlsx",
        "artifact_path": str(final_path)
    }


with dag:
    skip_chunks = create_skipcheck_batches()
    chunk_results = process_chunk.expand(chunk=skip_chunks)
    final_excel = build_final_excel_from_updates(chunk_results)
