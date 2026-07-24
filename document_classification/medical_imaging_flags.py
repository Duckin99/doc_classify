"""
medical_imaging_flags.py
================================
Clean, modular multi-label tagging for Medical Imaging (X-Ray, Ultrasound, ECG).
Supports joint Multimodal Synthesis (Text + Low-Detail Image) and toggleable Chain-of-Thought.
Optimized for `gpt-54mini-swe-d-clab-model01`.
Logs latency and token counts for performance comparison reports.

Retry/error/checkpoint handling mirrors document_classifier.py (and reuses its
call_with_retry directly) -- see _error_fallback below for why that matters here.
"""

import csv
import time
import math
import threading
import base64
import mimetypes
import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x
from typing import Optional, Type
from pydantic import BaseModel, create_model
from openai import AzureOpenAI, BadRequestError

from document_classifier import call_with_retry, run_cascade_batch_checkpointed

# ==========================================
# 1. CORE PROMPT DEFINITIONS
# ==========================================

FLAGS_BASE = """You are a Medical Imaging & Diagnostics Tagging Agent. You receive OCR/markdown text from a document already classified as medical. Your job is multi-label tagging, NOT single-choice classification -- a document can contain zero, one, two, or all three of these report types together.

Set each flag to true only if there is genuine, specific evidence of that report type's actual findings -- not just a passing mention that the test was ordered.

- contains_xray: true if the text OR image shows actual X-ray findings/impressions (e.g. chest X-ray, radiographic findings, "CXR", lung fields, bone/joint imaging).
- contains_ultrasound: true if the text OR image shows actual ultrasound findings/impressions (e.g. abdominal ultrasound, echo findings, sonography report content).
- contains_ecg: true if the text OR image shows actual ECG/EKG findings. *CRITICAL:* If the image visually contains an electrocardiogram waveform grid, you MUST set this to true, even if the OCR text is empty.

These three flags are independent. A plain outpatient note with no imaging/ECG content at all should have all three false."""

VLM_RULE = """
### Multimodal Joint Synthesis Rule
Synthesize BOTH the OCR text and the Document Image simultaneously. Do not treat the image merely as a fallback. 
The visual structure is critical context for identifying visual findings that lack textual description, such as raw ECG waveform graphs or X-Ray film sheets."""

# ==========================================
# 2. DYNAMIC SCHEMA (CoT Toggling)
# ==========================================

def get_flags_schema(use_cot: bool) -> Type[BaseModel]:
    fields = {}
    if use_cot:
        fields["chain_of_thought"] = (str, ...)
    fields["contains_xray"] = (bool, ...)
    fields["contains_ultrasound"] = (bool, ...)
    fields["contains_ecg"] = (bool, ...)
    return create_model("MedicalImagingFlagsOutput", **fields)

def build_prompt(use_vision: bool, use_cot: bool) -> str:
    prompt = FLAGS_BASE
    if use_vision:
        prompt += f"\n\n{VLM_RULE}"
    if use_cot:
        prompt += "\n\nIn chain_of_thought: Name the specific evidence found (e.g., 'Found visual ECG waveform graph in image' or 'Found normal sinus rhythm in text'). Explicitly note if a report type is only mentioned as ordered/scheduled, which should NOT count as true."
    return prompt

# ==========================================
# 3. HELPERS
# ==========================================

def encode_image(image_path: str) -> Optional[str]:
    if not image_path or not os.path.isfile(image_path): return None
    try:
        mime_type = mimetypes.guess_type(image_path)[0] or 'image/jpeg'
        with open(image_path, "rb") as f:
            return f"data:{mime_type};base64,{base64.b64encode(f.read()).decode('utf-8')}"
    except Exception:
        return None

def get_field_confidence(logprobs, field_name: str) -> float:
    if not logprobs or not logprobs.content: return 0.0
    tokens = logprobs.content
    search_key = f'"{field_name}"'
    
    running, start_idx = "", None
    for i, t in enumerate(tokens):
        running += t.token
        if search_key in running:
            start_idx = i
            break
            
    if start_idx is None: return 0.0
    
    val_probs, started = [], False
    for i in range(start_idx, len(tokens)):
        if not started:
            if ":" in tokens[i].token: started = True
            continue
        if "," in tokens[i].token or "}" in tokens[i].token: break
        if tokens[i].logprob is not None:
            val_probs.append(tokens[i].logprob)

    return round(math.exp(sum(val_probs) / len(val_probs)) * 100, 2) if val_probs else 0.0

# ==========================================
# 4. CORE EXECUTION
# ==========================================

IMAGING_RESULT_KEYS = [
    "contains_xray", "contains_xray_confidence",
    "contains_ultrasound", "contains_ultrasound_confidence",
    "contains_ecg", "contains_ecg_confidence",
    "flags_reason", "latency_sec", "tokens_in", "tokens_out",
]

_checkpoint_lock = threading.Lock()


def _error_fallback(reason: str) -> dict:
    """Complete row matching IMAGING_RESULT_KEYS, with False/0 defaults for
    anything not knowable from an error. Every row -- success or failure --
    must have every key: previously a failed row only carried {"error": ...},
    so after joining onto df the missing contains_xray/ultrasound/ecg columns
    were NaN -- and pandas' `.astype(bool)` (used in flag_review.py) turns NaN
    into True, silently reporting every failed document as a positive finding."""
    return {
        "contains_xray": False, "contains_xray_confidence": 0.0,
        "contains_ultrasound": False, "contains_ultrasound_confidence": 0.0,
        "contains_ecg": False, "contains_ecg_confidence": 0.0,
        "flags_reason": reason, "latency_sec": 0.0, "tokens_in": 0, "tokens_out": 0,
    }


def run_imaging_flags(ocr_text: str, image_path: Optional[str], client: AzureOpenAI,
                      model: str = "gpt-54mini-swe-d-clab-model01",
                      use_vision: bool = False, use_cot: bool = True) -> dict:

    schema = get_flags_schema(use_cot)
    sys_prompt = build_prompt(use_vision, use_cot)

    content = [{"type": "text", "text": f"Document Text:\n{ocr_text}"}]
    if use_vision and image_path and (img_b64 := encode_image(image_path)):
        content.append({"type": "image_url", "image_url": {"url": img_b64, "detail": "low"}})

    start_time = time.time()
    # Shared retry helper from document_classifier.py: exponential backoff + jitter,
    # respects the server's Retry-After header on rate limits. BadRequestError (e.g.
    # content-filter blocks) is intentionally not retryable here -- it propagates up
    # to the caller, same as document_classifier.py's cascade.
    res = call_with_retry(
        client.beta.chat.completions.parse,
        model=model,
        messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": content}],
        response_format=schema,
        logprobs=True, top_logprobs=1, temperature=0.0, seed=42,
    )

    latency = round(time.time() - start_time, 3)
    data = res.choices[0].message.parsed
    logprobs = res.choices[0].logprobs

    usage = res.usage
    tokens_in = usage.prompt_tokens if usage else 0
    tokens_out = usage.completion_tokens if usage else 0

    return {
        "contains_xray": data.contains_xray,
        "contains_xray_confidence": get_field_confidence(logprobs, "contains_xray"),
        "contains_ultrasound": data.contains_ultrasound,
        "contains_ultrasound_confidence": get_field_confidence(logprobs, "contains_ultrasound"),
        "contains_ecg": data.contains_ecg,
        "contains_ecg_confidence": get_field_confidence(logprobs, "contains_ecg"),
        "flags_reason": getattr(data, "chain_of_thought", "CoT Disabled"),
        "latency_sec": latency,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out
    }


def run_imaging_flags_batch(df: pd.DataFrame, client: AzureOpenAI, text_col="ocr_text", img_col="filepath",
                            model="gpt-54mini-swe-d-clab-model01", use_vision=False, use_cot=True, max_workers=5,
                            checkpoint_path: Optional[str] = None, resume: bool = True) -> pd.DataFrame:
    """Runs the imaging flags agent over a pandas DataFrame using multi-threading.

    If checkpoint_path is given, each row is written to disk as it completes
    (mirrors document_classifier.py's run_cascade_batch_checkpointed) -- a crash
    mid-run doesn't lose progress, and rerunning with the same checkpoint_path
    resumes, skipping rows whose img_col value is already checkpointed. Every
    original column of df (including ground-truth *_gt columns) is preserved in
    the checkpoint file, not just the new prediction columns.

    Without checkpoint_path, behaves as before: results are held in memory and
    joined back onto df at the end.
    """
    already_done = set()
    if checkpoint_path and resume and os.path.isfile(checkpoint_path):
        prior = pd.read_csv(checkpoint_path)
        if img_col in prior.columns:
            already_done = set(prior[img_col].dropna().astype(str))
        print(f"Resuming: {len(already_done)} document(s) already in checkpoint, will be skipped.")

    pending = [(idx, row) for idx, row in df.iterrows() if str(row.get(img_col)) not in already_done]

    def process_row(idx, row):
        ocr_text = str(row.get(text_col, ""))
        img_path = str(row.get(img_col, "")) if pd.notna(row.get(img_col)) else None
        try:
            res = run_imaging_flags(ocr_text, img_path, client, model, use_vision, use_cot)
        except BadRequestError as e:
            error_message = str(e).lower()
            if "content management policy" in error_message or "jailbreak" in error_message:
                print(f"[WARNING] Doc {idx + 1} blocked by content filter. Skipping...")
                res = _error_fallback("Azure OpenAI Content Filter flagged this text.")
            else:
                print(f"[ERROR] Doc {idx + 1} failed (BadRequestError): {e}")
                res = _error_fallback(str(e))
        except Exception as e:
            print(f"[ERROR] Doc {idx + 1} failed: {e}")
            res = _error_fallback(str(e))
        return idx, row, res

    fieldnames = None

    def write_row(row, res):
        nonlocal fieldnames
        combined_row = {**row.to_dict(), **res}  # full original columns + predictions, self-contained on disk
        with _checkpoint_lock:
            file_exists = os.path.isfile(checkpoint_path)
            if fieldnames is None:
                fieldnames = list(pd.read_csv(checkpoint_path, nrows=0).columns) if file_exists else list(combined_row.keys())
            with open(checkpoint_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()
                writer.writerow(combined_row)

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_row, idx, row): idx for idx, row in pending}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Tagging Imaging Findings"):
            idx, row, res = future.result()
            if checkpoint_path:
                write_row(row, res)
            else:
                res = {**res, "_idx": idx}  # predictions only -- df.join() below adds them onto the existing columns
                results.append(res)

    if checkpoint_path:
        print(f"\nDone. Full results in '{checkpoint_path}'.")
        return pd.read_csv(checkpoint_path)

    results_df = pd.DataFrame(results).set_index("_idx")
    return df.join(results_df)


# ==========================================
# 5. IMAGING ROUTING RECALL (upstream cascade coverage check)
# ==========================================
# run_imaging_flags_batch (above) only ever runs on documents the macro+specialist
# cascade already routed into one of these three subclasses -- medical_others and
# every non-medical domain are never passed to the imaging tagger at all. So a
# document that genuinely contains an X-ray/ultrasound/ECG finding but gets
# misrouted upstream (wrong domain, or dumped in medical_others) never reaches the
# imaging tagger to begin with -- that's a miss the flags model itself can't see or
# be blamed for. evaluate_imaging_routing_recall measures exactly that upstream gap.
IMAGING_ELIGIBLE_SUBCATEGORIES = {"medical_clinical", "medical_healthcheck", "medical_lab"}


def evaluate_imaging_routing_recall(df_sample: pd.DataFrame, client: AzureOpenAI,
                                     text_col="ocr_text", img_col="filepath",
                                     macro_model: str = "gpt-54mini-swe-d-clab-model01",
                                     specialist_model: str = "gpt-54mini-swe-d-clab-model01",
                                     use_vision: bool = False, use_cot: bool = True,
                                     max_workers: int = 5,
                                     checkpoint_path: str = "imaging_routing_recall.csv",
                                     resume: bool = True):
    """Runs df_sample -- documents already known to have at least one true imaging
    flag (xray_gt / ultrasound_gt / ecg_gt) -- through the full document_classifier.py
    macro->specialist cascade, then reports what fraction land in one of
    IMAGING_ELIGIBLE_SUBCATEGORIES (i.e. would actually reach the imaging tagger).

    This is a recall metric ("of documents that truly have imaging content, how many
    does the pipeline funnel somewhere the imaging tagger can see them"), not an
    imaging-tagging accuracy metric -- it never looks at contains_xray/etc. at all.

    Reuses run_cascade_batch_checkpointed directly rather than reimplementing the
    cascade, so it gets the same retry/error-handling/checkpoint-resume behavior as
    every other cascade run. checkpoint_path defaults to a name distinct from your
    main results CSV -- pass your own to avoid colliding with an unrelated run.

    Returns (routed_df, recall_stats) where routed_df is df_sample with the cascade's
    output columns (macro_decision, final_subcategory, etc.) joined on, and
    recall_stats is {"recall": float, "n": int, "captured": int} -- or None if
    df_sample has no rows with a true imaging flag.
    """
    routed_df = run_cascade_batch_checkpointed(
        df_sample, client, text_col=text_col, img_col=img_col,
        macro_model=macro_model, specialist_model=specialist_model,
        use_vision=use_vision, use_cot=use_cot, max_workers=max_workers,
        checkpoint_path=checkpoint_path, resume=resume,
    )

    gt_cols = [c for c in ("xray_gt", "ultrasound_gt", "ecg_gt") if c in routed_df.columns]
    if not gt_cols:
        raise ValueError(
            "df_sample has none of xray_gt/ultrasound_gt/ecg_gt -- evaluate_imaging_routing_recall "
            "expects a sample already filtered to documents with a known imaging ground-truth flag."
        )
    has_flag = routed_df[gt_cols].fillna(False).astype(bool).any(axis=1)
    positive = routed_df[has_flag]
    if positive.empty:
        return routed_df, None

    captured = positive["final_subcategory"].isin(IMAGING_ELIGIBLE_SUBCATEGORIES)
    recall_stats = {"recall": float(captured.mean()), "n": len(positive), "captured": int(captured.sum())}
    return routed_df, recall_stats