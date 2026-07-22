"""
medical_imaging_flags.py
================================
Clean, modular multi-label tagging for Medical Imaging (X-Ray, Ultrasound, ECG).
Supports joint Multimodal Synthesis (Text + Low-Detail Image) and toggleable Chain-of-Thought.
Optimized for `gpt-54mini-swe-d-clab-model01`.
Logs latency and token counts for performance comparison reports.
"""

import time
import math
import random
import base64
import mimetypes
import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x
from typing import Optional, Type, Tuple
from pydantic import BaseModel, create_model
from openai import AzureOpenAI, RateLimitError, APIConnectionError, APITimeoutError

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

def run_imaging_flags(ocr_text: str, image_path: Optional[str], client: AzureOpenAI, 
                      model: str = "gpt-54mini-swe-d-clab-model01",
                      use_vision: bool = False, use_cot: bool = True) -> dict:
    
    schema = get_flags_schema(use_cot)
    sys_prompt = build_prompt(use_vision, use_cot)
    
    content = [{"type": "text", "text": f"Document Text:\n{ocr_text}"}]
    if use_vision and image_path and (img_b64 := encode_image(image_path)):
        content.append({"type": "image_url", "image_url": {"url": img_b64, "detail": "low"}})

    start_time = time.time()
    
    for attempt in range(4):
        try:
            res = client.beta.chat.completions.parse(
                model=model,
                messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": content}],
                response_format=schema,
                logprobs=True, top_logprobs=1, temperature=0.0, seed=42
            )
            break
        except RateLimitError:
            time.sleep(2 ** attempt + random.uniform(0, 1))
        except (APIConnectionError, APITimeoutError):
            time.sleep(1.5)
    else:
        raise RuntimeError("LLM Call Failed after retries.")

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
                            model="gpt-54mini-swe-d-clab-model01", use_vision=False, use_cot=True, max_workers=5) -> pd.DataFrame:
    """Runs the imaging flags agent over a pandas DataFrame using multi-threading and returns a full metrics DataFrame."""
    results = []
    
    def process_row(idx, row):
        ocr_text = str(row.get(text_col, ""))
        img_path = str(row.get(img_col, "")) if pd.notna(row.get(img_col)) else None
        try:
            res = run_imaging_flags(ocr_text, img_path, client, model, use_vision, use_cot)
            res["_idx"] = idx
            return res
        except Exception as e:
            return {"_idx": idx, "error": str(e)}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_row, idx, row): idx for idx, row in df.iterrows()}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Tagging Imaging Findings"):
            results.append(future.result())
    
    results_df = pd.DataFrame(results).set_index("_idx")
    return df.join(results_df)