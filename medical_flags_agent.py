"""
medical_imaging_flags.py
================================
Clean, modular multi-label tagging for Medical Imaging.
Supports optional Vision Fallback and toggleable Chain-of-Thought for production cost savings.
Optimized for `gpt-54mini-swe-d-clab-model01`.
Logs latency for performance comparison reporting.
"""

import time
import math
import random
import base64
import mimetypes
import os
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

VLM_FALLBACK_RULE = """
### Multimodal Analysis Rule
Start by evaluating the OCR text first. If the OCR text is heavily garbled, extremely sparse, or lacks semantic meaning (e.g., pure waveform graphs, messy scans), seamlessly fallback to relying on the visual structure of the Document Image to verify findings."""

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
        prompt += f"\n{VLM_FALLBACK_RULE}"
    if use_cot:
        prompt += "\n\nIn chain_of_thought: Name the specific evidence found (e.g., 'Found visual ECG waveform graph in image' or 'Found normal sinus rhythm in text'). Explicitly note if a report type is only mentioned as ordered/scheduled, which should NOT count as true."
    return prompt

# ==========================================
# 3. HELPERS
# ==========================================

def encode_image(image_path: str) -> Optional[str]:
    if not image_path or not os.path.isfile(image_path): return None
    mime_type = mimetypes.guess_type(image_path)[0] or 'image/jpeg'
    with open(image_path, "rb") as f:
        return f"data:{mime_type};base64,{base64.b64encode(f.read()).decode('utf-8')}"

def get_field_confidence(logprobs, field_name: str) -> float:
    """Averages logprobs specifically for a target JSON key."""
    if not logprobs or not logprobs.content: return 0.0
    tokens = logprobs.content
    search_key = f'"{field_name}"'
    
    running = ""
    start_idx = None
    for i, t in enumerate(tokens):
        running += t.token
        if search_key in running:
            start_idx = i
            break
            
    if start_idx is None: return 0.0
    
    val_probs = []
    started = False
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
        content.append({"type": "image_url", "image_url": {"url": img_b64, "detail": "auto"}})

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
        except RateLimitError as e:
            time.sleep(2 ** attempt + random.uniform(0, 1))
        except (APIConnectionError, APITimeoutError):
            time.sleep(1.5)
    else:
        raise RuntimeError("LLM Call Failed after retries.")

    latency = round(time.time() - start_time, 3)
    data = res.choices[0].message.parsed
    logprobs = res.choices[0].logprobs

    return {
        "contains_xray": data.contains_xray,
        "contains_xray_confidence": get_field_confidence(logprobs, "contains_xray"),
        "contains_ultrasound": data.contains_ultrasound,
        "contains_ultrasound_confidence": get_field_confidence(logprobs, "contains_ultrasound"),
        "contains_ecg": data.contains_ecg,
        "contains_ecg_confidence": get_field_confidence(logprobs, "contains_ecg"),
        "flags_reason": getattr(data, "chain_of_thought", "CoT Disabled"),
        "latency_sec": latency
    }

# Usage in Jupyter:
# client = AzureOpenAI(...)
# result = run_imaging_flags("raw text here", "path/to/img.jpg", client, use_vision=True, use_cot=False)