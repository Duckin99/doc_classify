"""
Document Classification Pipeline (Multimodal + Independent Step Metrics)
======================================================================
Features:
- Independent Latency & Token Tracking (Macro vs. Specialist).
- Dynamic Chain-of-Thought (CoT) Toggle (`use_cot=True/False`).
- Joint Multimodal Synthesis (Text + Low-Detail Image).
- Multithreaded DataFrame batch execution with auto-checkpointing.
"""

import math
import time
import random
import os
import csv
import threading
import base64
import mimetypes
import pandas as pd
from typing import Literal, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from pydantic import create_model, BaseModel
from openai import AzureOpenAI, BadRequestError, RateLimitError, APIConnectionError, APITimeoutError

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# ==========================================
# 1. BASE SCHEMAS (Factory for CoT Toggle)
# ==========================================

def get_schema(use_cot: bool, schema_type: str):
    fields = {}
    if use_cot:
        fields["chain_of_thought"] = (str, ...)
        
    if schema_type == "macro":
        fields["macro_class"] = (Literal["medical", "financial", "identification", "not_for_underwriting"], ...)
    elif schema_type == "financial":
        fields["subcategory"] = (Literal["financial_bankstatement", "financial_bookbank", "financial_companyregistration", "financial_selfincomedeclaration", "financial_receipt", "financial_others"], ...)
    elif schema_type == "id":
        fields["subcategory"] = (Literal["id_driverlicense", "id_fatca", "id_foreignerconfirmationform", "id_foreigner_nationalid", "id_visastamp", "id_thaievisa", "id_passport", "id_statelessid", "id_thainationalid", "id_workpermit", "id_houseregistration", "id_marriagecertificate", "id_birthcertificate", "id_others"], ...)
    elif schema_type == "medical":
        fields["subcategory"] = (Literal["medical_clinical", "medical_healthcheck", "medical_lab", "medical_others"], ...)
        
    return create_model(f"{schema_type.capitalize()}Output", **fields)

# ==========================================
# 2. PROMPTS (Preserving Original Tuning + Joint Multimodal Rule)
# ==========================================

MULTIMODAL_INSTRUCTION = """

### Joint Multimodal Rule (Text + Image Synthesis)
You receive BOTH the OCR/markdown text and the raw document image. Do not treat the image merely as a fallback. 
The visual structure is critical context for:
1. Low-text visual artifacts (e.g., foreign scripts like Lao IDs where OCR scrambles text, or raw ECG waveforms/X-ray films).
2. Official stamps, seals, or layouts that provide ground-truth document structure."""

MACRO_BASE = """You are a Document Macro Classifier for an insurance underwriting pipeline. Classify raw OCR/markdown text into exactly one of: `medical`, `financial`, `identification`, `not_for_underwriting`. Treat the patterns below as supporting evidence, not an exact-string checklist -- read holistically. Ignore PII.

### Ignore Watermarks
Insurance disclaimers (+ date/time stamps) are never evidence of document type -- ignore them completely.

### 1. medical
Route here for genuine clinical content: clinical notes, lab metrics, health-check parameters, prescriptions, test ranges, normal/abnormal flags. Garbled OCR with technical-looking terms next to numbers/units/ranges still counts as low-confidence medical (flag the uncertainty).
- EXCLUSION: Do NOT route here for a hospital/pharmacy receipt or billing statement -- those are `financial` regardless of medical context. Do NOT route here for correspondence that merely discusses or requests medical information without containing the clinical data itself -- that is `not_for_underwriting`.

### 2. financial
- Bank/account identity: Account number + bank name/branch, or a native transaction ledger (dated money movements, ideally a markdown table).
- Corporate registration: Registration number + certification/signature block.
- Self income declaration & Income Overrides: The person reporting, clarifying, or declaring their own income ("รายได้" / income). 
  - CRITICAL OVERRIDE: If income ("รายได้") details are present, you MUST route to `financial`, even if the document looks like an insurance e-Form containing premium payment keywords ("ชำระเบี้ยประกัน") or company headers.
- Receipts/Proof-of-payment: Including hospital/pharmacy/medical treatment receipts (ค่ายา, ค่ารักษา, ค่าห้อง), government permit fee receipts, retail receipts, or any other billing statement. All receipts are financial, regardless of what the payment was for.
- General agreements/contracts: Lease, sale, or other business contracts.

### 3. identification
A genuine ID/travel/civil-registry document with its own native structure (not a field filled into someone else's form):
- National/Stateless ID: Issuing country name/code (e.g., "LAO PDR", "RDP LAO"), a genuine name+DOB or name+issue/expiry block. Look for "รับรองสำเนาถูกต้อง" (certified true copy) or `<figure>` tags wrapping personal-detail blocks. Stateless-ID text or garbled non-Thai script (e.g., Lao misread as Thai) combined with a country-code header also counts.
- Passports/Visas: MRZ lines (letters/digits + long "<" runs, e.g., "PA123<<<<<<<"), or immigration/visa keywords ("VISA", "VISACLASS", "IMMIGRATION", "DEPARTED", "ADMITTED", "ENTRY") near messy digits/dates.
- Thai E-Visa: The literal term "E-VISA" printed on the document is a strong, standalone anchor.
- Tax Forms: FATCA-related tax forms (W-9, W-4, W-8BEN) or similar withholding declarations.
- Work Permits: Labor authorization booklets, employer/employee details, type-of-work (ประเภทงาน), Department of Employment text. Includes renewals/amendments (รายการเปลี่ยนเพิ่มประเภทงาน).
- Civil Registry: Household registration (ทะเบียนบ้าน), Marriage certificate (ใบสำคัญการสมรส / ทะเบียนสมรส), Birth certificate (สูติบัตร).
- Other IDs: Student IDs, employee badges, or any physical identity card with a photo placeholder/ID number.
- Reminder: An ID document stamped with an insurance watermark/date is still `identification`.

### 4. not_for_underwriting
Default when nothing above applies. The underwriter will not use this document.
- Insurance application/policy paperwork (eForm Trap): Forms containing applicant fields (เบี้ยประกัน, policy details). 
  - SPECIFIC ANCHORS: Documents opening with "บันทึกคำชี้แจงเกี่ยวกับใบคำขอประกันภัย", consent letters, OR documents containing "ชำระเบี้ยประกัน" (premium payment) mixed with a company name/address in the PageHeader. 
  - EXCEPTION: If the document explicitly declares actual income details ("รายได้"), route to `financial` instead.
- Litmus test for eForms: Strip out the applicant's name/ID/policy number -- is any standalone financial, clinical, or identity statement left? If nothing remains but "applying for / updating a policy," it belongs here.
- Correspondence: Letters/memos between the insured and underwriter requesting documents (administrative communication).
- Envelope/mailing metadata: Sender name, policy codes (e.g., "T1348646"), addressing text.
- Photos/Noise: Portrait photos or ID placeholders without issuing text, or completely illegible noise."""

AGENT1_USER_PROMPT = "Classify the following raw OCR text into medical, financial, identification, or not_for_underwriting:\n{ocr_text}"

FINANCIAL_BASE = """You are an expert Financial Document Specialist. You receive text already confirmed as financial. Perform granular classification into one of six leaf classes. Use markdown structure (tables, headers) where present. Do not process PII.

- financial_bankstatement: running transactional ledgers, money transfers, account balances, mobile banking markers -- ideally a markdown table of dated transaction rows.
- financial_bookbank: static account identity headers (bilingual accounts, branches, bank names) without a long transaction ledger.
- financial_companyregistration: formal commercial corporate registry verbiage and certification signatures.
- financial_selfincomedeclaration: the person reporting, clarifying, or declaring their own income, salary, or source of income -- regardless of whether it's laid out as a form or questionnaire.
- financial_receipt: any proof-of-payment document -- hospital/pharmacy/medical treatment receipts, government license/permit fee receipts, retail receipts, or any other billing statement, regardless of what the payment was for.
- financial_others: financial in nature but doesn't cleanly match the above -- also covers general agreements/contracts (lease, sale, business contracts)."""

AGENT_FINANCIAL_USER_PROMPT = "Perform deep financial classification on this verified text:\n{ocr_text}"

ID_BASE = """You are an expert Identification Document Specialist. You receive text already confirmed as an identification/travel document. Perform granular classification. Use markdown structure where present. Do not process PII.
 
Reminder: insurance watermarks/disclaimers (+ date/time stamps) are never evidence of document type -- ignore them when classifying.
 
- id_thainationalid / id_foreigner_nationalid: national demographic card headers and identity issuing text blocks. Foreign-script documents (e.g. a Lao national ID) still count even if largely unreadable by OCR -- look for legible fragments like issuing country name/code, name, or a DOB pattern.
- id_passport: the passport bio data page -- photo/data block, passport number, nationality, an MRZ line (letters/digits + long "<" runs).
- id_visastamp: a Thai visa page and/or immigration control stamp -- visa category codes, "DEPARTED"/"ADMITTED"/"ENTRY" keywords, and (often garbled, since stamp text is curved) date-like digit strings near them. Does NOT include a Thai E-Visa (see id_thaievisa below).
- id_thaievisa: a Thai Electronic Visa (E-Visa) -- look for the literal term "E-VISA" printed on the document; this alone is a reliable, standalone anchor for this class.
- id_workpermit: labor authorization booklets and official employment permission context. This includes work permit RENEWALS and amendment/endorsement documents such as "Change or addition of category of work" (รายการเปลี่ยนเพิ่มประเภทงาน).
- id_fatca: any FATCA-related US tax form -- W-9, W-4, W-8BEN, or similar FATCA/tax-status declaration or backup-withholding form, entity/individual tax certification sections.
- id_foreignerconfirmationform: an official government-issued confirmation/declaration form attesting to a foreign national's identity or status -- issued BY a government/authority ABOUT the person.
- id_statelessid: a stateless-person identity card -- card layout similar to a national ID (photo placeholder, ID number, name, DOB) but with issuing text indicating stateless/no-registered-nationality status.
- id_driverlicense: driver's license.
- id_houseregistration: household registration document (ทะเบียนบ้าน) -- household registration book header, house/address registry formatting, list of household members.
- id_marriagecertificate: marriage certificate (ใบทะเบียนสมรส) -- marriage registration terminology, groom/bride names, registrar signature, marriage date.
- id_birthcertificate: birth certificate (สูติบัตร) -- birth registration terminology, parents' names, date/place of birth, registrar signature/seal.
- id_others: clearly an ID-type artifact that doesn't cleanly match any subtype above -- e.g. a student ID card, employee badge."""

AGENT_ID_USER_PROMPT = "Perform deep identification classification on this verified text:\n{ocr_text}"

MEDICAL_BASE = """You are a strict Medical Document Specialist. Categorize pre-verified medical OCR text into the exact clinical subclass. Focus on clinical structures and specific medical anchors.
 
### Subclass Priority & Evaluation Guides
Evaluate in this priority order -- if multiple patterns are present on the same page, the higher-priority match wins.

- medical_lab (highest priority): Laboratory test results. Contains tabular test data with columns for Test Name, Result, Units, and Reference Range. Keywords: CBC, BUN, Creatinine, Lipid Profile, Glucose, mg/dL, mmol/L, x10^3/uL. (Note: waveform studies go to clinical).
- medical_healthcheck (second priority): Health check / physical examination data. Any page containing extractable vital signs or physical examination findings -- Blood Pressure, BMI, Heart Rate, Weight, Height, or a structured wellness summary.
- medical_clinical (third priority): Everything genuinely medical that isn't extractable lab data or extractable vitals. Includes narrative notes, pathology reports, imaging (X-ray, ultrasound), and functional diagnostics (ECG/EKG, EEG, spirometry) with no extractable lab or vitals data.
- medical_others: Medical documents that don't structurally fit any category above. SPECIFIC INCLUSION: always route Sleep Test reports here."""

AGENT3_USER_PROMPT = "Perform deep clinical classification on this verified medical text:\n{ocr_text}"

def build_prompt(base_prompt: str, use_vision: bool, use_cot: bool) -> str:
    prompt = base_prompt
    if use_vision:
        prompt += f"\n\n{MULTIMODAL_INSTRUCTION}"
    if use_cot:
        prompt += "\n\nIn chain_of_thought: Provide step-by-step structural rationale linking specific evidence to your decision."
    return prompt

# ==========================================
# 3. HELPERS & MULTIMODAL CORE
# ==========================================

def encode_image(image_path: str) -> Optional[str]:
    if not image_path or not os.path.isfile(image_path):
        return None
    try:
        mime_type, _ = mimetypes.guess_type(image_path)
        mime_type = mime_type or 'image/jpeg'
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return f"data:{mime_type};base64,{encoded_string}"
    except Exception as e:
        print(f"Failed to encode image {image_path}: {e}")
        return None

def get_field_confidence(logprobs, field_value: str) -> float:
    if not logprobs or not logprobs.content: return 0.0
    tokens = logprobs.content
    token_strs = [t.token for t in tokens]
    
    running, span_start, span_end = "", None, None
    for i, tok in enumerate(token_strs):
        prev_len = len(running)
        running += tok
        if field_value in running[max(0, prev_len - len(field_value)):]:
            end_idx = i
            partial, start_idx = "", i
            for j in range(i, -1, -1):
                partial = token_strs[j] + partial
                start_idx = j
                if field_value in partial: break
            span_start, span_end = start_idx, end_idx
            
    if span_start is None: return 0.0
    val_probs = [tokens[i].logprob for i in range(span_start, span_end + 1) if tokens[i].logprob is not None]
    return round(math.exp(sum(val_probs) / len(val_probs)) * 100, 2) if val_probs else 0.0

def call_with_retry(fn, *args, **kwargs):
    max_retries = 6
    base_delay = 1.0
    max_delay = 60.0
    for attempt in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except RateLimitError as e:
            retry_after = float(e.response.headers.get("retry-after", base_delay)) if e.response else base_delay
            time.sleep(retry_after + random.uniform(0, 1))
        except (APIConnectionError, APITimeoutError) as e:
            time.sleep(min(max_delay, base_delay * (2 ** attempt)) + random.uniform(0, 1))
    raise RuntimeError("Exceeded max retries for API call.")

def call_agent_multimodal(client, model, system_prompt, user_prompt, ocr_text, image_path, use_vision, schema, temperature=0.0, seed=42):
    user_content = [{"type": "text", "text": user_prompt.format(ocr_text=ocr_text)}]
    encoded_img = encode_image(image_path)
    if use_vision and encoded_img:
        user_content.append({"type": "image_url", "image_url": {"url": encoded_img, "detail": "low"}})

    start_time = time.time()
    response = call_with_retry(
        client.beta.chat.completions.parse,
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ],
        response_format=schema,
        logprobs=True,
        top_logprobs=1,
        temperature=temperature,
        seed=seed
    )
    latency = round(time.time() - start_time, 3)
    usage = response.usage
    tokens_in = usage.prompt_tokens if usage else 0
    tokens_out = usage.completion_tokens if usage else 0

    parsed_data = response.choices[0].message.parsed
    val_to_check = getattr(parsed_data, "macro_class", getattr(parsed_data, "subcategory", ""))
    conf = get_field_confidence(response.choices[0].logprobs, str(val_to_check))
    
    return parsed_data, conf, latency, tokens_in, tokens_out

# ==========================================
# 4. CASCADE PIPELINE WITH SEPARATE METRICS
# ==========================================

def run_cascade_pipeline(ocr_text: str, image_path: str, client: AzureOpenAI,
                         macro_model: str = "gpt-5.4-mini-swe-d-clab-model01",
                         specialist_model: str = "gpt-5.4-mini-swe-d-clab-model01",
                         use_vision: bool = False,
                         use_cot: bool = True) -> dict:
    
    # --- STAGE 1: MACRO AGENT ---
    macro_schema = get_schema(use_cot, "macro")
    macro_prompt = build_prompt(MACRO_BASE, use_vision, use_cot)
    macro_data, macro_conf, macro_lat, macro_in, macro_out = call_agent_multimodal(
        client, macro_model, macro_prompt, AGENT1_USER_PROMPT, ocr_text, image_path, use_vision, macro_schema
    )
    
    macro_class = macro_data.macro_class
    macro_reason = getattr(macro_data, "chain_of_thought", "CoT disabled")

    base_result = {
        "macro_decision": macro_class,
        "macro_reason": macro_reason,
        "macro_confidence": macro_conf,
        "macro_latency_sec": macro_lat,
        "macro_tokens_in": macro_in,
        "macro_tokens_out": macro_out,
        "specialist_latency_sec": 0.0,
        "specialist_tokens_in": 0,
        "specialist_tokens_out": 0,
    }

    if macro_class == "not_for_underwriting":
        return {
            **base_result,
            "final_subcategory": "not_for_underwriting",
            "specialist_reason": "Terminal at macro classifier.",
            "specialist_confidence": macro_conf,
        }

    # --- STAGE 2: SPECIALIST AGENT ---
    if macro_class == "medical":
        spec_schema, sys_p, user_p = get_schema(use_cot, "medical"), MEDICAL_BASE, AGENT3_USER_PROMPT
    elif macro_class == "financial":
        spec_schema, sys_p, user_p = get_schema(use_cot, "financial"), FINANCIAL_BASE, AGENT_FINANCIAL_USER_PROMPT
    else: # identification
        spec_schema, sys_p, user_p = get_schema(use_cot, "id"), ID_BASE, AGENT_ID_USER_PROMPT

    spec_prompt = build_prompt(sys_p, use_vision, use_cot)
    spec_data, spec_conf, spec_lat, spec_in, spec_out = call_agent_multimodal(
        client, specialist_model, spec_prompt, user_p, ocr_text, image_path, use_vision, spec_schema
    )

    return {
        **base_result,
        "final_subcategory": spec_data.subcategory,
        "specialist_reason": getattr(spec_data, "chain_of_thought", "CoT disabled"),
        "specialist_confidence": spec_conf,
        "specialist_latency_sec": spec_lat,
        "specialist_tokens_in": spec_in,
        "specialist_tokens_out": spec_out,
    }

# ==========================================
# 5. MULTITHREADED BATCH DATAFRAME RUNNER
# ==========================================

_checkpoint_lock = threading.Lock()

def _process_one_row(idx, row, client, macro_model, specialist_model, use_vision, use_cot):
    ocr_text = str(row.get("ocr_text", ""))
    filepath = str(row.get("filepath", ""))
    
    try:
        res = run_cascade_pipeline(ocr_text, filepath, client, macro_model, specialist_model, use_vision, use_cot)
    except BadRequestError as e:
        error_message = str(e).lower()
        if "content management policy" in error_message or "jailbreak" in error_message:
            print(f"[WARNING] Doc {idx + 1} blocked by Azure Jailbreak Filter. Skipping...")
            res = {
                "macro_decision": "blocked_by_firewall",
                "macro_reason": "Azure OpenAI Content Filter flagged this OCR text as a jailbreak attempt.",
                "macro_confidence": 0.0,
                "macro_latency_sec": 0.0,
                "macro_tokens_in": 0,
                "macro_tokens_out": 0,
                "final_subcategory": "error_content_filter",
                "specialist_reason": "Skipped due to API security policy.",
                "specialist_confidence": 0.0,
                "specialist_latency_sec": 0.0,
                "specialist_tokens_in": 0,
                "specialist_tokens_out": 0,
            }
        else:
            raise e
    except Exception as e:
        print(f"[ERROR] Doc {idx + 1} failed due to unexpected error: {e}")
        res = {
            "macro_decision": "api_error",
            "macro_reason": str(e),
            "macro_confidence": 0.0,
            "macro_latency_sec": 0.0,
            "macro_tokens_in": 0,
            "macro_tokens_out": 0,
            "final_subcategory": "api_error",
            "specialist_reason": "API disconnected or failed.",
            "specialist_confidence": 0.0,
            "specialist_latency_sec": 0.0,
            "specialist_tokens_in": 0,
            "specialist_tokens_out": 0,
        }

    combined_row = {"filepath": filepath, **res}
    return idx, combined_row

def run_cascade_batch(csv_path: str, endpoint: str, api_key: str,
                      macro_model: str = "gpt-5.4-mini-swe-d-clab-model01",
                      specialist_model: str = "gpt-5.4-mini-swe-d-clab-model01",
                      use_vision: bool = False,
                      use_cot: bool = True, max_workers: int = 5,
                      checkpoint_path: str = "cascade_results.csv",
                      resume: bool = True) -> pd.DataFrame:
    
    client = AzureOpenAI(azure_endpoint=endpoint, api_key=api_key, api_version="2024-08-01-preview")
    df = pd.read_csv(csv_path)

    already_done = set()
    if resume and os.path.isfile(checkpoint_path):
        prior = pd.read_csv(checkpoint_path)
        if 'filepath' in prior.columns:
            already_done = set(prior['filepath'].dropna().astype(str))
        print(f"Resuming: {len(already_done)} document(s) already in checkpoint, will be skipped.")

    pending = [(idx, row) for idx, row in df.iterrows() if str(row.get('filepath')) not in already_done]
    mode_str = "MULTIMODAL VLM (Text + Image)" if use_vision else "TEXT-ONLY"
    print(f"Running cascade agent in [{mode_str}] mode on {len(pending)} document(s) with max_workers={max_workers}...")

    fieldnames = None

    def write_row(combined_row):
        nonlocal fieldnames
        with _checkpoint_lock:
            file_exists = os.path.isfile(checkpoint_path)
            if fieldnames is None:
                if file_exists:
                    fieldnames = list(pd.read_csv(checkpoint_path, nrows=0).columns)
                else:
                    fieldnames = list(combined_row.keys())
            with open(checkpoint_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()
                writer.writerow(combined_row)

    completed = 0
    iterator_kwargs = dict(total=len(pending)) if HAS_TQDM else {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_process_one_row, idx, row, client, macro_model, specialist_model, use_vision, use_cot): idx 
            for idx, row in pending
        }
        progress = tqdm(as_completed(futures), **iterator_kwargs) if HAS_TQDM else as_completed(futures)
        for future in progress:
            idx, res_row = future.result()
            write_row(res_row)
            completed += 1
            if not HAS_TQDM:
                print(f"[{completed}/{len(pending)}] {res_row.get('filepath')} "
                      f"Macro: {res_row['macro_decision']} -> Leaf: {res_row['final_subcategory']}")

    print(f"\nDone. {completed} document(s) processed this run. Results in '{checkpoint_path}'.")
    return pd.read_csv(checkpoint_path)