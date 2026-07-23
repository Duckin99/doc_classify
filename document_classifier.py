"""
Document Classification Pipeline (Multimodal + Independent Step Metrics)
======================================================================
Features:
- Independent Latency & Token Tracking (Macro vs. Specialist).
- Dynamic Chain-of-Thought (CoT) Toggle (`use_cot=True/False`).
- Joint Multimodal Synthesis (Text + Low-Detail Image).
- Multithreaded DataFrame batch execution with checkpointing + resume.
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
# 2. PROMPTS
# ==========================================
# CHANGED (this round): logic fixes to MACRO_BASE / MEDICAL_BASE / ID_BASE based on
# reviewing actual vision+text failures (see comments at each change site). Also
# replaced the single generic MULTIMODAL_INSTRUCTION with PER-AGENT vision guidance
# (MACRO_VISION_NOTE / MEDICAL_VISION_NOTE / ID_VISION_NOTE / FINANCIAL_VISION_NOTE) --
# that's the actual "not customized per agent" gap you flagged. build_prompt now takes
# a vision_note argument instead of always appending the same generic block.

GENERIC_VISION_NOTE = """

### Joint Multimodal Rule (Text + Image Synthesis)
You receive BOTH the OCR/markdown text and the raw document image. Do not treat the image merely as a fallback.
The visual structure is critical context for:
1. Low-text visual artifacts (e.g., foreign scripts like Lao IDs where OCR scrambles text, or raw ECG waveforms/X-ray films).
2. Official stamps, seals, or layouts that provide ground-truth document structure."""

# --- Macro: reverted MACRO_BASE back to the pre-insight-round wording (highest
# measured macro accuracy, 96.81%), then added 3 new targeted fixes there instead of
# stacking more rules on top of a version that had regressed. Kept only the
# portrait-holding-ID vision fix here since it wasn't implicated in the regression and
# isn't covered by the new text-level fixes; dropped the logo/formal-layout vision
# bullets since that content no longer has a matching anchor in the reverted base.
MACRO_VISION_NOTE = GENERIC_VISION_NOTE + """

### Vision-specific guidance for macro classification
- If the image's main visible subject is a PERSON -- e.g. a selfie/portrait of someone holding up an ID card to the camera -- this is a portrait/liveness photo, not an identification document. Route to `not_for_underwriting`. A genuine identification document has the ID card/document itself as the flat, primary subject filling the frame, not a person holding it."""

FINANCIAL_VISION_NOTE = GENERIC_VISION_NOTE

# --- Identification: fixes for passport vs foreigner_nationalid confusion, and the
# stateless-ID exact-phrase disambiguation.
ID_VISION_NOTE = GENERIC_VISION_NOTE + """

### Vision-specific guidance for identification classification
- Passports have a very distinct visual layout: a booklet biodata page with a fixed photo position, a structured data block, and a printed MRZ band at the bottom. If the image shows this layout, trust it strongly -- even if the OCR text is sparse or a country name in the text seems to suggest a national ID card instead. A country name/code by itself only tells you the person's NATIONALITY, not the document TYPE -- it does not distinguish a passport from a national ID card. When passport-specific structural signals (MRZ artifacts, booklet biodata layout) are present, they win over an assumption based on country name alone."""

# --- Medical: fixes for checkbox-symptom-tables leaking into healthcheck, and patient
# history documents needing an explicit clinical-narrative home.
MEDICAL_VISION_NOTE = GENERIC_VISION_NOTE + """

### Vision-specific guidance for medical classification
- A checkbox-style table or checklist of disease/symptom names (checked/unchecked, yes/no) is NOT sufficient for medical_healthcheck on its own -- that class requires genuine MEASURED vitals (an actual number for BP, weight, height, BMI, or heart rate). If the image shows only checkboxes/ticks against condition names with no such numeric vitals present, do not let the checkup-form-shaped visual layout alone push you toward medical_healthcheck -- classify by what's actually on the page per the priority order below."""

MACRO_BASE = """You are a Document Macro Classifier for an insurance underwriting pipeline. Classify raw OCR/markdown text into exactly one of: `medical`, `financial`, `identification`, `not_for_underwriting`. Treat the patterns below as supporting evidence, not an exact-string checklist -- read holistically. Ignore PII.

### Ignore Watermarks
Insurance disclaimers (+ date/time stamps) are never evidence of document type -- ignore them completely.

### 1. medical
Route here for genuine clinical content: clinical notes, lab metrics, health-check parameters, prescriptions, test ranges, normal/abnormal flags. Garbled OCR with technical-looking terms next to numbers/units/ranges still counts as low-confidence medical (flag the uncertainty).
- EXCLUSION: Do NOT route here for a hospital/pharmacy receipt or billing statement -- those are `financial` regardless of medical context. Do NOT route here for correspondence that merely discusses or requests medical information without containing the clinical data itself -- that is `not_for_underwriting`. This includes an envelope/mailing label that mentions what's being sent (e.g. "ส่งใบตรวจสุขภาพ" -- sending a health check report) but has no actual clinical data on the page itself -- still `not_for_underwriting`, not medical, regardless of the hospital name being present.

### 2. financial
- Bank/account identity: Account number + bank name/branch, or a native transaction ledger (dated money movements, ideally a markdown table).
- Corporate registration: Registration number + certification/signature block.
- Self income declaration & Income Overrides: The person reporting, clarifying, or declaring their own income ("รายได้" / income). 
  - CRITICAL OVERRIDE: If income ("รายได้") details are present, you MUST route to `financial`, even if the document looks like an insurance e-Form containing premium payment keywords ("ชำระเบี้ยประกัน") or company headers.
- Receipts/Proof-of-payment: Including hospital/pharmacy/medical treatment receipts (ค่ายา, ค่ารักษา, ค่าห้อง), government permit fee receipts, retail receipts, or any other billing statement. All receipts are financial, regardless of what the payment was for.
- General agreements/contracts: Lease, sale, or other business contracts.
- NOT financial: a LETTER or memo requesting/claiming reimbursement for a fee -- e.g. "ขอเบิกค่าตรวจ" (requesting to claim an examination fee) -- is a request ABOUT a payment, not a financial record itself. Route to `not_for_underwriting` (Correspondence). Likewise, money figures describing an insurance policy's own terms (ทุนประกัน, เบี้ยประกัน) inside an application form are not a financial transaction record -- see the eForm Trap below.

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
  - SPECIFIC ANCHORS: A title/heading matching the PATTERN "บันทึกคำชี้แจ้ง/คำชี้แจง...เกี่ยวกับใบคำขอ(เอา)ประกัน(ภัย)" -- a memo clarifying details for an insurance application. Wording and spelling vary -- e.g. "บันทึกคำชี้แจ้งเกี่ยวกับใบคำขอเอาประกัน", "บันทึกคำชี้แจงเกี่ยวกับใบคำขอประกันภัย" -- match the pattern, not one exact string. Also: consent letters, OR documents containing "ชำระเบี้ยประกัน" (premium payment) mixed with a company name/address in the PageHeader.
  - EXCEPTION: If the document explicitly declares actual income details ("รายได้"), route to `financial` instead.
- Litmus test for eForms: Strip out the applicant's name/ID/policy number and any policy-term figures (ทุนประกัน, เบี้ยประกัน) -- is any standalone financial, clinical, or identity statement left? If nothing remains but "applying for / updating a policy," it belongs here.
- Correspondence: Letters/memos between the insured and underwriter requesting documents, or requesting/claiming reimbursement for a fee (administrative communication, not a financial record).
- Envelope/mailing metadata: Sender name, policy codes (e.g., "T1348646"), addressing text -- including when it references sending a medical or financial document without containing that document's actual content.
- Photos/Noise: Portrait photos or ID placeholders without issuing text, or completely illegible noise."""

AGENT1_USER_PROMPT = "Classify the following raw OCR text into medical, financial, identification, or not_for_underwriting:\n{ocr_text}"

FINANCIAL_BASE = """You are an expert Financial Document Specialist. You receive text already confirmed as financial. Perform granular classification into one of six leaf classes. Use markdown structure (tables, headers) where present. Do not process PII.
 
- financial_bankstatement: running transactional ledgers, money transfers, account balances, mobile banking markers -- ideally a markdown table of dated transaction rows.
- financial_bookbank: static account identity headers (bilingual accounts, branches, bank names) without a long transaction ledger.
- financial_companyregistration: formal commercial corporate registry verbiage and certification signatures.
- financial_selfincomedeclaration: the person reporting, clarifying, or declaring their own income, salary, or source of income -- regardless of whether it's laid out as a form or questionnaire.
- financial_receipt: any proof-of-payment document -- hospital/pharmacy/medical treatment receipts, government license/permit fee receipts, retail receipts, or any other billing statement, regardless of what the payment was for.
- financial_others: financial in nature but doesn't cleanly match the above -- also covers general agreements/contracts (lease, sale, business contracts).
"""


AGENT_FINANCIAL_USER_PROMPT = "Perform deep financial classification on this verified text:\n{ocr_text}"

ID_BASE = """You are an expert Identification Document Specialist. You receive text already confirmed as an identification/travel document. Perform granular classification. Use markdown structure where present. Do not process PII.
 
Reminder: insurance watermarks/disclaimers (+ date/time stamps) are never evidence of document type -- ignore them when classifying.

Reminder: a country name/code (e.g. "LAO PDR", "Union of Myanmar") tells you the person's NATIONALITY, not the document TYPE. It does not by itself mean id_foreigner_nationalid rather than id_passport -- check for passport-specific structural signals (MRZ garbled artifact patterns like "<<<<", a booklet biodata-page layout, a passport number field) before deciding. When those are present, they win over a bare country-name association.
 
- id_thainationalid / id_foreigner_nationalid: national demographic card headers and identity issuing text blocks. Foreign-script documents (e.g. a Lao national ID) still count even if largely unreadable by OCR -- look for legible fragments like issuing country name/code, name, or a DOB pattern. Do NOT use this class just because a country name appears in the text -- confirm it's a national ID card layout, not a passport (see id_passport) or a stateless ID (see id_statelessid).
- id_passport: the passport bio data page -- photo/data block, passport number, nationality, an MRZ line (letters/digits + long "<" runs, including garbled OCR fragments like "SACSDWsd<<<<" -- this pattern alone is a strong passport anchor even without a clean country name). If an image is available, a booklet biodata-page layout is a strong confirming signal.
- id_visastamp: a Thai visa page and/or immigration control stamp -- visa category codes, "DEPARTED"/"ADMITTED"/"ENTRY" keywords, and (often garbled, since stamp text is curved) date-like digit strings near them. Does NOT include a Thai E-Visa (see id_thaievisa below).
- id_thaievisa: a Thai Electronic Visa (E-Visa) -- look for the literal term "E-VISA" printed on the document; this alone is a reliable, standalone anchor for this class.
- id_workpermit: labor authorization booklets and official employment permission context. This includes work permit RENEWALS and amendment/endorsement documents such as "Change or addition of category of work" (รายการเปลี่ยนเพิ่มประเภทงาน).
- id_fatca: any FATCA-related US tax form -- W-9, W-4, W-8BEN, or similar FATCA/tax-status declaration or backup-withholding form, entity/individual tax certification sections.
- id_foreignerconfirmationform: an official government-issued confirmation/declaration form attesting to a foreign national's identity or status -- issued BY a government/authority ABOUT the person.
- id_statelessid: a stateless-person identity card -- card layout similar to a national ID (photo placeholder, ID number, name, DOB) but with issuing text indicating stateless/no-registered-nationality status. The exact phrase "บัตรประจำตัวคนไม่มีสัญชาติไทย" (identity card for a person without Thai nationality) ALWAYS means id_statelessid -- never id_foreigner_nationalid. This is a distinct Thai legal status (no registered nationality at all), not the same as a foreign national holding their own country's ID.
- id_driverlicense: driver's license.
- id_houseregistration: household registration document (ทะเบียนบ้าน) -- household registration book header, house/address registry formatting, list of household members.
- id_marriagecertificate: marriage certificate (ใบทะเบียนสมรส) -- marriage registration terminology, groom/bride names, registrar signature, marriage date.
- id_birthcertificate: birth certificate (สูติบัตร) -- birth registration terminology, parents' names, date/place of birth, registrar signature/seal.
- id_others: clearly an ID-type artifact that doesn't cleanly match any subtype above -- e.g. a student ID card, employee badge."""

AGENT_ID_USER_PROMPT = "Perform deep identification classification on this verified text:\n{ocr_text}"

MEDICAL_BASE = """You are a strict Medical Document Specialist. Categorize pre-verified medical OCR text into the exact clinical subclass. Focus on clinical structures and specific medical anchors.
 
### Subclass Priority & Evaluation Guides
Evaluate in this priority order -- if multiple patterns are present on the same page, the higher-priority match wins.
 
- **medical_lab (highest priority):** Laboratory test results. Contains tabular test data with columns for Test Name, Result, Units, and Reference Range.
  Keywords: CBC, BUN, Creatinine, Lipid Profile, Glucose, mg/dL, mmol/L, x10^3/uL.
  NOT included: waveform studies, signal recordings, or functional diagnostics (e.g., ECG, EKG, EMG, EEG, spirometry graphs, imaging reports) -- these go to medical_clinical instead (see below), unless the same page also carries extractable lab values, in which case medical_lab still wins.
 
- **medical_healthcheck (second priority):** Health check / physical examination data. Requires genuine MEASURED vitals -- an actual number/value for Blood Pressure, BMI, Heart Rate, Weight, Height, or a structured wellness summary with real measurements. Applies regardless of whether the source document is a dedicated checkup report (e.g., "Annual Health Checkup", "Executive Health Screening") or a clinical encounter record (OPD/IPD) that includes a Physical Examination (PE) section with structured vitals.
  NOT included: a checkbox/checklist table of disease or symptom names (checked/unchecked, yes/no) with no accompanying numeric vitals -- that has the visual shape of a checkup form but no actual measured values, so it does not qualify here. Route it per medical_clinical or medical_others below instead.

- **medical_clinical (third priority):** Everything genuinely medical that isn't extractable lab data or extractable vitals. This is deliberately a broad category: medical_lab and medical_healthcheck feed a downstream extraction node that pulls structured values (test results, BMI, BP, weight) -- medical_clinical is the landing spot for real clinical content that node doesn't need structured values from. Includes:
  - Clinical narrative/notes: doctor consultation notes, progress notes, admission or discharge summaries, diagnosis lists, prescriptions, treatment plans, operative notes, hospital course documentation. Typically has visit/admission dates, narrative assessments, plans, and medication orders.
  - Patient history forms (ประวัติผู้ป่วย) -- including ones laid out as a checkbox/table of past conditions or symptoms -- these are history/narrative content, not a real-time physical exam with measured vitals.
  - Pathology reports.
  - Imaging or functional diagnostic reports with no extractable lab or vitals data on the page -- X-ray, ultrasound, ECG/EKG, EMG, EEG, spirometry, or similar studies.
  NOT included: pages whose primary content is a table of vital signs or PE measurements with real numeric values (-> medical_healthcheck) or a lab results table (-> medical_lab).
 
- **medical_others:** Medical documents that don't structurally fit any category above.
  SPECIFIC INCLUSION: always route Sleep Test reports here.
"""

AGENT3_USER_PROMPT = "Perform deep clinical classification on this verified medical text:\n{ocr_text}"

def build_prompt(base_prompt: str, use_vision: bool, use_cot: bool, vision_note: str = None) -> str:
    prompt = base_prompt
    if use_vision:
        prompt += vision_note if vision_note is not None else GENERIC_VISION_NOTE
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

def call_with_retry(fn, *args, max_retries: int = 6, base_delay: float = 1.0,
                     max_delay: float = 60.0, **kwargs):
    """CHANGED: the previous version slept the same flat delay on every RateLimitError
    retry attempt (no exponential growth), which under sustained rate limiting just
    re-hammers the endpoint every ~1s until max_retries gives up -- exactly the "lots
    of max retries errors" pattern. Now backs off exponentially (with jitter, so
    concurrent threads don't retry in lockstep) whenever the server doesn't tell us
    exactly how long to wait via Retry-After."""
    for attempt in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except RateLimitError as e:
            retry_after = None
            try:
                retry_after = float(e.response.headers.get("retry-after")) if e.response else None
            except (TypeError, ValueError):
                retry_after = None
            delay = retry_after if retry_after is not None else min(max_delay, base_delay * (2 ** attempt))
            delay += random.uniform(0, delay * 0.25)
            print(f"[RATE LIMIT] attempt {attempt + 1}/{max_retries}, sleeping {delay:.1f}s"
                  f"{' (server Retry-After)' if retry_after is not None else ' (backoff)'}")
            time.sleep(delay)
        except (APIConnectionError, APITimeoutError) as e:
            delay = min(max_delay, base_delay * (2 ** attempt)) + random.uniform(0, 1)
            print(f"[TRANSIENT ERROR] {type(e).__name__} -- attempt {attempt + 1}/{max_retries}, sleeping {delay:.1f}s")
            time.sleep(delay)
    raise RuntimeError(f"Exceeded max_retries ({max_retries}) calling {getattr(fn, '__name__', fn)}")

def call_agent_multimodal(client, model, system_prompt, user_prompt, ocr_text, image_path, use_vision, schema, temperature=0.0, seed=42):
    user_content = [{"type": "text", "text": user_prompt.format(ocr_text=ocr_text)}]
    # CHANGED: only touch disk / base64-encode when the image is actually going to be
    # used. Previously this ran unconditionally, even on text-only calls -- 2 wasted
    # disk reads + encodes per document (macro + specialist), across every worker
    # thread, for no benefit at all when use_vision=False.
    if use_vision:
        encoded_img = encode_image(image_path)
        if encoded_img:
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
# CHANGED: each build_prompt call now passes its own agent-specific vision_note.

def run_cascade_pipeline(ocr_text: str, image_path: Optional[str], client: AzureOpenAI,
                         macro_model: str = "gpt-5.4-mini-swe-d-clab-model01",
                         specialist_model: str = "gpt-5.4-mini-swe-d-clab-model01",
                         use_vision: bool = False,
                         use_cot: bool = True) -> dict:

    # --- STAGE 1: MACRO AGENT ---
    macro_schema = get_schema(use_cot, "macro")
    macro_prompt = build_prompt(MACRO_BASE, use_vision, use_cot, vision_note=MACRO_VISION_NOTE)
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
        spec_schema, sys_p, user_p, vnote = get_schema(use_cot, "medical"), MEDICAL_BASE, AGENT3_USER_PROMPT, MEDICAL_VISION_NOTE
    elif macro_class == "financial":
        spec_schema, sys_p, user_p, vnote = get_schema(use_cot, "financial"), FINANCIAL_BASE, AGENT_FINANCIAL_USER_PROMPT, FINANCIAL_VISION_NOTE
    else: # identification
        spec_schema, sys_p, user_p, vnote = get_schema(use_cot, "id"), ID_BASE, AGENT_ID_USER_PROMPT, ID_VISION_NOTE

    spec_prompt = build_prompt(sys_p, use_vision, use_cot, vision_note=vnote)
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
# 5. MULTITHREADED BATCH RUNNER WITH CHECKPOINT + RESUME
# ==========================================
_RESULT_KEYS = [
    "macro_decision", "macro_reason", "macro_confidence", "macro_latency_sec",
    "macro_tokens_in", "macro_tokens_out",
    "final_subcategory", "specialist_reason", "specialist_confidence",
    "specialist_latency_sec", "specialist_tokens_in", "specialist_tokens_out",
]

_checkpoint_lock = threading.Lock()


def _error_fallback(decision: str, reason: str) -> dict:
    """Complete row matching _RESULT_KEYS, with None/0 for anything not knowable from
    an error. Every row -- success or failure -- has the same columns this way."""
    fallback = {k: None for k in _RESULT_KEYS}
    fallback.update({
        "macro_decision": decision, "macro_reason": reason, "macro_confidence": 0.0,
        "macro_latency_sec": 0.0, "macro_tokens_in": 0, "macro_tokens_out": 0,
        "final_subcategory": decision, "specialist_reason": reason, "specialist_confidence": 0.0,
        "specialist_latency_sec": 0.0, "specialist_tokens_in": 0, "specialist_tokens_out": 0,
    })
    return fallback


def _process_one_row(idx, row, client, text_col, img_col, macro_model, specialist_model, use_vision, use_cot):
    ocr_text = str(row.get(text_col, ""))
    img_path = str(row.get(img_col, "")) if pd.notna(row.get(img_col)) else None

    try:
        res = run_cascade_pipeline(ocr_text, img_path, client, macro_model, specialist_model, use_vision, use_cot)
    except BadRequestError as e:
        error_message = str(e).lower()
        if "content management policy" in error_message or "jailbreak" in error_message:
            print(f"[WARNING] Doc {idx + 1} blocked by content filter. Skipping...")
            res = _error_fallback("blocked_by_firewall", "Azure OpenAI Content Filter flagged this text.")
        else:
            raise
    except Exception as e:
        print(f"[ERROR] Doc {idx + 1} failed: {e}")
        res = _error_fallback("api_error", str(e))

    combined_row = {
        "filepath": img_path,
        "ocr_text": ocr_text,
        **res,
    }
    return idx, combined_row


def run_cascade_batch_checkpointed(df: pd.DataFrame, client: AzureOpenAI,
                                    text_col: str = "ocr_text", img_col: str = "filepath",
                                    macro_model: str = "gpt-54-mini-swe-d-clab-model01",
                                    specialist_model: str = "gpt-54-mini-swe-d-clab-model01",
                                    use_vision: bool = False, use_cot: bool = True,
                                    max_workers: int = 5,
                                    checkpoint_path: str = "eval2/gpt54mini_cls_result.csv",
                                    resume: bool = True) -> pd.DataFrame:
    """Runs the cascade pipeline over df, writing each completed row to
    checkpoint_path as it finishes rather than joining an in-memory results frame back
    onto df at the end. Rerun with the same checkpoint_path to resume -- rows whose
    img_col value is already in the checkpoint are skipped.

    Returns the full checkpoint (including any rows resumed from a prior run), read
    back from disk -- this is always the complete, ground-truth result set, not just
    what this particular call processed.
    """
    already_done = set()
    if resume and os.path.isfile(checkpoint_path):
        prior = pd.read_csv(checkpoint_path)
        if img_col in prior.columns:
            already_done = set(prior[img_col].dropna().astype(str))
        print(f"Resuming: {len(already_done)} document(s) already in checkpoint, will be skipped.")

    pending = [(idx, row) for idx, row in df.iterrows() if str(row.get(img_col)) not in already_done]

    mode_str = "MULTIMODAL VLM (Text + Image)" if use_vision else "TEXT-ONLY"
    print(f"Running cascade agent in [{mode_str}] mode on {len(pending)} document(s) "
          f"(of {len(df)} total, {len(already_done)} already done) with max_workers={max_workers}...")

    fieldnames = None

    def write_row(combined_row):
        nonlocal fieldnames
        with _checkpoint_lock:
            file_exists = os.path.isfile(checkpoint_path)
            if fieldnames is None:
                fieldnames = list(pd.read_csv(checkpoint_path, nrows=0).columns) if file_exists else list(combined_row.keys())
            with open(checkpoint_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()
                writer.writerow(combined_row)

    completed = 0
    iterator_kwargs = dict(total=len(pending), desc="Classifying Docs") if HAS_TQDM else {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_process_one_row, idx, row, client, text_col, img_col,
                             macro_model, specialist_model, use_vision, use_cot): idx
            for idx, row in pending
        }
        progress = tqdm(as_completed(futures), **iterator_kwargs) if HAS_TQDM else as_completed(futures)
        for future in progress:
            idx, combined_row = future.result()
            write_row(combined_row)
            completed += 1
            if not HAS_TQDM:
                print(f"[{completed}/{len(pending)}] {combined_row.get(img_col)} -> {combined_row.get('final_subcategory')}")

    print(f"\nDone. {completed} document(s) processed this run. Full results in '{checkpoint_path}'.")
    return pd.read_csv(checkpoint_path)

# To run in your notebook cell:
#
# df_text = run_cascade_batch_checkpointed(
#     df, client, use_vision=False, checkpoint_path="results_text_only.csv"
# )
# df_image = run_cascade_batch_checkpointed(
#     df, client, use_vision=True, checkpoint_path="results_text_image.csv"
# )