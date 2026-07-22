"""
document_classifier.py
================================
Clean, modular cascade pipeline for Document Classification.
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
from typing import Literal, Optional, Tuple, Type
from pydantic import BaseModel, create_model
from openai import AzureOpenAI, RateLimitError, APIConnectionError, APITimeoutError

# ==========================================
# 1. CORE PROMPT DEFINITIONS (Restored & Tuned)
# ==========================================

VLM_FALLBACK_RULE = """
### Multimodal Analysis Rule
Always start by classifying based on the OCR text first. If the OCR text is heavily garbled, completely trash, or lacks semantic meaning (e.g., noisy scans or purely visual data), seamlessly fallback to relying on the visual structure of the Document Image to make your decision."""

MACRO_BASE = """You are a Document Macro Classifier for an insurance underwriting pipeline. Classify raw OCR/markdown text into exactly one of: `medical`, `financial`, `identification`, `not_for_underwriting`. Treat the patterns below as supporting evidence, not an exact-string checklist -- read holistically. Ignore PII.

### Ignore Watermarks
Insurance disclaimers (+ date/time stamps) are never evidence of document type -- ignore them completely.

### medical
Route here for genuine clinical content: clinical notes, lab metrics, health-check parameters, prescriptions, test ranges, normal/abnormal flags. Garbled OCR with technical-looking terms next to numbers/units/ranges still counts as low-confidence medical (flag the uncertainty).
Do NOT route here for a hospital/pharmacy receipt or billing statement -- those are financial_receipt regardless of medical context (see financial, below). Do NOT route here for correspondence that merely discusses or requests medical information without containing the clinical data itself -- that's not_for_underwriting.

### financial
- Bank/account identity (account number + bank name/branch), or a native transaction ledger (dated money movements, ideally a markdown table).
- Corporate registration number + certification/signature block.
- Self income declaration: the person reporting, clarifying, or declaring their own income or source of income -- even if laid out as a form or questionnaire.
- Any receipt or proof-of-payment document -- including hospital/pharmacy/medical treatment receipts (ค่ายา, ค่ารักษา, ค่าห้อง), government license/permit fee receipts, retail receipts, or any other billing statement. All receipts are financial, regardless of what the payment was for.
- General agreements/contracts: lease, sale, or other business contracts.

### identification
A genuine ID/travel/civil-registry document with its own native structure (not a field filled into someone else's form):
- Issuing country name/code (e.g. "LAO PDR", "RDP LAO"), a genuine name+DOB or name+issue/expiry block, "รับรองสำเนาถูกต้อง" (certified true copy), or a `<figure>` region actually wrapping such a personal-detail block -- not just a name/reference number sitting near a `<figure>` tag with no other identity structure.
- Stateless-ID text, or garbled non-Thai script (e.g. Lao misread as Thai) combined with a country-code header.
- MRZ lines (letters/digits + long "<" runs, e.g. "PA123SD464987<<<<<<<"), or immigration/visa keywords -- "VISA", "VISA CLASS"/"VISACLASS", "IMMIGRATION", "DEPARTED", "ADMITTED", "ENTRY" -- near messy digits/dates. Standardized terms, safe to trust literally even amid heavy OCR noise; don't let a "looks like noise" impression override them.
- FATCA/W-9 tax-withholding declaration forms.
- Work permit documents: labor authorization booklets, employer/employee details, type-of-work (ประเภทงาน) fields, Department of Employment issuing text. This ALSO includes work permit renewals and amendment/endorsement documents such as "Change or addition of category of work" (รายการเปลี่ยนเพิ่มประเภทงาน) -- these are still the same underlying document type, just an update to it, not a new document category.
- Household registration (ทะเบียนบ้าน): household registration book header, house/address registry formatting, list of household members.
- Marriage certificate (ใบสำคัญการสมรส / ทะเบียนสมรส): marriage registration terminology, groom/bride names, registrar signature, marriage date.
- If the document is clearly a personal ID-type card/document (has a photo placeholder, ID number, name, and an issuing context) but doesn't match any pattern above -- e.g. a student ID card, employee badge, or other institutional ID -- it STILL counts as `identification` here. Don't route it to not_for_underwriting just because it isn't one of the specific types listed; the specialist downstream has its own catch-all for exactly this case.
- Reminder: watermark text (see Global Rules above) applies here too. An ID document with an insurance watermark/date stamped near it is still identification -- don't let the watermark, or a date sitting close to it, push you toward not_for_underwriting.

### not_for_underwriting
Default when nothing above applies -- the underwriter will not use this document for processing. This includes several distinct patterns, all landing in this one class:
- Insurance application/policy paperwork (eForm) with no independent financial or identity substance -- fields filled in about an applicant (เบี้ยประกัน, policy details), documents opening with "บันทึกคำชี้แจงเกี่ยวกับใบคำขอประกันภัย", or consent letters for data collection. Litmus test: strip out the applicant's name/ID/policy number -- is any standalone financial, clinical, or identity statement left? If nothing remains but "applying for / updating / requesting details about a policy," it belongs here, regardless of medical or financial-sounding language elsewhere.
- Correspondence: letters or memos between the insured and the underwriter requesting or forwarding additional documents/information -- administrative communication, not a document in its own right.
- Envelope/mailing metadata -- sender name, a policy-style reference code (e.g. "T1348646"), addressing text. Describes or routes a document without containing it.
- A portrait photo or ID placeholder with no accompanying issuing text.
- Illegible noise, or genuinely unclear content that doesn't fit any of the above."""

FINANCIAL_BASE = """You are an expert Financial Document Specialist. You receive text already confirmed as financial. Perform granular classification into one of six leaf classes. Use markdown structure (tables, headers) where present. Do not process PII.

- financial_bankstatement: running transactional ledgers, money transfers, account balances, mobile banking markers -- ideally a markdown table of dated transaction rows.
- financial_bookbank: static account identity headers (bilingual accounts, branches, bank names) without a long transaction ledger.
- financial_companyregistration: formal commercial corporate registry verbiage and certification signatures.
- financial_selfincomedeclaration: the person reporting, clarifying, or declaring their own income, salary, or source of income -- regardless of whether it's laid out as a form or questionnaire.
- financial_receipt: any proof-of-payment document -- hospital/pharmacy/medical treatment receipts, government license/permit fee receipts, retail receipts, or any other billing statement, regardless of what the payment was for.
- financial_others: financial in nature but doesn't cleanly match the above -- also covers general agreements/contracts (lease, sale, business contracts)."""

ID_BASE = """You are an expert Identification Document Specialist. You receive text already confirmed as an identification/travel document. Perform granular classification. Use markdown structure where present. Do not process PII.

Reminder: insurance watermarks/disclaimers (+ date/time stamps) are never evidence of document type -- ignore them when classifying.

- id_thainationalid / id_foreigner_nationalid: national demographic card headers and identity issuing text blocks. Foreign-script documents (e.g. a Lao national ID) still count even if largely unreadable by OCR -- look for legible fragments like issuing country name/code, name, or a DOB pattern.
- id_passport: the passport bio data page -- photo/data block, passport number, nationality, an MRZ line (letters/digits + long "<" runs).
- id_visastamp: a Thai visa page and/or immigration control stamp -- visa category codes, "DEPARTED"/"ADMITTED"/"ENTRY" keywords, and (often garbled, since stamp text is curved) date-like digit strings near them. Does NOT include a Thai E-Visa (see id_thaievisa below) -- that's a distinct electronically-issued document, not a physical stamp/page in a passport.
- id_thaievisa: a Thai Electronic Visa (E-Visa) -- look for the literal term "E-VISA" printed on the document; this alone is a reliable, standalone anchor for this class.
- id_workpermit: labor authorization booklets and official employment permission context. This includes work permit RENEWALS and amendment/endorsement documents such as "Change or addition of category of work" (รายการเปลี่ยนเพิ่มประเภทงาน) -- these are the same underlying document type as the original permit, just an update to it.
- id_fatca_w9: US tax withholding terminology, backup withholding rules, entity declaration sections.
- id_foreignerconfirmationform: an official government-issued confirmation/declaration form attesting to a foreign national's identity or status -- issued BY a government/authority ABOUT the person, not filled in by an agent for a policy application.
- id_statelessid: a stateless-person identity card -- card layout similar to a national ID (photo placeholder, ID number, name, DOB) but with issuing text indicating stateless/no-registered-nationality status.
- id_driverlicense: driver's license.
- id_houseregistration: household registration document (ทะเบียนบ้าน) -- household registration book header, house/address registry formatting, list of household members.
- id_marriagecertificate: marriage certificate (ใบทะเบียนสมรส) -- marriage registration terminology, groom/bride names, registrar signature, marriage date.
- id_birthcertificate: birth certificate (สูติบัตร) -- birth registration terminology, parents' names, date/place of birth, registrar signature/seal.
- id_others: clearly an ID-type artifact that doesn't cleanly match any subtype above -- e.g. a student ID card, employee badge, or other institutional ID card. Expected, valid outcome for these and other borderline documents -- don't force a wrong specific subtype."""

MEDICAL_BASE = """You are a strict Medical Document Specialist. Categorize pre-verified medical OCR text into the exact clinical subclass. Focus on clinical structures and specific medical anchors.

### Subclass Priority & Evaluation Guides
Evaluate in this priority order -- if multiple patterns are present on the same page, the higher-priority match wins.

- **medical_lab (highest priority):** Laboratory test results. Contains tabular test data with columns for Test Name, Result, Units, and Reference Range.
  Keywords: CBC, BUN, Creatinine, Lipid Profile, Glucose, mg/dL, mmol/L, x10^3/uL.
  NOT included: waveform studies, signal recordings, or functional diagnostics (e.g., ECG, EKG, EMG, EEG, spirometry graphs, imaging reports) -- these go to medical_clinical instead (see below), unless the same page also carries extractable lab values, in which case medical_lab still wins.

- **medical_healthcheck (second priority):** Health check / physical examination data. Any page containing extractable vital signs or physical examination findings -- Blood Pressure, BMI, Heart Rate, Weight, Height, or a structured wellness summary. Applies regardless of whether the source document is a dedicated checkup report (e.g., "Annual Health Checkup", "Executive Health Screening") or a clinical encounter record (OPD/IPD) that includes a Physical Examination (PE) section with structured vitals.

- **medical_clinical (third priority):** Everything genuinely medical that isn't extractable lab data or extractable vitals. This is deliberately a broad category: medical_lab and medical_healthcheck feed a downstream extraction node that pulls structured values (test results, BMI, BP, weight) -- medical_clinical is the landing spot for real clinical content that node doesn't need structured values from. Includes:
  - Clinical narrative/notes: doctor consultation notes, progress notes, admission or discharge summaries, diagnosis lists, prescriptions, treatment plans, operative notes, hospital course documentation. Typically has visit/admission dates, narrative assessments, plans, and medication orders.
  - Pathology reports.
  - Imaging or functional diagnostic reports with no extractable lab or vitals data on the page -- X-ray, ultrasound, ECG/EKG, EMG, EEG, spirometry, or similar studies.
  NOT included: pages whose primary content is a table of vital signs or PE measurements (-> medical_healthcheck) or a lab results table (-> medical_lab).

- **medical_others:** Medical documents that don't structurally fit any category above.
  SPECIFIC INCLUSION: always route Sleep Test reports here."""

# ==========================================
# 2. DYNAMIC SCHEMAS (Handles CoT Toggling)
# ==========================================

def get_schema(use_cot: bool, agent_type: str) -> Type[BaseModel]:
    """Dynamically builds Pydantic schema based on CoT requirements."""
    fields = {}
    if use_cot:
        fields["chain_of_thought"] = (str, ...)

    if agent_type == "macro":
        fields["macro_class"] = (Literal["medical", "financial", "identification", "not_for_underwriting"], ...)
    elif agent_type == "financial":
        fields["subcategory"] = (Literal["financial_bankstatement", "financial_bookbank", "financial_companyregistration", "financial_selfincomedeclaration", "financial_receipt", "financial_others"], ...)
    elif agent_type == "id":
        fields["subcategory"] = (Literal["id_driverlicense", "id_fatca_w9", "id_foreignerconfirmationform", "id_foreigner_nationalid", "id_visastamp", "id_thaievisa", "id_passport", "id_statelessid", "id_thainationalid", "id_workpermit", "id_houseregistration", "id_marriagecertificate", "id_birthcertificate", "id_others"], ...)
    elif agent_type == "medical":
        fields["subcategory"] = (Literal["medical_clinical", "medical_healthcheck", "medical_lab", "medical_others"], ...)

    return create_model(f"{agent_type.capitalize()}Output", **fields)

def build_prompt(base: str, use_vision: bool, use_cot: bool) -> str:
    """Combines the base prompt with situational instructions."""
    prompt = base
    if use_vision:
        prompt += f"\n{VLM_FALLBACK_RULE}"
    if use_cot:
        prompt += "\n\nIn chain_of_thought: Provide your step-by-step structural rationale based on the rules above before selecting the final class."
    return prompt

# ==========================================
# 3. CORE HELPERS
# ==========================================

def encode_image(image_path: str) -> Optional[str]:
    if not image_path or not os.path.isfile(image_path): return None
    mime_type = mimetypes.guess_type(image_path)[0] or 'image/jpeg'
    with open(image_path, "rb") as f:
        return f"data:{mime_type};base64,{base64.b64encode(f.read()).decode('utf-8')}"

def get_confidence(logprobs, key: str = "") -> float:
    """Averages logprobs safely. If specific key searching is needed, expand here."""
    if not logprobs or not logprobs.content: return 0.0
    toks = [t.logprob for t in logprobs.content if t.logprob is not None]
    return round(math.exp(sum(toks) / len(toks)) * 100, 2) if toks else 0.0

def call_agent(client: AzureOpenAI, model: str, sys_prompt: str, ocr_text: str, 
               image_path: Optional[str], use_vision: bool, schema: Type[BaseModel]) -> Tuple[BaseModel, float, float]:
    """Universal LLM caller tracking execution latency."""
    content = [{"type": "text", "text": f"Document Text:\n{ocr_text}"}]
    if use_vision and image_path and (img_b64 := encode_image(image_path)):
        content.append({"type": "image_url", "image_url": {"url": img_b64, "detail": "auto"}})

    for attempt in range(4):
        try:
            start_time = time.time()
            res = client.beta.chat.completions.parse(
                model=model,
                messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": content}],
                response_format=schema,
                logprobs=True, top_logprobs=1, temperature=0.0, seed=42
            )
            latency = round(time.time() - start_time, 3)
            return res.choices[0].message.parsed, get_confidence(res.choices[0].logprobs), latency
        except RateLimitError as e:
            time.sleep(2 ** attempt + random.uniform(0, 1))
        except (APIConnectionError, APITimeoutError):
            time.sleep(1.5)
    raise RuntimeError("LLM Call Failed after retries.")

# ==========================================
# 4. MAIN PIPELINE EXECUTION
# ==========================================

def run_cascade(ocr_text: str, image_path: Optional[str], client: AzureOpenAI, 
                model: str = "gpt-54mini-swe-d-clab-model01", 
                use_vision: bool = False, use_cot: bool = True) -> dict:
    
    # 1. MACRO STAGE
    macro_schema = get_schema(use_cot, "macro")
    macro_prompt = build_prompt(MACRO_BASE, use_vision, use_cot)
    macro_data, macro_conf, mac_lat = call_agent(client, model, macro_prompt, ocr_text, image_path, use_vision, macro_schema)
    
    res = {
        "macro_decision": macro_data.macro_class,
        "macro_reason": getattr(macro_data, "chain_of_thought", "CoT Disabled"),
        "macro_confidence": macro_conf,
        "macro_latency_sec": mac_lat,
        "specialist_latency_sec": 0.0,
        "total_latency_sec": mac_lat
    }

    # 2. SPECIALIST STAGE
    spec_schema, spec_base = None, None
    if macro_data.macro_class == "medical":
        spec_schema, spec_base = get_schema(use_cot, "medical"), MEDICAL_BASE
    elif macro_data.macro_class == "financial":
        spec_schema, spec_base = get_schema(use_cot, "financial"), FINANCIAL_BASE
    elif macro_data.macro_class == "identification":
        spec_schema, spec_base = get_schema(use_cot, "id"), ID_BASE

    if spec_schema:
        spec_prompt = build_prompt(spec_base, use_vision, use_cot)
        spec_data, spec_conf, spec_lat = call_agent(client, model, spec_prompt, ocr_text, image_path, use_vision, spec_schema)
        res.update({
            "final_subcategory": spec_data.subcategory,
            "specialist_reason": getattr(spec_data, "chain_of_thought", "CoT Disabled"),
            "specialist_confidence": spec_conf,
            "specialist_latency_sec": spec_lat,
            "total_latency_sec": round(mac_lat + spec_lat, 3)
        })
    else:
        # Terminal branch
        res.update({
            "final_subcategory": "not_for_underwriting",
            "specialist_reason": "Terminal at macro level.",
            "specialist_confidence": macro_conf
        })

    return res

# Usage in Jupyter:
# client = AzureOpenAI(...)
# result = run_cascade("raw text here", "path/to/img.jpg", client, use_vision=True, use_cot=False)