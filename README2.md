# Document Classification — Text-only vs Multimodal Comparison

**Date:** `YYYY-MM-DD`
**Sample size:** 1,127 documents
**Model:** `gpt-54mini-swe-d-clab-model01` (same deployment, both runs)
**Pipeline version:** `cascade_pipeline_v24.py` (or note actual version used)

---

## ⚠️ Worth double-checking before drawing conclusions

Multimodal shows **lower** average latency (2.84s vs 4.18s) and **lower** average input tokens (3,567 vs 4,280) than text-only, despite sending an image on top of the same OCR text. That's the opposite of what adding an image would normally do, so before treating the latency/token numbers below as real findings, worth ruling out:
- Different time-of-day / API load between the two runs (text-only and multimodal runs happening at different times can easily produce a 1.3s latency gap on their own, unrelated to the mode itself).
- Whether routing actually matched between runs — if a document routes to `medical` in one run and `financial` in the other (different specialist prompt lengths), that alone shifts the average token_in independent of vision.
- Whether `image_detail="low"` is quietly capping image token cost low enough that it's not the dominant factor, while something else (e.g. retry overhead from the backoff bug fixed earlier) inflated the text-only run's latency instead of image processing costing multimodal anything extra.

If this holds up under a rerun, it's a genuinely interesting result worth calling out prominently in the summary. If it doesn't hold up, treat it as an artifact of when/how each run was executed.

---

## Summary

| Metric | Multimodal | Text-only |
|---|---|---|
| Macro accuracy | **97.78%** | 96.72% |
| End-to-end accuracy | **88.79%** | 87.67% |
| Medical specialist accuracy | **82.58%** | 82.30% |
| Financial specialist accuracy | 95.97% | **96.67%** |
| Identification specialist accuracy | **92.20%** | 91.94% |
| Avg. end-to-end latency | 2.8426 sec | 4.1815 sec |
| Avg. end-to-end input tokens | 3,567.19 | 4,279.71 |
| Avg. end-to-end output tokens | 28.30 | 28.20 |

**Headline:** multimodal wins on 4 of 5 accuracy metrics (macro, e2e, medical, identification); text-only wins on financial specifically. Net effect on financial is small (−0.70pp) relative to the e2e gain (+1.12pp), so multimodal looks like the better default — but see the latency/token caveat above before treating this as fully settled.

---

## 1. Macro Classification (by domain)

`medical` / `financial` / `identification` / `not_for_underwriting`

| Metric | Multimodal | Text-only |
|---|---|---|
| Overall accuracy | **97.78%** | 96.72% |

### 1.1 Per-domain breakdown

`[Fill in from your confusion matrix per mode -- precision/recall/F1 per domain aren't derivable from a single accuracy number]`

| Domain | Precision (MM) | Recall (MM) | F1 (MM) | Precision (Text) | Recall (Text) | F1 (Text) | Support |
|---|---|---|---|---|---|---|---|
| medical | | | | | | | |
| financial | | | | | | | |
| identification | | | | | | | |
| not_for_underwriting | | | | | | | |

### 1.2 Confusion matrix — Multimodal
`[Embed image or paste table here]`

### 1.3 Confusion matrix — Text-only
`[Embed image or paste table here]`

**Notable differences between modes:** `[e.g. "eForm-vs-financial confusion mostly resolved in multimodal but N cases remain in text-only"]`

---

## 2. End-to-End (Leaf-level)

`ground_truth (gt)` vs `final_subcategory` — full taxonomy across all domains

| Metric | Multimodal | Text-only |
|---|---|---|
| Overall accuracy | **88.79%** | 87.67% |

### 2.1 Confusion matrix — Multimodal
`[Embed image or paste table here -- likely large (~24 classes), an image export is more readable than a markdown table at this size]`

### 2.2 Confusion matrix — Text-only
`[Embed image or paste table here]`

**Notable differences between modes:** `[fill in]`

---

## 3. Specialist Breakdown

### 3.1 Medical Specialist

`medical_clinical` / `medical_healthcheck` / `medical_lab` / `medical_others`

| Metric | Multimodal | Text-only |
|---|---|---|
| Accuracy | **82.58%** | 82.30% |

#### Confusion matrix — Multimodal
`[Embed image or paste table here]`

#### Confusion matrix — Text-only
`[Embed image or paste table here]`

---

### 3.2 Financial Specialist

`financial_bankstatement` / `financial_bookbank` / `financial_companyregistration` / `financial_selfincomedeclaration` / `financial_receipt` / `financial_others`

| Metric | Multimodal | Text-only |
|---|---|---|
| Accuracy | 95.97% | **96.67%** |

**The one metric where text-only wins.** Worth a specific look at which financial documents flip between modes — e.g. does the image ever introduce a false visual cue (a logo, a stamp) that pulls a genuinely financial document toward a wrong leaf class?

#### Confusion matrix — Multimodal
`[Embed image or paste table here]`

#### Confusion matrix — Text-only
`[Embed image or paste table here]`

---

### 3.3 Identification Specialist

`id_thainationalid` / `id_foreigner_nationalid` / `id_passport` / `id_visastamp` / `id_thaievisa` / `id_workpermit` / `id_fatca` / `id_foreignerconfirmationform` / `id_statelessid` / `id_driverlicense` / `id_houseregistration` / `id_marriagecertificate` / `id_birthcertificate` / `id_others`

| Metric | Multimodal | Text-only |
|---|---|---|
| Accuracy | **92.20%** | 91.94% |

#### Confusion matrix — Multimodal
`[Embed image or paste table here]`

#### Confusion matrix — Text-only
`[Embed image or paste table here]`

---

## 4. Latency & Token Usage

| Metric | Multimodal | Text-only | Δ (MM − Text) |
|---|---|---|---|
| Avg. e2e latency (sec) | 2.8426 | 4.1815 | −1.3389 |
| Avg. e2e input tokens | 3,567.19 | 4,279.71 | −712.52 |
| Avg. e2e output tokens | 28.30 | 28.20 | +0.10 |

Output tokens are nearly identical between modes, which makes sense — that's mostly a function of `chain_of_thought` length (or its absence) and the classification label itself, not the input modality. The input token and latency gap is the part flagged in the caveat above as worth re-confirming before relying on it.

`[If useful: cost comparison here too, once you have $/1K token pricing for this deployment -- e2e input tokens x price gives a rough per-document cost delta between modes.]`

---

## Appendix: Sample-level disagreements

`[Documents where multimodal and text-only disagree on the final classification -- useful for spot-checking whether the image is actually adding signal or just changing the answer without improving it. Pull from your review app's drill-down.]`

| Filepath | GT | Multimodal Pred | Text-only Pred | Notes |
|---|---|---|---|---|
| | | | | |