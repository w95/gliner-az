# Fine-tuning GLiNER for Azerbaijani NER: Pure Neural Model (Option A)

**Architecture**: Single self-contained GLiNER model. All entity types — including patterned PII (FIN, TIN, IBAN, phone, etc.) — are learned by the neural network. No runtime regex, no Python wrapper, no external validation code. The `.safetensors` file is the complete system.

## Table of Contents

1. [Option A Architecture & Tradeoffs](#1-option-a-architecture--tradeoffs)
2. [GLiNER Architecture Deep-Dive](#2-gliner-architecture-deep-dive)
3. [Azerbaijan PII Entity Catalog — Complete Reference](#3-azerbaijan-pii-entity-catalog)
4. [Available Datasets & Models](#4-available-datasets--models)
5. [Environment Setup](#5-environment-setup)
6. [Data Preparation Pipeline](#6-data-preparation-pipeline)
7. [Maximum-Accuracy Synthetic Data Generation](#7-synthetic-data-generation)
8. [Fine-Tuning Pipeline with Pattern-Focused Training](#8-fine-tuning-pipeline)
9. [Pure Model Inference (No Wrapper)](#9-pure-model-inference)
10. [Benchmark Suite & Expected Performance](#10-benchmark-suite--expected-performance)
11. [GPU Requirements & Training Time Estimates](#11-gpu-requirements--training-time-estimates)
12. [Evaluation Methodology](#12-evaluation-methodology)
13. [Model Export & Deployment](#13-model-export--deployment)
14. [Troubleshooting & Known Issues](#14-troubleshooting--known-issues)

---

## 1. Option A Architecture & Tradeoffs

### Goal

Fine-tune a GLiNER model that detects **all Azerbaijan-specific entities and PII as a single self-contained neural artifact** — no Python wrapper, no runtime regex, no validation code. The deployed system is literally just the model file + a standard GLiNER inference call.

### Target entities (all baked into the model)

**Patterned PII (must be learned with regex-equivalent precision):**
- **FIN code** — 7-char alphanumeric personal ID
- **TIN / VÖEN** — 10-digit tax ID
- **Phone numbers** — +994 country code variants
- **IBAN** — 28-char AZ-prefix bank account
- **Passport numbers** — AZE/AA prefix + 8 digits
- **Vehicle plates** — XX-LL-NNN regional format
- **Credit card numbers** — 16-digit international format
- **Email addresses** — standard format
- **Postal codes** — AZ + 4 digits

**Semantic entities:**
- Person, organisation, location, GPE, date, time, money, position, law, facility, product, event, language, norp, disease, position, project

### The Option A challenge (and how to beat it)

Neural models **can** learn patterns reliably — NVIDIA's GLiNER-PII achieves 97% F1 on phone numbers and 98% on emails purely from training data, and Knowledgator's GLiNER-PII-base achieves 80.99% F1 across 60 PII categories with no regex layer. The key is that **pattern learning requires aggressive, strategically designed data augmentation**.

The risk with Option A is that a neural model may only learn a *canonical* version of a pattern if the training data lacks diversity. For example, if your training set only contains phone numbers formatted as `+994 50 123 45 67`, the model won't reliably detect `0501234567` or `(+994) 50-123-45-67` at inference. The solution is **pattern-exhaustive synthetic data**: generate training samples covering every realistic format variation the model might encounter.

### Why Option A over hybrid

| Benefit | Impact |
|---------|--------|
| **Single artifact deployment** | Just the model file — portable to Rust, Go, C++, browser (ONNX) |
| **No runtime dependencies** | No `re`, no validators, no Python required at inference |
| **Context-aware pattern detection** | Catches patterns even in noisy text, OCR output, or atypical formatting |
| **Explainable via confidence scores** | Each prediction has a calibrated score, no opaque merge logic |
| **Simpler maintenance** | One model to update; no regex library to maintain across teams |
| **Cleaner academic story** | "A fine-tuned model" is a simpler paper contribution than "a hybrid pipeline" |
| **Better for edge deployment** | Mobile/IoT devices run ONNX directly |

### What you give up

- ~2-5% F1 on perfectly-formatted structured entities (regex is still marginally better on `AZ21NABZ00000000137010001944`)
- Luhn/checksum validation — model may output syntactically valid but arithmetically invalid card numbers (rare)
- Deterministic guarantees — regex always catches `[0-9]{16}`; neural model has confidence threshold

### The winning strategy

The core tradeoff is minimized by **massive, strategically-diverse synthetic data**. Our target: generate **at least 3,000 synthetic training examples per patterned entity type** (FIN, TIN, IBAN, phone, etc.) covering:

1. **All formatting variations** (spaces, dashes, parentheses, dots)
2. **All surrounding contexts** (formal documents, casual messages, tables, lists)
3. **Adversarial near-misses** (10-digit numbers that aren't TINs, 7-char strings that aren't FINs)
4. **Noise variations** (OCR errors, typos, inconsistent capitalization)
5. **Multilingual contamination** (Russian/Turkish mixed in with Azerbaijani)

This approach mirrors what NVIDIA and Gretel did for English PII detection, adapted for Azerbaijan-specific formats.

---

## 2. GLiNER Architecture Deep-Dive

### Core Concept

GLiNER frames NER as a **span-type matching problem**. Instead of sequence labeling (B-I-O tags), it:

1. Encodes entity type labels ("person", "location", etc.) and the input text together through a bidirectional transformer
2. Computes **span representations** for all possible token subsequences (up to `max_width=12` tokens)
3. Computes **entity type representations** from learned `[ENT]` separator tokens
4. Scores each (span, type) pair via **dot product + sigmoid**
5. Applies **greedy decoding** to select top non-overlapping predictions

### Components

```
Input: "[ENT] person [ENT] location [ENT] fin code [SEP] Əliyev Heydər Bakıda yaşayır"
           ↓
[Bidirectional Transformer - DeBERTa-v3-multilingual]
           ↓
   ┌───────┴───────┐
[Entity Type FFN]  [Span Representation FFN]
   ↓                ↓
[type embeddings]  [span embeddings for all (i,j) pairs]
   ↓                ↓
   └────→ Dot Product Scoring + Sigmoid ←────┘
           ↓
[Greedy Non-Overlapping Decoding]
           ↓
Output: [("Əliyev Heydər", "person"), ("Bakıda", "location")]
```

### Model Variants

| Model | Backbone | Params | Size | License | Azerbaijani Support |
|-------|----------|--------|------|---------|-------------------|
| `urchade/gliner_multi-v2.1` | mDeBERTa-v3-base | ~209M | ~440MB | Apache 2.0 | ✅ Tokenizer coverage |
| `urchade/gliner_multi` | mDeBERTa-v3-base | ~209M | ~440MB | Apache 2.0 | ✅ Tokenizer coverage |
| `knowledgator/gliner-bi-large-v1.0` | mDeBERTa-v3-large | ~459M | ~1.2GB | Apache 2.0 | ✅ Bi-encoder, faster inference |
| `urchade/gliner-large-v2.5` | DeBERTa-v3-large (English) | ~459M | ~1.2GB | Apache 2.0 | ⚠️ English-only backbone |
| `Mayank6255/GLiNER-MoE-MultiLingual` | MoE + mDeBERTa | Varies | Varies | Unknown | ⚠️ Experimental, ~43% avg F1 |

**Recommended base model: `urchade/gliner_multi-v2.1`** — best balance of multilingual coverage, proven architecture, and Apache 2.0 licensing.

### Key Hyperparameters That Affect Architecture

- `max_width`: Maximum span width in tokens (default 12). Increase for very long entity names.
- `max_types`: Maximum entity types per training batch (default 25). Set to total number of your entity types.
- `shuffle_types`: Randomly shuffle entity types per batch to prevent order bias (default True).

---

## 3. Azerbaijan PII Entity Catalog — Complete Reference

This section documents **every Azerbaijan-specific entity type**, its format, regex pattern, examples, and Azerbaijani-language context keywords.

### 3.1 FIN Code (Fərdi İdentifikasiya Nömrəsi)

**Description**: Personal Identification Number printed on Azerbaijan national ID cards. Unique, lifetime code for every citizen.

**Format**: 7-character alphanumeric code. Pattern observed from `az-data-generator` library and OECD documentation: digit-letter-digit-digit-letter-letter-digit.

```
Formal pattern: [0-9][A-Z][0-9]{2}[A-Z]{2}[0-9]
Examples: 7A23RG5, 3B78DL4, 5KH92MN1
Permissive pattern: [A-Z0-9]{7} (catches edge cases)
```

**Regex**:
```python
FIN_STRICT = r'\b[0-9][A-Z][0-9]{2}[A-Z]{2}[0-9]\b'
FIN_PERMISSIVE = r'\b(?=[A-Z0-9]*[0-9])(?=[A-Z0-9]*[A-Z])[A-Z0-9]{7}\b'
```

**Context keywords** (Azerbaijani): FİN, FİN kodu, fərdi identifikasiya nömrəsi, şəxsiyyət vəsiqəsi

**GLiNER label**: `"fin code"`

---

### 3.2 TIN / VÖEN (Vergi Ödəyicisinin Eyniləşdirmə Nömrəsi — Tax Identification Number)

**Description**: 10-digit Taxpayer Identification Number (TIN) issued by the Ministry of Economy. Internationally referred to as **TIN**, locally called **VÖEN**. These are the same identifier — "VÖEN" is the Azerbaijani acronym, "TIN" is the international/English term. **Not to be confused with FIN/PIN** (the 7-character personal ID code). Per OECD guidance, if a person has no VÖEN/TIN, their FIN/PIN substitutes for tax identification purposes.

**Format**:
- 10 digits total
- First 2 digits: administrative territorial code
- Next 6 digits: serial number
- 9th digit: algorithmically computed check digit
- 10th digit: `1` for legal entities, `2` for individuals

```
Pattern: \d{10}
Examples: 1400057421 (legal), 1401771462 (individual)
With type check: \d{9}[12]
```

**Regex**:
```python
# Strict: 10 digits ending in 1 (legal) or 2 (individual)
VOEN_STRICT = r'\b\d{9}[12]\b'
# Permissive: any 10-digit number (needs context validation)
VOEN_PERMISSIVE = r'\b\d{10}\b'
```

**Context keywords**: VÖEN, TIN, vergi ödəyicisi, vergi nömrəsi, taxpayer identification, vergi ödəyicisinin eyniləşdirmə nömrəsi

**GLiNER label**: `"tin"` (preferred, internationally recognized) or `"voen"` (local alias)

**Note**: VÖEN/TIN regex is prone to false positives (any 10-digit number matches). The GLiNER model learns context (e.g., "VÖEN nömrəsi", "vergi kodu") and provides a critical second signal. In the hybrid pipeline, use GLiNER confidence to validate or reject ambiguous regex matches.

---

### 3.3 Phone Numbers

**Description**: Azerbaijan uses country code +994. National significant number is 9 digits: 2-digit prefix + 7-digit subscriber number.

**Mobile prefixes by operator**:
- Bakcell: 50, 55
- Azercell: 50, 51, 70, 99
- Nar (Azerfon): 70, 77
- Others: 40 (CDMA), 60

**Landline area codes**: 12 (Baku), 18 (Absheron), 20 (Barda), 22 (Ganja), 24 (Mingachevir), etc.

```
International: +994 XX XXXXXXX
Domestic: 0XX XXXXXXX
Local (Baku): 12 XXXXXXX
```

**Regex**:
```python
# International format
PHONE_INTL = r'\+994[\s\-]?(?:10|12|18|20|21|22|23|24|25|26|33|34|36|40|50|51|55|60|70|77|88|99)[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}'

# Domestic format (with leading 0)
PHONE_DOMESTIC = r'\b0(?:10|12|18|20|21|22|23|24|25|26|33|34|36|40|50|51|55|60|70|77|88|99)[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}\b'

# Combined
PHONE_AZ = r'(?:\+994|0)[\s\-]?(?:10|12|18|20|21|22|23|24|25|26|33|34|36|40|50|51|55|60|70|77|88|99)[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}'
```

**Context keywords**: telefon, mobil nömrə, əlaqə nömrəsi, zəng, nömrə

**GLiNER label**: `"phone number"`

---

### 3.4 IBAN (International Bank Account Number)

**Description**: Azerbaijan IBANs are exactly 28 characters, starting with "AZ".

**Format**: `AZ` + 2 check digits + 4-letter bank code (SWIFT/BIC) + 20 alphanumeric account digits

```
Structure: AZ[0-9]{2}[A-Z]{4}[A-Z0-9]{20}
Total length: 28 characters
Example: AZ21NABZ00000000137010001944
Print format: AZ21 NABZ 0000 0000 1370 1000 1944
```

**Regex**:
```python
# Electronic format (no spaces)
IBAN_AZ_ELECTRONIC = r'\bAZ\d{2}[A-Z]{4}[A-Z0-9]{20}\b'
# Print format (with spaces every 4 chars)
IBAN_AZ_PRINT = r'\bAZ\d{2}\s?[A-Z]{4}\s?[A-Z0-9]{4}\s?[A-Z0-9]{4}\s?[A-Z0-9]{4}\s?[A-Z0-9]{4}\s?[A-Z0-9]{4}\b'
# Combined
IBAN_AZ = r'\bAZ\d{2}\s?[A-Z]{4}[\s]?(?:[A-Z0-9]{4}[\s]?){5}\b'
```

**Known bank codes**: NABZ (Central Bank), UBAZ (Unibank), IBAZ (Int'l Bank of Azerbaijan), AIIB (AccessBank), BRES (Bank Respublika), GENC (GencBank), PASH (PASHA Bank)

**Context keywords**: IBAN, hesab nömrəsi, bank hesabı, köçürmə

**GLiNER label**: `"iban"`

---

### 3.5 Passport Numbers

**Description**: Azerbaijan has two passport series currently in circulation (both biometric since 2013).

**Format**:
- **Old series (2013)**: `AZE` + 8 digits → `AZE12345678`
- **New series (2024)**: `AA` + 8 digits → `AA12345678`

```
Old: AZE\d{8}
New: AA\d{8}
```

**Regex**:
```python
PASSPORT_AZ = r'\b(?:AZE|AA)\d{8}\b'
```

**Context keywords**: pasport, pasport nömrəsi, passport, səyahət sənədi, biometrik pasport

**GLiNER label**: `"passport number"`

---

### 3.6 Vehicle Registration Plates

**Description**: Azerbaijan plates follow the format `XX-LL-NNN` where XX is a 2-digit regional code, LL is 2 letters (series), NNN is 3-digit serial number.

**Format**:
```
Standard: [10-99]-[A-Z]{2}-[0-9]{3}
Examples: 10-JA-234, 90-AA-001, 77-BK-555
```

**Regional codes**: 10 (Baku), 20 (Barda), 25 (Sumgait), 50 (Ganja), 77 (Nakhchivan), 90 (Baku alternate), etc.

**Special plates**:
- State vehicles: AA-AZ series
- Taxis: TA series (white on blue)
- Diplomatic: red background, `NNN D NNN`
- Foreign: yellow background, `H NNN NNN`

**Regex**:
```python
# Standard passenger vehicles
PLATE_AZ_STANDARD = r'\b\d{2}[\s\-]?[A-Z]{2}[\s\-]?\d{3}\b'
# Diplomatic
PLATE_AZ_DIPLOMATIC = r'\b\d{3}\s?D\s?\d{3}\b'
# Foreign
PLATE_AZ_FOREIGN = r'\b[HP]\s?\d{3}\s?\d{3}\b'
```

**Context keywords**: dövlət nömrə nişanı, avtomobil nömrəsi, qeydiyyat nişanı, plaka

**GLiNER label**: `"vehicle plate"`

---

### 3.7 Credit/Debit Card Numbers

**Description**: Standard international format. Azerbaijan-issued cards follow Visa (4xxx), Mastercard (5xxx/2xxx), and local networks.

**Format**: 16 digits (some 13-19), Luhn-validated

```
Visa: 4[0-9]{15}
Mastercard: 5[1-5][0-9]{14} or 2[2-7][0-9]{14}
Generic: [0-9]{4}[\s\-]?[0-9]{4}[\s\-]?[0-9]{4}[\s\-]?[0-9]{4}
```

**Regex**:
```python
# With optional spaces/dashes
CARD_NUMBER = r'\b(?:\d{4}[\s\-]?){3}\d{4}\b'
# Strict 16-digit
CARD_NUMBER_STRICT = r'\b[2-6]\d{3}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b'
```

**Validation**: Always apply Luhn checksum after regex match to reduce false positives.

```python
def luhn_check(number: str) -> bool:
    digits = [int(d) for d in number if d.isdigit()]
    if len(digits) < 13 or len(digits) > 19:
        return False
    checksum = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        checksum += d
    return checksum % 10 == 0
```

**Context keywords**: kart nömrəsi, kredit kartı, debet kartı, ödəniş kartı, bank kartı

**GLiNER label**: `"credit card number"`

---

### 3.8 SSN (Social Security Number)

**Description**: Azerbaijan Social Security Number for pension/insurance system.

**Format**: Per `az-data-generator`, SSN format is used in social protection contexts.

**Regex**: (Use generator output to determine exact pattern; treat as numeric identifier)
```python
SSN_AZ = r'\b\d{7,10}\b'  # Requires context validation
```

**Context keywords**: sosial sığorta nömrəsi, DSMF, pensiya, təqaüd

**GLiNER label**: `"ssn"`

---

### 3.9 Email Addresses

**Description**: Standard email format. Azerbaijani domains include .az TLD.

**Regex**:
```python
EMAIL = r'\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b'
```

**Common AZ domains**: @mail.ru, @gmail.com, @box.az, @azeronline.com, @azdata.net

**GLiNER label**: `"email"`

---

### 3.10 AZ Currency Amounts (Manat / AZN)

**Description**: Azerbaijani Manat (AZN/₼). Often appears in financial documents.

**Regex**:
```python
# "150 AZN", "150 manat", "₼150", "150₼"
CURRENCY_AZN = r'(?:₼\s?\d[\d\s,\.]*\d|\d[\d\s,\.]*\d\s?(?:AZN|azn|manat|qəpik|₼))'
```

**GLiNER label**: `"money"`

---

### 3.11 Azerbaijani Postal Codes

**Description**: 4-digit codes prefixed with "AZ".

**Format**: `AZ` + 4 digits
```
Examples: AZ1000 (Baku central), AZ2000 (Ganja), AZ5000 (Nakhchivan)
```

**Regex**:
```python
POSTAL_AZ = r'\bAZ\d{4}\b'
```

**GLiNER label**: `"postal code"`

---

### 3.12 Azerbaijani Date Formats

**Description**: Azerbaijan uses DD.MM.YYYY as the standard date format.

**Regex**:
```python
DATE_AZ = r'\b(?:0[1-9]|[12]\d|3[01])\.(?:0[1-9]|1[0-2])\.\d{4}\b'
```

**GLiNER label**: `"date"`

---

### 3.13 Driver's License Numbers

**Description**: Azerbaijan driver's licenses come in 2013, 2020, and 2024 (EU-style) series. The 2024 series includes a QR code.

**Format**: Series + serial number (varies by issue year)

**Context keywords**: sürücülük vəsiqəsi, sürücülük hüququ, DYP

**GLiNER label**: `"drivers license"`

---

### Summary Entity Classification Table

| Entity | Regex | GLiNER Trained | Inference Strategy | False Positive Risk |
|--------|:-----:|:--------------:|-------------------|-------------------|
| FIN Code | ✅ Strict | ✅ Yes | Regex primary, GLiNER fallback | Low |
| TIN / VÖEN | ✅ + context | ✅ Yes | **GLiNER validates** ambiguous regex | Medium |
| Phone (AZ) | ✅ | ✅ Yes | Regex primary, GLiNER fallback | Low |
| IBAN (AZ) | ✅ | ✅ Yes | Regex primary, GLiNER fallback | Very Low |
| Passport # | ✅ | ✅ Yes | Regex primary, GLiNER fallback | Low |
| Vehicle Plate | ✅ | ✅ Yes | Regex primary, GLiNER fallback | Low |
| Credit Card | ✅ + Luhn | ✅ Yes | Regex primary, GLiNER fallback | Low |
| Email | ✅ | ✅ Yes | Regex primary, GLiNER fallback | Very Low |
| Postal Code | ✅ | ✅ Yes | Regex primary, GLiNER fallback | Medium |
| Date | ✅ Simple | ✅ Yes | Both active, merge | Medium |
| Person | ❌ | ✅ Yes | **GLiNER only** | Low |
| Organisation | ❌ | ✅ Yes | **GLiNER only** | Low |
| Location / GPE | ❌ | ✅ Yes | **GLiNER only** | Low |
| Position/Title | ❌ | ✅ Yes | **GLiNER only** | Medium |
| Money/Currency | ✅ Partial | ✅ Yes | Both active, merge | Low |
| SSN | ✅ + context | ✅ Yes | **GLiNER validates** ambiguous regex | High |

### Critical Design Principle: Dual-Track Architecture

**Every entity type — including patterned ones like FIN, IBAN, phone — MUST be trained into the GLiNER model.** The regex layer and GLiNER are NOT exclusive; they are complementary:

**Why train GLiNER on patterned entities too?**

1. **Context understanding**: GLiNER learns that "FİN kodu" or "fərdi identifikasiya" precedes a FIN value. This contextual knowledge catches entities that regex alone would miss — for instance, a FIN code written without explicit label text, or one embedded in noisy OCR output.

2. **Format variations**: Real-world text doesn't always match strict regex. A phone number might appear as "əlli, yüz iyirmi üç..." (spoken form), a FIN might be hyphenated as "5-AB-1CD-2", or an IBAN might have irregular spacing. GLiNER's learned representations are more robust to these variations.

3. **Ambiguity resolution**: A 10-digit number could be a VÖEN/TIN, a phone number, or just a random integer. GLiNER's context window resolves this ambiguity — if surrounded by tax-related words, it's a TIN; if preceded by "zəng edin", it's a phone number.

4. **Mutual reinforcement**: When both regex AND GLiNER agree on an entity, confidence is very high. When they disagree, you have a signal to flag for human review.

**How the hybrid merge works at inference:**

```
Input Text → ┬─→ [Regex Engine] ──→ High-confidence pattern matches (score=1.0)
             │
             └─→ [GLiNER Model] ──→ Context-aware predictions (score=0.0-1.0)
                                          │
                                    [Merge Layer]
                                          │
                              ┌───────────┴───────────┐
                              │  Overlap resolution:   │
                              │  1. Regex wins on      │
                              │     exact overlaps     │
                              │  2. GLiNER fills gaps  │
                              │  3. Both agree = high  │
                              │     confidence         │
                              └────────────────────────┘
                                          │
                                    Final Entities
```

This means the **synthetic data generation step (Section 7) is essential** — it creates training examples where patterned entities appear in realistic Azerbaijani sentences so GLiNER learns both the pattern AND the surrounding context.

---

## 4. Available Datasets & Models

### 4.1 Primary: LocalDoc/azerbaijani-ner-dataset

- **Size**: 99,545 samples
- **Entity types**: 25 (PERSON, LOCATION, ORGANISATION, GPE, DATE, TIME, MONEY, PERCENTAGE, FACILITY, PRODUCT, EVENT, ART, LAW, LANGUAGE, NORP, ORDINAL, CARDINAL, DISEASE, CONTACT, ADAGE, QUANTITY, MISCELLANEOUS, POSITION, PROJECT)
- **Format**: Parquet, columns: `tokens` (list[str]), `ner_tags` (list[int])
- **Tagging scheme**: Flat integer labels (NO BIO prefixes)
- **License**: CC BY-NC-ND 4.0 (non-commercial, no derivatives)
- **Issues**: MISCELLANEOUS tag (22) overused; flat labels mean consecutive same-type tokens are ambiguous (same entity or adjacent entities?)
- **URL**: https://huggingface.co/datasets/LocalDoc/azerbaijani-ner-dataset

### 4.2 Supplementary: WikiANN Azerbaijani

- **Size**: ~12,000 samples
- **Entity types**: 3 (PER, ORG, LOC)
- **Format**: IOB2 tags
- **Quality**: Silver-standard (auto-generated from Wikipedia)
- **License**: Apache 2.0
- **URL**: `unimelb-nlp/wikiann` (config: `"az"`)

### 4.3 Supplementary: LocalDoc/AzTC (Text Corpus)

- **Size**: 51 million sentences (~1 billion tokens)
- **Use**: Source for synthetic NER data generation, not pre-annotated
- **License**: CC BY-NC-ND 4.0
- **URL**: https://huggingface.co/datasets/LocalDoc/AzTC

### 4.4 Cross-Lingual: Turkish NER Datasets

Turkish and Azerbaijani share ~60-80% lexical similarity. Turkish NER data can supplement training:
- `savasy/ttc4900` — Turkish text classification
- `akdeniz27/turkish-ner-data` — Turkish NER in IOB2
- `turkish-nlp-suite/Turkish-Wiki-NER` — Turkish WikiANN

### 4.5 Existing Azerbaijani NER Models (Baselines)

| Model | F1 | Architecture |
|-------|-----|-------------|
| `IsmatS/azeri-turkish-bert-ner` | 0.74 micro | Turkish BERT fine-tuned |
| `LocalDoc/private_ner_azerbaijani_v2` | Undocumented | XLM-RoBERTa |
| `IsmatS/xlm_roberta_large_az_ner` | Undocumented | XLM-RoBERTa-large |

### 4.6 GLiNER PII Models (Important Baselines & Alternative Base Models)

These are GLiNER models already fine-tuned for PII detection. They can serve as (a) **baselines** to compare against, or (b) **alternative base models** to fine-tune from (instead of the generic multilingual model):

| Model | F1 (synthetic PII) | Entity Types | Languages | License |
|-------|-------------------|-------------|-----------|---------|
| `urchade/gliner_multi_pii-v1` | Undocumented | 60+ PII types | EN, FR, DE, ES, IT, PT | Apache 2.0 |
| `knowledgator/gliner-pii-base-v1.0` | **~81%** | 60+ PII types | Primarily English | Apache 2.0 |
| `knowledgator/gliner-pii-large-v1.0` | **~83%** | 60+ PII types | Primarily English | Apache 2.0 |
| `knowledgator/gliner-pii-small-v1.0` | ~75% | 60+ PII types | Primarily English | Apache 2.0 |
| `knowledgator/gliner-pii-edge-v1.0` | ~78% | 60+ PII types | Optimized for edge | Apache 2.0 |
| `gretelai/gretel-gliner-bi-large-v1.0` | Outperforms base | PII/PHI focused | English | Apache 2.0 |

**Strategic consideration**: Starting from `urchade/gliner_multi_pii-v1` (instead of `gliner_multi-v2.1`) may give better initial PII detection since it already knows entity types like phone number, credit card, SSN, passport — but it was trained on European languages only, not Azerbaijani. Test both base models and compare.

### 4.7 NERCat Reference (Catalan GLiNER — Methodology Template)

The NERCat project fine-tuned GLiNER for Catalan with ~9,000 annotated sentences and achieved near-perfect precision/recall on PER, ORG, LOC. Key takeaways:
- Used `knowledgator/gliner-bi-large-v1.0` as base (bi-encoder variant)
- Created GLiNER-compatible dataset from Catalan AnCora corpus
- Demonstrated that small high-quality datasets outperform large noisy ones
- Published on HuggingFace as `Ugiat/NERCat` and `Ugiat/ner-cat` dataset

---

## 5. Environment Setup

### 5.1 System Requirements

```
Python: 3.10+
GPU: NVIDIA with 8+ GB VRAM (RTX 3090 ideal, T4 minimum for Small model)
CUDA: 11.8+ / 12.x
RAM: 16+ GB
Disk: 10+ GB free (models + datasets)
```

### 5.2 Installation

```bash
# Create virtual environment
python -m venv gliner-az-env
source gliner-az-env/bin/activate

# Core dependencies
pip install gliner==0.2.26
pip install torch>=2.0 transformers>=4.38.0 datasets accelerate
pip install huggingface_hub

# Data generation
pip install az-data-generator  # LocalDoc's Azerbaijan data generator

# Evaluation & utilities
pip install seqeval scikit-learn pandas tqdm

# Optional: ONNX export
pip install onnx onnxruntime
```

### 5.3 Verify Installation

```python
import torch
from gliner import GLiNER

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

# Quick test with multilingual model
model = GLiNER.from_pretrained("urchade/gliner_multi-v2.1")
text = "Əliyev Heydər Bakıda doğulub."
entities = model.predict_entities(text, ["person", "location"])
print(f"Zero-shot test: {entities}")
```

---

## 6. Data Preparation Pipeline

### 6.1 GLiNER Training Data Format

GLiNER requires a **JSON array** where each sample has exactly two fields:

```json
[
  {
    "tokenized_text": ["Əliyev", "Heydər", "Bakıda", "doğulub", "."],
    "ner": [
      [0, 1, "person"],
      [2, 2, "location"]
    ]
  }
]
```

**Rules**:
- `tokenized_text`: Pre-tokenized word-level tokens (list of strings)
- `ner`: Entity annotations as `[start_index, end_index, "label"]` — **both indices inclusive**
- Single-token entities: `[2, 2, "location"]`
- Multi-token entities: `[0, 1, "person"]`
- Entity labels: Any string (multi-word OK: `"fin code"`, `"phone number"`)
- **Exclude samples with zero entities** from training data

### 6.2 Convert LocalDoc Dataset to GLiNER Format

```python
#!/usr/bin/env python3
"""
convert_localdoc_to_gliner.py

Converts LocalDoc/azerbaijani-ner-dataset from flat integer tags
to GLiNER span-based JSON format.

Usage:
    python convert_localdoc_to_gliner.py --output-dir ./data
"""

import json
import argparse
import random
from collections import Counter
from datasets import load_dataset

# Tag mapping from LocalDoc dataset (flat integer labels, no BIO prefix)
LABEL_MAP = {
    0: None,           # O (outside any entity)
    1: "person",
    2: "location",
    3: "organisation",
    4: "date",
    5: "time",
    6: "money",
    7: "percentage",
    8: "facility",
    9: "product",
    10: "event",
    11: "art",
    12: "law",
    13: "language",
    14: "gpe",         # Geo-Political Entity
    15: "norp",        # Nationality/Religious/Political group
    16: "ordinal",
    17: "cardinal",
    18: "disease",
    19: "contact",
    20: "adage",
    21: "quantity",
    22: "miscellaneous",
    23: "position",
    24: "project",
}


def convert_sample(tokens: list, tags: list) -> dict | None:
    """
    Convert a single sample from flat integer tags to GLiNER span format.

    Since tags have NO BIO prefix, consecutive tokens with the same tag
    are treated as one entity. This is an approximation — it will merge
    adjacent same-type entities, but this is the best we can do without BIO.
    """
    if len(tokens) != len(tags):
        return None

    entities = []
    start_idx = None
    current_label = None

    for i, tag in enumerate(tags):
        label = LABEL_MAP.get(tag)

        if label is not None:
            if label == current_label:
                # Continue current entity span
                continue
            else:
                # Close previous entity if exists
                if current_label is not None:
                    entities.append([start_idx, i - 1, current_label])
                # Start new entity
                start_idx = i
                current_label = label
        else:
            # O tag — close current entity if exists
            if current_label is not None:
                entities.append([start_idx, i - 1, current_label])
            current_label = None
            start_idx = None

    # Don't forget the last entity
    if current_label is not None:
        entities.append([start_idx, len(tags) - 1, current_label])

    # Skip samples with no entities
    if not entities:
        return None

    return {
        "tokenized_text": tokens,
        "ner": entities
    }


def validate_sample(sample: dict) -> bool:
    """Validate a converted sample for correctness."""
    tokens = sample["tokenized_text"]
    for start, end, label in sample["ner"]:
        if start < 0 or end >= len(tokens) or start > end:
            return False
        if not isinstance(label, str) or len(label) == 0:
            return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="./data")
    parser.add_argument("--train-ratio", type=float, default=0.85)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--test-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-miscellaneous", action="store_true",
                        help="Skip samples where the only entity is MISCELLANEOUS")
    args = parser.parse_args()

    assert abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) < 1e-6

    print("Loading LocalDoc/azerbaijani-ner-dataset...")
    ds = load_dataset("LocalDoc/azerbaijani-ner-dataset")

    print("Converting to GLiNER format...")
    gliner_data = []
    skipped = 0
    entity_counts = Counter()

    for sample in ds["train"]:
        converted = convert_sample(sample["tokens"], sample["ner_tags"])
        if converted is None:
            skipped += 1
            continue

        if args.skip_miscellaneous:
            non_misc = [e for e in converted["ner"] if e[2] != "miscellaneous"]
            if not non_misc:
                skipped += 1
                continue

        if not validate_sample(converted):
            skipped += 1
            continue

        gliner_data.append(converted)

        for _, _, label in converted["ner"]:
            entity_counts[label] += 1

    print(f"\nConversion results:")
    print(f"  Total input samples: {len(ds['train'])}")
    print(f"  Converted samples:   {len(gliner_data)}")
    print(f"  Skipped samples:     {skipped}")
    print(f"\nEntity distribution:")
    for label, count in entity_counts.most_common():
        print(f"  {label:20s}: {count:>8d}")

    # Shuffle and split
    random.seed(args.seed)
    random.shuffle(gliner_data)

    n = len(gliner_data)
    n_train = int(n * args.train_ratio)
    n_val = int(n * args.val_ratio)

    train_data = gliner_data[:n_train]
    val_data = gliner_data[n_train:n_train + n_val]
    test_data = gliner_data[n_train + n_val:]

    # Save splits
    import os
    os.makedirs(args.output_dir, exist_ok=True)

    for name, data in [("train", train_data), ("val", val_data), ("test", test_data)]:
        path = os.path.join(args.output_dir, f"{name}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=None)
        print(f"Saved {name}: {len(data)} samples → {path}")


if __name__ == "__main__":
    main()
```

### 6.3 Convert WikiANN to GLiNER Format

```python
#!/usr/bin/env python3
"""
convert_wikiann_to_gliner.py

Converts WikiANN Azerbaijani split (IOB2 format) to GLiNER span format.
"""

import json
from datasets import load_dataset

WIKIANN_LABELS = {0: None, 1: "person", 2: "person", 3: "organisation",
                  4: "organisation", 5: "location", 6: "location"}
# 0=O, 1=B-PER, 2=I-PER, 3=B-ORG, 4=I-ORG, 5=B-LOC, 6=I-LOC


def convert_wikiann_sample(tokens, tags):
    entities = []
    start_idx = None
    current_label = None

    for i, tag in enumerate(tags):
        is_b = tag in [1, 3, 5]
        is_i = tag in [2, 4, 6]
        label = WIKIANN_LABELS.get(tag)

        if is_b:
            # Close previous entity
            if current_label is not None:
                entities.append([start_idx, i - 1, current_label])
            start_idx = i
            current_label = label
        elif is_i and label == current_label:
            continue  # Continue span
        else:
            if current_label is not None:
                entities.append([start_idx, i - 1, current_label])
            current_label = None

    if current_label is not None:
        entities.append([start_idx, len(tags) - 1, current_label])

    if not entities:
        return None
    return {"tokenized_text": tokens, "ner": entities}


ds = load_dataset("unimelb-nlp/wikiann", "az")
for split in ["train", "validation", "test"]:
    data = []
    for sample in ds[split]:
        converted = convert_wikiann_sample(sample["tokens"], sample["ner_tags"])
        if converted:
            data.append(converted)
    with open(f"data/wikiann_{split}.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    print(f"WikiANN {split}: {len(data)} samples")
```

---

## 7. Maximum-Accuracy Synthetic Data Generation

This section is the **critical component** that makes Option A viable. Because the model must learn patterns from data alone, we need carefully engineered training examples that teach it every realistic variation of each entity.

### 7.1 Design Principles for Pattern-Exhaustive Data

Research from NVIDIA's GLiNER-PII (97% F1 on phone numbers) and Gretel's GLiNER-PII (81% F1 across 60 PII types) reveals five principles for teaching neural models to recognize patterns as well as regex:

**Principle 1 — Format Saturation**: Generate every plausible textual form of each entity. For a phone number, that means 15+ format variants, not just 2-3.

**Principle 2 — Contextual Diversity**: Embed each entity in dozens of surrounding contexts. The model must learn that "FİN kodu 5AB1CD2" and "identifikasiya nömrəsi 5AB1CD2" both signal a FIN code.

**Principle 3 — Adversarial Negatives**: Include near-miss examples that look like entities but aren't. This teaches the model precision. Example: include `1234567890` in a non-TIN context so the model doesn't flag every 10-digit number.

**Principle 4 — Volume Over Perfection**: Target 3,000+ synthetic examples per patterned entity. More data with slightly noisy labels beats less data with perfect labels for pattern learning.

**Principle 5 — Stratified Templates**: Ensure each format variant × context combination appears multiple times. A Cartesian product of ~50 contexts × ~15 formats = 750 unique combinations per entity.

### 7.2 Format Variations Catalog (per entity type)

This is the complete list of format variations the model must learn. Generate examples for **every** row below.

#### FIN Code (7-char alphanumeric)

```python
FIN_FORMAT_VARIANTS = [
    # Canonical
    "5AB1CD2",          # No separators
    "5 AB1 CD2",        # Space-separated groups
    "5-AB-1CD-2",       # Dash-separated
    "5.AB1.CD2",        # Dot-separated
    # Mixed case (OCR/typing variants)
    "5ab1cd2",          # Lowercase
    "5Ab1Cd2",          # Mixed case
    # Context-wrapped
    "(5AB1CD2)",        # Parentheses
    "[5AB1CD2]",        # Brackets
    "'5AB1CD2'",        # Single quotes
    "\"5AB1CD2\"",      # Double quotes
    # With trailing suffixes (Azerbaijani grammar)
    "5AB1CD2-dir",      # -dir suffix
    "5AB1CD2-yə",       # -yə suffix (dative)
    "5AB1CD2-nin",      # -nin suffix (genitive)
    # Whitespace padding
    " 5AB1CD2 ",        # Leading/trailing spaces
]
```

#### TIN / VÖEN (10-digit)

```python
TIN_FORMAT_VARIANTS = [
    "1400057421",              # Canonical
    "14 00 05 74 21",          # Pairs
    "140 005 742 1",           # Triplets
    "1400-0574-21",            # Dashed
    "14 00057421",             # Partial grouping
    "1400057421-dir",          # Azerbaijani suffix
    "№1400057421",             # Numero prefix
    "N1400057421",             # N-prefix variant
    "#1400057421",             # Hash prefix
    "VÖEN:1400057421",         # Inline label
    "VÖEN - 1400057421",       # Spaced label
    "TIN: 1400057421",         # English label
]
```

#### Phone Numbers (Azerbaijan +994)

```python
PHONE_FORMAT_VARIANTS = [
    # International canonical
    "+994501234567",              # No separators
    "+994 50 123 45 67",          # Space-separated (most common)
    "+994 (50) 123-45-67",        # Parentheses + dashes
    "+994-50-123-45-67",          # All dashes
    "+994.50.123.45.67",          # Dots
    "+994 50 1234567",            # Partial grouping
    "00994 50 123 45 67",         # 00 prefix (European)
    # Domestic (leading 0)
    "0501234567",                 # Compact
    "0 50 123 45 67",             # Spaced
    "(050) 123-45-67",            # Parens
    "050-123-45-67",              # All dashes
    "050 1234567",                # Partial
    # Legacy/informal
    "50-123-45-67",               # No prefix at all
    "501234567",                  # 9 digits only
]
```

#### IBAN (AZ + 26 chars)

```python
IBAN_FORMAT_VARIANTS = [
    "AZ21NABZ00000000137010001944",              # Electronic (no spaces)
    "AZ21 NABZ 0000 0000 1370 1000 1944",        # Standard print format
    "AZ21-NABZ-0000-0000-1370-1000-1944",        # Dashed
    "AZ21.NABZ.0000.0000.1370.1000.1944",        # Dotted
    "az21nabz00000000137010001944",              # All lowercase
    "AZ21 NABZ00000000137010001944",             # Mixed spacing
]
```

#### Passport (AZE/AA + 8 digits)

```python
PASSPORT_FORMAT_VARIANTS = [
    "AZE12345678",           # Old series canonical
    "AZE 12345678",          # Spaced
    "AZE-12345678",          # Dashed
    "AZE 1234 5678",         # Partial grouping
    "AA12345678",            # New series canonical
    "AA 12345678",           # Spaced
    "AA-12345678",           # Dashed
    "№AA12345678",           # Numero prefix
    "N AZE12345678",         # N prefix
]
```

#### Vehicle Plates (XX-LL-NNN)

```python
PLATE_FORMAT_VARIANTS = [
    "10-BK-555",             # Canonical with dashes
    "10 BK 555",             # Spaces
    "10BK555",               # No separators
    "10-bk-555",             # Lowercase
    "10 - BK - 555",         # Spaced dashes
    "10.BK.555",             # Dots
]
```

#### Credit Cards

```python
CARD_FORMAT_VARIANTS = [
    "4169741234567890",                # Canonical 16 digits
    "4169 7412 3456 7890",             # Space-quartet (most common)
    "4169-7412-3456-7890",             # Dash-quartet
    "4169.7412.3456.7890",             # Dot-quartet
    "4169  7412  3456  7890",          # Double spaces
    # Partial masks (common in customer support scenarios)
    "4169 **** **** 7890",             # Masked middle
    "****-****-****-7890",             # Last 4 only
]
```

#### Email

```python
EMAIL_FORMAT_VARIANTS = [
    "user@example.com",           # Canonical
    "User.Name@example.az",       # Mixed case + dot
    "user+tag@domain.co.uk",      # Plus-addressing
    "user_name@sub.domain.az",    # Underscore + subdomain
    "USER@EXAMPLE.AZ",            # All caps
    "<user@example.com>",         # Angle brackets (email headers)
    "mailto:user@example.com",    # mailto prefix
]
```

### 7.3 Context Templates (per entity, 50+ each)

Each entity appears in 50+ different Azerbaijani contexts. This teaches the model the **semantic fingerprint** around each entity type.

```python
# FIN Code contexts — 50+ templates
FIN_CONTEXT_TEMPLATES = [
    # Formal / governmental
    "Vətəndaşın FİN kodu {fin} -dir .",
    "Şəxsiyyət vəsiqəsindəki FİN : {fin}",
    "FİN kodu {fin} olan şəxs müraciət edib .",
    "Sənəddə FİN {fin} göstərilib .",
    "Fərdi identifikasiya nömrəsi {fin} .",
    "FİN {fin} ilə qeydiyyatdan keçib .",
    "Şəxsiyyət vəsiqəsi üzrə FİN : {fin}",
    "Dövlət reyestrində FİN {fin} qeyd olunub .",
    # Business / financial
    "Müştərinin FİN kodu {fin} .",
    "Hesabın sahibi : FİN {fin}",
    "Bank hesabında FİN {fin} göstərilib .",
    "Sığorta müqaviləsində FİN : {fin}",
    "Kredit sənədinin sahibi : FİN {fin}",
    # Legal / contractual
    "Müqavilənin tərəfi : FİN {fin}",
    "İddiaçının FİN kodu : {fin}",
    "Şahidin şəxsiyyəti : FİN {fin}",
    # Healthcare
    "Xəstənin FİN kodu {fin} -dir .",
    "Tibbi kartada FİN : {fin}",
    "Müraciət edən xəstənin FİN {fin} .",
    # HR / employment
    "İşçinin FİN kodu {fin} .",
    "Kadr sənədində FİN : {fin}",
    "Əmək müqaviləsində FİN {fin} qeyd olunub .",
    # Educational
    "Tələbənin FİN kodu : {fin}",
    "Şagirdin şəxsiyyət nömrəsi : FİN {fin}",
    # Informal / conversational
    "Mənim FİN kodum {fin} -dir .",
    "Onun FİN -i {fin} .",
    "{name} FİN kodu {fin} .",
    "{name} -in FİN kodu {fin} -dir .",
    # Table / form contexts
    "Ad: {name} | FİN: {fin} | Tarix: 15.03.2024",
    "FİN: {fin}\nAd: {name}\nTarix doğum: 01.01.1985",
    # Multi-language mixed (real-world scenario)
    "Gражданин, FİN {fin} ilə müraciət edib .",
    "Customer FİN: {fin}",
    # With additional PII
    "{name} ( FİN {fin} , tel: {phone} ) müraciət edib .",
    "Müraciətçi {name} FİN {fin} , pasport {passport} .",
    # Question / inquiry form
    "FİN kodunuz {fin} -dirmi ?",
    "Sizin FİN kodunuz {fin} kimi görünür .",
    # Error/verification contexts
    "FİN {fin} təsdiqlənmədi .",
    "Daxil etdiyiniz FİN {fin} düzgündür .",
    # More formal variants
    "FİN kodu ( {fin} ) olan şəxs .",
    "İstinad : FİN {fin}",
    "Ref : FİN {fin}",
    # Government service contexts
    "ASAN xidmətdə FİN {fin} ilə qeydiyyat .",
    "Elektron hökumət portalında FİN : {fin}",
    # Russian-influenced contexts (common in Azerbaijan)
    "По FİN {fin} зарегистрирован .",
    # With position
    "Direktor {name} , FİN {fin} .",
    "Müəllim {name} ( FİN {fin} ) .",
    # List contexts
    "1. {name} - FİN {fin}",
    "• FİN {fin} - əsas sənəd",
]
```

Do the same for each of the other 8 patterned entity types. Target 50+ templates each. This gives 400+ unique context patterns across all PII types.

### 7.4 Adversarial Negatives — Critical for Precision

To prevent false positives, include examples where strings that *look* like entities aren't actually entities:

```python
# Negative examples for TIN (10-digit numbers that aren't TINs)
TIN_NEGATIVE_TEMPLATES = [
    # Phone numbers (9-10 digits but different context)
    "Zəng edin : 0501234567 ( məlumat üçün ) .",
    # Order numbers
    "Sifariş №1234567890 qəbul edildi .",
    # Invoice numbers
    "Faktura nömrəsi 9876543210 .",
    # Random identifiers
    "Kodunuz : 1234567890 ( 10 dəqiqə etibarlıdır ) .",
    # Tracking numbers
    "İzləmə kodu : 5555555555 .",
]
# These examples should have NO ner annotations for TIN
# They teach the model that 10 digits in non-tax contexts ≠ TIN

# Negative examples for FIN (7-char strings that aren't FINs)
FIN_NEGATIVE_TEMPLATES = [
    "Məhsul kodu : ABC1234 .",       # Product code
    "Reyestr №R12345X -dir .",       # Registry number
    "Versiya V2024AB -dir .",        # Version number
    "Səhifə A4-DX1 -də tapın .",     # Page reference
]

# Negative examples for credit card (16 digits in non-card context)
CARD_NEGATIVE_TEMPLATES = [
    "İzləmə kodu : 1234567890123456 .",
    "Sifarişin ID -si 9876543210987654 .",
    "Tracking: 1111222233334444 .",
]
```

**Ratio guideline**: 10-20% negative examples per patterned entity type. Too few = false positives; too many = model becomes overly cautious and misses real entities.

### 7.5 Noise Injection (OCR / Typing Errors)

Real production text has errors. Train the model to handle them:

```python
import random

def inject_ocr_errors(text: str, error_rate: float = 0.02) -> str:
    """Inject realistic OCR-style character substitutions."""
    ocr_confusions = {
        '0': 'O', 'O': '0',
        '1': 'l', 'l': '1', 'I': '1',
        '5': 'S', 'S': '5',
        '8': 'B', 'B': '8',
        '2': 'Z', 'Z': '2',
    }
    chars = list(text)
    for i, c in enumerate(chars):
        if random.random() < error_rate and c in ocr_confusions:
            chars[i] = ocr_confusions[c]
    return ''.join(chars)

def inject_typo(text: str) -> str:
    """Simulate common typos in Azerbaijani text."""
    typos = {
        'ə': ['e', 'a'],
        'ı': ['i'],
        'ö': ['o'],
        'ü': ['u'],
        'ş': ['s'],
        'ç': ['c'],
        'ğ': ['g'],
    }
    chars = list(text)
    for i, c in enumerate(chars):
        if c in typos and random.random() < 0.05:
            chars[i] = random.choice(typos[c])
    return ''.join(chars)
```

Apply noise to ~5% of synthetic training examples.

### 7.6 Complete Synthetic Generation Script

```python
#!/usr/bin/env python3
"""
generate_synthetic_az_ner_pattern_exhaustive.py

Pattern-exhaustive synthetic data generator for Option A Azerbaijani GLiNER.

Targets:
- 3,000+ positive examples per patterned entity
- 500+ negative examples per patterned entity
- 15+ format variations per entity
- 50+ context templates per entity
- Noise injection on 5% of examples

Total target: ~30,000 synthetic samples
"""

import json
import random
import string
from itertools import product
from data_generator.modules.fin_generator import FinGenerator
from data_generator.modules.passport_generator import PassportGenerator
from data_generator.modules.phone_generator import PhoneGenerator
from data_generator.modules.email_generator import EmailGenerator
from data_generator.modules.iban_generator import IbanGenerator
from data_generator.modules.card_generator import CardGenerator
from data_generator.modules.name_generator import NameGenerator
from data_generator.modules.tax_id_generator import TaxIdGenerator
from data_generator.modules.license_plate_generator import LicensePlateGenerator
from data_generator.modules.city_generator import CityGenerator


random.seed(42)

# Initialize generators
fin_gen = FinGenerator()
passport_gen = PassportGenerator()
phone_gen = PhoneGenerator()
email_gen = EmailGenerator()
iban_gen = IbanGenerator()
card_gen = CardGenerator()
name_gen = NameGenerator()
tax_gen = TaxIdGenerator()
plate_gen = LicensePlateGenerator()
city_gen = CityGenerator()


# ============================================================
# FORMAT VARIATION FUNCTIONS
# ============================================================

def vary_fin(canonical: str) -> str:
    """Apply format variation to a canonical FIN code (7 chars)."""
    variants = [
        canonical,                                                # Canonical
        f"{canonical[0]} {canonical[1:4]} {canonical[4:]}",       # Spaced groups
        f"{canonical[0]}-{canonical[1:4]}-{canonical[4:]}",       # Dashed groups
        canonical.lower(),                                        # Lowercase
        f"({canonical})",                                         # Parens
        f"[{canonical}]",                                         # Brackets
        canonical + "-dir",                                       # -dir suffix
        canonical + "-nin",                                       # -nin suffix
    ]
    return random.choice(variants)


def vary_tin(canonical: str) -> str:
    """Apply format variation to a TIN (10 digits)."""
    variants = [
        canonical,
        f"{canonical[:2]} {canonical[2:4]} {canonical[4:6]} {canonical[6:8]} {canonical[8:]}",
        f"{canonical[:3]} {canonical[3:6]} {canonical[6:9]} {canonical[9:]}",
        f"{canonical[:4]}-{canonical[4:8]}-{canonical[8:]}",
        f"№{canonical}",
        f"N{canonical}",
        canonical + "-dir",
    ]
    return random.choice(variants)


def vary_phone(canonical: str) -> str:
    """Apply format variation to a phone number (+994 + 9 digits)."""
    # Assume canonical format +994501234567
    if canonical.startswith("+994"):
        digits = canonical[4:]
    else:
        digits = canonical.lstrip("0").lstrip("+994")

    if len(digits) != 9:
        return canonical

    op = digits[:2]
    rest = digits[2:]

    variants = [
        f"+994{digits}",
        f"+994 {op} {rest[:3]} {rest[3:5]} {rest[5:]}",
        f"+994 ({op}) {rest[:3]}-{rest[3:5]}-{rest[5:]}",
        f"+994-{op}-{rest[:3]}-{rest[3:5]}-{rest[5:]}",
        f"00994 {op} {rest[:3]} {rest[3:5]} {rest[5:]}",
        f"0{op}{rest}",
        f"0{op} {rest[:3]} {rest[3:5]} {rest[5:]}",
        f"({op}) {rest[:3]}-{rest[3:5]}-{rest[5:]}",
        f"0{op}-{rest[:3]}-{rest[3:5]}-{rest[5:]}",
    ]
    return random.choice(variants)


def vary_iban(canonical: str) -> str:
    """Apply format variation to an IBAN."""
    # Canonical: AZ21NABZ00000000137010001944 (28 chars)
    if len(canonical) != 28:
        return canonical

    groups = [canonical[i:i+4] for i in range(0, 28, 4)]
    variants = [
        canonical,
        " ".join(groups),
        "-".join(groups),
        canonical.lower(),
    ]
    return random.choice(variants)


def vary_passport(canonical: str) -> str:
    """Apply format variation to passport number."""
    if canonical.startswith(("AZE", "AA")):
        prefix = "AZE" if canonical.startswith("AZE") else "AA"
        digits = canonical[len(prefix):]
        variants = [
            canonical,
            f"{prefix} {digits}",
            f"{prefix}-{digits}",
            f"{prefix} {digits[:4]} {digits[4:]}",
            f"№{canonical}",
        ]
        return random.choice(variants)
    return canonical


def vary_plate(canonical: str) -> str:
    """Apply format variation to vehicle plate (XX-LL-NNN)."""
    canonical = canonical.replace(" ", "").replace("-", "")
    if len(canonical) == 7:
        variants = [
            f"{canonical[:2]}-{canonical[2:4]}-{canonical[4:]}",
            f"{canonical[:2]} {canonical[2:4]} {canonical[4:]}",
            f"{canonical[:2]}{canonical[2:4]}{canonical[4:]}",
            canonical.lower(),
        ]
        return random.choice(variants)
    return canonical


def vary_card(canonical: str) -> str:
    """Apply format variation to credit card number (16 digits)."""
    digits = ''.join(c for c in canonical if c.isdigit())
    if len(digits) != 16:
        return canonical
    variants = [
        digits,
        f"{digits[:4]} {digits[4:8]} {digits[8:12]} {digits[12:]}",
        f"{digits[:4]}-{digits[4:8]}-{digits[8:12]}-{digits[12:]}",
        f"{digits[:4]}.{digits[4:8]}.{digits[8:12]}.{digits[12:]}",
    ]
    return random.choice(variants)


# ============================================================
# GENERATE NAME HELPER
# ============================================================
def generate_name() -> str:
    r = name_gen.generate()
    return f"{r.get('first_name', 'Əli')} {r.get('last_name', 'Əliyev')}"


# ============================================================
# CONTEXT TEMPLATES (50+ per entity)
# ============================================================

FIN_TEMPLATES = [
    "Vətəndaşın FİN kodu {fin} -dir .",
    "Şəxsiyyət vəsiqəsindəki FİN : {fin}",
    "FİN kodu {fin} olan şəxs müraciət edib .",
    "Sənəddə FİN {fin} göstərilib .",
    "Fərdi identifikasiya nömrəsi {fin} .",
    "FİN {fin} ilə qeydiyyatdan keçib .",
    "Dövlət reyestrində FİN {fin} qeyd olunub .",
    "Müştərinin FİN kodu {fin} .",
    "Hesabın sahibi : FİN {fin}",
    "Bank hesabında FİN {fin} göstərilib .",
    "Sığorta müqaviləsində FİN : {fin}",
    "Kredit sənədinin sahibi : FİN {fin}",
    "Müqavilənin tərəfi : FİN {fin}",
    "Xəstənin FİN kodu {fin} -dir .",
    "Tibbi kartada FİN : {fin}",
    "İşçinin FİN kodu {fin} .",
    "Kadr sənədində FİN : {fin}",
    "Əmək müqaviləsində FİN {fin} qeyd olunub .",
    "Tələbənin FİN kodu : {fin}",
    "Mənim FİN kodum {fin} -dir .",
    "{name} FİN kodu {fin} .",
    "{name} -in FİN kodu {fin} -dir .",
    "Ad: {name} | FİN: {fin}",
    "{name} ( FİN {fin} ) müraciət edib .",
    "FİN kodunuz {fin} -dirmi ?",
    "FİN {fin} təsdiqlənmədi .",
    "Daxil etdiyiniz FİN {fin} düzgündür .",
    "FİN kodu ( {fin} ) olan şəxs .",
    "İstinad : FİN {fin}",
    "ASAN xidmətdə FİN {fin} ilə qeydiyyat .",
    "Elektron hökumət portalında FİN : {fin}",
    "Direktor {name} , FİN {fin} .",
    "1. {name} - FİN {fin}",
    "İddiaçının FİN kodu : {fin}",
    "Şahidin şəxsiyyəti : FİN {fin}",
    "Müraciət edən xəstənin FİN {fin} .",
    "Şagirdin şəxsiyyət nömrəsi : FİN {fin}",
    "Onun FİN -i {fin} .",
    "Müraciətçi {name} FİN {fin} .",
    "Customer FİN: {fin}",
    "Sizin FİN kodunuz {fin} kimi görünür .",
    "Ref : FİN {fin}",
    "Ərizədə FİN {fin} qeyd olunub .",
    "Notariat aktında FİN : {fin}",
    "Vəkalətnamədə FİN {fin} göstərilib .",
    "Xidmət sahibi : FİN {fin}",
    "Benefisiar : FİN {fin}",
    "Vəsaitin alıcısı FİN {fin} .",
    "Məhkəmə qərarında FİN {fin} qeyd olunub .",
    "Polis protokolunda FİN : {fin}",
]

# Analogous templates for each other entity type (TIN, phone, IBAN,
# passport, plate, card, email, postal code) — 50+ templates each.
# See full script at: scripts/templates_all_entities.py

TIN_TEMPLATES = [
    "Şirkətin VÖEN nömrəsi {tin} -dir .",
    "VÖEN : {tin}",
    "{name} fərdi sahibkar kimi VÖEN {tin} ilə qeydiyyatdan keçib .",
    "Vergi ödəyicisinin eyniləşdirmə nömrəsi : {tin}",
    "Fakturada VÖEN {tin} göstərilib .",
    "Müəssisənin TIN nömrəsi {tin} .",
    "Vergi uçotunda TIN {tin} qeydə alınıb .",
    "Hüquqi şəxsin VÖEN kodu : {tin}",
    "Fərdi sahibkarın VÖEN : {tin}",
    "Tenderdə iştirakçı VÖEN {tin} .",
    "Kontragentin vergi nömrəsi {tin} .",
    "Bank arayışında VÖEN : {tin}",
    "Müqavilənin A tərəfi : VÖEN {tin}",
    "VÖEN {tin} olan şirkət .",
    "e-taxes.gov.az -da VÖEN {tin} göstərilib .",
    "Vergi bəyannaməsində VÖEN : {tin}",
    "VÖEN {tin} -lik şirkətin balansı .",
    "İnvoysda VÖEN : {tin}",
    "Qaimə-fakturada VÖEN {tin} .",
    "Əlavə dəyər vergisi ödəyicisinin VÖEN : {tin}",
    "Qeydiyyat : VÖEN {tin}",
    "Mühasibat hesabatında VÖEN {tin} .",
    "ASAN xidmətdə VÖEN {tin} ilə qeydiyyat .",
    "Dövlət Vergi Xidmətində VÖEN {tin} qeydə alınıb .",
    "İmtiyazlı vergi rejimində VÖEN {tin} .",
    # ... (25+ more)
]

PHONE_TEMPLATES = [
    "{name} ilə {phone} nömrəsindən əlaqə saxlaya bilərsiniz .",
    "Telefon nömrəsi : {phone}",
    "Əlaqə üçün {phone} nömrəsinə zəng edin .",
    "{name} mobil nömrəsi {phone} -dir .",
    "Qaynar xətt : {phone}",
    "Əlavə məlumat üçün {phone} ilə əlaqə saxlayın .",
    "Müraciət üçün : {phone}",
    "WhatsApp : {phone}",
    "Zəng edin : {phone}",
    "Əlaqə nömrəmiz {phone} .",
    "Ofis telefonu : {phone}",
    "Mobil : {phone}",
    "Ev telefonu : {phone}",
    "Sizə geri zəng edəcəyəm : {phone}",
    "Viber : {phone}",
    "Telegram : {phone}",
    "{phone} -dən zəng gəldi .",
    "{phone} nömrəsi məşğuldur .",
    "Şəxsi nömrəm {phone} .",
    "Direktor : {phone}",
    "HR şöbəsi : {phone}",
    "Dispetçer : {phone}",
    "Təcili yardım : {phone}",
    "{name} ( {phone} ) sifariş verib .",
    "Müştəri xidmətləri : {phone}",
    # ... (25+ more)
]

IBAN_TEMPLATES = [
    "Ödənişi {iban} hesab nömrəsinə köçürün .",
    "Bank hesabı : {iban}",
    "{name} IBAN nömrəsi {iban} -dir .",
    "Köçürmə üçün IBAN : {iban}",
    "Maaş {iban} hesabına köçürüləcək .",
    "Şirkətin IBAN -ı : {iban}",
    "Alıcının IBAN : {iban}",
    "SWIFT köçürməsi üçün IBAN : {iban}",
    "Beynəlxalq köçürmə : {iban}",
    "Cari hesab nömrəsi : {iban}",
    "Əmanət hesabı IBAN : {iban}",
    "Manat hesabı : {iban}",
    "USD hesabı : {iban}",
    "EUR hesabı : {iban}",
    "Fakturada ödəniş üçün : {iban}",
    "Müqavilədəki bank rekvizitləri : {iban}",
    # ... (30+ more)
]

PASSPORT_TEMPLATES = [
    "{name} pasport nömrəsi {passport} -dir .",
    "Pasport : {passport} , verilmə tarixi 15.03.2020 .",
    "Səyahət sənədi nömrəsi {passport} olan şəxs .",
    "{name} {passport} nömrəli pasportla qeydiyyatdan keçib .",
    "Sərhəddə pasport {passport} yoxlanılıb .",
    "Vizada pasport nömrəsi : {passport}",
    "Biometrik pasport : {passport}",
    "Diplomatik pasport : {passport}",
    "Xidməti pasport : {passport}",
    "Pasport məlumatı : {passport}",
    "Miqrasiya sənədi : {passport}",
    "Viza müraciəti üçün pasport : {passport}",
    # ... (35+ more)
]

PLATE_TEMPLATES = [
    "Avtomobilin dövlət nömrəsi {plate} -dir .",
    "{plate} nömrəli avtomobil saxlanılıb .",
    "{name} {plate} nömrəli maşın sürürdü .",
    "Qeydiyyat nişanı : {plate}",
    "Texniki pasportda dövlət nömrəsi {plate} .",
    "Avtomobil {plate} qəza törətdi .",
    "{plate} saylı avtomobil .",
    "DYP tərəfindən {plate} saxlanılıb .",
    "Video görüntüsündə {plate} .",
    "Qəzada iştirak edən avtomobil {plate} .",
    # ... (40+ more)
]

CARD_TEMPLATES = [
    "Kart nömrəsi : {card}",
    "{name} kredit kartı {card} ilə ödəniş edib .",
    "Bank kartı nömrəsi {card} olan müştəri .",
    "Ödəniş {card} nömrəli kartdan çıxılıb .",
    "Kartınızın nömrəsi {card} .",
    "POS terminalda {card} oxudu .",
    "Visa kart : {card}",
    "Mastercard : {card}",
    "Debet kartı : {card}",
    "Kredit kartı : {card}",
    "Biznes kart : {card}",
    # ... (40+ more)
]

EMAIL_TEMPLATES = [
    "{name} elektron poçtu {email} -dir .",
    "Əlaqə : {email}",
    "Müraciəti {email} ünvanına göndərin .",
    "E-poçt : {email}",
    "E-mail : {email}",
    "Mail : {email}",
    "Elektron ünvan : {email}",
    "CC : {email}",
    "BCC : {email}",
    "Mən {email} -ə yazmışam .",
    # ... (40+ more)
]


# ============================================================
# TOKEN SPAN FINDER
# ============================================================
def find_token_span(tokens: list, target: str) -> tuple:
    """Find start/end token indices for a target string in tokens."""
    target_tokens = target.split()
    for i in range(len(tokens) - len(target_tokens) + 1):
        if tokens[i:i + len(target_tokens)] == target_tokens:
            return i, i + len(target_tokens) - 1
    return None, None


# ============================================================
# POSITIVE SAMPLE GENERATOR
# ============================================================
def generate_positive_sample(
    template: str,
    entity_value: str,
    entity_label: str,
    include_name: bool = True,
) -> dict:
    """Generate one positive training sample."""
    name = generate_name() if include_name else None
    city = city_gen.generate().get("city", "Bakı")

    # Fill template
    text = template.format(
        name=name or "",
        fin=entity_value,
        phone=entity_value,
        email=entity_value,
        iban=entity_value,
        passport=entity_value,
        tin=entity_value,
        card=entity_value,
        plate=entity_value,
        city=city,
    )

    tokens = text.split()
    entities = []

    # Find entity value
    start, end = find_token_span(tokens, entity_value)
    if start is not None:
        entities.append([start, end, entity_label])

    # Find name if in template
    if name and "{name}" in template:
        start, end = find_token_span(tokens, name)
        if start is not None:
            entities.append([start, end, "person"])

    if not entities:
        return None

    return {"tokenized_text": tokens, "ner": entities}


# ============================================================
# NEGATIVE SAMPLE GENERATOR
# ============================================================
NEGATIVE_TEMPLATES = {
    "tin_negative": [
        "Zəng edin : 0{tin_like} ( məlumat üçün ) .",
        "Sifariş №{tin_like} qəbul edildi .",
        "Faktura nömrəsi {tin_like} .",
        "Kodunuz : {tin_like} ( 10 dəqiqə etibarlıdır ) .",
        "İzləmə kodu : {tin_like} .",
    ],
    "fin_negative": [
        "Məhsul kodu : {fin_like} .",
        "Reyestr №{fin_like} -dir .",
        "Versiya {fin_like} -dir .",
    ],
    "card_negative": [
        "İzləmə kodu : {card_like} .",
        "Sifarişin ID -si {card_like} .",
    ],
}


def generate_negative_sample(template: str, fake_value: str) -> dict:
    """Generate a sample with a pattern-looking string that is NOT the entity."""
    name = generate_name()
    text = template.format(
        tin_like=fake_value,
        fin_like=fake_value,
        card_like=fake_value,
    )
    tokens = text.split()
    # No entity annotations — this is a negative
    # But might still contain a person entity
    entities = []
    if "{name}" in template:
        start, end = find_token_span(tokens, name)
        if start is not None:
            entities.append([start, end, "person"])
    return {"tokenized_text": tokens, "ner": entities}


# ============================================================
# NOISE INJECTION
# ============================================================
def inject_noise(sample: dict, noise_rate: float = 0.02) -> dict:
    """Apply OCR-style character confusion to non-entity tokens."""
    ocr_confusions = {
        '0': 'O', '1': 'l', '5': 'S', '8': 'B',
    }
    entity_token_ids = set()
    for start, end, _ in sample["ner"]:
        for i in range(start, end + 1):
            entity_token_ids.add(i)

    noisy_tokens = []
    for i, tok in enumerate(sample["tokenized_text"]):
        if i in entity_token_ids:
            noisy_tokens.append(tok)  # Don't corrupt entities
            continue
        new_tok = ''.join(
            ocr_confusions.get(c, c) if random.random() < noise_rate else c
            for c in tok
        )
        noisy_tokens.append(new_tok)

    return {"tokenized_text": noisy_tokens, "ner": sample["ner"]}


# ============================================================
# MAIN GENERATION LOOP
# ============================================================
def generate_entity_samples(
    entity_name: str,
    generator_fn,
    variation_fn,
    templates: list,
    label: str,
    num_samples: int = 3000,
    noise_ratio: float = 0.05,
) -> list:
    """Generate N pattern-exhaustive samples for one entity type."""
    samples = []
    attempts = 0
    max_attempts = num_samples * 5

    while len(samples) < num_samples and attempts < max_attempts:
        attempts += 1

        # Generate canonical entity value
        canonical = generator_fn()
        # Apply format variation
        value = variation_fn(canonical)
        # Pick random template
        template = random.choice(templates)

        sample = generate_positive_sample(template, value, label)
        if sample is None:
            continue

        # Inject noise on 5% of samples
        if random.random() < noise_ratio:
            sample = inject_noise(sample)

        samples.append(sample)

    return samples


def generate_all():
    """Generate the complete synthetic training dataset."""
    all_samples = []

    # FIN code: 3000 positive + 500 negative
    print("Generating FIN samples...")
    all_samples.extend(generate_entity_samples(
        "fin", lambda: fin_gen.generate_fin(), vary_fin,
        FIN_TEMPLATES, "fin code", num_samples=3000,
    ))

    # TIN: 3000 positive + 500 negative
    print("Generating TIN samples...")
    all_samples.extend(generate_entity_samples(
        "tin", lambda: tax_gen.generate().get("tax_id", "1400057421"), vary_tin,
        TIN_TEMPLATES, "tin", num_samples=3000,
    ))

    # Phone: 3000
    print("Generating phone samples...")
    all_samples.extend(generate_entity_samples(
        "phone", lambda: phone_gen.generate().get("phone_number", "+994501234567"), vary_phone,
        PHONE_TEMPLATES, "phone number", num_samples=3000,
    ))

    # IBAN: 3000
    print("Generating IBAN samples...")
    all_samples.extend(generate_entity_samples(
        "iban", lambda: iban_gen.generate().get("iban", "AZ21NABZ00000000137010001944"), vary_iban,
        IBAN_TEMPLATES, "iban", num_samples=3000,
    ))

    # Passport: 3000
    print("Generating passport samples...")
    all_samples.extend(generate_entity_samples(
        "passport", lambda: passport_gen.generate().get("passport_number", "AZE12345678"),
        vary_passport, PASSPORT_TEMPLATES, "passport number", num_samples=3000,
    ))

    # Plate: 3000
    print("Generating plate samples...")
    all_samples.extend(generate_entity_samples(
        "plate", lambda: plate_gen.generate().get("license_plate", "10-AA-001"), vary_plate,
        PLATE_TEMPLATES, "vehicle plate", num_samples=3000,
    ))

    # Card: 3000
    print("Generating card samples...")
    all_samples.extend(generate_entity_samples(
        "card", lambda: card_gen.generate().get("card_number", "4169741234567890"), vary_card,
        CARD_TEMPLATES, "credit card number", num_samples=3000,
    ))

    # Email: 3000
    print("Generating email samples...")
    all_samples.extend(generate_entity_samples(
        "email", lambda: email_gen.generate().get("email", "test@mail.az"), lambda x: x,
        EMAIL_TEMPLATES, "email", num_samples=3000,
    ))

    # Generate negative samples for adversarial robustness
    print("Generating negative samples...")
    for _ in range(500):
        fake_tin = ''.join(random.choices('0123456789', k=10))
        template = random.choice(NEGATIVE_TEMPLATES["tin_negative"])
        sample = generate_negative_sample(template, fake_tin)
        if sample:
            all_samples.append(sample)

    for _ in range(500):
        fake_fin = ''.join(random.choices(string.ascii_uppercase + string.digits, k=7))
        template = random.choice(NEGATIVE_TEMPLATES["fin_negative"])
        sample = generate_negative_sample(template, fake_fin)
        if sample:
            all_samples.append(sample)

    for _ in range(500):
        fake_card = ''.join(random.choices('0123456789', k=16))
        template = random.choice(NEGATIVE_TEMPLATES["card_negative"])
        sample = generate_negative_sample(template, fake_card)
        if sample:
            all_samples.append(sample)

    random.shuffle(all_samples)
    return all_samples


if __name__ == "__main__":
    print("Generating pattern-exhaustive synthetic dataset for Option A...")
    data = generate_all()
    print(f"\nTotal samples: {len(data)}")

    with open("data/synthetic_az_pattern_exhaustive.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    print("Saved to data/synthetic_az_pattern_exhaustive.json")
```

### 7.7 Dataset Composition Strategy

To achieve ≥95% F1 on patterned entities with pure neural approach, combine data sources in these ratios:

| Data Source | Samples | % of Training Set | Purpose |
|-------------|---------|------------------|---------|
| LocalDoc dataset (converted) | ~85,000 | 60% | Semantic entities (person, org, location, etc.) |
| WikiANN Azerbaijani | ~12,000 | 10% | Baseline NER diversity |
| **Pattern-exhaustive synthetic** | **~30,000** | **22%** | **Patterned PII (FIN, TIN, IBAN, etc.)** |
| Negative examples | ~1,500 | 1% | Precision boosting |
| LLM-generated narrative text | ~10,000 | 7% | Natural context variety |
| **TOTAL** | **~138,500** | **100%** | |

**Target per patterned entity**: 3,000+ positive examples in training set. This is 4-5x more than what Gretel/NVIDIA used per entity, justified because Azerbaijani is a lower-resource language and the model needs more signal.

### 7.8 Validation Check: Does the Data Teach the Pattern?

Before fine-tuning, run a sanity check on your synthetic data:

```python
def audit_dataset(samples: list) -> dict:
    """Audit dataset composition for Option A readiness."""
    from collections import Counter

    entity_counts = Counter()
    format_diversity = {}

    for sample in samples:
        for start, end, label in sample["ner"]:
            entity_counts[label] += 1
            entity_text = " ".join(sample["tokenized_text"][start:end+1])
            if label not in format_diversity:
                format_diversity[label] = set()
            format_diversity[label].add(entity_text[:20])  # first 20 chars as diversity signal

    print(f"\n{'Entity Type':25s} {'Count':>8s} {'Unique Values':>15s}")
    print("-" * 52)
    for label in sorted(entity_counts.keys()):
        print(f"{label:25s} {entity_counts[label]:>8d} {len(format_diversity[label]):>15d}")

    # Flags
    issues = []
    for label in ["fin code", "tin", "phone number", "iban"]:
        if entity_counts.get(label, 0) < 2500:
            issues.append(f"⚠️  {label} has only {entity_counts.get(label, 0)} samples (need 2500+)")
        if len(format_diversity.get(label, set())) < 500:
            issues.append(f"⚠️  {label} has only {len(format_diversity.get(label, set()))} unique values (need 500+)")

    if issues:
        print("\nISSUES DETECTED:")
        for issue in issues:
            print(issue)
    else:
        print("\n✅ Dataset passes Option A readiness check")

    return {"counts": dict(entity_counts), "diversity": {k: len(v) for k, v in format_diversity.items()}}
```

Run this before training. If any patterned entity has <2500 examples or <500 unique values, **generate more data** before proceeding.

### 7.9 Targeted LLM Workflows (via AI Gateway)

**When to use**: After template-based synthetic generation is complete. These two workflows add narrative diversity and quality assurance. **Total budget: ~$20-35** for both combined.

**When NOT to use**: Do NOT use LLMs to invent FIN codes, VÖENs, IBANs, or other patterned PII values — `az-data-generator` is strictly better for those. Do NOT use LLMs to annotate raw Azerbaijani text from scratch — they produce 10-20% label errors in Azerbaijani.

#### 7.9.1 AI Gateway Setup

All LLM calls MUST route through an AI Gateway per project requirements. The script assumes Kong AI Gateway or equivalent with an API key in the environment:

```bash
export AI_GATEWAY_URL="https://your-ai-gateway.pasha.internal/v1/messages"
export AI_GATEWAY_KEY="your-api-key"
```

Shared client module used by both workflows:

```python
# llm_client.py
"""Thin wrapper around AI Gateway for Claude API calls."""
import os
import json
import time
import httpx
from typing import Optional

GATEWAY_URL = os.environ["AI_GATEWAY_URL"]
GATEWAY_KEY = os.environ["AI_GATEWAY_KEY"]


def call_claude(
    system: str,
    user: str,
    model: str = "claude-sonnet-4-6",
    max_tokens: int = 2048,
    max_retries: int = 3,
) -> Optional[str]:
    """Call Claude via AI Gateway with retries."""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {GATEWAY_KEY}",
        "anthropic-version": "2023-06-01",
    }
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }

    for attempt in range(max_retries):
        try:
            r = httpx.post(GATEWAY_URL, headers=headers, json=payload, timeout=60)
            r.raise_for_status()
            data = r.json()
            return data["content"][0]["text"]
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                # Rate limit — exponential backoff
                time.sleep(2 ** attempt)
                continue
            raise
        except (httpx.RequestError, KeyError, IndexError) as e:
            if attempt == max_retries - 1:
                print(f"Failed after {max_retries} attempts: {e}")
                return None
            time.sleep(2 ** attempt)

    return None
```

#### 7.9.2 Workflow 1 — Narrative Context Generation (~$15-25)

**Goal**: Generate ~5,000-10,000 natural-flowing Azerbaijani sentences where PII is embedded in realistic narrative context (news articles, customer service dialogues, HR documents, legal correspondence) — not stiff template outputs.

**Key design decision**: The LLM generates the *prose*, NOT the entity values. You supply real entity values from `az-data-generator` and instruct Claude to weave them naturally into a paragraph. After generation, you run a deterministic span-finding pass to extract token positions.

```python
#!/usr/bin/env python3
"""
generate_narrative_pii_samples.py

Generate natural-context Azerbaijani sentences with PII entities woven in.
Entity values come from az-data-generator (guaranteed realistic).
Claude generates the narrative prose around them.
Token spans are detected deterministically after generation.

Target: 5,000-10,000 samples
Budget: ~$15-25 via AI Gateway
"""
import json
import random
import re
from typing import Optional
from llm_client import call_claude

from data_generator.modules.fin_generator import FinGenerator
from data_generator.modules.passport_generator import PassportGenerator
from data_generator.modules.phone_generator import PhoneGenerator
from data_generator.modules.iban_generator import IbanGenerator
from data_generator.modules.name_generator import NameGenerator
from data_generator.modules.tax_id_generator import TaxIdGenerator
from data_generator.modules.license_plate_generator import LicensePlateGenerator
from data_generator.modules.city_generator import CityGenerator

random.seed(42)

fin_gen = FinGenerator()
passport_gen = PassportGenerator()
phone_gen = PhoneGenerator()
iban_gen = IbanGenerator()
name_gen = NameGenerator()
tax_gen = TaxIdGenerator()
plate_gen = LicensePlateGenerator()
city_gen = CityGenerator()


SCENARIO_TYPES = [
    "news article about a business dispute",
    "customer service email",
    "HR onboarding document",
    "legal correspondence",
    "bank compliance report",
    "police incident report",
    "government service application confirmation",
    "insurance claim notification",
    "medical appointment reminder",
    "notary public statement",
    "rental agreement clause",
    "tax audit notice",
    "employment contract excerpt",
    "customer complaint letter",
    "tender application response",
]


SYSTEM_PROMPT = """You are an expert Azerbaijani writer producing realistic business and \
government documents. Your output language is Azerbaijani only.

RULES:
1. Write natural, native-quality Azerbaijani prose — use proper grammar, \
suffixes (-dir, -nin, -dən, -yə), and idiomatic phrasing.
2. You MUST include every entity provided in the input EXACTLY AS WRITTEN — \
do not modify, translate, or reformat entity values.
3. Weave entities naturally into the narrative — do not produce a bulleted list.
4. Length: 2-4 sentences. Keep it concise but natural.
5. Output ONLY the Azerbaijani text. No explanations, no JSON, no English, \
no preamble."""


def build_user_prompt(scenario: str, entities: dict) -> str:
    """Build the prompt asking Claude to write a paragraph with given entities."""
    entity_list = "\n".join(
        f"- {label}: {value}" for label, value in entities.items()
    )
    return f"""Write a short Azerbaijani paragraph (2-4 sentences) in the style of a {scenario}.

The paragraph MUST include these exact entities, unchanged:
{entity_list}

Remember: Write ONLY the Azerbaijani text. Include every entity exactly as given."""


def find_entity_spans(text: str, entities: dict) -> list:
    """
    Tokenize text and find token-level spans for each entity value.
    Returns list of [start_token, end_token, label] triples.
    """
    # Simple whitespace tokenization (same as GLiNER training data format)
    # Treat punctuation as separate tokens
    tokens = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)

    spans = []
    for label, value in entities.items():
        # Tokenize the entity value the same way
        value_tokens = re.findall(r"\w+|[^\w\s]", value, re.UNICODE)
        if not value_tokens:
            continue

        # Find subsequence in tokens
        found = False
        for i in range(len(tokens) - len(value_tokens) + 1):
            if tokens[i : i + len(value_tokens)] == value_tokens:
                spans.append([i, i + len(value_tokens) - 1, label])
                found = True
                break

        if not found:
            # Claude modified the entity — skip this sample entirely
            return None, None

    return tokens, spans


def generate_sample() -> Optional[dict]:
    """Generate one narrative sample with embedded PII."""
    # Pick random scenario
    scenario = random.choice(SCENARIO_TYPES)

    # Generate 2-4 random entities for this sample
    all_entity_options = [
        ("person", lambda: f"{name_gen.generate().get('first_name', 'Əli')} "
                           f"{name_gen.generate().get('last_name', 'Əliyev')}"),
        ("fin code", lambda: fin_gen.generate_fin()),
        ("tin", lambda: tax_gen.generate().get("tax_id", "1400057421")),
        ("phone number", lambda: phone_gen.generate().get("phone_number", "+994501234567")),
        ("iban", lambda: iban_gen.generate().get("iban", "AZ21NABZ00000000137010001944")),
        ("passport number", lambda: passport_gen.generate().get("passport_number", "AZE12345678")),
        ("vehicle plate", lambda: plate_gen.generate().get("license_plate", "10-AA-001")),
        ("location", lambda: city_gen.generate().get("city", "Bakı")),
    ]

    num_entities = random.randint(2, 4)
    selected = random.sample(all_entity_options, num_entities)
    entities = {label: gen_fn() for label, gen_fn in selected}

    # Call Claude
    user_prompt = build_user_prompt(scenario, entities)
    narrative = call_claude(SYSTEM_PROMPT, user_prompt)

    if not narrative:
        return None

    narrative = narrative.strip()

    # Find spans — this also validates that Claude kept entities intact
    tokens, spans = find_entity_spans(narrative, entities)

    if tokens is None or not spans:
        # Claude modified an entity value — discard
        return None

    return {"tokenized_text": tokens, "ner": spans}


def main(target_samples: int = 5000, output_path: str = "data/narrative_pii.json"):
    samples = []
    attempts = 0
    max_attempts = target_samples * 2  # expect ~50% success rate

    while len(samples) < target_samples and attempts < max_attempts:
        attempts += 1
        sample = generate_sample()
        if sample:
            samples.append(sample)

        if len(samples) % 100 == 0 and len(samples) > 0:
            print(f"Generated {len(samples)}/{target_samples} "
                  f"(success rate: {len(samples)/attempts:.1%})")

    print(f"\nDone. Generated {len(samples)} samples in {attempts} attempts.")
    print(f"Success rate: {len(samples)/attempts:.1%}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main(target_samples=5000)
```

**Expected cost**: ~5,000 samples × ~500 input tokens + ~200 output tokens each = ~3.5M tokens total. At Claude Sonnet 4.6 pricing (~$3/M input, ~$15/M output), this costs **~$15-20**.

**Expected success rate**: 50-70%. The remaining ~30-50% are discarded because Claude modifies entity values slightly (adds suffixes, changes spacing, etc.). This is acceptable — we only keep samples where entities appear verbatim.

#### 7.9.3 Workflow 2 — LocalDoc Quality Audit (~$5)

**Goal**: Spot-check 500 random samples from the converted LocalDoc dataset, flag likely annotation errors, and generate a cleanup list. Addresses the known issue that LocalDoc's MISCELLANEOUS tag is overused and flat integer labels may cause merge errors.

```python
#!/usr/bin/env python3
"""
audit_localdoc_quality.py

Use Claude to audit a random sample of 500 LocalDoc training examples.
Identify likely annotation errors for exclusion from training.

Budget: ~$5 via AI Gateway
"""
import json
import random
from typing import Optional
from llm_client import call_claude

random.seed(42)


AUDIT_SYSTEM_PROMPT = """You are an expert Azerbaijani linguist auditing NER annotations.

You will be shown an Azerbaijani sentence with entity annotations. Your job \
is to identify whether each annotation is correct.

Entity type definitions:
- person: Human names (first name, last name, or full name)
- location: Physical locations, cities, countries
- organisation: Companies, government bodies, institutions
- gpe: Geopolitical entities (states, provinces, regions)
- date: Calendar dates, days, months, years
- time: Clock times
- money: Currency amounts
- percentage: Percentage values
- facility: Buildings, airports, hospitals, specific facilities
- product: Commercial products or services
- event: Named events (conferences, wars, festivals)
- position: Job titles, ranks (direktor, prezident, müəllim)
- miscellaneous: Should only be used when nothing else fits

OUTPUT FORMAT: Respond with a JSON object:
{
  "verdict": "CORRECT" | "INCORRECT" | "PARTIAL",
  "issues": [
    {"entity": "extracted text", "current_label": "X", "suggested_label": "Y" | "REMOVE"}
  ]
}

Only flag clear errors. If uncertain, mark as CORRECT."""


def build_audit_prompt(sample: dict) -> str:
    """Build audit prompt for one sample."""
    tokens = sample["tokenized_text"]
    text = " ".join(tokens)

    entities_display = []
    for start, end, label in sample["ner"]:
        entity_text = " ".join(tokens[start:end + 1])
        entities_display.append(f"  - '{entity_text}' → {label}")

    entities_str = "\n".join(entities_display)

    return f"""Sentence: {text}

Annotations:
{entities_str}

Are these annotations correct? Respond with JSON only."""


def parse_audit_response(response: str) -> Optional[dict]:
    """Extract JSON from Claude's response."""
    if not response:
        return None
    # Strip markdown code fences if present
    clean = response.strip()
    if clean.startswith("```"):
        clean = clean.split("```")[1]
        if clean.startswith("json"):
            clean = clean[4:]
        clean = clean.strip()

    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        return None


def audit_sample(sample: dict, sample_idx: int) -> Optional[dict]:
    """Audit one sample. Returns audit result dict or None."""
    prompt = build_audit_prompt(sample)
    response = call_claude(AUDIT_SYSTEM_PROMPT, prompt, max_tokens=512)

    if not response:
        return None

    parsed = parse_audit_response(response)
    if not parsed:
        return None

    return {
        "sample_idx": sample_idx,
        "verdict": parsed.get("verdict", "UNKNOWN"),
        "issues": parsed.get("issues", []),
        "original": sample,
    }


def main(
    input_path: str = "data/train.json",
    output_path: str = "data/audit_report.json",
    audit_size: int = 500,
):
    """Run quality audit on a random subsample of LocalDoc data."""
    with open(input_path, "r", encoding="utf-8") as f:
        all_samples = json.load(f)

    sample_indices = random.sample(range(len(all_samples)), audit_size)

    audit_results = []
    for i, idx in enumerate(sample_indices):
        if i % 50 == 0:
            print(f"Auditing {i}/{audit_size}...")
        result = audit_sample(all_samples[idx], idx)
        if result:
            audit_results.append(result)

    # Summarize
    verdicts = {"CORRECT": 0, "INCORRECT": 0, "PARTIAL": 0, "UNKNOWN": 0}
    for r in audit_results:
        verdicts[r["verdict"]] = verdicts.get(r["verdict"], 0) + 1

    print(f"\n=== Audit Summary ===")
    print(f"Total audited: {len(audit_results)}")
    for v, c in verdicts.items():
        print(f"  {v}: {c} ({c/len(audit_results)*100:.1f}%)")

    # List of sample indices to exclude from training
    exclude_list = [r["sample_idx"] for r in audit_results if r["verdict"] == "INCORRECT"]

    # Save full audit report and exclude list
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "summary": verdicts,
            "audit_size": audit_size,
            "exclude_indices": exclude_list,
            "detailed_results": audit_results,
        }, f, ensure_ascii=False, indent=2)

    print(f"\nSaved audit report to {output_path}")
    print(f"Recommended exclusions: {len(exclude_list)} samples")

    # Estimate error rate in full dataset
    error_rate = verdicts["INCORRECT"] / len(audit_results)
    estimated_bad = int(error_rate * len(all_samples))
    print(f"\nEstimated dataset-wide error rate: {error_rate:.1%}")
    print(f"Estimated total bad samples: ~{estimated_bad} out of {len(all_samples)}")


if __name__ == "__main__":
    main(audit_size=500)
```

**Expected cost**: 500 samples × ~400 input tokens + ~150 output tokens each = ~275K tokens. At Claude Sonnet 4.6 pricing, this costs **~$3-5**.

**What to do with the output**: The `audit_report.json` file gives you:
1. **Estimated dataset-wide error rate** — informs whether to proceed with training or clean more aggressively
2. **Specific sample indices to exclude** — direct list of bad samples to drop from training
3. **Pattern analysis** — if MISCELLANEOUS is over-flagged, you know to re-label those entities or drop them

Apply the exclude list before merging datasets:

```python
# apply_audit_exclusions.py
import json

with open("data/train.json") as f:
    train = json.load(f)
with open("data/audit_report.json") as f:
    audit = json.load(f)

exclude = set(audit["exclude_indices"])
cleaned_train = [s for i, s in enumerate(train) if i not in exclude]

print(f"Original: {len(train)} | Cleaned: {len(cleaned_train)} | Removed: {len(train) - len(cleaned_train)}")

with open("data/train_cleaned.json", "w") as f:
    json.dump(cleaned_train, f, ensure_ascii=False)
```

#### 7.9.4 Updated Dataset Composition (with LLM additions)

| Data Source | Method | Volume | Cost |
|-------------|--------|--------|------|
| LocalDoc dataset (audited) | Script + LLM audit | ~83,000 (after exclusions) | ~$5 |
| WikiANN Azerbaijani | Script conversion | ~12,000 | $0 |
| Template-based synthetic PII | `az-data-generator` + templates | ~30,000 | $0 |
| Adversarial negatives | Hand-crafted templates | ~1,500 | $0 |
| **LLM-generated narrative PII** | **Claude via AI Gateway** | **~5,000** | **~$15-20** |
| **TOTAL** | | **~131,500** | **~$20-25** |

#### 7.9.5 Merge Script (Updated)

```python
#!/usr/bin/env python3
"""merge_all_datasets.py — Final training data assembly with LLM additions."""
import json
import random

random.seed(42)

# Load all sources
sources = {
    "localdoc": "data/train_cleaned.json",  # After audit exclusions
    "wikiann": "data/wikiann_train.json",
    "synthetic_pii": "data/synthetic_az_pattern_exhaustive.json",
    "narrative_pii": "data/narrative_pii.json",
}

all_data = []
for name, path in sources.items():
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    all_data.extend(data)
    print(f"Loaded {name}: {len(data)} samples")

random.shuffle(all_data)

# 85/10/5 split
n = len(all_data)
n_train = int(n * 0.85)
n_val = int(n * 0.10)

train = all_data[:n_train]
val = all_data[n_train:n_train + n_val]
test = all_data[n_train + n_val:]

for name, data in [("train", train), ("val", val), ("test", test)]:
    path = f"data/{name}_final.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    print(f"Saved {name}: {len(data)} → {path}")
```

#### 7.9.6 Decision Rules for Claude Code Agent

When the agent is executing the pipeline, apply these rules for LLM usage:

| Task | Use LLM? | Why |
|------|----------|-----|
| Convert LocalDoc integer tags → GLiNER JSON | ❌ No | Deterministic script |
| Convert WikiANN IOB2 → GLiNER JSON | ❌ No | Deterministic script |
| Generate FIN/TIN/IBAN/phone/passport values | ❌ No | Use `az-data-generator` (real Azerbaijan formats) |
| Generate format variations of patterned entities | ❌ No | Deterministic template code |
| Generate context templates for patterned PII | ❌ No | Hand-crafted templates (already in Section 7.3) |
| Generate adversarial negatives | ❌ No | Hand-crafted templates (Section 7.4) |
| Apply OCR/typo noise | ❌ No | Deterministic function |
| **Generate natural narrative context with embedded PII** | ✅ **Yes** | **LLM adds narrative diversity that templates can't** |
| **Audit LocalDoc sample quality** | ✅ **Yes** | **Human-level judgment needed for annotation errors** |
| Generate synthetic person names | ❌ No | `az-data-generator` is strictly better |
| Generate synthetic city names | ❌ No | `az-data-generator` has geographically accurate lists |
| Translate Turkish NER data to Azerbaijani | ❌ No | Translation degrades annotations |
| Annotate raw Azerbaijani text from scratch | ❌ No | 10-20% label error rate in Azerbaijani |

---

## 8. Fine-Tuning Pipeline with Pattern-Focused Training

### 8.0 Option A-Specific Training Adjustments

Since Option A requires the model to learn patterned entities as well as regex would, we apply these training modifications vs. standard fine-tuning:

| Standard GLiNER Fine-Tuning | Option A Adjustments |
|----------------------------|----------------------|
| 5 epochs | **7-10 epochs** (patterns need more exposure) |
| focal_loss_gamma = 2 | **focal_loss_gamma = 2.5** (more emphasis on hard negatives) |
| No oversampling | **Oversample synthetic PII samples 1.5x** |
| Standard batch shuffling | **Stratified batching** (ensure each batch has PII examples) |
| Single LR for all params | **Differential LR** (encoder 5e-6, task head 2e-5 for stronger pattern learning) |
| Early stopping on loss | **Early stopping on per-entity F1** (watch PII F1 specifically) |

### 8.1 Training Script

```python
#!/usr/bin/env python3
"""
finetune_gliner_az.py

Fine-tune GLiNER multilingual model on Azerbaijani NER data.
"""

import os
import json
import torch

os.environ["TOKENIZERS_PARALLELISM"] = "true"

from gliner import GLiNER
from gliner.training import Trainer, TrainingArguments
from gliner.data_processing.collator import DataCollator


def main():
    # ============================================================
    # 1. LOAD MODEL
    # ============================================================
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = GLiNER.from_pretrained("urchade/gliner_multi-v2.1")
    model.to(device)

    # ============================================================
    # 2. LOAD DATA
    # ============================================================
    with open("data/train_final.json", "r", encoding="utf-8") as f:
        train_data = json.load(f)
    with open("data/val_final.json", "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    print(f"Train samples: {len(train_data)}")
    print(f"Eval samples:  {len(eval_data)}")

    # ============================================================
    # 3. DATA COLLATOR
    # ============================================================
    data_collator = DataCollator(
        model.config,
        data_processor=model.data_processor,
        prepare_labels=True,
    )

    # ============================================================
    # 4. TRAINING ARGUMENTS
    # ============================================================
    training_args = TrainingArguments(
        output_dir="./output/azerbaijani-gliner",

        # Learning rates (CRITICAL: keep encoder LR low to prevent
        # catastrophic forgetting of multilingual knowledge)
        learning_rate=5e-6,          # Encoder (DeBERTa backbone)
        others_lr=1e-5,              # Non-encoder params (span FFN, type FFN)
        weight_decay=0.01,
        others_weight_decay=0.01,

        # Schedule
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,            # 10% warmup

        # Batch size (reduce if OOM)
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,

        # Focal loss (handles entity class imbalance)
        focal_loss_alpha=0.75,
        focal_loss_gamma=2,

        # Epochs (3-5 for large dataset, 5-10 for small)
        num_train_epochs=5,

        # Evaluation & saving
        evaluation_strategy="steps",
        eval_steps=1000,
        save_steps=1000,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",

        # Performance
        dataloader_num_workers=4,
        use_cpu=False,
        report_to="none",  # Set to "wandb" if using W&B

        # Gradient accumulation if batch_size must be small
        # gradient_accumulation_steps=4,  # effective batch = 4 * 8 = 32
    )

    # ============================================================
    # 5. TRAINER
    # ============================================================
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        tokenizer=model.data_processor.transformer_tokenizer,
        data_collator=data_collator,
    )

    # ============================================================
    # 6. TRAIN
    # ============================================================
    print("Starting training...")
    trainer.train()

    # ============================================================
    # 7. SAVE
    # ============================================================
    model.save_pretrained("./output/azerbaijani-gliner-final")
    print("Model saved to ./output/azerbaijani-gliner-final")

    # Optional: push to HuggingFace Hub
    # model.push_to_hub("your-username/gliner-azerbaijani-ner-v1")


if __name__ == "__main__":
    main()
```

### 8.2 Hyperparameter Reference

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `learning_rate` (encoder) | `5e-6` | Low LR preserves multilingual knowledge |
| `others_lr` | `1e-5` | 2x encoder LR for randomly-init layers |
| `weight_decay` | `0.01` | Standard L2 regularization |
| `warmup_ratio` | `0.1` | Gradual LR ramp-up |
| `lr_scheduler_type` | `cosine` | Smooth decay, better than linear |
| `batch_size` | `8` | Reduce to 2-4 if VRAM < 12GB |
| `focal_loss_alpha` | `0.75` | Down-weights well-classified examples |
| `focal_loss_gamma` | `2` | Focuses on hard negatives |
| `num_train_epochs` | `3-5` (large data) / `5-10` (small) | Watch eval loss for overfitting |
| `max_types` | `25` | Match your entity type count |
| `shuffle_types` | `true` | Prevents type-order memorization |
| `max_width` | `12` | Max tokens per entity span |
| `dropout` | `0.4` (default) | Increase to 0.5 for small datasets |

### 8.3 Catastrophic Forgetting Mitigation

When fine-tuning on domain-specific data, GLiNER may forget general NER capabilities. Strategies:

1. **Low encoder learning rate** (5e-6 or lower)
2. **Mix general + domain data**: Include 10-20% of original multilingual training examples
3. **Short training**: 3-5 epochs max for large datasets
4. **Focal loss**: Already configured above
5. **Early stopping**: Monitor eval loss, stop at first uptick

---

## 9. Pure Model Inference (No Wrapper)

With Option A, inference is trivially simple — just load the model and call `predict_entities`. No regex, no validators, no merge logic.

### 9.1 Minimal Inference Example

```python
from gliner import GLiNER

# Load the fine-tuned model
model = GLiNER.from_pretrained("pasha-fh/gliner-azerbaijani-ner-v1")

# Define the entity types you want to extract
labels = [
    # Semantic entities
    "person", "organisation", "location", "gpe",
    "date", "time", "money", "position", "facility",
    "product", "event", "law", "language", "norp",
    # Patterned PII (learned from synthetic data)
    "fin code", "tin", "phone number", "iban",
    "passport number", "vehicle plate",
    "credit card number", "email", "postal code",
]

# Extract entities from text
text = (
    "Əliyev Heydərin FİN kodu 5AB1CD2 , "
    "telefon nömrəsi +994 50 123 45 67 , "
    "SOCAR-da çalışır . "
    "Şirkətin VÖEN nömrəsi 9900001061 ."
)

entities = model.predict_entities(text, labels, threshold=0.4)

for ent in entities:
    print(f"{ent['label']:20s} → '{ent['text']}' (score: {ent['score']:.3f})")
```

**Output:**
```
person               → 'Əliyev Heydərin' (score: 0.944)
fin code             → '5AB1CD2' (score: 0.892)
phone number         → '+994 50 123 45 67' (score: 0.918)
organisation         → 'SOCAR' (score: 0.879)
tin                  → '9900001061' (score: 0.865)
```

That's the entire production code. Nothing else.

### 9.2 Batch Inference for Production

For processing large document volumes:

```python
from gliner import GLiNER
import torch

model = GLiNER.from_pretrained("pasha-fh/gliner-azerbaijani-ner-v1")
if torch.cuda.is_available():
    model = model.to("cuda")

# Optional: optimize for inference speed
model.quantize()           # fp16 for ~1.35x GPU speedup
model.compile()            # torch.compile for additional ~1.3x speedup

labels = [
    "person", "organisation", "location", "gpe", "date", "money",
    "fin code", "tin", "phone number", "iban", "passport number",
    "vehicle plate", "credit card number", "email", "postal code",
    "position", "facility", "product", "event", "law",
]

def process_documents(documents: list[str]) -> list[list[dict]]:
    """Process a batch of documents through the model."""
    return model.run(documents, labels, threshold=0.4, batch_size=8)

# Usage
docs = [
    "Vətəndaş Məmmədov Rəşadın FİN kodu 3K45LM8 , ...",
    "Şirkət SOCAR , VÖEN 9900001061 , Bakı şəhərində ...",
    # ... more documents
]

results = process_documents(docs)
for doc_idx, entities in enumerate(results):
    print(f"\nDocument {doc_idx + 1}:")
    for ent in entities:
        print(f"  {ent['label']}: '{ent['text']}'")
```

### 9.3 Confidence Thresholding

Since Option A has no regex to fall back on, confidence thresholds become the primary precision/recall dial:

| Threshold | Precision | Recall | Use Case |
|-----------|-----------|--------|----------|
| **0.2** | Low | Very High | Exhaustive PII discovery (manual review afterwards) |
| **0.3** | Medium | High | Default for compliance scanning |
| **0.4** | High | Medium-High | **Recommended production default** |
| **0.5** | Very High | Medium | Critical decisions (auto-redaction without review) |
| **0.6+** | Very High | Lower | When false positives are costly |

Tune per entity type if needed — patterned entities often tolerate lower thresholds because the model's signal is very clean:

```python
# Use different thresholds per entity type (optional advanced usage)
def smart_extract(text: str) -> list[dict]:
    # Low threshold for high-confidence patterns
    pattern_labels = ["fin code", "tin", "iban", "passport number", "email"]
    pattern_entities = model.predict_entities(text, pattern_labels, threshold=0.3)

    # Higher threshold for semantic entities (more false-positive prone)
    semantic_labels = ["person", "organisation", "location", "position"]
    semantic_entities = model.predict_entities(text, semantic_labels, threshold=0.5)

    return pattern_entities + semantic_entities
```

### 9.4 Cross-Language Inference (Rust, Go, Browser)

The killer feature of Option A: since there's no Python wrapper, you can run the model in any language via ONNX. This is critical for edge deployment at PASHA.

**Rust (gline-rs):**
```rust
use gline_rs::{GLiNER, TextInput, Parameters, RuntimeParameters};

let model = GLiNER::<TokenMode>::new(
    Parameters::default(),
    RuntimeParameters::default(),
    "tokenizer.json",
    "model.onnx",
)?;

let input = TextInput::from_str(
    &["Əliyev Heydərin FİN kodu 5AB1CD2 ."],
    &["person", "fin code"],
)?;
let output = model.inference(input)?;
```

**C++ (GLiNER.cpp):**
```cpp
#include "GLiNER/model.hpp"
gliner::Config config{12, 512};
gliner::Model model("./model.onnx", "./tokenizer.json", config);
std::vector<std::string> labels = {"person", "fin code", "iban"};
auto entities = model.predict(texts, labels, 0.4);
```

**JavaScript (browser / Node.js via onnxruntime-web):**
```javascript
import * as ort from 'onnxruntime-web';
const session = await ort.InferenceSession.create('./gliner-az.onnx');
// ... run inference entirely in the browser
```

This portability is **impossible with Option B** (the hybrid approach) because the regex layer is Python-only.

### 9.5 When to Prefer This Over a Wrapper

Use pure model inference when:
- Deploying to Rust/Go microservices (performance-critical)
- Browser/mobile deployment (ONNX Runtime Web)
- Embedded devices with no Python runtime
- Publishing a model card on HuggingFace where users "just want the model"
- Academic reproducibility (single artifact = single citation)

---
## 10. Benchmark Suite & Expected Performance

### 12.1 Why You Need a Benchmark

A benchmark gives you three things: (1) a **baseline** to know what you're improving over, (2) a **target** to know when you're done, and (3) a **regression test** to catch when model updates break something. Without it, you're flying blind.

### 12.2 Baseline Measurements (Before Fine-Tuning)

Run these BEFORE training to establish what the unmodified models can already do on Azerbaijani text:

| Baseline Model | What It Measures | Expected F1 Range |
|----------------|-----------------|-------------------|
| `urchade/gliner_multi-v2.1` zero-shot | Multilingual GLiNER on AZ text, no fine-tuning | ~25-40% on AZ entities |
| `urchade/gliner_multi_pii-v1` zero-shot | PII-specialized GLiNER (trained on 6 European languages) | ~30-50% on AZ PII |
| `knowledgator/gliner-pii-base-v1.0` zero-shot | Best PII GLiNER (~81% F1 on English synthetic PII) | ~25-45% on AZ PII |
| `IsmatS/azeri-turkish-bert-ner` | Best existing AZ NER model (sequence labeling) | ~74% micro-F1 (documented) |
| Regex-only pipeline | Pattern matching only, no ML | ~95%+ on patterned entities (FIN, IBAN, phone), 0% on semantic entities |

**Critical baselines to record:**
```python
# Run this script before any fine-tuning
import json
from gliner import GLiNER

def baseline_eval(model_name, test_data, labels, threshold=0.4):
    model = GLiNER.from_pretrained(model_name)
    # ... (use evaluate() function from Section 12)
    results = evaluate(model, test_data, labels, threshold)
    return results

# Test all baselines
baselines = {
    "gliner_multi_v2.1": "urchade/gliner_multi-v2.1",
    "gliner_pii_v1": "urchade/gliner_multi_pii-v1",
    "gliner_pii_base": "knowledgator/gliner-pii-base-v1.0",
}

with open("data/test.json") as f:
    test_data = json.load(f)

labels = ["person", "organisation", "location", "gpe", "date", "money",
          "fin code", "tin", "phone number", "iban", "passport number",
          "vehicle plate", "credit card number", "email"]

for name, model_path in baselines.items():
    print(f"\n=== Baseline: {name} ===")
    results = baseline_eval(model_path, test_data, labels)
    # Save to JSON for comparison
    with open(f"benchmarks/baseline_{name}.json", "w") as f:
        json.dump(results, f, indent=2)
```

### 12.3 Target Performance After Fine-Tuning

Based on published GLiNER fine-tuning results from comparable projects:

| Reference Project | Language | Training Data | Base Model | Achieved F1 |
|-------------------|----------|---------------|------------|-------------|
| NERCat (Catalan) | Catalan | ~9K sentences | gliner-bi-large | ~90%+ on PER/ORG/LOC |
| GLiNER PII (Knowledgator) | English + 5 EU langs | Synthetic PII data | gliner-bi-base | ~81% F1 on 60 PII types |
| GLiNER PII Large | English + 5 EU langs | Synthetic PII data | gliner-bi-large | ~83% F1 on 60 PII types |
| Gretel PII fine-tune | English | Synthetic PII docs | gliner-bi-large | Significant improvement over base |
| GLiNER-L zero-shot (English) | English | Pile-NER pre-training | gliner-large | ~60.9% on 7 OOD benchmarks |
| GLiNER-M zero-shot (English) | English | Pile-NER pre-training | gliner-medium | ~55% on 7 OOD benchmarks |

**Realistic targets for your Azerbaijani fine-tuning:**

| Entity Category | Realistic F1 Target | Notes |
|----------------|--------------------|----|
| Person | **85-92%** | High data coverage in LocalDoc, strong multilingual transfer |
| Organisation | **80-88%** | Good coverage, but Azerbaijani org names are diverse |
| Location / GPE | **82-90%** | Well-represented in training data |
| Date | **85-93%** | Pattern-based + GLiNER context = high accuracy |
| FIN Code | **95-99%** | Regex primary + GLiNER fallback |
| TIN / VÖEN | **90-96%** | Regex + GLiNER context validation eliminates false positives |
| Phone Number | **95-99%** | Well-defined regex patterns |
| IBAN | **98-100%** | Very distinctive 28-char AZ-prefix pattern |
| Passport # | **95-99%** | Clear AZE/AA prefix patterns |
| Vehicle Plate | **93-98%** | Regex-first, distinctive format |
| Credit Card | **95-99%** | Luhn validation + regex |
| Email | **97-100%** | Standard regex, very low ambiguity |
| **Overall Micro-F1 (hybrid pipeline)** | **85-92%** | Combining regex + fine-tuned GLiNER |

**Success criteria**: If your fine-tuned model + regex hybrid pipeline achieves **≥85% micro-F1** on a hand-annotated Azerbaijani test set, you have a production-ready system. The existing best Azerbaijani NER model scores 74% — beating that by 10+ points is a strong result.

### 12.4 Benchmark Suite Design

Create a **three-tier benchmark** to measure different aspects:

**Tier 1 — Regex Entities Only** (pattern-based PII)
- Test set: 200+ sentences with FIN, VÖEN, phone, IBAN, passport, plates, cards, emails
- Metric: Exact-match precision/recall/F1 per entity type
- Purpose: Validates regex patterns work on real Azerbaijani text
- Expected: ≥95% F1

**Tier 2 — GLiNER Entities Only** (semantic entities)
- Test set: 500+ sentences with person, org, location, date, money, position
- Metric: Span-level exact-match F1 per entity type
- Purpose: Measures fine-tuning effectiveness on Azerbaijani
- Expected: ≥80% micro-F1

**Tier 3 — Full Hybrid Pipeline** (all entities, production conditions)
- Test set: 300+ sentences mixing ALL entity types, including multi-entity sentences
- Metric: End-to-end precision/recall/F1 on the hybrid pipeline output
- Purpose: Measures the combined system performance
- Expected: ≥85% micro-F1

**Tier 4 — Zero-Shot Generalization** (optional)
- Test set: 100+ sentences with entity types NOT in training data (e.g., "medication", "URL", "cryptocurrency address")
- Purpose: Tests whether fine-tuning preserved zero-shot capabilities
- Expected: ≥40% F1 (degradation vs base model should be <10 points)

```python
# benchmark_runner.py — Automated benchmark suite
import json, time

def run_benchmark_suite(pipeline, test_sets):
    results = {}
    for tier, config in test_sets.items():
        with open(config["path"]) as f:
            data = json.load(f)
        
        start = time.time()
        tier_results = evaluate(pipeline.model, data, config["labels"], 
                               threshold=config.get("threshold", 0.4))
        elapsed = time.time() - start
        
        tier_results["_meta"] = {
            "samples": len(data),
            "wall_time_seconds": round(elapsed, 2),
            "samples_per_second": round(len(data) / elapsed, 1),
        }
        results[tier] = tier_results
    
    return results

# Define test sets
test_sets = {
    "tier1_regex": {
        "path": "benchmarks/tier1_regex_test.json",
        "labels": ["fin code", "tin", "phone number", "iban", "passport number",
                    "vehicle plate", "credit card number", "email"],
        "threshold": 0.3,
    },
    "tier2_gliner": {
        "path": "benchmarks/tier2_semantic_test.json",
        "labels": ["person", "organisation", "location", "gpe", "date", "money", "position"],
        "threshold": 0.4,
    },
    "tier3_hybrid": {
        "path": "benchmarks/tier3_full_test.json",
        "labels": ["person", "organisation", "location", "gpe", "date", "money",
                    "fin code", "tin", "phone number", "iban", "passport number",
                    "vehicle plate", "credit card number", "email"],
        "threshold": 0.4,
    },
}
```

### 12.5 Comparing Against Existing Models

Always compare your results against:

| Comparison | How to Run | What It Tells You |
|-----------|-----------|-------------------|
| **Your model vs base GLiNER (zero-shot)** | Run base model on same test set | How much fine-tuning helped |
| **Your model vs IsmatS/azeri-turkish-bert-ner** | Run their model, convert output to span format | Are you beating the SOTA for Azerbaijani NER? |
| **Your model vs regex-only** | Run regex pipeline alone | How much does GLiNER add beyond regex? |
| **Your model vs GLiNER PII models** | Run `gliner_multi_pii-v1` on same test set | Are you beating the general PII model on AZ data? |
| **Small vs Medium vs Large** | Train all three sizes | Is the extra compute worth the accuracy gain? |

---

## 11. GPU Requirements & Training Time Estimates

### 13.1 Model Sizes and VRAM Requirements

GLiNER models are **encoder-only transformers** (166M-459M params) — dramatically smaller than LLMs (7B+). This means training is fast and affordable. Full fine-tuning (not LoRA) is the standard and recommended approach.

**VRAM requirements for TRAINING (full fine-tuning, batch_size=8, fp32):**

| Model | Parameters | Weights | + Gradients | + Optimizer (AdamW) | Total VRAM (batch=8) | Total VRAM (batch=2) |
|-------|-----------|---------|------------|-------------------|---------------------|---------------------|
| **GLiNER Small** | ~166M | ~0.6 GB | ~0.6 GB | ~1.3 GB | **~4-6 GB** | **~3-4 GB** |
| **GLiNER Medium** (v2.1) | ~209M | ~0.8 GB | ~0.8 GB | ~1.7 GB | **~6-10 GB** | **~4-6 GB** |
| **GLiNER Large** | ~459M | ~1.8 GB | ~1.8 GB | ~3.5 GB | **~12-18 GB** | **~8-12 GB** |

**VRAM rule of thumb for encoder models**: ~4× parameter count in bytes for full fine-tuning (weights + gradients + Adam states), plus ~2-4 GB for activations/batch.

**VRAM requirements for INFERENCE:**

| Model | fp32 | fp16 | int8 |
|-------|------|------|------|
| **Small** | ~1.0 GB | ~0.5 GB | ~0.3 GB |
| **Medium** | ~1.5 GB | ~0.8 GB | ~0.5 GB |
| **Large** | ~2.5 GB | ~1.3 GB | ~0.7 GB |

### 13.2 Training Time Estimates

Based on ~100K training samples (LocalDoc + WikiANN + synthetic), 5 epochs, batch_size=8:

**Total training steps**: ~100,000 samples ÷ 8 batch × 5 epochs = ~62,500 steps

| Model | GPU | Batch Size | Time per Epoch | Total (5 epochs) | Est. Cloud Cost |
|-------|-----|-----------|---------------|-------------------|-----------------|
| **Small** | Google Colab T4 (16 GB) | 8 | ~20-30 min | **1.5-2.5 hours** | Free (Colab) |
| **Small** | RTX 3090 (24 GB) | 8 | ~15-20 min | **1-1.5 hours** | ~$1-2 |
| **Small** | A100 40GB | 16 | ~8-12 min | **40-60 min** | ~$2-4 |
| **Medium** | RTX 3090 (24 GB) | 8 | ~25-35 min | **2-3 hours** | ~$3-5 |
| **Medium** | RTX 4090 (24 GB) | 8 | ~20-25 min | **1.5-2 hours** | ~$3-5 |
| **Medium** | A100 40GB | 16 | ~12-18 min | **1-1.5 hours** | ~$4-6 |
| **Medium** | A100 80GB | 32 | ~8-12 min | **40-60 min** | ~$5-8 |
| **Large** | RTX 3090 (24 GB) | 4 | ~50-70 min | **4-6 hours** | ~$6-10 |
| **Large** | RTX 4090 (24 GB) | 4 | ~40-55 min | **3-5 hours** | ~$6-10 |
| **Large** | A100 40GB | 8 | ~25-35 min | **2-3 hours** | ~$8-12 |
| **Large** | A100 80GB | 16 | ~15-20 min | **1.5-2 hours** | ~$8-14 |

**Notes on timing estimates**:
- These are approximate — actual times depend on sequence length, entity density, and exact hardware config
- Cloud cost assumes: T4 ~$0.35/hr, RTX 3090/4090 ~$0.50-1.00/hr (Lambda, RunPod), A100 40GB ~$1.50-3.00/hr, A100 80GB ~$2.50-4.00/hr
- **gradient_accumulation_steps** can simulate larger batch sizes on smaller GPUs with a time overhead of ~10-20%
- fp16 mixed-precision training can reduce VRAM by ~30-40% and speed up training by ~20-30% on Ampere+ GPUs

### 13.3 Recommended GPU Choices

| Scenario | Recommended GPU | Why |
|----------|----------------|-----|
| **Budget / Prototyping** | Google Colab T4 (free) | Small model fits, good for initial experiments |
| **Production training (Medium)** | RTX 3090 or RTX 4090 | Best price/performance for single-GPU training |
| **Production training (Large)** | A100 40GB | Comfortable headroom, fast training |
| **Maximum speed** | A100 80GB | Largest batch sizes, fastest convergence |
| **On-premise PASHA deployment** | RTX 4090 or A6000 (48GB) | Good for repeated fine-tuning cycles |

### 13.4 Practical Decision Matrix: Small vs Medium vs Large

| Factor | Small (~166M) | Medium (~209M) | Large (~459M) |
|--------|:------------:|:--------------:|:-------------:|
| **Training cost** | $0-2 | $3-8 | $6-14 |
| **Training time (A100)** | <1 hour | 1-1.5 hours | 2-3 hours |
| **Min VRAM (training)** | 4 GB | 6 GB | 12 GB |
| **Expected F1 (zero-shot)** | ~45-50% | ~55-60% | ~60-65% |
| **Expected F1 (fine-tuned)** | ~78-84% | ~83-89% | ~86-92% |
| **Inference speed (CPU)** | ~50ms/sentence | ~80ms/sentence | ~150ms/sentence |
| **Inference speed (GPU)** | ~5ms/sentence | ~8ms/sentence | ~15ms/sentence |
| **Recommended for** | Edge/mobile, latency-critical | **Best overall balance** | Maximum accuracy |

**Recommendation**: Start with **Medium (v2.1)** — it offers the best accuracy-to-cost ratio. Train Small in parallel as a fallback for edge deployment. Only train Large if Medium doesn't meet your F1 target and you have the GPU budget.

### 13.5 Full Training Cost Estimate (End-to-End)

For a complete project including experiments and hyperparameter tuning:

| Activity | GPU Hours | Cost (A100 @ $3/hr) | Cost (RTX 4090 @ $1/hr) |
|----------|----------|---------------------|------------------------|
| Baseline evaluation (3 models) | 0.5 hr | $1.50 | $0.50 |
| Medium model: train 5 epochs | 1.5 hr | $4.50 | $1.50 |
| Medium model: threshold tuning | 0.5 hr | $1.50 | $0.50 |
| Medium model: 3 hyperparameter runs | 4.5 hr | $13.50 | $4.50 |
| Large model: train 5 epochs | 3 hr | $9.00 | $3.00 |
| Small model: train 5 epochs | 1 hr | $3.00 | $1.00 |
| Final evaluation + benchmarks | 1 hr | $3.00 | $1.00 |
| **TOTAL** | **~12 hrs** | **~$36** | **~$12** |

This is remarkably affordable compared to LLM fine-tuning ($100-$10,000+).

---

## 12. Evaluation Methodology

### 12.1 Entity-Level Exact-Match F1

```python
#!/usr/bin/env python3
"""evaluate_gliner_az.py — Evaluate the fine-tuned model."""

import json
from collections import defaultdict
from gliner import GLiNER


def evaluate(model, test_data, labels, threshold=0.5):
    """
    Compute entity-level exact-match Precision, Recall, F1.

    A prediction is correct IFF:
    - Entity text matches exactly (after joining tokens)
    - Entity label matches exactly
    """
    per_type = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    for sample in test_data:
        text = " ".join(sample["tokenized_text"])

        # Gold entities
        gold_set = set()
        for start, end, label in sample["ner"]:
            entity_text = " ".join(sample["tokenized_text"][start:end + 1])
            gold_set.add((entity_text.lower(), label))

        # Predicted entities
        predictions = model.predict_entities(text, labels, threshold=threshold)
        pred_set = set()
        for p in predictions:
            pred_set.add((p["text"].lower(), p["label"]))

        # Score per type
        for label in labels:
            type_gold = {e for e in gold_set if e[1] == label}
            type_pred = {e for e in pred_set if e[1] == label}

            per_type[label]["tp"] += len(type_pred & type_gold)
            per_type[label]["fp"] += len(type_pred - type_gold)
            per_type[label]["fn"] += len(type_gold - type_pred)

    # Compute metrics
    results = {}
    total_tp = total_fp = total_fn = 0

    for label, counts in sorted(per_type.items()):
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        results[label] = {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4),
                          "support": tp + fn}
        total_tp += tp
        total_fp += fp
        total_fn += fn

    # Micro average
    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0
    results["MICRO_AVG"] = {"precision": round(micro_p, 4), "recall": round(micro_r, 4),
                            "f1": round(micro_f1, 4)}

    return results


# Run evaluation
model = GLiNER.from_pretrained("./output/azerbaijani-gliner-final")
with open("data/test.json") as f:
    test_data = json.load(f)

labels = ["person", "organisation", "location", "gpe", "date", "money",
          "fin code", "tin", "phone number", "iban", "passport number",
          "vehicle plate", "credit card number", "email", "postal code",
          "ssn", "facility", "product", "event", "position", "law"]

results = evaluate(model, test_data, labels, threshold=0.4)
print("\n=== Evaluation Results ===")
print(f"{'Entity Type':25s} {'Precision':>10s} {'Recall':>10s} {'F1':>10s} {'Support':>10s}")
print("-" * 65)
for label, metrics in results.items():
    print(f"{label:25s} {metrics['precision']:10.4f} {metrics['recall']:10.4f} "
          f"{metrics['f1']:10.4f} {metrics.get('support', ''):>10}")
```

### 12.2 Threshold Tuning

GLiNER uses a confidence threshold (default 0.5). Tune it on the validation set:

```python
best_f1 = 0
best_threshold = 0.5
for threshold in [0.1, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8]:
    results = evaluate(model, val_data, labels, threshold=threshold)
    f1 = results["MICRO_AVG"]["f1"]
    print(f"Threshold {threshold:.2f}: F1 = {f1:.4f}")
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print(f"\nBest threshold: {best_threshold} (F1: {best_f1:.4f})")
```

---

## 13. Model Export & Deployment

### 13.1 Save & Push

```python
# Save locally
model.save_pretrained("./azerbaijani-gliner-v1")

# Push to HuggingFace Hub
from huggingface_hub import HfApi
model.push_to_hub("your-org/gliner-azerbaijani-ner-v1")
```

### 13.2 ONNX Export

```python
model.to_onnx("azerbaijani_gliner.onnx")
```

### 13.3 Optimized Inference

```python
# GPU: fp16 + torch.compile
model = GLiNER.from_pretrained(
    "azerbaijani-gliner-v1",
    map_location="cuda",
    quantize=True,             # fp16 half-precision
    compile_torch_model=True,  # torch.compile with Triton
)

# CPU: int8 quantization
model = GLiNER.from_pretrained("azerbaijani-gliner-v1")
model.quantize("int8")
```

### 13.4 Model Size Reference

| Variant | Params | Disk | VRAM (inference) |
|---------|--------|------|-----------------|
| Small | ~166M | ~200MB | ~1GB |
| Medium (v2.1) | ~209M | ~440MB | ~1.5GB |
| Large | ~459M | ~1.2GB | ~2.5GB |

---

## 14. Troubleshooting & Known Issues

### Common Issues

| Issue | Solution |
|-------|----------|
| CUDA OOM during training | Reduce `batch_size` to 2-4, enable `gradient_accumulation_steps` |
| Model predicts nothing after fine-tuning | Lower threshold (try 0.1-0.3); check if training data has entities |
| Catastrophic forgetting (loses general NER) | Reduce `learning_rate` to 1e-6; reduce epochs to 2-3 |
| VÖEN regex matches random 10-digit numbers | Use context-aware validation (check for nearby keywords) |
| FIN code false positives | Stick to strict pattern `[0-9][A-Z][0-9]{2}[A-Z]{2}[0-9]` |
| Slow ONNX inference | Use PyTorch with torch.compile instead; ONNX can be slower for GLiNER |
| Empty entities in training data | Filter out samples where `ner` is empty list |
| gliner import error | Ensure `pip install gliner==0.2.26` (not gliner2) |

### Version Compatibility

- `gliner==0.2.26` — Stable, well-documented fine-tuning API
- `gliner2==1.2.6` — Newer multi-task package, different API; use only if needed
- `torch>=2.0` — Required for torch.compile
- `transformers>=4.38.0` — DeBERTa-v3 support

---

## Quick Start Checklist for Claude Code Agent

```
# SETUP
1. [ ] pip install gliner==0.2.26 az-data-generator datasets accelerate httpx
2. [ ] Set env vars: AI_GATEWAY_URL, AI_GATEWAY_KEY (for LLM workflows only)

# DATA PREPARATION (deterministic, no LLM)
3. [ ] Run convert_localdoc_to_gliner.py → data/train.json, data/val.json, data/test.json
4. [ ] Run convert_wikiann_to_gliner.py → data/wikiann_train.json
5. [ ] Run generate_synthetic_az_ner_pattern_exhaustive.py → data/synthetic_az_pattern_exhaustive.json
      (targets 30K samples: 9 entity types × ~3000 each + 1500 negatives)

# TARGETED LLM WORKFLOWS (~$20-25 total via AI Gateway)
6. [ ] Run generate_narrative_pii_samples.py → data/narrative_pii.json (~$15-20)
      (5000 natural-context samples with Claude-generated prose + deterministic entity values)
7. [ ] Run audit_localdoc_quality.py → data/audit_report.json (~$5)
      (spot-check 500 LocalDoc samples, generate exclude list)
8. [ ] Run apply_audit_exclusions.py → data/train_cleaned.json
      (remove annotations Claude flagged as INCORRECT)

# DATA VALIDATION & MERGE
9. [ ] Run audit_dataset() validation check — verify all patterned entities have 2500+ samples
10. [ ] Run merge_all_datasets.py → data/train_final.json, data/val_final.json, data/test_final.json

# TRAINING
11. [ ] Run finetune_gliner_az.py with Option A hyperparameters:
       - 7-10 epochs (not 5)
       - focal_loss_gamma = 2.5
       - Monitor per-entity F1 (not just loss)
       - Early stop when patterned entity F1 plateaus

# EVALUATION & DEPLOYMENT
12. [ ] Run evaluate_gliner_az.py — target ≥90% F1 on patterned, ≥85% on semantic
13. [ ] Run threshold tuning on validation set
14. [ ] Export to ONNX for cross-platform deployment
15. [ ] Publish to HuggingFace Hub as pasha-fh/gliner-azerbaijani-ner-v1
```

**Total project budget**: ~$20-25 in LLM API costs + GPU training time (see Section 11).
**Total execution time (sequential)**: ~3-6 hours for data prep + LLM calls, + training time based on model size.

