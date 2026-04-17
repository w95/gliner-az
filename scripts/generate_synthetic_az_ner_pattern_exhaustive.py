#!/usr/bin/env python3
"""Pattern-exhaustive synthetic data generator for Option A Azerbaijani GLiNER.

Targets per guide §7:
- 3,000 positive samples per patterned entity (8 types) = 24,000
- ~1,500 adversarial negatives (TIN-like, FIN-like, card-like)
- 15+ format variations, 30+ context templates per entity
- Noise injection on 5% of examples

Self-contained: inline generators for all Azerbaijan-specific PII formats.
No external `az-data-generator` dependency.

Output: data/synthetic_az_pattern_exhaustive.json
        + push to HF Hub as {HF_USER}/azerbaijani-ner-synthetic
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import string

from hf_utils import push_gliner_dataset

SEED = 42
random.seed(SEED)


# ============================================================
# Inline entity generators — Azerbaijani real-world formats
# ============================================================

AZ_FIRST_NAMES = [
    "Əli", "Elçin", "Rəşad", "Vüqar", "Orxan", "Turan", "Kamran", "Səbuhi",
    "Elvin", "Fəxri", "Ramin", "Nicat", "Murad", "Nihad", "Ceyhun", "Anar",
    "Leyla", "Aysel", "Günel", "Nigar", "Səbinə", "Aygün", "Xədicə", "Mələk",
    "Lalə", "Ülviyyə", "Nərgiz", "Pərvanə", "Zəhra", "Gülşən",
]
AZ_LAST_NAMES = [
    "Əliyev", "Məmmədov", "Hüseynov", "Həsənov", "Quliyev", "Rəhimov", "İsmayılov",
    "Abbasov", "Kərimov", "Mustafayev", "Nəsibov", "Süleymanov", "Hacıyev", "Rzayev",
    "Əhmədov", "Məhərrəmov", "Qurbanov", "Cəfərov", "Vəliyev", "Zeynalov",
]
AZ_CITIES = [
    "Bakı", "Gəncə", "Sumqayıt", "Mingəçevir", "Lənkəran", "Şəki", "Naxçıvan",
    "Quba", "Şirvan", "Zaqatala", "Qəbələ", "Şamaxı", "Qazax", "Ağdam", "Şuşa",
    "Füzuli", "Xankəndi", "Xaçmaz", "Astara", "Lerik", "Masallı", "Cəlilabad",
]
AZ_BANK_CODES = ["NABZ", "IBAZ", "JBAZ", "RZBZ", "XALQ", "KAPZ", "PAKB", "BRSB"]
AZ_MOBILE_PREFIXES = ["50", "51", "55", "70", "77", "99"]
AZ_REGION_CODES = ["10", "20", "40", "50", "60", "70", "77", "80", "90", "99"]
EMAIL_DOMAINS = ["mail.az", "gmail.com", "yahoo.com", "outlook.com", "bakubank.az",
                 "azercell.com", "bakcell.com", "box.az", "inbox.az", "list.az"]
EMAIL_LOCAL_CHARS = string.ascii_lowercase + string.digits + "._-"


def gen_fin() -> str:
    """7-char alphanumeric, uppercase."""
    alphabet = string.ascii_uppercase + string.digits
    return "".join(random.choices(alphabet, k=7))


def gen_tin() -> str:
    """10-digit, ending in 1 (legal) or 2 (individual) per guide §3.2."""
    body = "".join(random.choices(string.digits, k=9))
    return body + random.choice(["1", "2"])


def gen_phone() -> str:
    """+994 + mobile prefix + 7 digits, canonical form."""
    prefix = random.choice(AZ_MOBILE_PREFIXES)
    digits = "".join(random.choices(string.digits, k=7))
    return f"+994{prefix}{digits}"


def _iban_checksum(bban: str) -> str:
    """Compute IBAN check digits for country AZ + bban."""
    rearranged = bban + "AZ00"
    numeric = ""
    for c in rearranged:
        if c.isdigit():
            numeric += c
        else:
            numeric += str(ord(c.upper()) - 55)  # A=10, B=11, ...
    check = 98 - (int(numeric) % 97)
    return f"{check:02d}"


def gen_iban() -> str:
    """AZ + 2 check + 4-char bank + 20 digits = 28 chars, with valid checksum."""
    bank = random.choice(AZ_BANK_CODES)
    account = "".join(random.choices(string.digits, k=20))
    bban = bank + account
    check = _iban_checksum(bban)
    return f"AZ{check}{bban}"


def gen_passport() -> str:
    prefix = random.choice(["AZE", "AA"])
    digits = "".join(random.choices(string.digits, k=8))
    return prefix + digits


def gen_plate() -> str:
    """XX-LL-NNN: 2 region digits, 2 uppercase letters, 3 digits."""
    region = random.choice(AZ_REGION_CODES)
    letters = "".join(random.choices(string.ascii_uppercase, k=2))
    digits = "".join(random.choices(string.digits, k=3))
    return f"{region}-{letters}-{digits}"


def _luhn_check_digit(digits_15: str) -> str:
    total = 0
    for i, c in enumerate(reversed(digits_15)):
        d = int(c)
        if i % 2 == 0:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return str((10 - total % 10) % 10)


def gen_card() -> str:
    """16-digit Luhn-valid card (Visa-like prefix)."""
    prefix = "4" + "".join(random.choices(string.digits, k=14))
    return prefix + _luhn_check_digit(prefix)


def gen_email() -> str:
    local_len = random.randint(4, 12)
    local = "".join(random.choices(EMAIL_LOCAL_CHARS, k=local_len)).strip("._-") or "user"
    return f"{local}@{random.choice(EMAIL_DOMAINS)}"


def gen_name() -> str:
    return f"{random.choice(AZ_FIRST_NAMES)} {random.choice(AZ_LAST_NAMES)}"


def gen_city() -> str:
    return random.choice(AZ_CITIES)


# ============================================================
# Format variation functions (guide §7.6)
# ============================================================

def vary_fin(canonical: str) -> str:
    v = [
        canonical,
        f"{canonical[0]} {canonical[1:4]} {canonical[4:]}",
        f"{canonical[0]}-{canonical[1:4]}-{canonical[4:]}",
        canonical.lower(),
        f"({canonical})",
        f"[{canonical}]",
        canonical + "-dir",
        canonical + "-nin",
    ]
    return random.choice(v)


def vary_tin(canonical: str) -> str:
    v = [
        canonical,
        f"{canonical[:2]} {canonical[2:4]} {canonical[4:6]} {canonical[6:8]} {canonical[8:]}",
        f"{canonical[:3]} {canonical[3:6]} {canonical[6:9]} {canonical[9:]}",
        f"{canonical[:4]}-{canonical[4:8]}-{canonical[8:]}",
        f"№{canonical}",
        f"N{canonical}",
        canonical + "-dir",
    ]
    return random.choice(v)


def vary_phone(canonical: str) -> str:
    # canonical = "+994" + 9 digits
    if not (canonical.startswith("+994") and len(canonical) == 13):
        return canonical
    digits = canonical[4:]
    op, rest = digits[:2], digits[2:]
    v = [
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
    return random.choice(v)


def vary_iban(canonical: str) -> str:
    if len(canonical) != 28:
        return canonical
    groups = [canonical[i:i + 4] for i in range(0, 28, 4)]
    v = [
        canonical,
        " ".join(groups),
        "-".join(groups),
        canonical.lower(),
    ]
    return random.choice(v)


def vary_passport(canonical: str) -> str:
    if canonical.startswith(("AZE", "AA")):
        prefix = "AZE" if canonical.startswith("AZE") else "AA"
        digits = canonical[len(prefix):]
        v = [
            canonical,
            f"{prefix} {digits}",
            f"{prefix}-{digits}",
            f"{prefix} {digits[:4]} {digits[4:]}",
            f"№{canonical}",
        ]
        return random.choice(v)
    return canonical


def vary_plate(canonical: str) -> str:
    compact = canonical.replace(" ", "").replace("-", "")
    if len(compact) != 7:
        return canonical
    v = [
        f"{compact[:2]}-{compact[2:4]}-{compact[4:]}",
        f"{compact[:2]} {compact[2:4]} {compact[4:]}",
        compact,
        canonical.lower(),
    ]
    return random.choice(v)


def vary_card(canonical: str) -> str:
    digits = "".join(c for c in canonical if c.isdigit())
    if len(digits) != 16:
        return canonical
    v = [
        digits,
        f"{digits[:4]} {digits[4:8]} {digits[8:12]} {digits[12:]}",
        f"{digits[:4]}-{digits[4:8]}-{digits[8:12]}-{digits[12:]}",
        f"{digits[:4]}.{digits[4:8]}.{digits[8:12]}.{digits[12:]}",
    ]
    return random.choice(v)


def vary_email(canonical: str) -> str:
    v = [
        canonical,
        canonical.upper(),
        canonical.capitalize(),
        f"<{canonical}>",
        f"mailto:{canonical}",
    ]
    return random.choice(v)


# ============================================================
# Context templates — 30+ per entity
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
    "1 . {name} - FİN {fin}",
    "İddiaçının FİN kodu : {fin}",
    "Şahidin şəxsiyyəti : FİN {fin}",
    "Şagirdin şəxsiyyət nömrəsi : FİN {fin}",
    "Onun FİN -i {fin} .",
    "Müraciətçi {name} FİN {fin} .",
    "Sizin FİN kodunuz {fin} kimi görünür .",
    "Ərizədə FİN {fin} qeyd olunub .",
    "Notariat aktında FİN : {fin}",
    "Vəkalətnamədə FİN {fin} göstərilib .",
    "Polis protokolunda FİN : {fin}",
    "Məhkəmə qərarında FİN {fin} qeyd olunub .",
]

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
    "Qaimə - fakturada VÖEN {tin} .",
    "Əlavə dəyər vergisi ödəyicisinin VÖEN : {tin}",
    "Qeydiyyat : VÖEN {tin}",
    "Mühasibat hesabatında VÖEN {tin} .",
    "ASAN xidmətdə VÖEN {tin} ilə qeydiyyat .",
    "Dövlət Vergi Xidmətində VÖEN {tin} qeydə alınıb .",
    "İmtiyazlı vergi rejimində VÖEN {tin} .",
    "Kiçik sahibkarlıq subyekti VÖEN {tin} .",
    "Sadələşdirilmiş vergi VÖEN : {tin}",
    "İxracatçının VÖEN -i {tin} .",
    "İdxalatçı VÖEN {tin} bəyannamə verib .",
    "VÖEN {tin} yoxlanışdan keçib .",
    "Gömrük bəyannaməsində VÖEN {tin} .",
    "Tender iştirakçısı VÖEN {tin} .",
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
    "Çağrı mərkəzi : {phone}",
    "{phone} nömrəsinə SMS göndərin .",
    "Kuryer nömrəsi : {phone}",
    "Faks : {phone}",
    "Sabahkı görüş üçün {phone} .",
    "Telefonla sifariş : {phone}",
    "{phone} — əlaqə saxlamaq üçün .",
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
    "IBAN {iban} doğrulanıb .",
    "Bank köçürməsi — IBAN : {iban}",
    "{name} hesabı : {iban}",
    "Kontragent IBAN : {iban}",
    "Maliyyə departamentinin hesabı : {iban}",
    "Əmək haqqı hesabı : {iban}",
    "Dövlət xəzinədarlığı hesabı : {iban}",
    "İcarə ödənişi üçün IBAN : {iban}",
    "Abunəlik hesabı : {iban}",
    "Təqaüd hesabı IBAN : {iban}",
    "Vergi ödənişi IBAN : {iban}",
    "Kredit hesabı : {iban}",
    "Xeyriyyə ödənişi IBAN : {iban}",
    "Komisyon haqqı hesabı : {iban}",
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
    "Otel qeydiyyatında pasport : {passport}",
    "Aviabilet üçün pasport : {passport}",
    "Konsulluqda pasport {passport} qeydə alınıb .",
    "Pasport nömrəsini yoxlayın : {passport}",
    "{name} pasport : {passport} .",
    "Şəngən vizası üçün pasport : {passport}",
    "Yeni pasport : {passport}",
    "Keçmiş pasport nömrəsi : {passport}",
    "Uşağın pasport nömrəsi : {passport}",
    "Pasport etibarsızdır : {passport}",
    "Pasport {passport} saxta sayılıb .",
    "Polis arayışında pasport : {passport}",
    "DMX -də pasport {passport} qeydə alınıb .",
    "Bank hesabı üçün pasport : {passport}",
    "Bələdçi pasport {passport} təqdim edib .",
    "Qeydiyyat üçün pasport : {passport}",
    "Sərnişin pasport {passport} ilə yolçuluq edir .",
    "Elektron viza üçün pasport : {passport}",
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
    "Oğurlanmış avtomobilin nömrəsi {plate} .",
    "Parkinqdə {plate} saylı maşın .",
    "Tıxac videosunda {plate} .",
    "Sığorta polisində {plate} .",
    "Kirayə avtomobili {plate} .",
    "Taksi {plate} müştəri qəbul etdi .",
    "Yük maşını {plate} gömrükdə .",
    "Avtobus {plate} marşrutda .",
    "Elektrik avtomobili {plate} .",
    "Diplomatik nömrə {plate} .",
    "{plate} plakalı avtomobilin sahibi {name} .",
    "Qeydiyyata alınan {plate} .",
    "Sürücü {name} , avtomobil {plate} .",
    "Sərhəd keçidində {plate} .",
    "Xüsusi qeydiyyat : {plate}",
    "Yarış avtomobili {plate} .",
    "Motosiklet {plate} .",
    "{plate} nömrəli yük .",
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
    "Məzuniyyət kartı : {card}",
    "İctimai nəqliyyat kartı : {card}",
    "Kart bloklandı : {card}",
    "{card} — əlavə təsdiq tələb olunur .",
    "Mağazada {card} ilə ödəniş .",
    "Bankomatdan {card} ilə çıxarış .",
    "Onlayn alış üçün {card} .",
    "Rekvizitlər : {card} .",
    "Maaş kartı : {card}",
    "Virtual kart : {card}",
    "Şirkət kartı : {card}",
    "Təhlükəsizlik mesajı : {card} .",
    "Tranzaksiya reyestrində {card} .",
    "{name} {card} ilə sifariş verib .",
    "İnternet alış {card} ilə reallaşdı .",
    "ATM -də {card} oxunmadı .",
    "Avtomatik ödəniş kartı : {card}",
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
    "Müştəri dəstəyi : {email}",
    "Satış komandası : {email}",
    "HR ünvanı : {email}",
    "Müavin direktor : {email}",
    "Abunə olmaq üçün : {email}",
    "Şikayət üçün : {email}",
    "Texniki dəstək : {email}",
    "Marketinq : {email}",
    "Media əlaqələri : {email}",
    "Tenderlər üçün : {email}",
    "Karyera : {email}",
    "Tədbir qeydiyyatı : {email}",
    "Faktura : {email}",
    "Vergi bəyannaməsi : {email}",
    "{name} {email} ünvanından yazıb .",
    "CV -ni {email} -ə göndərin .",
    "Məktubu {email} -ə göndərdim .",
    "Cavab {email} ünvanından gələcək .",
    "Yalnız {email} vasitəsilə .",
]


# ============================================================
# Token span finding + sample construction
# ============================================================

_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Unicode-aware whitespace+punctuation tokenizer (same as narrative script)."""
    return _TOKEN_RE.findall(text)


def find_span(tokens: list[str], target: str) -> tuple[int | None, int | None]:
    target_tokens = tokenize(target)
    if not target_tokens:
        return None, None
    n, m = len(tokens), len(target_tokens)
    for i in range(n - m + 1):
        if tokens[i:i + m] == target_tokens:
            return i, i + m - 1
    return None, None


def build_positive(template: str, entity_value: str, entity_label: str) -> dict | None:
    name = gen_name()
    text = template.format(
        name=name, fin=entity_value, phone=entity_value, email=entity_value,
        iban=entity_value, passport=entity_value, tin=entity_value,
        card=entity_value, plate=entity_value,
    )
    tokens = tokenize(text)
    entities: list[list] = []

    start, end = find_span(tokens, entity_value)
    if start is None:
        return None
    entities.append([start, end, entity_label])

    if "{name}" in template:
        n_start, n_end = find_span(tokens, name)
        if n_start is not None:
            # avoid overlap
            if not any(not (n_end < s or n_start > e) for s, e, _ in entities):
                entities.append([n_start, n_end, "person"])

    return {"tokenized_text": tokens, "ner": entities}


# ============================================================
# Adversarial negatives — 10/7/16 digits in non-entity contexts
# ============================================================

NEGATIVE_TEMPLATES = {
    "tin_negative": [
        "Zəng edin : 0{blob} ( məlumat üçün ) .",
        "Sifariş № {blob} qəbul edildi .",
        "Faktura nömrəsi {blob} .",
        "Kodunuz : {blob} ( 10 dəqiqə etibarlıdır ) .",
        "İzləmə kodu : {blob} .",
        "Kampaniya ID : {blob} .",
        "Tracking: {blob} .",
        "Mağaza məhsul kodu {blob} .",
    ],
    "fin_negative": [
        "Məhsul kodu : {blob} .",
        "Reyestr № {blob} -dir .",
        "Versiya {blob} -dir .",
        "Modul kodu : {blob} .",
        "Sxem kodu {blob} .",
        "API açarı fraqmenti : {blob} .",
    ],
    "card_negative": [
        "İzləmə kodu : {blob} .",
        "Sifarişin ID -si {blob} .",
        "Tracking: {blob} .",
        "Məhsul seriya nömrəsi {blob} .",
        "Konteyner nömrəsi {blob} .",
    ],
}


def build_negative(template: str, blob: str) -> dict:
    name = gen_name() if "{name}" in template else None
    text = template.format(blob=blob, name=name or "")
    tokens = tokenize(text)
    entities: list[list] = []
    if name:
        s, e = find_span(tokens, name)
        if s is not None:
            entities.append([s, e, "person"])
    return {"tokenized_text": tokens, "ner": entities}


# ============================================================
# Noise injection — OCR-style substitutions on non-entity tokens
# ============================================================

_OCR_CONFUSIONS = {"0": "O", "1": "l", "5": "S", "8": "B"}


def inject_noise(sample: dict, rate: float = 0.02) -> dict:
    entity_ids: set[int] = set()
    for s, e, _ in sample["ner"]:
        entity_ids.update(range(s, e + 1))
    noisy = []
    for i, tok in enumerate(sample["tokenized_text"]):
        if i in entity_ids:
            noisy.append(tok)
            continue
        noisy.append("".join(
            _OCR_CONFUSIONS.get(c, c) if random.random() < rate else c
            for c in tok
        ))
    return {"tokenized_text": noisy, "ner": sample["ner"]}


# ============================================================
# Main generation loop
# ============================================================

ENTITY_CONFIGS = [
    # (entity_name, generator, variation_fn, templates, gliner_label, n_samples)
    ("fin",      gen_fin,      vary_fin,      FIN_TEMPLATES,      "fin code",           3000),
    ("tin",      gen_tin,      vary_tin,      TIN_TEMPLATES,      "tin",                3000),
    ("phone",    gen_phone,    vary_phone,    PHONE_TEMPLATES,    "phone number",       3000),
    ("iban",     gen_iban,     vary_iban,     IBAN_TEMPLATES,     "iban",               3000),
    ("passport", gen_passport, vary_passport, PASSPORT_TEMPLATES, "passport number",    3000),
    ("plate",    gen_plate,    vary_plate,    PLATE_TEMPLATES,    "vehicle plate",      3000),
    ("card",     gen_card,     vary_card,     CARD_TEMPLATES,     "credit card number", 3000),
    ("email",    gen_email,    vary_email,    EMAIL_TEMPLATES,    "email",              3000),
]


def generate_positives(gen, vary, templates, label, n, noise_rate=0.05) -> list[dict]:
    out: list[dict] = []
    attempts = 0
    while len(out) < n and attempts < n * 6:
        attempts += 1
        canonical = gen()
        value = vary(canonical)
        template = random.choice(templates)
        sample = build_positive(template, value, label)
        if sample is None:
            continue
        if random.random() < noise_rate:
            sample = inject_noise(sample)
        out.append(sample)
    return out


def generate_negatives(n_per_kind: int = 500) -> list[dict]:
    out: list[dict] = []
    for _ in range(n_per_kind):
        blob = "".join(random.choices(string.digits, k=10))
        out.append(build_negative(random.choice(NEGATIVE_TEMPLATES["tin_negative"]), blob))
    for _ in range(n_per_kind):
        blob = "".join(random.choices(string.ascii_uppercase + string.digits, k=7))
        out.append(build_negative(random.choice(NEGATIVE_TEMPLATES["fin_negative"]), blob))
    for _ in range(n_per_kind):
        blob = "".join(random.choices(string.digits, k=16))
        out.append(build_negative(random.choice(NEGATIVE_TEMPLATES["card_negative"]), blob))
    return out


def audit_dataset(samples: list[dict]) -> dict:
    from collections import Counter
    counts: Counter = Counter()
    diversity: dict[str, set[str]] = {}
    for s in samples:
        for start, end, label in s["ner"]:
            counts[label] += 1
            diversity.setdefault(label, set()).add(
                " ".join(s["tokenized_text"][start:end + 1])[:40]
            )
    print(f"\n{'Entity':25s} {'Count':>8s} {'Unique':>10s}")
    print("-" * 47)
    for label in sorted(counts):
        print(f"{label:25s} {counts[label]:>8d} {len(diversity[label]):>10d}")
    issues = []
    for label in ["fin code", "tin", "phone number", "iban"]:
        if counts.get(label, 0) < 2500:
            issues.append(f"WARN {label}: {counts.get(label, 0)} (need >=2500)")
    if issues:
        print("\nISSUES:")
        for i in issues:
            print(" -", i)
    else:
        print("\nAll patterned entities meet 2500+ threshold.")
    return {"counts": dict(counts), "diversity": {k: len(v) for k, v in diversity.items()}}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="data")
    p.add_argument("--repo-name", default="azerbaijani-ner-synthetic")
    p.add_argument("--no-push", action="store_true")
    p.add_argument("--noise-rate", type=float, default=0.05)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    all_samples: list[dict] = []

    for name, gen, vary, templates, label, n in ENTITY_CONFIGS:
        print(f"Generating {n} {name} ({label}) samples...")
        batch = generate_positives(gen, vary, templates, label, n, noise_rate=args.noise_rate)
        all_samples.extend(batch)
        print(f"  got {len(batch)}")

    print("Generating adversarial negatives (500 per kind)...")
    all_samples.extend(generate_negatives(500))

    random.shuffle(all_samples)

    path = os.path.join(args.output_dir, "synthetic_az_pattern_exhaustive.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(all_samples, f, ensure_ascii=False)
    print(f"\nTotal: {len(all_samples)} samples → {path}")

    audit_dataset(all_samples)

    if not args.no_push:
        push_gliner_dataset(all_samples, args.repo_name, private=True)


if __name__ == "__main__":
    main()
