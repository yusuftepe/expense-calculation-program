"""
Fiş Metni Ayrıştırma Modülü (receipt_parser.py).

OCR'dan gelen satırları Türkiye'deki ÖKC (yazar kasa) fiş düzenine göre
yorumlayarak işletme adı, tarih, genel toplam ve ürün kalemlerini çıkarır.

Tipik fiş düzeni:
    ŞOK MARKETLER TİC. A.Ş.
    ...adres / vergi dairesi...
    TARİH : 07.11.2026     SAAT : 14:32
    FİŞ NO : 0045
    2 AD X 22,50                 <- miktar satırı (bir sonraki ürüne aittir)
    SÜT 1L            %1  *45,00
    EKMEK             %1  *15,00
    TOPKDV                 *0,59
    TOPLAM                *60,00
    NAKİT                 *100,00
"""

from __future__ import annotations

import difflib
import re
from datetime import date, datetime
from functools import lru_cache
from typing import Any, Optional

from backend.schemas import OCRProcessingError
from ocr_ml.text_utils import AMOUNT_PATTERN, find_trailing_amount, fix_digit_token, fold, parse_amount

UNKNOWN_MERCHANT = "Bilinmeyen İşletme"

# Ürün satırı OLMAYAN satırları tanıyan anahtar kelimeler (diakritiksiz, küçük harf)
NON_ITEM_KEYWORDS: tuple[str, ...] = (
    "toplam", "topkdv", "topkov", "kdv", "kov", "ara toplam", "nakit", "kredi", "kart", "banka", "para ustu",
    "ustu", "odenen", "fis no", "fis", "z no", "eku", "mersis", "vergi", "v.d", "vd:",
    "vkn", "tckn", "tel", "saat", "tarih", "kasiyer", "kasa", "pos", "onay", "provizyon",
    "iade", "tesekkur", "www", "adres", "mf", "bilgi fis", "ettn", "no:", "sicil",
    # Ara toplam kısaltmaları ve POS slipi alanları
    "ara top", "top", "top.", "ver", "sn", "batch", "stan", "rrn", "aid", "terminal", "isyeri",
    "referans", "parafpara", "hesap", "puan", "bonus", "chip", "worldpuan", "maxipuan",
)
SUBTOTAL_KEYWORDS: tuple[str, ...] = ("ara toplam", "ara top", "ara top.", "top")
DISCOUNT_KEYWORDS: tuple[str, ...] = ("indirim", "iskonto", "kampanya", "promosyon")
TOTAL_KEYWORDS: tuple[str, ...] = ("genel toplam", "toplam", "top.", "tutar", "tutari", "islem tutari")
MERCHANT_SKIP_KEYWORDS: tuple[str, ...] = (
    "hos geldiniz", "hosgeldiniz", "hos geldin", "welcome", "iyi gunler", "tesekkur",
    "tarih", "saat", "fis", "tel", "vergi", "v.d", "vkn", "mersis", "no:", "www", "mah", "cad",
    # "sok." kullanılmaz: "ŞOK." market adıyla çakışır
    "sokak", "sokagi", "sk.", "bulvar", "blv",
)

# İşletme adını tanıtan ifadeler: ilk satırlarda bunlardan biri varsa o satır tercih edilir
MERCHANT_HINTS: tuple[str, ...] = (
    "a.s", "a.s.", "ltd", "sti", "tic", "san", "anonim", "sirketi", "market", "marketler", "magaza",
    "magazacilik", "alisveris", "akaryakit", "petrol", "eczane", "eczanesi", "restoran", "kafe", "cafe",
    "gida", "unlu mamul", "firin", "kasap", "manav",
)
MIN_MERCHANT_CONFIDENCE = 0.5

NON_RECEIPT_DATE_KEYWORDS: tuple[str, ...] = (
    "kesim", "son odeme", "odeme tarihi", "vade", "gecerlilik", "son kullanma", "skt", "hesap",
)

DATE_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\b(\d{1,2})[./-](\d{1,2})[./-](\d{4})\b", "dmy"),
    # OCR ayraç hataları: "O1 12.2026", "26.08,2025"
    (r"\b(\d{1,2})[./\-, ](\d{1,2})[./\-,](\d{4})\b", "dmy"),
    (r"\b(\d{4})[./-](\d{1,2})[./-](\d{1,2})\b", "ymd"),
    (r"\b(\d{1,2})[./-](\d{1,2})[./-](\d{2})\b", "dmy2"),
)

# "%1", "%01", "%08", "%18", "%20", "% 10" gibi KDV oranı ifadeleri
VAT_PATTERN = re.compile(r"%\s?\d{1,2}\b")
# OCR'ın '%' işaretini », #, X, Y, S, 9, 8 veya 4; '1' rakamını I/l/| olarak okuduğu KDV kalıntıları
# ("»1", "X1", "41", "420", "410", "91", "YI", "S1", "820"). Yalnızca ürün adının sonunda ve Türkiye'deki
# KDV oranlarıyla (1, 8, 10, 18, 20) eşleşir.
VAT_REMNANT_PATTERN = re.compile(r"\s+(?:[%»«#xXyYsS]{1,2}|[489])?\s?(?:0?[1Il|]|0?8|[1Il|][08]|20)$")

# Ad sonundaki gramajda G harfinin 6 okunması: "MAKARNA 5006" -> "500G". Yalnızca yaygın
# gramajlarla eşleşir; "5006" gibi gerçek bir sayı nadiren ürün adının sonunda bulunur.
COMMON_GRAMS_MISREAD = re.compile(r"\b(100|125|150|200|250|300|330|400|450|500|600|700|750|800|900)6$")

# Miktarı bozuk okunmuş miktar satırı: birim + X + tutar ile biter ("poşsa KG X 579,00")
LOOSE_QUANTITY_LINE = re.compile(
    # '*' ayraç sayılmaz: Türk fişlerinde fiyat önekidir ("VİDA 5006 ADET *3,00" bir ürün satırıdır)
    rf"\b(?:KG|AD|ADET|ADT|LT|GR)\.?\s*[xX×]\s*{AMOUNT_PATTERN}\s*$", flags=re.IGNORECASE
)

# Barkod / ürün kodu satırı ("8680596220826", "(1)2706182") ve ürün adı başındaki kod
BARCODE_LINE = re.compile(r"^\W*(?:\(\d\))?\d{4,14}\W*$")
LEADING_CODE = re.compile(r"^\W*(?:\(\d\))?\d{4,14}\b\s*")

# Miktar satırı: "2 AD X 22,50", "0,750 KG X 60,00", "3 X 15,00"
QUANTITY_LINE = re.compile(
    rf"^\s*(\d+(?:[.,]\d{{1,3}})?)\s*(?:AD(?:ET)?|KG|GR|LT|L|ADT)?\.?\s*[xX*×]\s*({AMOUNT_PATTERN})\s*(?:TL|₺)?\s*$",
    flags=re.IGNORECASE,
)


# OCR'ın bozduğu anahtar kelimeleri ("TOPLAN", "TOPKODOV", "NAKLT") tanımak için kanonik liste
CANONICAL_KEYWORDS: tuple[str, ...] = (
    "toplam", "topkdv", "nakit", "kredi", "karti", "indirim", "tarih", "iskonto", "ustu",
)


@lru_cache(maxsize=4096)
def keyword_fold(text: str) -> str:
    """
    fold() + OCR hatalarına dayanıklı anahtar kelime düzeltmesi.
    5+ harfli kelimeler kanonik anahtar kelimelere benzerlik >= 0.8 ise o kelimeye çevrilir
    (4 harfte yanlış pozitif çok: "KEDİ" -> "kredi").
    """
    # Harflerin arasındaki 0, O harfidir: "T0PKDV" -> "topkdv"
    tokens = re.split(r"(\W+)", re.sub(r"(?<=[a-z])0|0(?=[a-z])", "o", fold(text)))
    for idx, token in enumerate(tokens):
        if len(token) >= 5 and token.isalpha() and token not in CANONICAL_KEYWORDS:
            close = difflib.get_close_matches(token, CANONICAL_KEYWORDS, n=1, cutoff=0.8)
            if close:
                tokens[idx] = close[0]
    return "".join(tokens)


def _contains_keyword(folded_line: str, keywords: tuple[str, ...]) -> bool:
    return any(re.search(rf"(?<![a-z]){re.escape(kw)}(?![a-z])", folded_line) for kw in keywords)


def _count_letters(text: str) -> int:
    return sum(1 for ch in text if ch.isalpha())


def _normalize_line(text: str) -> str:
    """OCR satırındaki gereksiz boşlukları ve kenar sembollerini temizler."""
    text = re.sub(r"\s+", " ", text).strip()
    return text.strip("|_~`'\"")


def extract_date(lines: list[str]) -> Optional[str]:
    """Satırlardaki ilk geçerli tarihi ISO formatında ('YYYY-MM-DD') döndürür."""
    # Öncelik: "TARİH" geçen satırlar > diğerleri; "SON ÖDEME", "HESAP KESİM" gibi fiş tarihi
    # olmayan tarihler en sona. Eşitlikte fişin üstündeki satır önce gelir.
    def rank(item: tuple[int, str]) -> tuple[int, int]:
        idx, line = item
        folded = keyword_fold(line)
        if _contains_keyword(folded, NON_RECEIPT_DATE_KEYWORDS):
            return 2, idx
        return (0 if "tarih" in folded else 1), idx

    ordered = [line for _, line in sorted(enumerate(lines), key=rank)]
    for line in ordered:
        # "O1.12.2026" gibi OCR hatalarını tarih biçimli parçalarda düzelt
        line = re.sub(
            r"(?<![\w])[\dOoIl|]{1,4}[./\-, ][\dOoIl|]{1,2}[./\-,][\dOoIl|]{2,4}(?![\w])",
            lambda m: fix_digit_token(m.group()), line,
        )
        for pattern, kind in DATE_PATTERNS:
            for match in re.finditer(pattern, line):
                a, b, c = (int(part) for part in match.groups())
                if kind == "dmy":
                    day, month, year = a, b, c
                elif kind == "ymd":
                    year, month, day = a, b, c
                else:
                    day, month, year = a, b, 2000 + c
                parsed = _valid_date(year, month, day) or _repair_date(year, month, day)
                if parsed and 2000 <= parsed.year <= 2100:
                    return parsed.isoformat()
    return None


def _valid_date(year: int, month: int, day: int) -> Optional[date]:
    try:
        return date(year, month, day)
    except ValueError:
        return None


# OCR'ın tarihte sık karıştırdığı rakamlar: 0 -> 6/8/9 ("66.08.2025" -> 06.08.2025)
_DATE_DIGIT_FIXES = {"6": "0", "8": "0", "9": "0", "7": "1"}


def _repair_date(year: int, month: int, day: int) -> Optional[date]:
    """Geçersiz gün/ayda tek rakam düzeltmesi dener; yalnız TEK geçerli sonuç varsa döndürür."""
    day_s, month_s = f"{day:02d}", f"{month:02d}"
    options: set[date] = set()
    for field in ("day", "month"):
        text = day_s if field == "day" else month_s
        for pos, char in enumerate(text):
            if char not in _DATE_DIGIT_FIXES:
                continue
            fixed = int(text[:pos] + _DATE_DIGIT_FIXES[char] + text[pos + 1:])
            candidate = _valid_date(year, month, fixed) if field == "day" else _valid_date(year, fixed, day)
            if candidate:
                options.add(candidate)
    return options.pop() if len(options) == 1 else None


def _merchant_candidate(line: str) -> Optional[str]:
    folded = keyword_fold(line)
    if _count_letters(line) < 3 or _contains_keyword(folded, MERCHANT_SKIP_KEYWORDS):
        return None
    if find_trailing_amount(line)[0] is not None:
        return None
    cleaned = re.sub(r"[^\wÇĞİÖŞÜçğıöşü&.\-' ]", "", line).strip(" -")
    # Harf oranı düşük satırlar (arka plan gürültüsü, barkod altı) işletme adı olamaz
    if _count_letters(cleaned) < 3 or _count_letters(cleaned) < 0.6 * len(cleaned.replace(" ", "")):
        return None
    return cleaned[:120]


def extract_merchant(lines: list[str], confidences: Optional[list[float]] = None) -> str:
    """
    Fişin üst kısmından işletme adını seçer.

    Selamlama ("HOŞ GELDİNİZ"), adres ve düşük güvenli (gürültü) satırlar atlanır.
    İlk satırlarda şirket ibaresi (A.Ş., LTD, MARKET, AKARYAKIT...) içeren satır varsa o tercih edilir.
    """
    head = lines[:7]
    confs = (confidences or [1.0] * len(lines))[:7]
    candidates = [
        (idx, cand)
        for idx, (line, conf) in enumerate(zip(head, confs))
        if conf >= MIN_MERCHANT_CONFIDENCE and (cand := _merchant_candidate(line))
    ]
    for _, cand in candidates:
        if _contains_keyword(keyword_fold(cand), MERCHANT_HINTS):
            return cand
    return candidates[0][1] if candidates else UNKNOWN_MERCHANT


def extract_total(lines: list[str]) -> Optional[float]:
    """Genel toplam tutarı (bkz. extract_total_with_source)."""
    return extract_total_with_source(lines)[0]


def extract_total_with_source(lines: list[str]) -> tuple[Optional[float], int, int]:
    """
    Returns: (tutar, öncelik, satır_no). Öncelik 1 ise yalnız ara toplam okunmuştur.

    Genel toplamı bulur. Öncelik: 'GENEL TOPLAM' > 'TOPLAM' > 'TUTAR' > 'ARA TOPLAM'.
    KDV satırları (TOPKDV, KDV TOPLAM) dikkate alınmaz. Birden fazla eşleşmede
    fişin en altındaki satır seçilir.
    """
    best: Optional[tuple[int, int, float]] = None  # (öncelik, satır_no, tutar)
    for idx, line in enumerate(lines):
        folded = keyword_fold(line)
        if "kdv" in folded or "kov" in folded:
            continue
        if not _contains_keyword(folded, TOTAL_KEYWORDS):
            continue
        amount, _ = find_trailing_amount(line)
        if amount is None:
            # POS sliplerinde tutar bir alt satırda olur: "İŞLEM TUTARI" / "1.234,50 TL"
            amount = _amount_on_next_lines(lines, idx)
        if amount is None or amount <= 0:
            continue
        if "genel toplam" in folded:
            priority = 4
        elif "ara" in folded or _contains_keyword(folded, ("top",)):
            priority = 1
        elif "toplam" in folded:
            priority = 3
        else:  # "TUTAR" / "İŞLEM TUTARI" (POS slipi bölümü)
            priority = 2
        candidate = (priority, idx, amount)
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    return (best[2], best[0], best[1]) if best else (None, 0, -1)


def _amount_on_next_lines(lines: list[str], idx: int, lookahead: int = 2) -> Optional[float]:
    """Satırdan sonraki 'lookahead' satırda yalnız tutardan oluşan bir satır varsa tutarı döndürür."""
    for line in lines[idx + 1: idx + 1 + lookahead]:
        # Yalnız tutardan oluşan satırda boşluklu binlik de kabul edilir: "1 050,25 TL"
        bare = re.sub(r"(?i)\s*(?:TL|₺|©|®|£|€)\s*$", "", line).strip(" *|#.")
        match = re.fullmatch(r"(\d{1,3}(?:[\s.,]\d{3})*)\s?[.,]\s?(\d{2})", bare)
        if match:
            return float(re.sub(r"\D", "", match.group(1)) + "." + match.group(2))
        amount, rest = find_trailing_amount(line)
        if amount is not None and not rest.strip(" *|#.-"):
            return amount
    return None


PAYMENT_KEYWORDS: tuple[str, ...] = ("kredi", "karti", "kart", "pos", "odeme", "banka")


def _payment_amounts(lines: list[str]) -> list[float]:
    """Ödeme satırlarındaki tutarlar (KREDİ KARTI *200,00, ORTAK POS *120,00). Nakit hariç (para üstü olur)."""
    amounts = []
    for line in lines:
        if _contains_keyword(keyword_fold(line), PAYMENT_KEYWORDS):
            amount, _ = find_trailing_amount(line)
            if amount and amount > 0:
                amounts.append(amount)
    return amounts


def extract_subtotal(lines: list[str]) -> Optional[float]:
    """'ARA TOPLAM' / 'ARA TOP.' satırındaki tutar."""
    for line in lines:
        folded = keyword_fold(line)
        if _contains_keyword(folded, SUBTOTAL_KEYWORDS) and "genel" not in folded and "kdv" not in folded:
            amount, _ = find_trailing_amount(line)
            if amount and amount > 0:
                return amount
    return None


def _find_items_section_end(lines: list[str]) -> int:
    """Ürün listesinin bittiği satırı (ilk toplam/KDV satırı) döndürür."""
    for idx, line in enumerate(lines):
        folded = keyword_fold(line)
        if ("toplam" in folded or "topkdv" in folded or "topkov" in folded
                or _contains_keyword(folded, SUBTOTAL_KEYWORDS)) and find_trailing_amount(line)[0]:
            return idx
    return len(lines)


def extract_items(ocr_lines: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], float]:
    """
    Ürün kalemlerini çıkarır.

    Returns:
        (ürün listesi, toplam indirim tutarı)
    """
    texts = [line["text"] for line in ocr_lines]
    end = _find_items_section_end(texts)

    items: list[dict[str, Any]] = []
    discount_total = 0.0
    pending_qty: Optional[tuple[float, float]] = None
    orphan_name: Optional[str] = None  # fiyatsız ürün adı satırı (buruşuk fişte fiyat alt satıra kayar)

    for line_info in ocr_lines[:end]:
        text = line_info["text"]
        confidence = float(line_info.get("confidence", 1.0))

        qty_match = QUANTITY_LINE.match(text)
        if not qty_match and LOOSE_QUANTITY_LINE.search(text):
            # Miktarı bozuk okunmuş miktar satırı ("poşsa KG X 579,00"): ürün değildir, birim fiyat bilinir
            pending_qty = None
            continue
        if qty_match:
            quantity = float(qty_match.group(1).replace(",", "."))
            unit_price = parse_amount(qty_match.group(2).replace(" ", ""))
            if quantity and unit_price and quantity > 0 and unit_price > 0:
                pending_qty = (quantity, unit_price)
            continue

        amount, name_part = find_trailing_amount(text)
        if amount is None:
            # Fiyatsız, harf ağırlıklı satır: bir sonraki fiyatlı satırın ürün adı olabilir
            letters = _count_letters(text)
            if letters >= 4 and letters >= 0.5 * len(text.replace(" ", "")) and not _contains_keyword(
                keyword_fold(text), NON_ITEM_KEYWORDS + DISCOUNT_KEYWORDS + TOTAL_KEYWORDS
            ):
                orphan_name = text
            elif not BARCODE_LINE.match(text):
                orphan_name = None
            continue

        folded = keyword_fold(name_part)
        if _contains_keyword(folded, DISCOUNT_KEYWORDS) or amount < 0:
            discount_total += abs(amount)
            pending_qty = None
            continue
        if _contains_keyword(folded, NON_ITEM_KEYWORDS + TOTAL_KEYWORDS):
            pending_qty = None
            continue

        # Ürün adının başındaki barkod / ürün kodu: "8680596220826 OTRA" -> "OTRA"
        had_code = bool(LEADING_CODE.match(name_part))
        name_part = LEADING_CODE.sub("", name_part)
        # Fiyat satırı barkodla başlıyor veya adı yoksa, ad bir üst satırda kalmıştır
        if orphan_name and (_count_letters(name_part) < 3 or (had_code and _count_letters(name_part) < _count_letters(orphan_name))):
            name_part = orphan_name + " " + name_part
        orphan_name = None
        name = VAT_PATTERN.sub(" ", name_part)
        name = re.sub(r"[\W_]+$", "", name)  # sondaki tire, uzun çizgi vb.
        name = re.sub(r"\s+[*xX#]$", "", name)  # fiyattan kopmuş '*' kalıntısı
        name = VAT_REMNANT_PATTERN.sub("", name)
        name = re.sub(r"[\W_]+$", "", name)
        name = re.sub(r"(?<=\d)[Oo](?=\d)", "0", name)  # "5O0MG" -> "500MG"
        name = COMMON_GRAMS_MISREAD.sub(lambda m: m.group(1) + "G", name)  # "5006" -> "500G"
        name = re.sub(r"\s+", " ", name).strip(" .:-*")
        if _count_letters(name) < 2 or amount <= 0:
            pending_qty = None
            continue

        quantity, unit_price = 1.0, amount
        line_total: Optional[float] = None  # miktar satırından hesaplanan tutar (sağlama için)
        if pending_qty:
            quantity, unit_price = pending_qty
            line_total = round(quantity * unit_price, 2)
            if abs(line_total - amount) > 0.05:
                unit_price = round(amount / quantity, 2)
            pending_qty = None

        item: dict[str, Any] = {
            "name": name[:120],
            "quantity": round(quantity, 3),
            "unit_price": round(unit_price, 2),
            "total_price": round(amount, 2),
            "confidence": round(max(0.0, min(1.0, confidence)), 4),
        }
        if line_total is not None and abs(line_total - amount) > 0.05:
            item["_line_total"] = line_total
        items.append(item)

    return items, round(discount_total, 2)


def reconcile_items(items: list[dict[str, Any]], total: float, discount: float) -> bool:
    """
    Ürün toplamı genel toplamı tutmuyorsa OCR hatalı kalemi düzeltmeye çalışır:
        1. Miktar satırıyla çelişen kalem (miktar x birim fiyat toplamı tutturuyorsa)
        2. Fazladan okunmuş rakam: '*' işaretinin rakam okunması ("*192,23" -> "2192,23")
           veya araya rakam girmesi ("594,64" -> "5094,64")
    Toplamı tam tutturan TEK bir düzeltme varsa uygular; birden fazla olasılık varsa dokunmaz.

    Returns:
        Düzeltme yapıldıysa True.
    """
    if not items:
        return False
    items_sum = round(sum(item["total_price"] for item in items) - discount, 2)
    excess = round(items_sum - total, 2)
    if abs(excess) <= 0.05:
        return False

    fixes: list[tuple[int, float]] = []
    # 1. Miktar satırıyla çelişen kalem: "37,67 LT X 43,59" = 1.642,04 ama satırda 1.642,64 okunmuş
    for idx, item in enumerate(items):
        line_total = item.get("_line_total")
        if line_total and abs(round(item["total_price"] - line_total, 2) - excess) <= 0.01:
            fixes.append((idx, line_total))

    # 2. Fazladan okunmuş rakam (yalnızca ürün toplamı fazlaysa)
    for idx, item in enumerate(items if excess > 0 and not fixes else []):
        integer_part, decimals = f"{item['total_price']:.2f}".split(".")
        if len(integer_part) < 2:
            continue
        seen: set[float] = set()
        for pos in range(len(integer_part)):
            corrected = float(f"{integer_part[:pos] + integer_part[pos + 1:]}.{decimals}")
            if corrected in seen or corrected <= 0:
                continue
            seen.add(corrected)
            if abs(round(item["total_price"] - corrected, 2) - excess) <= 0.01:
                fixes.append((idx, corrected))
    if len(fixes) != 1:
        return False

    idx, corrected = fixes[0]
    item = items[idx]
    item["total_price"] = corrected
    item["unit_price"] = round(corrected / item["quantity"], 2)
    item["confidence"] = round(min(item["confidence"], 0.5), 4)
    return True


def parse_receipt(ocr_lines: list[dict[str, Any]] | list[str] | str) -> dict[str, Any]:
    """
    OCR satırlarını yapılandırılmış fiş verisine çevirir.

    Args:
        ocr_lines: receipt_ocr.extract_text çıktısı, düz satır listesi veya düz metin.

    Returns:
        {
            "merchant": str, "date": "YYYY-MM-DD", "total": float,
            "items": [{"name", "quantity", "unit_price", "total_price", "confidence"}],
            "discount": float, "total_source": "ocr" | "items", "date_found": bool,
            "subtotal": float | None, "items_synthesized": bool,
            "warnings": [str], "raw_text": str
        }

    Raises:
        OCRProcessingError: Ne toplam ne de ürün okunabildiyse.
    """
    if isinstance(ocr_lines, str):
        ocr_lines = ocr_lines.splitlines()

    normalized: list[dict[str, Any]] = []
    for entry in ocr_lines or []:
        if isinstance(entry, dict):
            text, conf = str(entry.get("text") or ""), entry.get("confidence", 1.0)
        else:
            text, conf = str(entry), 1.0
        text = _normalize_line(text)
        if text:
            normalized.append({"text": text, "confidence": conf})

    if not normalized:
        raise OCRProcessingError("Fiş metni boş; ayrıştırılacak satır bulunamadı.")

    texts = [line["text"] for line in normalized]
    warnings: list[str] = []

    merchant = extract_merchant(texts, [float(line["confidence"]) for line in normalized])
    if merchant == UNKNOWN_MERCHANT:
        warnings.append("İşletme adı okunamadı; lütfen kontrol edin.")

    receipt_date = extract_date(texts)
    date_found = receipt_date is not None
    if receipt_date is None:
        receipt_date = date.today().isoformat()
        warnings.append("Fiş tarihi okunamadı; bugünün tarihi atandı.")

    items, discount = extract_items(normalized)
    total, total_priority, total_line = extract_total_with_source(texts)
    subtotal = extract_subtotal(texts)
    if total is not None and total_priority == 1:
        # Yalnız ara toplam okundu (iri puntolu TOPLAM satırı okunamamış olabilir):
        # ara toplamdan sonraki eksi tutarlar indirimdir; yoksa ödeme satırı ödenen tutardır
        reductions = [
            -amount for line in texts[total_line + 1:]
            if (amount := find_trailing_amount(line)[0]) is not None and amount < 0
        ]
        payments = [amount for amount in _payment_amounts(texts[total_line + 1:]) if amount <= total]
        if reductions:
            total = round(total - sum(reductions), 2)
        elif payments:
            total = max(set(payments), key=payments.count)
        if total != subtotal:
            warnings.append("Genel toplam okunamadı; ara toplam ve indirim/ödeme satırından hesaplandı.")
    # İndirim satırı okunamadıysa (ör. "*49, 5-D") ara toplam ile toplam farkından hesapla
    if not discount and subtotal and total and subtotal - total > 0.009:
        discount = round(subtotal - total, 2)
        warnings.append(f"İndirim ara toplam ile toplam farkından hesaplandı ({discount:.2f}).")
    # Toplam okunan ürünlerle çelişiyor ama ödeme satırı (KREDİ KARTI / POS) ürün toplamıyla birebir
    # tutuyorsa toplamda OCR rakam hatası vardır: "1.005,02" -> "1.605,02"
    if total is not None and items:
        expected = round(sum(item["total_price"] for item in items) - discount, 2)
        if abs(expected - total) > 0.01 and expected > 0 and any(
            abs(amount - expected) <= 0.01 for amount in _payment_amounts(texts)
        ):
            total = expected
            warnings.append("Genel toplam ödeme satırına göre düzeltildi; lütfen kontrol edin.")
    if total is not None and reconcile_items(items, total, discount):
        warnings.append("Bir ürün tutarı fiş toplamına göre düzeltildi; lütfen kontrol edin.")
    for item in items:
        item.pop("_line_total", None)
    total_source = "ocr"
    if total is None:
        total_source = "items"
        if not items:
            raise OCRProcessingError(
                "Fişte toplam tutar veya ürün okunamadı. Lütfen daha net bir fotoğraf yükleyin."
            )
        total = round(sum(item["total_price"] for item in items) - discount, 2)
        warnings.append("Genel toplam okunamadı; ürün tutarlarından hesaplandı.")

    items_synthesized = False
    if not items:
        # Ürün satırı olmayan belge (ör. POS slipi): toplamı işletme adıyla tek kalem yap;
        # kategori işletme adından tahmin edilir (AKARYAKIT -> Ulaşım). Kullanıcı onayda düzenleyebilir.
        items = [{
            "name": merchant if merchant != UNKNOWN_MERCHANT else "Fiş toplamı",
            "quantity": 1.0, "unit_price": total, "total_price": total, "confidence": 0.5,
        }]
        items_synthesized = True
        warnings.append("Fişte ürün satırı yok; toplam tek kalem olarak eklendi.")

    return {
        "merchant": merchant,
        "date": receipt_date,
        "total": total,
        "items": items,
        "discount": discount,
        "total_source": total_source,
        "subtotal": subtotal,
        "items_synthesized": items_synthesized,
        "date_found": date_found,
        "warnings": warnings,
        "raw_text": "\n".join(texts),
        "parsed_at": datetime.now().isoformat(timespec="seconds"),
    }
