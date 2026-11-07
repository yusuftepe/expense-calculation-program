"""
OCR/ML modülleri için ortak metin yardımcıları (text_utils.py).

Türkçe büyük/küçük harf dönüşümü, diakritik katlama (ş -> s, ı -> i ...) ve
Türk fişlerindeki tutar biçimlerini ("1.250,50", "*45,00", "135.00 TL")
float değere çevirme işlevlerini içerir.
"""

from __future__ import annotations

import re
from typing import Optional

_TR_FOLD_TABLE = str.maketrans(
    {
        "ç": "c", "ğ": "g", "ı": "i", "ö": "o", "ş": "s", "ü": "u",
        "â": "a", "î": "i", "û": "u",
    }
)

# Fiyat ifadesi: isteğe bağlı '*' öneki, binlik ayraçlı veya ayraçsız tutar, 2 ondalık.
# OCR binlik noktasını sık sık virgül okur ("2,019,03", "1.,005,02"); son ayraç ondalık, öncekiler binliktir.
# Boşluk binlik ayracı kabul edilmez: "SU 1 330,00" gibi satırlar 1330 TL okunmasın.
AMOUNT_PATTERN = r"[*xX#]?\s?(?:\d{1,3}(?:[.,]{1,2}\d{3})+|\d+)\s?[.,]\s?\d{2}"


def tr_lower(text: str) -> str:
    """Türkçe kurallarına göre küçük harfe çevirir (I -> ı, İ -> i)."""
    return text.replace("I", "ı").replace("İ", "i").lower()


def tr_upper(text: str) -> str:
    """Türkçe kurallarına göre büyük harfe çevirir (i -> İ, ı -> I)."""
    return text.replace("i", "İ").replace("ı", "I").upper()


def fold(text: str) -> str:
    """
    Metni karşılaştırma anahtarına çevirir: Türkçe küçük harf + diakritiksiz.
    OCR'ın 'Ş' yerine 'S' okuması gibi hatalara karşı eşleştirmeyi sağlamlaştırır.
    """
    return tr_lower(text).translate(_TR_FOLD_TABLE)


def parse_amount(raw: str) -> Optional[float]:
    """
    Fiş üzerindeki tutar metnini float değere çevirir.

    Örnekler:
        "*45,00" -> 45.0, "1.250,50" -> 1250.5, "135.00 TL" -> 135.0.
        Çözümlenemezse None döner.
    """
    if raw is None:
        return None
    cleaned = re.sub(r"(?i)\s*(tl|₺)\s*$", "", str(raw).strip())
    cleaned = cleaned.lstrip("*xX#").strip()
    negative = cleaned.startswith("-")
    cleaned = cleaned.lstrip("-").strip()

    # "45, 00" / "45 ,00" gibi OCR boşluklarını ayraç çevresinden temizle
    cleaned = re.sub(r"\s*([.,])\s*(\d{2})$", r"\1\2", cleaned)

    match = re.fullmatch(r"(\d{1,3}(?:[.,]{1,2}\d{3})+|\d+)(?:[.,](\d{1,2}))?", cleaned)
    if not match:
        return None

    integer_part = re.sub(r"[.,]", "", match.group(1))
    decimal_part = match.group(2) or "0"
    try:
        value = round(float(f"{integer_part}.{decimal_part}"), 2)
    except ValueError:
        return None
    return -value if negative else value


def find_trailing_amount(line: str) -> tuple[Optional[float], str]:
    """
    Satırın sonundaki tutarı bulur.

    Returns:
        (tutar, tutardan önceki metin). Satır sonunda tutar yoksa (None, line).
    """
    line = _clean_amount_tail(fix_amount_digits(line))

    # '*' önekli, boşlukla ayrılmış binlik: "*1 070,18" -> 1070.18
    match = re.search(r"\*\s?(\d{1,3})\s(\d{3})\s?[.,]\s?(\d{2})\s*$", line)
    if match:
        return float(f"{match.group(1)}{match.group(2)}.{match.group(3)}"), line[: match.start()].rstrip()

    match = re.search(rf"(-?\s?{AMOUNT_PATTERN})\s*$", line, flags=re.IGNORECASE)
    if match and _starts_cleanly(line, match):
        amount = parse_amount(match.group(1).replace(" ", ""))
        if amount is not None:
            return amount, line[: match.start()].rstrip()

    # Ondalık virgülü kaybolmuş tutar: "*424 22" -> 424.22 (yalnızca '*' önekiyle, yanlış eşleşmeyi önlemek için)
    match = re.search(r"\*\s?(\d{1,3}(?:[.,]\d{3})*|\d+)\s(\d{2})\s*$", line)
    if match:
        amount = parse_amount(f"{match.group(1)},{match.group(2)}")
        if amount is not None:
            return amount, line[: match.start()].rstrip()
    return None, line


def _clean_amount_tail(line: str) -> str:
    """
    Tutar sonrası para birimi ve OCR kalıntılarını temizler: "1.234,50 ©" (₺ -> ©), "45,00 TL".
    Fiyat önündeki '*' işaretinin '.' okunmasını düzeltir: " .424.99" -> " *424.99".
    """
    line = line.rstrip()
    # Fiş kenarından taşan çöp parçaları: "*120,00 >", "*46,95 | NV", "*1050,25 9" (en fazla 2 kısa
    # parça). Yalnızca temizlenmiş satır bir tutarla bitiyorsa uygulanır.
    candidate = line
    for _ in range(2):
        trimmed = re.sub(r"\s+(?:[^\w\s]{1,3}|[A-Za-zÇĞİÖŞÜçğıöşü]{1,2}|\w)\s*$", "", candidate)
        if trimmed == candidate:
            break
        candidate = trimmed
        if re.search(r"\d[.,]\s?\d{2}$", candidate):
            line = candidate
            break
    line = re.sub(r"(?<=\d)\s*(?:TL|₺|©|®|£|€|\$|&)\s*$", "", line, flags=re.IGNORECASE)
    return re.sub(r"(?<=\s)\.(?=\d[\d.,\s]*$)", "*", line)


def _starts_cleanly(line: str, match: re.Match) -> bool:
    """
    Tutar bir kelimenin ortasından başlamamalı: "VER:1.00.33" içindeki "00.33" fiyat değildir.
    '*' / '#' önekli tutarlar her zaman kabul edilir.
    """
    text = match.group(1).lstrip("- ")
    if text[:1] in "*#":
        return True
    start = match.start(1) + (len(match.group(1)) - len(match.group(1).lstrip("- ")))
    if text[:1] in "xX":
        start += 1
    previous = line[start - 1] if start > 0 else " "
    return previous.isspace() or previous in "%»-xX"


_DIGIT_LOOKALIKES = str.maketrans(
    {"O": "0", "o": "0", "D": "0", "l": "1", "I": "1", "|": "1", "S": "5", "B": "8", "b": "5", "G": "6"}
)


def fix_amount_digits(line: str) -> str:
    """
    Satır sonundaki '*' önekli tutarda rakama benzeyen harfleri düzeltir:
    "*1O5,00" -> "*105,00", "*4l,50" -> "*41,50". Yalnızca '*' sonrası parçaya dokunur.
    """
    match = re.search(r"\*\s?([\dOoDlI|SBbG][\dOoDlI|SBbG.,\s]*[\dOoDlI|SBbG])\s*(?:TL|₺)?\s*$", line)
    if not match or not any(ch.isdigit() for ch in match.group(1)):
        return line
    fixed = match.group(1).translate(_DIGIT_LOOKALIKES)
    return line[: match.start(1)] + fixed + line[match.end(1):]


def fix_digit_token(token: str) -> str:
    """Rakam ağırlıklı bir parçadaki O/o -> 0, l/I/| -> 1 düzeltmesi ("O1.12.2O26" -> "01.12.2026")."""
    return token.translate(str.maketrans({"O": "0", "o": "0", "l": "1", "I": "1", "|": "1"}))
