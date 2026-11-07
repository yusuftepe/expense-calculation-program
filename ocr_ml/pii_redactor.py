"""
Kişisel Veri Sansürleme Modülü (pii_redactor.py).

Fiş görsellerinde ve OCR metninde bulunan kişisel verileri (KVKK) otomatik gizler:
    - Kredi/banka kartı numarası (tam veya maskeli: 540061******1234, **** 1234)
    - IBAN (TR ile başlayan)
    - T.C. kimlik numarası (11 hane + algoritma kontrolü)
    - Cep telefonu numarası (05xx / +90 5xx)
    - E-posta adresi
    - Müşteri adı, üye/kart numarası gibi etiketli alanlar ("SAYIN ...", "MÜŞTERİ: ...")

İki katmanda çalışır:
    1. Görsel: Kişisel veri içeren kelimelerin üzeri siyah kutuyla kapatılır ve dosya
       yerinde yeniden yazılır. Yeniden kodlama EXIF bilgisini (GPS konumu, cihaz
       bilgisi) de siler.
    2. Metin: OCR çıktısındaki kişisel veriler "[KART]", "[TCKN]" gibi etiketlerle
       değiştirilir; ham hali ayrıştırıcıya, oturuma veya veritabanına ulaşmaz.

Not: Tespit OCR'a dayalıdır; çok bulanık veya kırışık fişlerde bir alan okunamazsa
gizlenemeyebilir. Bu modül hiçbir kişisel veriyi loglamaz.
"""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from typing import Any, Union

import cv2
import numpy as np

from backend.schemas import ImageProcessingError
from ocr_ml import image_processor, receipt_ocr
from ocr_ml.text_utils import fold

LABEL_CARD = "KART"
LABEL_IBAN = "IBAN"
LABEL_TCKN = "TCKN"
LABEL_PHONE = "TELEFON"
LABEL_EMAIL = "E-POSTA"
LABEL_PERSONAL = "KİŞİSEL"

EMAIL_PATTERN = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
IBAN_PATTERN = re.compile(r"\bTR\s?\d{2}(?:\s?[\d*Xx]){22}(?!\d)", flags=re.IGNORECASE)
# 12-19 karakterlik rakam / maske dizisi; aralarında tek boşluk veya tire olabilir
CARD_CANDIDATE = re.compile(r"(?<![\w*])[\d*Xx#](?:[ -]?[\d*Xx#]){11,18}(?![\w*])")
# Yalnız son 4 hanesi görünen kısa maskeli kart: "**** 1234", "****1234"
CARD_SHORT_MASKED = re.compile(r"(?<![\w*])\*{4,}[ -]?\d{4}(?!\d)")
TCKN_CANDIDATE = re.compile(r"(?<!\d)[1-9]\d{10}(?!\d)")
# OCR'ın bozduğu maskeli kart: "r400000xxxxxx9999" (öndeki harf), "400000x*****9999"
CARD_LOOSE_MASKED = re.compile(r"(?:\d{4,6}[xX*#]{3,}|\w{4,6}[xX*#]{4,})\d{4}(?!\d)")
# Satır sonundaki 4 haneli grup (tutarın parçası değil) ve maske benzeri parça ("XAXX", "Kxrx", "xxx4")
TRAILING_FOUR_DIGITS = re.compile(r"(?<![\d.,])\d{4}(?![\d.,])(?=[^\d]{0,14}$)")
MASK_LIKE_TOKEN = re.compile(r"(?<!\S)\S*[xX*#]\S*[xX*#]\S*(?!\S)")
# Kart sahibi adı satırı: yalnız harflerden oluşan 2-4 kelime
NAME_LINE = re.compile(r"^\s*[A-Za-zÇĞİÖŞÜçğıöşü]{2,}(?:\s+[A-Za-zÇĞİÖŞÜçğıöşü.]{1,}){1,3}\s*\W{0,2}\s*$")
NOT_A_NAME: tuple[str, ...] = (
    "mastercard", "visa", "kredi", "karti", "kart", "banka", "bankasi", "debit", "troy", "yurt", "satis",
    "tutar", "tutari", "islem", "temassiz", "onay", "kodu", "toplam", "nakit", "ziraat", "halk", "garanti",
    "akbank", "yapi", "kredi", "isbank", "is", "vakif", "vakifbank", "denizbank", "teb", "qnb", "enpara",
    "belgeyi", "saklayiniz", "nushasi", "nushasidir", "hamili", "mal", "hizmet", "aldim", "iade", "referans",
)
MOBILE_PATTERN = re.compile(r"(?<!\d)(?:\+?90[ -]?|0[ -]?)?\(?5\d{2}\)?[ -]?\d{3}[ -]?\d{2}[ -]?\d{2}(?!\d)")

# Ardından gelen değerin tamamı gizlenecek etiketler (diakritiksiz, küçük harf)
LABELED_FIELDS: tuple[str, ...] = (
    "sayin", "musteri adi soyadi", "musteri adi", "musteri no", "musteri", "ad soyad", "adi soyadi", "kart sahibi", "kart no", "card no",
    "uye no", "uye adi", "tc kimlik", "t.c. kimlik", "tckn", "tc no", "t.c. no", "gsm", "cep tel",
    "kasiyer",  # çalışan adı da kişisel veridir
)
LABELED_PATTERN = re.compile(
    # Uzun etiketler önce denenir: "musteri adi: X" -> yalnız X gizlenir
    rf"(?<![a-z])(?:{'|'.join(re.escape(f) for f in sorted(LABELED_FIELDS, key=len, reverse=True))})(?![a-z])\s*[:.]?\s*(\S.*)$"
)

Span = tuple[int, int, str]


def luhn_valid(digits: str) -> bool:
    """Kart numarası Luhn (mod 10) kontrolü."""
    total = 0
    for idx, char in enumerate(reversed(digits)):
        value = int(char)
        if idx % 2 == 1:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return total % 10 == 0


def tckn_valid(number: str) -> bool:
    """T.C. kimlik numarası algoritma kontrolü (10. ve 11. hane)."""
    if len(number) != 11 or not number.isdigit() or number[0] == "0":
        return False
    d = [int(ch) for ch in number]
    tenth = ((d[0] + d[2] + d[4] + d[6] + d[8]) * 7 - (d[1] + d[3] + d[5] + d[7])) % 10
    eleventh = sum(d[:10]) % 10
    return d[9] == tenth and d[10] == eleventh


def _is_card(candidate: str) -> bool:
    compact = re.sub(r"[ -]", "", candidate)
    digits = sum(ch.isdigit() for ch in compact)
    masks = len(compact) - digits
    if not 12 <= len(compact) <= 19:
        return False
    if masks == 0:
        return len(compact) >= 13 and luhn_valid(compact)
    return digits >= 4 and masks >= 4


def find_pii_spans(text: str) -> list[Span]:
    """
    Metindeki kişisel veri aralıklarını bulur.

    Returns:
        Çakışmaları birleştirilmiş, sıralı (başlangıç, bitiş, etiket) listesi.
    """
    if not text:
        return []
    spans: list[Span] = []

    spans += [(m.start(), m.end(), LABEL_EMAIL) for m in EMAIL_PATTERN.finditer(text)]
    spans += [(m.start(), m.end(), LABEL_IBAN) for m in IBAN_PATTERN.finditer(text)]
    spans += [(m.start(), m.end(), LABEL_CARD) for m in CARD_CANDIDATE.finditer(text) if _is_card(m.group())]
    spans += [(m.start(), m.end(), LABEL_CARD) for m in CARD_SHORT_MASKED.finditer(text)]
    spans += [(m.start(), m.end(), LABEL_CARD) for m in CARD_LOOSE_MASKED.finditer(text)]
    if _looks_like_masked_card_line(text):
        spans.append((0, len(text.rstrip()), LABEL_CARD))
    spans += [(m.start(), m.end(), LABEL_TCKN) for m in TCKN_CANDIDATE.finditer(text) if tckn_valid(m.group())]
    spans += [(m.start(), m.end(), LABEL_PHONE) for m in MOBILE_PATTERN.finditer(text)]

    folded = fold(text)
    if len(folded) == len(text):  # fold uzunluğu koruduğunda konumlar birebir eşleşir
        match = LABELED_PATTERN.search(folded)
        if match:
            spans.append((match.start(1), len(text.rstrip()), LABEL_PERSONAL))

    spans.sort()
    merged: list[Span] = []
    for start, end, label in spans:
        if merged and start <= merged[-1][1]:
            prev_start, prev_end, prev_label = merged[-1]
            # Genel "KİŞİSEL" etiketi yerine daha belirgin etiketi (KART, TCKN...) koru
            best_label = label if prev_label == LABEL_PERSONAL else prev_label
            merged[-1] = (prev_start, max(prev_end, end), best_label)
        else:
            merged.append((start, end, label))
    return merged


def _looks_like_masked_card_line(text: str) -> bool:
    """
    OCR'ın bozduğu maskeli kart satırı: sonda 4 haneli grup + önünde maske benzeri parça.
    Örn. "XAXX Haxa xxx4 9999", "KA <T NÖ SAĞE Doe Kxrx 1234 EE".
    """
    match = TRAILING_FOUR_DIGITS.search(text)
    return bool(match and MASK_LIKE_TOKEN.search(text[: match.start()]))


def _is_name_line(text: str) -> bool:
    if not NAME_LINE.match(text):
        return False
    words = [fold(word).strip(".") for word in text.split()]
    return not any(word in NOT_A_NAME for word in words)


def find_line_spans(texts: list[str]) -> list[list[Span]]:
    """
    Satır listesindeki kişisel veri aralıkları; satır içi kuralların yanında bağlama bakar:
    kart satırının hemen altındaki etiketsiz ad satırı ("AD SOYAD") kart sahibi adıdır.
    """
    result = [find_pii_spans(text) for text in texts]
    # Banka slibi: "SATIŞ" satırından sonraki 2 satırda 4 haneyle biten satır maskeli kart numarasıdır
    # (OCR maskeyi "SE40RAAK KAZA 1234" gibi harflere çevirse bile)
    for idx, text in enumerate(texts):
        if re.search(r"(?<![a-z])sat[it1l|][st](?![a-z])", fold(text)):  # SATIŞ, OCR: "SATTS"
            for nxt in range(idx + 1, min(idx + 3, len(texts))):
                candidate = texts[nxt]
                if any(label == LABEL_CARD for _, _, label in result[nxt]):
                    break  # kart satırı zaten bulundu
                if re.search(r"terminal|batch|isyeri|onay|ref|sira|stan|tarih|saat", fold(candidate)):
                    continue
                if not result[nxt] and TRAILING_FOUR_DIGITS.search(candidate) and not re.search(r"\d[.,]\d{2}", candidate):
                    result[nxt] = [(0, len(candidate.rstrip()), LABEL_CARD)]
                    break
    for idx in range(len(texts) - 1):
        if any(label == LABEL_CARD for _, _, label in result[idx]) and not result[idx + 1]:
            following = texts[idx + 1]
            if _is_name_line(following):
                result[idx + 1] = [(0, len(following.rstrip()), LABEL_PERSONAL)]
    return result


def _apply_spans(text: str, spans: list[Span]) -> str:
    for start, end, label in reversed(spans):
        text = f"{text[:start]}[{label}]{text[end:]}"
    return text


def redact_text(text: str) -> str:
    """Metindeki kişisel verileri "[KART]", "[TCKN]" gibi etiketlerle değiştirir (tek satır)."""
    return _apply_spans(text, find_pii_spans(text))


def redact_lines(texts: list[str]) -> list[str]:
    """Satır listesini bağlam kurallarıyla (kart sahibi adı vb.) birlikte maskeler."""
    return [_apply_spans(text, spans) for text, spans in zip(texts, find_line_spans(texts))]


def find_pii_boxes(lines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    receipt_ocr.extract_lines_with_boxes çıktısından kişisel veri içeren kelime kutularını bulur.

    Returns:
        [{"label": "KART", "left": .., "top": .., "width": .., "height": ..}, ...]
    """
    boxes: list[dict[str, Any]] = []
    joined = [" ".join(word["text"] for word in (line.get("words") or [])) for line in lines]
    for line, spans in zip(lines, find_line_spans(joined)):
        words = line.get("words") or []
        offsets: list[tuple[int, int]] = []
        cursor = 0
        for word in words:
            offsets.append((cursor, cursor + len(word["text"])))
            cursor += len(word["text"]) + 1
        for start, end, label in spans:
            for word, (w_start, w_end) in zip(words, offsets):
                if w_start < end and w_end > start:
                    boxes.append(
                        {
                            "label": label,
                            "left": word["left"],
                            "top": word["top"],
                            "width": word["width"],
                            "height": word["height"],
                        }
                    )
    return boxes


def redact_image(image: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """
    Görseldeki kişisel verilerin üzerini siyah kutuyla kapatır.

    Returns:
        (sansürlü görselin kopyası, bulunan etiketler listesi)
    """
    gray = image_processor.to_grayscale(image)
    scaled = image_processor.normalize_size(gray)
    scale = scaled.shape[1] / gray.shape[1]

    lines = receipt_ocr.extract_lines_with_boxes(image_processor.binarize(scaled))
    boxes = [
        {key: (value / scale if key != "label" else value) for key, value in box.items()}
        for box in find_pii_boxes(lines)
    ]
    return draw_boxes(image, boxes), [box["label"] for box in boxes]


def draw_boxes(image: np.ndarray, boxes: list[dict[str, Any]]) -> np.ndarray:
    """Verilen kutuların (görsel koordinatlarında) üzerini siyahla kapatır; kopya döndürür."""
    redacted = image.copy()
    height, width = image.shape[:2]
    color: Union[int, tuple[int, ...]] = 0 if image.ndim == 2 else (0,) * image.shape[2]
    for box in boxes:
        pad = max(3, int(box["height"] * 0.2))
        x0 = max(0, int(box["left"]) - pad)
        y0 = max(0, int(box["top"]) - pad)
        x1 = min(width - 1, int(box["left"] + box["width"]) + pad)
        y1 = min(height - 1, int(box["top"] + box["height"]) + pad)
        cv2.rectangle(redacted, (x0, y0), (x1, y1), color, thickness=-1)
    return redacted


def write_redacted_image(path: Path, image: np.ndarray, boxes: list[dict[str, Any]]) -> None:
    """
    Görselin kişisel veri kutularını kapatıp dosyanın üzerine yazar.
    Kutu olmasa bile yeniden yazar; EXIF (GPS, cihaz bilgisi) her durumda silinir.
    """
    _write_image(path, draw_boxes(image, boxes))


def _write_image(path: Path, image: np.ndarray) -> None:
    """Görseli aynı formatta, geçici dosya + atomik değiştirme ile yazar (EXIF yazılmaz)."""
    ext = path.suffix.lower() or ".jpg"
    params: list[int] = []
    if ext in (".jpg", ".jpeg"):
        params = [cv2.IMWRITE_JPEG_QUALITY, 92]
    elif ext == ".webp":
        params = [cv2.IMWRITE_WEBP_QUALITY, 92]

    ok, encoded = cv2.imencode(ext, image, params)
    if not ok:
        raise ImageProcessingError("Sansürlenen fiş görseli kaydedilemedi.")

    fd, tmp_name = tempfile.mkstemp(dir=path.parent, suffix=ext)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(encoded.tobytes())
        os.replace(tmp_name, path)
    except OSError as exc:
        Path(tmp_name).unlink(missing_ok=True)
        raise ImageProcessingError("Sansürlenen fiş görseli kaydedilemedi.", details=str(exc)) from exc


def redact_image_file(image_path: Union[str, Path]) -> tuple[np.ndarray, list[str]]:
    """
    Fiş görsel dosyasındaki kişisel verileri sansürler ve dosyayı yerinde yeniden yazar.

    Kişisel veri bulunmasa bile dosya yeniden kodlanır; böylece EXIF bilgisi
    (GPS konumu, cihaz modeli, çekim zamanı) her durumda silinir.

    Returns:
        (sansürlü görsel, bulunan etiketler listesi)
    """
    path = Path(image_path)
    image = image_processor.load_image(path)
    redacted, labels = redact_image(image)
    _write_image(path, redacted)
    return redacted, labels
