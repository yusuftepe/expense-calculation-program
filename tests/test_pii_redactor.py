"""
Kişisel Veri Sansürleme Testleri (tests/test_pii_redactor.py).

Tüm kart, IBAN, TCKN ve telefon numaraları sahte/test değerleridir:
    4111 1111 1111 1111 -> standart Visa test kartı (Luhn geçerli)
    10000000146         -> algoritmaya uygun örnek TCKN
    TR33 0006 1005 1978 6457 8413 26 -> dokümantasyonlarda kullanılan örnek IBAN
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from backend.schemas import OCRProcessingError
from ocr_ml import receipt_ocr
from ocr_ml.pii_redactor import (
    find_pii_boxes,
    luhn_valid,
    redact_image_file,
    redact_text,
    tckn_valid,
)


def _tesseract_ready() -> bool:
    try:
        receipt_ocr.check_tesseract()
        return True
    except OCRProcessingError:
        return False


requires_tesseract = pytest.mark.skipif(
    not _tesseract_ready(), reason="Tesseract veya Türkçe dil paketi kurulu değil"
)


# ============================================================================
# 1. Algoritma kontrolleri
# ============================================================================

def test_luhn() -> None:
    assert luhn_valid("4111111111111111")
    assert not luhn_valid("4111111111111112")


def test_tckn_checksum() -> None:
    assert tckn_valid("10000000146")
    assert not tckn_valid("10000000147")
    assert not tckn_valid("01234567890")
    assert not tckn_valid("1234")


# ============================================================================
# 2. Metin sansürü
# ============================================================================

@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("KART NO: 4111 1111 1111 1111", "KART NO: [KART]"),
        ("4111-1111-1111-1111", "[KART]"),
        ("540061******1234", "[KART]"),
        ("XXXX XXXX XXXX 1234", "[KART]"),
        ("**** 1234 SATIS", "[KART]"),  # satırın tamamı maskelenir (güvenli taraf)
        ("IBAN: TR33 0006 1005 1978 6457 8413 26", "IBAN: [IBAN]"),
        ("TCKN: 10000000146", "TCKN: [TCKN]"),
        ("10000000146", "[TCKN]"),
        ("+90 555 555 5555", "[TELEFON]"),
        ("0555 555 55 55", "[TELEFON]"),
        ("mail: ornek@example.com", "mail: [E-POSTA]"),
        ("SAYIN AHMET YILMAZ", "SAYIN [KİŞİSEL]"),
        ("MÜŞTERİ: Ayşe K.", "MÜŞTERİ: [KİŞİSEL]"),
        ("Musteri Adi: Can", "Musteri Adi: [KİŞİSEL]"),
        ("KASİYER: 7-ELİF Y.", "KASİYER: [KİŞİSEL]"),
        ("400000xxxxxx9999", "[KART]"),          # gevşek maskeli kart
        ("JJ) r400000xxxxxx9999", "[KART]"),  # önüne harf yapışmış
        ("XAXX Haxa xxx4 9999", "[KART]"),       # OCR'ın bozduğu maskeli kart satırı
    ],
)
def test_redact_text_masks_personal_data(text: str, expected: str) -> None:
    assert redact_text(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "TOPLAM *514,30",
        "KREDİ KARTI *514,30",
        "TARİH : 05.11.2026 SAAT : 14:32",
        "2 AD X 22,50",
        "SÜT 1L %1 *45,00",
        "KADIKÖY V.D. 1234567890",          # işletme vergi no (10 hane)
        "MERSİS NO: 0123456789000015",      # işletme MERSİS no
        "TEL: 0216 555 12 34",              # işletme sabit hattı
        "EKÜ NO: 0001 Z NO: 0123",
        "10000000147",                      # algoritmaya uymayan 11 hane
        "4111111111111112",                 # Luhn geçersiz 16 hane
    ],
)
def test_redact_text_keeps_receipt_and_business_data(text: str) -> None:
    assert redact_text(text) == text


def test_find_pii_boxes_maps_spans_to_words() -> None:
    words = [
        {"text": "KART", "left": 0, "top": 0, "width": 40, "height": 20},
        {"text": "NO:", "left": 45, "top": 0, "width": 30, "height": 20},
        {"text": "4111", "left": 80, "top": 0, "width": 40, "height": 20},
        {"text": "1111", "left": 125, "top": 0, "width": 40, "height": 20},
        {"text": "1111", "left": 170, "top": 0, "width": 40, "height": 20},
        {"text": "1111", "left": 215, "top": 0, "width": 40, "height": 20},
    ]
    boxes = find_pii_boxes([{"text": "", "confidence": 0.9, "words": words}])
    assert [box["left"] for box in boxes] == [80, 125, 170, 215]


# ============================================================================
# 3. Görsel sansürü
# ============================================================================

def _make_pii_receipt(path: Path) -> Path:
    lines = [
        "MIGROS TICARET A.S.",
        "TARIH : 06.11.2026",
        "MAKARNA          *25,00",
        "TOPLAM           *25,00",
        "KART NO: 4111 1111 1111 1111",
    ]
    font = ImageFont.load_default(size=28)
    image = Image.new("RGB", (700, 70 + len(lines) * 50), (240, 240, 240))
    draw = ImageDraw.Draw(image)
    for idx, text in enumerate(lines):
        draw.text((30, 35 + idx * 50), text, fill=(20, 20, 20), font=font)
    exif = Image.Exif()
    exif[0x010F] = "TestPhone"  # Make
    image.save(path, quality=92, exif=exif)
    return path


@requires_tesseract
def test_redact_image_file_hides_card_and_strips_exif(tmp_path: Path) -> None:
    path = _make_pii_receipt(tmp_path / "fis.jpg")
    assert dict(Image.open(path).getexif())

    _, labels = redact_image_file(path)
    # Kart no, sayı deseniyle [KART] veya "KART NO:" etiketiyle [KİŞİSEL] olarak yakalanabilir
    assert labels and set(labels) <= {"KART", "KİŞİSEL"}
    assert not dict(Image.open(path).getexif())

    # Sansür sonrası görselde kart numarası okunmamalı, fiş bilgileri okunmalı
    text = " ".join(line["text"] for line in receipt_ocr.extract_lines_with_boxes(str(path)))
    assert "4111" not in text
    assert "TOPLAM" in text
    assert "25,00" in text


@requires_tesseract
def test_preprocess_redacts_uploaded_file_in_place(tmp_path: Path) -> None:
    from ocr_ml.image_processor import preprocess_image

    path = _make_pii_receipt(tmp_path / "fis.png")
    preprocess_image(path)

    # Dosya düzeltilmiş (kırpılmış/dik) ve sansürlü haliyle yeniden yazılır
    stored = " ".join(line["text"] for line in receipt_ocr.extract_lines_with_boxes(str(path)))
    assert "4111" not in stored
    assert "TOPLAM" in stored
    assert (np.asarray(Image.open(path).convert("L")) < 10).sum() > 0  # siyah kutular var


@requires_tesseract
def test_extract_text_never_returns_raw_card_number(tmp_path: Path) -> None:
    """Görsel sansürü atlanıp doğrudan OCR yapılsa bile metin katmanı maskelemeli."""
    path = _make_pii_receipt(tmp_path / "fis.png")
    text = " ".join(line["text"] for line in receipt_ocr.extract_text(str(path)))
    assert "4111" not in text
    assert "[KART]" in text or "[KİŞİSEL]" in text


def test_cardholder_name_after_card_line_is_masked() -> None:
    """Banka slibinde kart satırının altındaki etiketsiz ad satırı gizlenmeli; banka adı gizlenmemeli."""
    from ocr_ml.pii_redactor import redact_lines

    lines = ["SATIŞ", "**** **** **** 9999", "AYŞE KAYA", "TUTAR", "1.050,25 TL"]
    assert redact_lines(lines) == ["SATIŞ", "[KART]", "[KİŞİSEL]", "TUTAR", "1.050,25 TL"]
    assert redact_lines(["**** **** **** 9999", "HALK BANKASI"])[1] == "HALK BANKASI"
    assert redact_lines(["**** **** **** 9999", "Mastercard/Kredi/Yurt İçi"])[1] == "Mastercard/Kredi/Yurt İçi"


@pytest.mark.parametrize(
    "text", ["ADE Xx01 x46,95", "Z NO:1794 EKÜ NO:0001", "POSET BSK 36x69+15 X20 *1,00", "MAX 9999 TL"]
)
def test_masked_card_heuristics_do_not_touch_receipt_lines(text: str) -> None:
    assert redact_text(text) == text
