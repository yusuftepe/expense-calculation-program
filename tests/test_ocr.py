"""
Görüntü Ön İşleme ve OCR Testleri (tests/test_ocr.py).

Test görselleri Pillow ile çalışma anında üretilir; repoya fiş görseli eklenmez.
Tesseract kurulu olmayan makinelerde OCR testleri otomatik atlanır (skip).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from backend.schemas import ImageProcessingError, OCRProcessingError
from ocr_ml import image_processor, receipt_ocr, receipt_reader


def _tesseract_ready() -> bool:
    try:
        receipt_ocr.check_tesseract()
        return True
    except OCRProcessingError:
        return False


requires_tesseract = pytest.mark.skipif(
    not _tesseract_ready(), reason="Tesseract veya Türkçe dil paketi kurulu değil"
)

RECEIPT_LINES = [
    "MIGROS TICARET A.S.",
    "TARIH : 06.11.2026",
    "MAKARNA          *25,00",
    "PEYNIR           *90,50",
    "TOPLAM          *115,50",
]


def _make_receipt_image(path: Path, lines: list[str] = RECEIPT_LINES, angle: float = 0.0) -> Path:
    """Basit, ASCII karakterli sentetik fiş görseli üretir (her makinede aynı font)."""
    font = ImageFont.load_default(size=28)
    image = Image.new("L", (640, 70 + len(lines) * 50), 240)
    draw = ImageDraw.Draw(image)
    for idx, text in enumerate(lines):
        draw.text((30, 35 + idx * 50), text, fill=20, font=font)
    if angle:
        image = image.rotate(angle, expand=True, fillcolor=240, resample=Image.BICUBIC)
    image.save(path)
    return path


# ============================================================================
# 1. Görüntü ön işleme (image_processor.py)
# ============================================================================

@requires_tesseract  # dosya yolu verilince kişisel veri sansürü için OCR çalışır
def test_preprocess_returns_binary_grayscale(tmp_path: Path) -> None:
    result = image_processor.preprocess_image(_make_receipt_image(tmp_path / "fis.png"))
    assert isinstance(result, image_processor.PreparedImage)
    assert result.dtype == np.uint8 and result.ndim in (2, 3)
    assert result.ocr_lines  # en iyi okuma önceden hesaplanmış olmalı


@requires_tesseract  # dosya yolu verilince kişisel veri sansürü için OCR çalışır
def test_preprocess_supports_turkish_file_names(tmp_path: Path) -> None:
    path = _make_receipt_image(tmp_path / "şok_fişi_çarşamba.png")
    assert image_processor.preprocess_image(path).size > 0


def test_preprocess_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ImageProcessingError):
        image_processor.preprocess_image(tmp_path / "yok.jpg")


def test_preprocess_rejects_corrupt_file(tmp_path: Path) -> None:
    corrupt = tmp_path / "bozuk.jpg"
    corrupt.write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xd9")
    with pytest.raises(ImageProcessingError):
        image_processor.preprocess_image(corrupt)


def test_preprocess_rejects_tiny_image() -> None:
    with pytest.raises(ImageProcessingError):
        image_processor.preprocess_image(np.full((20, 20, 3), 255, dtype=np.uint8))


def test_deskew_corrects_rotated_receipt(tmp_path: Path) -> None:
    gray = image_processor.to_grayscale(
        image_processor.load_image(_make_receipt_image(tmp_path / "egik.png", angle=4.0))
    )
    gray = image_processor.normalize_size(gray)
    angle = image_processor.estimate_skew_angle(gray)
    assert abs(abs(angle) - 4.0) < 1.5

    straightened = image_processor.estimate_skew_angle(image_processor.deskew(gray))
    assert abs(straightened) < 1.0


# ============================================================================
# 2. OCR (receipt_ocr.py)
# ============================================================================

@requires_tesseract
def test_check_tesseract_reports_turkish() -> None:
    info = receipt_ocr.check_tesseract()
    assert "tur" in info["languages"]
    assert info["version"]


@requires_tesseract
def test_extract_text_returns_lines_with_confidence(tmp_path: Path) -> None:
    processed = image_processor.preprocess_image(_make_receipt_image(tmp_path / "fis.png"))
    lines = receipt_ocr.extract_text(processed)

    assert len(lines) >= 4
    assert [line["line_no"] for line in lines] == list(range(len(lines)))
    assert all(0.0 <= line["confidence"] <= 1.0 for line in lines)
    joined = " ".join(line["text"] for line in lines)
    assert "TOPLAM" in joined
    assert "115,50" in joined


@requires_tesseract
def test_extract_text_on_blank_image_raises() -> None:
    blank = np.full((400, 600), 255, dtype=np.uint8)
    with pytest.raises(OCRProcessingError):
        receipt_ocr.extract_text(blank)


@requires_tesseract
def test_full_pipeline_on_synthetic_receipt(tmp_path: Path) -> None:
    """Ön işleme -> OCR -> ayrıştırma -> kategori -> doğrulama zinciri uçtan uca çalışmalı."""
    from ocr_ml.categorizer_ml import predict_category
    from ocr_ml.receipt_parser import parse_receipt
    from ocr_ml.receipt_validator import validate_receipt
    from ocr_ml.text_utils import fold

    path = _make_receipt_image(tmp_path / "fis.png")
    receipt = parse_receipt(receipt_ocr.extract_text(image_processor.preprocess_image(path)))
    for item in receipt["items"]:
        item["category"] = predict_category(item["name"])
    result = validate_receipt(receipt)

    assert result["is_valid"] is True
    data = result["data"]
    assert data["date"] == "2026-11-06"
    assert data["total"] == 115.50
    # Türkçe model "PEYNIR" yerine "PEYNİR" okuyabilir; diakritiksiz karşılaştır
    assert [fold(item["name"]) for item in data["items"]] == ["makarna", "peynir"]
    assert all(item["category"] == "Gıda" for item in data["items"])


def test_missing_tesseract_binary_raises_clear_error(monkeypatch) -> None:
    monkeypatch.setenv("OCR_ENGINE", "tesseract")
    monkeypatch.setenv("TESSERACT_CMD", "/yok/tesseract")
    original_cmd = receipt_ocr.pytesseract.pytesseract.tesseract_cmd
    try:
        with pytest.raises(OCRProcessingError, match="Tesseract"):
            receipt_ocr.extract_text(np.full((100, 100), 255, dtype=np.uint8))
    finally:
        receipt_ocr.pytesseract.pytesseract.tesseract_cmd = original_cmd


# ============================================================================
# 3. Zorlu çekim koşulları (her biri gerçek hayatta sık görülen bir durum)
# ============================================================================

def _shadow(image: Image.Image) -> Image.Image:
    arr = np.asarray(image.convert("L")).astype(np.float32)
    height, width = arr.shape
    light = 0.35 + 0.65 * np.clip(np.linspace(0, 1, width)[None, :] * 0.7 + np.linspace(0, 1, height)[:, None] * 0.5, 0, 1)
    return Image.fromarray(np.clip(arr * light, 0, 255).astype(np.uint8))


def _noise(image: Image.Image) -> Image.Image:
    arr = np.asarray(image.convert("L")).astype(np.float32)
    arr += np.random.default_rng(0).normal(0, 22, arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def _faded(image: Image.Image) -> Image.Image:
    arr = np.asarray(image.convert("L")).astype(np.float32)
    return Image.fromarray((150 + arr / 255.0 * 85).astype(np.uint8))


def _on_table(image: Image.Image) -> Image.Image:
    """Fişi koyu bir masa üstüne, hafif dönük yerleştirir (telefonla çekilmiş fotoğraf)."""
    receipt = image.convert("RGB").rotate(5, expand=True, fillcolor=(70, 90, 120))
    table = Image.new("RGB", (int(receipt.width * 1.8), int(receipt.height * 1.4)), (70, 90, 120))
    table.paste(receipt, ((table.width - receipt.width) // 2, (table.height - receipt.height) // 2))
    return table


SCENARIOS = {
    "ters_cekilmis": lambda im: im.rotate(180, expand=True),
    "yan_cekilmis": lambda im: im.rotate(90, expand=True),
    "golgeli": _shadow,
    "gurultulu": _noise,
    "soluk_termal": _faded,
    "masa_ustunde": _on_table,
}


ENGINES = ["tesseract"] + (["rapidocr"] if receipt_ocr.rapidocr_available() else [])


@requires_tesseract
@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_pipeline_handles_difficult_conditions(tmp_path: Path, scenario: str, engine: str, monkeypatch) -> None:
    from ocr_ml.receipt_parser import parse_receipt

    monkeypatch.setenv("OCR_ENGINE", engine)
    path = _make_receipt_image(tmp_path / "base.png")
    degraded = SCENARIOS[scenario](Image.open(path))
    target = tmp_path / f"{scenario}.jpg"
    degraded.convert("RGB").save(target, quality=90)

    receipt = parse_receipt(receipt_ocr.extract_text(image_processor.preprocess_image(target)))
    assert receipt["total"] == 115.50
    assert receipt["date"] == "2026-11-06"
    assert sorted(item["total_price"] for item in receipt["items"]) == [25.0, 90.5]


@requires_tesseract
def test_stored_image_is_upright_after_processing(tmp_path: Path) -> None:
    """Ters yüklenen fiş, uploads/ altına düzeltilmiş (dik) haliyle yazılmalı."""
    path = _make_receipt_image(tmp_path / "base.png")
    upside = tmp_path / "ters.png"
    Image.open(path).rotate(180, expand=True).save(upside)
    image_processor.preprocess_image(upside)

    text = " ".join(line["text"] for line in receipt_ocr.extract_lines_with_boxes(str(upside)))
    assert "TOPLAM" in text


@requires_tesseract
def test_prepared_image_reuses_ocr_result(tmp_path: Path, monkeypatch) -> None:
    """preprocess_image OCR'ı zaten çalıştırdıysa extract_text tesseract'ı tekrar çağırmamalı."""
    prepared = image_processor.preprocess_image(_make_receipt_image(tmp_path / "fis.png"))
    assert isinstance(prepared, image_processor.PreparedImage) and prepared.ocr_lines

    def fail(*_args, **_kwargs):
        raise AssertionError("OCR ikinci kez çalıştırıldı")

    monkeypatch.setattr(receipt_ocr, "extract_lines_with_boxes", fail)
    assert receipt_ocr.extract_text(prepared)


def test_decompression_bomb_is_rejected(tmp_path: Path) -> None:
    """Küçük dosya boyutlu ama dev çözünürlüklü PNG belleğe açılmadan reddedilmeli."""
    bomb = tmp_path / "bomba.png"
    Image.new("L", (12000, 12000), 255).save(bomb, optimize=True)
    assert bomb.stat().st_size < 5 * 1024 * 1024
    with pytest.raises(ImageProcessingError, match="çözünürlüğü"):
        image_processor.preprocess_image(bomb)


def test_reader_survives_ocr_timeouts(monkeypatch) -> None:
    """Tüm OCR denemeleri zaman aşımına uğrarsa anlaşılır bir hata verilmeli, çökmemeli."""
    def timeout(*_args, **_kwargs):
        raise receipt_ocr.OCRTimeoutError("zaman aşımı")

    monkeypatch.setattr(receipt_reader, "extract_lines_with_boxes", timeout)
    image = np.full((600, 400, 3), 255, dtype=np.uint8)
    with pytest.raises(receipt_ocr.OCRTimeoutError):
        receipt_reader.read_best(image)


# ============================================================================
# 4. OCR motoru seçimi ve RapidOCR yardımcıları
# ============================================================================

def test_engine_selection(monkeypatch) -> None:
    monkeypatch.setenv("OCR_ENGINE", "tesseract")
    assert receipt_ocr.get_engine_name() == "tesseract"
    monkeypatch.setenv("OCR_ENGINE", "auto")
    expected = "rapidocr" if receipt_ocr.rapidocr_available() else "tesseract"
    assert receipt_ocr.get_engine_name() == expected


def test_engine_falls_back_to_tesseract_without_rapidocr(monkeypatch) -> None:
    monkeypatch.setattr(receipt_ocr, "rapidocr_available", lambda: False)
    monkeypatch.setenv("OCR_ENGINE", "auto")
    assert receipt_ocr.get_engine_name() == "tesseract"
    monkeypatch.setenv("OCR_ENGINE", "rapidocr")
    with pytest.raises(OCRProcessingError, match="rapidocr"):
        receipt_ocr.get_engine_name()


def test_group_segments_into_lines_joins_name_and_price() -> None:
    segments = [
        {"text": "*45,00", "confidence": 0.9, "left": 400, "top": 102, "width": 80, "height": 20},
        {"text": "SÜT 1L", "confidence": 0.8, "left": 10, "top": 100, "width": 90, "height": 22},
        {"text": "TOPLAM", "confidence": 0.95, "left": 10, "top": 140, "width": 90, "height": 22},
        {"text": "*45,00", "confidence": 0.95, "left": 400, "top": 141, "width": 80, "height": 21},
    ]
    lines = receipt_ocr.group_segments_into_lines(segments)
    assert [line["text"] for line in lines] == ["SÜT 1L *45,00", "TOPLAM *45,00"]
    assert len(lines[0]["words"]) == 2 and lines[0]["confidence"] == 0.85


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("TAM BUČDAY EKMEK", "TAM BUÇDAY EKMEK"),
        ("KAŞAR PEYNİRÍ", "KAŞAR PEYNİRİ"),
        ("ANTEP FISTIğI", "ANTEP FISTIĞI"),
        ("大大大水大大大大 9999", "******** 9999"),  # maske yıldızları CJK karakter okunmuş
        ("Kredi Kartı", "Kredi Kartı"),
    ],
)
def test_normalize_turkish(raw: str, expected: str) -> None:
    assert receipt_ocr.normalize_turkish(raw) == expected
