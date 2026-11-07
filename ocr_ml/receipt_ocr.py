"""
Fiş Metin Çıkarma Modülü (receipt_ocr.py).

Fiş görselinden satır satır metin, satır bazlı güven skoru ve kelime kutuları çıkarır.
İki OCR motoru desteklenir:

    rapidocr  (varsayılan): PaddleOCR modellerinin ONNX sürümü (rapidocr + onnxruntime).
              Telefonla çekilmiş buruşuk / soluk fişlerde Tesseract'tan çok daha başarılı
              (11 gerçek fişte toplam 11/11'e karşı 8/11) ve ~8 kat hızlı. Modeller paketle
              gelir, internet gerekmez; görsel makineden çıkmaz.
    tesseract (yedek): rapidocr kurulu değilse otomatik kullanılır.

Motor seçimi: OCR_ENGINE = auto (varsayılan) | rapidocr | tesseract

rapidocr kurulumu (opencv-python ile çakışmaması için bağımlılıksız kurulur):
    pip install --no-deps rapidocr && pip install onnxruntime pyclipper shapely omegaconf colorlog

Tesseract gereksinimi: Sistemde Tesseract ve Türkçe dil paketi kurulu olmalıdır.
    - Arch:    sudo pacman -S tesseract tesseract-data-tur
    - Ubuntu:  sudo apt install tesseract-ocr tesseract-ocr-tur
    - macOS:   brew install tesseract tesseract-lang
    - Windows: UB-Mannheim kurulumu; PATH'te değilse .env içine
               TESSERACT_CMD=C:\\Program Files\\Tesseract-OCR\\tesseract.exe

İsteğe bağlı ortam değişkenleri:
    OCR_ENGINE    : auto | rapidocr | tesseract
    TESSERACT_CMD : tesseract çalıştırılabilir dosyasının tam yolu
    OCR_LANG      : Tesseract dil kodu (varsayılan: 'tur')
"""

from __future__ import annotations

import importlib.util
import os
import re
import threading
from pathlib import Path
from typing import Any, Union

import numpy as np
import pytesseract
from PIL import Image

from backend.schemas import OCRProcessingError

DEFAULT_LANG = "tur"
# --oem 1: LSTM motoru, --psm 6: tek düzgün metin bloğu (fiş satırlarını bir arada tutar)
DEFAULT_CONFIG = "--oem 1 --psm 6"

OCRInput = Union[str, Path, np.ndarray, Image.Image]

DEFAULT_TIMEOUT = 20  # saniye; tek bir tesseract çağrısı için üst sınır


class OCRTimeoutError(OCRProcessingError):
    """Tesseract çağrısı zaman sınırını aştığında fırlatılır."""


def _configure_tesseract() -> None:
    """TESSERACT_CMD ortam değişkeni varsa pytesseract'a bildirir."""
    custom_cmd = (os.getenv("TESSERACT_CMD") or "").strip()
    if custom_cmd:
        pytesseract.pytesseract.tesseract_cmd = custom_cmd


def _get_lang() -> str:
    return (os.getenv("OCR_LANG") or DEFAULT_LANG).strip() or DEFAULT_LANG


def check_tesseract() -> dict[str, Any]:
    """
    Tesseract kurulumunu ve dil paketini kontrol eder.

    Returns:
        {"version": "5.5.4", "languages": [...], "lang": "tur"}

    Raises:
        OCRProcessingError: Tesseract bulunamazsa veya dil paketi eksikse.
    """
    _configure_tesseract()
    lang = _get_lang()
    try:
        version = str(pytesseract.get_tesseract_version())
        languages = pytesseract.get_languages(config="")
    except pytesseract.TesseractNotFoundError as exc:
        raise OCRProcessingError(
            "Tesseract OCR bulunamadı. Lütfen Tesseract'ı kurun veya TESSERACT_CMD ayarlayın.",
            details=str(exc),
        ) from exc

    missing = [code for code in lang.split("+") if code not in languages]
    if missing:
        raise OCRProcessingError(
            f"Tesseract dil paketi eksik: {', '.join(missing)}.",
            details=f"Kurulu diller: {', '.join(languages)}",
        )
    return {"version": version, "languages": languages, "lang": lang}


def _to_ocr_image(image: OCRInput) -> Union[str, np.ndarray, Image.Image]:
    if isinstance(image, Path):
        return str(image)
    if isinstance(image, (str, np.ndarray, Image.Image)):
        return image
    raise OCRProcessingError(f"Desteklenmeyen görsel tipi: {type(image).__name__}")


ENGINE_RAPIDOCR = "rapidocr"
ENGINE_TESSERACT = "tesseract"

_rapid_engine: Any = None
_rapid_lock = threading.Lock()


def rapidocr_available() -> bool:
    return importlib.util.find_spec("rapidocr") is not None and importlib.util.find_spec("onnxruntime") is not None


def get_engine_name() -> str:
    """OCR_ENGINE ortam değişkenine ve kurulu paketlere göre kullanılacak motoru döndürür."""
    choice = (os.getenv("OCR_ENGINE") or "auto").strip().lower()
    if choice == ENGINE_TESSERACT:
        return ENGINE_TESSERACT
    if rapidocr_available():
        return ENGINE_RAPIDOCR
    if choice == ENGINE_RAPIDOCR:
        raise OCRProcessingError(
            "OCR_ENGINE=rapidocr seçili ama rapidocr kurulu değil.",
            details="pip install --no-deps rapidocr && pip install onnxruntime pyclipper shapely omegaconf colorlog",
        )
    return ENGINE_TESSERACT


def _get_rapid_engine() -> Any:
    global _rapid_engine
    with _rapid_lock:
        if _rapid_engine is None:
            try:
                from rapidocr import RapidOCR
            except ImportError as exc:  # pragma: no cover - get_engine_name kontrol eder
                raise OCRProcessingError("rapidocr yüklenemedi.", details=str(exc)) from exc
            _rapid_engine = RapidOCR(params={"Global.log_level": "error"})
        return _rapid_engine


def _to_bgr_array(image: OCRInput) -> np.ndarray:
    import cv2

    if isinstance(image, (str, Path)):
        from ocr_ml.image_processor import load_image

        return load_image(image)
    if isinstance(image, Image.Image):
        return cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    array = np.asarray(image)
    if array.ndim == 2:
        return cv2.cvtColor(array, cv2.COLOR_GRAY2BGR)
    return array[:, :, :3]


# Modelin Türkçe yerine ürettiği benzer harfler
_TURKISH_LOOKALIKES = str.maketrans(
    {"Č": "Ç", "č": "ç", "Ć": "Ç", "ć": "ç", "Í": "İ", "Ì": "İ", "Î": "İ", "Ș": "Ş", "ș": "ş",
     "Ǧ": "Ğ", "ǧ": "ğ", "Ő": "Ö", "ő": "ö", "Ű": "Ü", "ű": "ü"}
)
_TR_UPPER = {"ç": "Ç", "ğ": "Ğ", "ı": "I", "i": "İ", "ö": "Ö", "ş": "Ş", "ü": "Ü"}


def normalize_turkish(text: str) -> str:
    """
    OCR çıktısındaki Türkçe olmayan benzer harfleri düzeltir ("BUČDAY" -> "BUÇDAY"), CJK
    karakterleri maske yıldızına çevirir ("大大大大 9999" -> "**** 9999") ve
    büyük harfli kelimelerin içindeki tek küçük Türkçe harfi büyütür ("FISTIğI" -> "FISTIĞI").
    """
    text = text.translate(_TURKISH_LOOKALIKES)
    # Varsayılan model Çince ağırlıklıdır; fişteki maske yıldızlarını "大", "天", "水" gibi
    # karakterlere çevirebilir. Türk fişinde CJK karakterin başka anlamı olmadığından '*' yapılır.
    text = re.sub(r"[\u2e80-\u9fff\uf900-\ufaff\uff00-\uffef]", "*", text)

    def fix_word(match: "re.Match[str]") -> str:
        word = match.group(0)
        letters = [ch for ch in word if ch.isalpha()]
        uppers = sum(ch.isupper() for ch in letters)
        if len(letters) >= 3 and uppers >= len(letters) - 1 and uppers < len(letters):
            return "".join(_TR_UPPER.get(ch, ch) if ch in _TR_UPPER else ch for ch in word)
        return word

    return re.sub(r"\w+", fix_word, text)


def group_segments_into_lines(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Metin parçalarını (kutularıyla) satırlara gruplar: dikey merkezleri örtüşen parçalar
    aynı satırdır; satır içinde soldan sağa sıralanır. Fişte ürün adı ile fiyat ayrı parça
    olarak algılanır, bu adım onları tek satırda birleştirir.
    """
    ordered = sorted(segments, key=lambda seg: seg["top"] + seg["height"] / 2)
    lines: list[dict[str, Any]] = []
    for seg in ordered:
        center = seg["top"] + seg["height"] / 2
        height = max(1.0, float(seg["height"]))
        for line in lines:
            if abs(line["center"] - center) < 0.5 * min(height, line["height"]):
                line["words"].append(seg)
                count = len(line["words"])
                line["center"] = (line["center"] * (count - 1) + center) / count
                break
        else:
            lines.append({"center": center, "height": height, "words": [seg]})

    result: list[dict[str, Any]] = []
    for line in sorted(lines, key=lambda item: item["center"]):
        words = sorted(line["words"], key=lambda seg: seg["left"])
        result.append(
            {
                "text": " ".join(word["text"] for word in words),
                "confidence": round(sum(word["confidence"] for word in words) / len(words), 4),
                "words": [{key: word[key] for key in ("text", "left", "top", "width", "height")} for word in words],
            }
        )
    return result


def _rapidocr_lines_with_boxes(image: OCRInput) -> list[dict[str, Any]]:
    engine = _get_rapid_engine()
    array = _to_bgr_array(image)
    try:
        with _rapid_lock:  # motor nesnesi iş parçacıkları arasında paylaşılıyor
            output = engine(array)
    except Exception as exc:  # noqa: BLE001 - onnxruntime çok çeşitli hata fırlatabilir
        raise OCRProcessingError("Fiş görselinden metin çıkarılamadı.", details=str(exc)) from exc
    if output is None or output.txts is None:
        return []
    segments = []
    for box, text, score in zip(output.boxes, output.txts, output.scores):
        text = normalize_turkish((text or "").strip())
        if not text:
            continue
        points = np.asarray(box, dtype=float).reshape(-1, 2)
        left, top = float(points[:, 0].min()), float(points[:, 1].min())
        segments.append(
            {
                "text": text,
                "confidence": float(max(0.0, min(1.0, score))),
                "left": int(left),
                "top": int(top),
                "width": int(points[:, 0].max() - left),
                "height": int(points[:, 1].max() - top),
            }
        )
    return group_segments_into_lines(segments)


def extract_lines_with_boxes(
    image: OCRInput,
    lang: str | None = None,
    config: str = DEFAULT_CONFIG,
    timeout: float = DEFAULT_TIMEOUT,
    engine: str | None = None,
) -> list[dict[str, Any]]:
    """
    Görseldeki metni satır satır, kelime (parça) kutularıyla birlikte çıkarır.

    Args:
        engine: 'rapidocr' veya 'tesseract'; verilmezse get_engine_name() seçer.
        lang, config, timeout: yalnızca Tesseract için geçerlidir.

    Returns:
        [{"text": ..., "confidence": 0-1, "words": [{"text", "left", "top", "width", "height"}]}]
    """
    if (engine or get_engine_name()) == ENGINE_RAPIDOCR:
        return _rapidocr_lines_with_boxes(image)
    return _tesseract_lines_with_boxes(image, lang=lang, config=config, timeout=timeout)


def _tesseract_lines_with_boxes(
    image: OCRInput, lang: str | None = None, config: str = DEFAULT_CONFIG, timeout: float = DEFAULT_TIMEOUT
) -> list[dict[str, Any]]:
    """
    Görseldeki metni satır satır, kelime koordinatlarıyla birlikte çıkarır.

    Returns:
        Yukarıdan aşağı sıralı satır listesi:
        [{"text": "KART NO 5400 ...", "confidence": 0.91,
          "words": [{"text": "KART", "left": 10, "top": 40, "width": 60, "height": 22}, ...]}]
        Görselde metin yoksa boş liste döner.

    Raises:
        OCRTimeoutError: Tesseract 'timeout' saniyede bitmezse.
        OCRProcessingError: Tesseract bulunamazsa veya çalışmazsa.
    """
    _configure_tesseract()
    ocr_image = _to_ocr_image(image)

    try:
        data = pytesseract.image_to_data(
            ocr_image,
            lang=lang or _get_lang(),
            config=config,
            output_type=pytesseract.Output.DICT,
            timeout=max(1, int(timeout)),
        )
    except pytesseract.TesseractNotFoundError as exc:
        raise OCRProcessingError(
            "Tesseract OCR bulunamadı. Lütfen Tesseract'ı kurun veya TESSERACT_CMD ayarlayın.",
            details=str(exc),
        ) from exc
    except RuntimeError as exc:
        if "timeout" in str(exc).lower():
            raise OCRTimeoutError("Fiş okuma zaman aşımına uğradı.", details=str(exc)) from exc
        raise OCRProcessingError("Fiş görselinden metin çıkarılamadı.", details=str(exc)) from exc
    except (pytesseract.TesseractError, OSError) as exc:
        raise OCRProcessingError("Fiş görselinden metin çıkarılamadı.", details=str(exc)) from exc

    # Kelimeleri (blok, paragraf, satır) anahtarına göre satırlarda topla
    grouped: dict[tuple[int, int, int], dict[str, Any]] = {}
    for idx, word in enumerate(data.get("text", [])):
        word = (word or "").strip()
        if not word:
            continue
        try:
            conf = float(data["conf"][idx])
        except (TypeError, ValueError):
            conf = -1.0
        if conf < 0:
            continue

        key = (data["block_num"][idx], data["par_num"][idx], data["line_num"][idx])
        entry = grouped.setdefault(key, {"words": [], "confs": [], "top": data["top"][idx]})
        entry["words"].append(
            {
                "text": word,
                "left": int(data["left"][idx]),
                "top": int(data["top"][idx]),
                "width": int(data["width"][idx]),
                "height": int(data["height"][idx]),
            }
        )
        entry["confs"].append(conf)
        entry["top"] = min(entry["top"], data["top"][idx])

    lines: list[dict[str, Any]] = []
    for entry in sorted(grouped.values(), key=lambda entry: entry["top"]):
        text = " ".join(word["text"] for word in entry["words"]).strip()
        if not text:
            continue
        confidence = round(sum(entry["confs"]) / len(entry["confs"]) / 100.0, 4)
        lines.append({"text": text, "confidence": max(0.0, min(1.0, confidence)), "words": entry["words"]})
    return lines


def extract_text(image: OCRInput, lang: str | None = None, config: str = DEFAULT_CONFIG) -> list[dict[str, Any]]:
    """
    Görseldeki metni satır satır çıkarır.

    Kişisel veriler (kart no, IBAN, TCKN, cep telefonu, e-posta, müşteri adı)
    pii_redactor ile maskelenir; dönen metinde ham hali bulunmaz.

    Args:
        image: Ön işlenmiş görsel (numpy dizisi), PIL görseli veya dosya yolu.
        lang: Tesseract dil kodu. Verilmezse OCR_LANG veya 'tur' kullanılır.
        config: Tesseract parametreleri.

    Returns:
        Yukarıdan aşağı sıralı satır listesi:
        [{"line_no": 0, "text": "ŞOK MARKET", "confidence": 0.93}, ...]
        confidence 0.0-1.0 aralığındadır (satırdaki kelimelerin ortalaması).

    Raises:
        OCRProcessingError: Tesseract çalışmazsa veya görselde metin bulunamazsa.
    """
    from ocr_ml.pii_redactor import redact_lines

    # preprocess_image en iyi okumayı zaten seçtiyse (PreparedImage) OCR'ı tekrar çalıştırma
    cached = getattr(image, "ocr_lines", None)
    raw_lines = cached if cached is not None else extract_lines_with_boxes(image, lang=lang, config=config)
    masked = redact_lines([line["text"] for line in raw_lines])
    lines = [
        {"line_no": idx, "text": text, "confidence": line["confidence"]}
        for idx, (line, text) in enumerate(zip(raw_lines, masked))
    ]
    if not lines:
        raise OCRProcessingError(
            "Fiş görselinde okunabilir metin bulunamadı. Lütfen daha net ve aydınlık bir fotoğraf yükleyin."
        )
    return lines


def extract_plain_text(image: OCRInput, lang: str | None = None) -> str:
    """Hata ayıklama için OCR çıktısını düz metin olarak döndürür (kişisel veriler maskeli)."""
    return "\n".join(line["text"] for line in extract_text(image, lang=lang))
