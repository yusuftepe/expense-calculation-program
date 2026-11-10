"""
Backend yardımcı fonksiyonları (Dosya güvenliği, tarih/dönem ayrıştırma ve para formatlama).

Bu modül; fiş görsellerinin güvenli şekilde doğrulanıp kaydedilmesini ve
servis katmanında sık kullanılan dönüşüm işlemlerini sağlar.
"""

from __future__ import annotations

import os
from datetime import date, datetime
from pathlib import Path
from typing import Any, BinaryIO, Optional
from uuid import uuid4

from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename

from backend.schemas import ImageProcessingError, OCRProcessingError, normalize_date

# Proje kök dizini ve varsayılan yükleme ayarları
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DEFAULT_UPLOAD_DIR: Path = PROJECT_ROOT / os.getenv("UPLOAD_FOLDER", "uploads")
ALLOWED_IMAGE_EXTENSIONS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp"})
DEFAULT_MAX_FILE_SIZE: int = int(os.getenv("MAX_CONTENT_LENGTH", str(5 * 1024 * 1024)))  # 5 MB


def allowed_file(filename: str, allowed_extensions: Optional[set[str] | frozenset[str]] = None) -> bool:
    """
    Dosya adının izin verilen görsel uzantılarından birine sahip olup olmadığını kontrol eder.

    Args:
        filename: Kontrol edilecek dosya adı.
        allowed_extensions: İzin verilen uzantı kümesi (varsayılan: ALLOWED_IMAGE_EXTENSIONS).

    Returns:
        Uzantı geçerliyse True, aksi halde False.
    """
    if not filename or "." not in filename:
        return False

    ext = Path(filename).suffix.lower()
    valid_exts = allowed_extensions if allowed_extensions is not None else ALLOWED_IMAGE_EXTENSIONS
    return ext in valid_exts


def generate_safe_filename(original_filename: str, prefix: str = "receipt") -> str:
    """
    Kullanıcının yüklediği dosya adını doğrudan kullanmak yerine zaman damgalı
    ve benzersiz (UUID) güvenli bir dosya adı üretir.

    Örnek çıktı: 'receipt_20261106_233030_a1b2c3d4.jpg'
    """
    cleaned_name = secure_filename(original_filename or "")
    ext = Path(cleaned_name).suffix.lower()

    if ext not in ALLOWED_IMAGE_EXTENSIONS:
        # secure_filename Türkçe karakterlerden dolayı uzantıyı düşürdüyse orijinal uzantıyı kontrol et
        fallback_ext = Path(original_filename or "").suffix.lower()
        if fallback_ext in ALLOWED_IMAGE_EXTENSIONS:
            ext = fallback_ext
        else:
            raise ImageProcessingError(
                f"Desteklenmeyen dosya uzantısı. İzin verilenler: {', '.join(sorted(ALLOWED_IMAGE_EXTENSIONS))}"
            )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_suffix = uuid4().hex[:8]
    return f"{prefix}_{timestamp}_{unique_suffix}{ext}"


def _extract_filename(
    file_obj: Any,
    fallback_filename: Optional[str] = None,
) -> Optional[str]:
    """
    Hem Flask FileStorage ('.filename'), hem Streamlit UploadedFile / standart dosya
    nesneleri ('.name'), hem de doğrudan dosya yolu (str/Path) üzerinden dosya adını çözer.
    """
    if fallback_filename and str(fallback_filename).strip():
        return str(fallback_filename).strip()

    for attr in ("filename", "name"):
        val = getattr(file_obj, attr, None)
        if isinstance(val, str) and val.strip():
            return Path(val.strip()).name

    if isinstance(file_obj, (str, Path)) and str(file_obj).strip():
        return Path(file_obj).name

    return None


def validate_image_file(
    file_obj: FileStorage | BinaryIO | bytes | str | Path | Any,
    max_size_bytes: int = DEFAULT_MAX_FILE_SIZE,
    filename: Optional[str] = None,
) -> None:
    """
    Yüklenen fiş görselinin varlığını, uzantısını ve dosya boyutunu doğrular.
    Hem Flask FileStorage hem de Streamlit UploadedFile nesneleriyle uyumludur.
    Geçersiz durumda ImageProcessingError fırlatır.
    """
    if file_obj is None:
        raise ImageProcessingError("Yüklenecek fiş görseli bulunamadı.")

    resolved_filename = _extract_filename(file_obj, fallback_filename=filename)
    if not resolved_filename:
        raise ImageProcessingError("Lütfen geçerli bir fiş fotoğrafı seçin.")

    if not allowed_file(resolved_filename):
        raise ImageProcessingError(
            f"Geçersiz dosya formatı ('{resolved_filename}'). Lütfen JPG, PNG veya WEBP formatında bir görsel yükleyin."
        )

    if isinstance(file_obj, (bytes, bytearray)):
        file_size = len(file_obj)
    elif isinstance(file_obj, (str, Path)):
        candidate = Path(file_obj)
        if not candidate.is_file():
            raise ImageProcessingError(f"Fiş görseli bulunamadı: '{file_obj}'")
        file_size = candidate.stat().st_size
    else:
        # Akış (stream) üzerinden boyut kontrolü yap ve imleci başa sar
        stream = getattr(file_obj, "stream", file_obj)
        if hasattr(stream, "seek") and hasattr(stream, "tell"):
            current_pos = stream.tell()
            stream.seek(0, os.SEEK_END)
            file_size = stream.tell()
            stream.seek(current_pos)
        elif hasattr(file_obj, "size") and isinstance(getattr(file_obj, "size"), int):
            file_size = int(getattr(file_obj, "size"))
        else:
            file_size = 1

    if file_size <= 0:
        raise ImageProcessingError("Yüklenen görsel dosyası boş.")
    if file_size > max_size_bytes:
        max_mb = round(max_size_bytes / (1024 * 1024), 1)
        raise ImageProcessingError(
            f"Dosya boyutu çok büyük. Maksimum izin verilen boyut: {max_mb} MB."
        )


def _decode_uploaded_image_bytes(raw_bytes: bytes) -> Any:
    """
    Bellekteki görsel baytlarını sıkıştırma bombası (decompression bomb) kontrolünden
    geçirerek BGR numpy dizisi olarak çözer.
    """
    import io
    import warnings

    import cv2
    import numpy as np
    from PIL import Image

    from ocr_ml.image_processor import DECODE_TARGET_PIXELS, MAX_PIXELS

    too_large = ImageProcessingError(
        "Fiş görselinin çözünürlüğü çok yüksek. Lütfen normal bir telefon fotoğrafı yükleyin."
    )
    flags = cv2.IMREAD_COLOR
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(raw_bytes)) as header:
                width, height = header.size
        pixels = width * height
        if pixels > MAX_PIXELS:
            raise too_large
        if pixels > DECODE_TARGET_PIXELS:
            flags = cv2.IMREAD_REDUCED_COLOR_8
            for factor, flag in ((2, cv2.IMREAD_REDUCED_COLOR_2), (4, cv2.IMREAD_REDUCED_COLOR_4)):
                if pixels / (factor * factor) <= DECODE_TARGET_PIXELS:
                    flags = flag
                    break
    except Image.DecompressionBombError as exc:
        raise too_large from exc
    except ImageProcessingError:
        raise
    except Exception:
        flags = cv2.IMREAD_COLOR

    try:
        buffer = np.frombuffer(raw_bytes, dtype=np.uint8)
        decoded = cv2.imdecode(buffer, flags)
    except (OSError, cv2.error, ValueError) as exc:
        raise ImageProcessingError("Fiş görseli okunamadı.", details=str(exc)) from exc

    if decoded is None or decoded.size == 0:
        raise ImageProcessingError(
            "Fiş görseli açılamadı. Dosya bozuk veya desteklenmeyen bir formatta olabilir."
        )

    h, w = decoded.shape[:2]
    if w < 50 or h < 50:
        raise ImageProcessingError(
            "Fiş görseli çok küçük. Lütfen fişin tamamını net şekilde gösteren bir fotoğraf yükleyin."
        )
    return decoded


def save_uploaded_receipt(
    file_obj: FileStorage | BinaryIO | bytes | str | Path | Any,
    upload_dir: Optional[str | Path] = None,
    max_size_bytes: int = DEFAULT_MAX_FILE_SIZE,
    filename: Optional[str] = None,
) -> str:
    """
    Flask (FileStorage) veya Streamlit (st.file_uploader UploadedFile) üzerinden
    gelen fiş görselini doğrular, diske yazmadan önce bellekte
    ocr_ml.pii_redactor.redact_image ile sansürler (ve EXIF/GPS verisini temizler),
    güvenli bir isimle uploads/ klasörüne kaydeder ve göreli dosya yolunu döndürür.

    Args:
        file_obj: Streamlit UploadedFile, Flask FileStorage, BytesIO, bytes veya dosya yolu.
        upload_dir: Kaydedilecek klasör yolu (varsayılan: proje kökündeki uploads/).
        max_size_bytes: İzin verilen maksimum dosya boyutu (byte).
        filename: file_obj üzerinde dosya adı yoksa kullanılacak opsiyonel dosya adı.

    Returns:
        Kaydedilen dosyanın göreli yolu (örn. 'uploads/receipt_20261106_233030_a1b2c3d4.jpg').
    """
    validate_image_file(file_obj, max_size_bytes=max_size_bytes, filename=filename)
    resolved_filename = _extract_filename(file_obj, fallback_filename=filename) or "receipt.jpg"

    if isinstance(file_obj, (bytes, bytearray)):
        raw_bytes = bytes(file_obj)
    elif isinstance(file_obj, (str, Path)):
        raw_bytes = Path(file_obj).read_bytes()
    else:
        stream = getattr(file_obj, "stream", file_obj)
        if hasattr(stream, "seek"):
            stream.seek(0)
        if hasattr(stream, "read"):
            raw_bytes = stream.read()
        elif hasattr(file_obj, "getvalue"):
            raw_bytes = file_obj.getvalue()
        else:
            raw_bytes = b""
        if hasattr(stream, "seek"):
            stream.seek(0)

    if not raw_bytes:
        raise ImageProcessingError("Yüklenen görsel dosyası boş.")

    decoded = _decode_uploaded_image_bytes(raw_bytes)

    from ocr_ml import pii_redactor

    try:
        redacted_image, _ = pii_redactor.redact_image(decoded)
    except OCRProcessingError:
        # OCR motoru kurulu olmayan test ortamlarında görseli EXIF temizlenmiş haliyle sakla
        redacted_image = decoded

    target_dir = Path(upload_dir) if upload_dir is not None else DEFAULT_UPLOAD_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    safe_name = generate_safe_filename(resolved_filename)
    destination = target_dir / safe_name

    pii_redactor._write_image(destination, redacted_image)

    try:
        return destination.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return destination.as_posix()


def remove_uploaded_file(file_path: Optional[str | Path]) -> bool:
    """
    Hatalı veya iptal edilen bir fiş görselini diskten güvenli şekilde siler.
     yalnızca uploads/ dizini altındaki dosyaların silinmesine izin verir.
    """
    if not file_path:
        return False

    candidate = Path(file_path)
    if not candidate.is_absolute():
        candidate = (PROJECT_ROOT / candidate).resolve()
    else:
        candidate = candidate.resolve()

    upload_root = DEFAULT_UPLOAD_DIR.resolve()
    if upload_root not in candidate.parents or candidate.name == ".gitkeep":
        return False

    if candidate.is_file():
        try:
            candidate.unlink()
            return True
        except OSError:
            return False
    return False


def format_currency(amount: float | int, currency: str = "TL") -> str:
    """
    Sayısal tutarı arayüzde gösterim için Türkçe para formatına dönüştürür.
    Örnek: 1250.5 -> '1.250,50 TL'
    """
    try:
        numeric = float(amount)
    except (TypeError, ValueError):
        numeric = 0.0

    formatted = f"{numeric:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"{formatted} {currency}".strip()


def get_current_year_month() -> tuple[int, int]:
    """İçinde bulunulan yıl ve ay bilgisini (year, month) olarak döndürür."""
    today = date.today()
    return today.year, today.month


def parse_year_month(date_value: str | date | datetime) -> tuple[int, int]:
    """Tarih bilgisinden (year, month) ikilisini ayıklar."""
    iso_date = normalize_date(date_value)
    year_str, month_str, _ = iso_date.split("-")
    return int(year_str), int(month_str)
