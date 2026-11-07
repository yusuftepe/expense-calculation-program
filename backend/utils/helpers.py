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

from backend.schemas import ImageProcessingError, normalize_date

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


def validate_image_file(
    file_obj: FileStorage | BinaryIO | Any,
    max_size_bytes: int = DEFAULT_MAX_FILE_SIZE,
) -> None:
    """
    Yüklenen fiş görselinin varlığını, uzantısını ve dosya boyutunu doğrular.
    Geçersiz durumda ImageProcessingError fırlatır.
    """
    if file_obj is None:
        raise ImageProcessingError("Yüklenecek fiş görseli bulunamadı.")

    filename = getattr(file_obj, "filename", None)
    if not filename or not str(filename).strip():
        raise ImageProcessingError("Lütfen geçerli bir fiş fotoğrafı seçin.")

    if not allowed_file(str(filename)):
        raise ImageProcessingError(
            f"Geçersiz dosya formatı ('{filename}'). Lütfen JPG, PNG veya WEBP formatında bir görsel yükleyin."
        )

    # Akış (stream) üzerinden boyut kontrolü yap ve imleci başa sar
    stream = getattr(file_obj, "stream", file_obj)
    if hasattr(stream, "seek") and hasattr(stream, "tell"):
        current_pos = stream.tell()
        stream.seek(0, os.SEEK_END)
        file_size = stream.tell()
        stream.seek(current_pos)

        if file_size <= 0:
            raise ImageProcessingError("Yüklenen görsel dosyası boş.")
        if file_size > max_size_bytes:
            max_mb = round(max_size_bytes / (1024 * 1024), 1)
            raise ImageProcessingError(
                f"Dosya boyutu çok büyük. Maksimum izin verilen boyut: {max_mb} MB."
            )


def save_uploaded_receipt(
    file_obj: FileStorage,
    upload_dir: Optional[str | Path] = None,
    max_size_bytes: int = DEFAULT_MAX_FILE_SIZE,
) -> str:
    """
    Flask üzerinden gelen fiş görselini doğrular, güvenli bir isimle uploads/ klasörüne
    kaydeder ve veritabanında saklanacak göreli (relative) dosya yolunu döndürür.

    Args:
        file_obj: Flask request.files içinden gelen FileStorage nesnesi.
        upload_dir: Kaydedilecek klasör yolu (varsayılan: proje kökündeki uploads/).
        max_size_bytes: İzin verilen maksimum dosya boyutu (byte).

    Returns:
        Kaydedilen dosyanın göreli yolu (örn. 'uploads/receipt_20261106_233030_a1b2c3d4.jpg').
    """
    validate_image_file(file_obj, max_size_bytes=max_size_bytes)

    target_dir = Path(upload_dir) if upload_dir is not None else DEFAULT_UPLOAD_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    safe_name = generate_safe_filename(str(file_obj.filename))
    destination = target_dir / safe_name

    try:
        file_obj.save(destination)
    except OSError as exc:
        raise ImageProcessingError(
            "Fiş görseli diske kaydedilirken bir hata oluştu.", details=str(exc)
        ) from exc

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
