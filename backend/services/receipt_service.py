"""
Fiş Tarama ve Onaylama Servisi (receipt_service.py).

Bu modül; yüklenen fiş fotoğrafını OCR/ML boru hattına (image_processor ->
receipt_ocr -> receipt_parser -> receipt_validator -> categorizer_ml) ileterek
yapılandırılmış önizleme verisi üretir ve kullanıcı onayından sonra fişi,
kalemlerini ve ana harcama kaydını TEK BİR TRANSACTION içinde kaydeder.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Optional

from backend.decorators import execution_timer
from backend.schemas import (
    DEFAULT_CATEGORY,
    ImageProcessingError,
    OCRProcessingError,
    ReceiptSchema,
    ReceiptValidationError,
    validate_receipt_payload,
)
from backend.services.expense_service import PROJECT_ROOT, get_db_session


def _resolve_image_path(image_path: str | Path) -> Path:
    """Görsel dosyasının tam yolunu çözümler ve dosyanın varlığını kontrol eder."""
    if not image_path or not str(image_path).strip():
        raise ImageProcessingError("Fiş görseli dosya yolu boş olamaz.")

    candidate = Path(image_path)
    if not candidate.is_absolute():
        candidate = (PROJECT_ROOT / candidate).resolve()
    else:
        candidate = candidate.resolve()

    if not candidate.is_file():
        raise ImageProcessingError(f"Fiş görseli bulunamadı: '{image_path}'")

    return candidate


def _run_ocr_ml_pipeline(resolved_image: Path, relative_path: str) -> dict[str, Any]:
    """
    ocr_ml paketindeki 5 adımlı boru hattını sırasıyla çalıştırır.
    Eğer ocr_ml modülleri ekip arkadaşı tarafından henüz tamamlanmamışsa
    (fonksiyonlar None döndürüyorsa), sistemin uçtan uca test edilebilmesi için
    standart veri sözleşmesine uygun bir şablon sonuç üretir.
    """
    try:
        from ocr_ml import (  # type: ignore[import-not-found]
            categorizer_ml,
            image_processor,
            receipt_ocr,
            receipt_parser,
            receipt_validator,
        )
    except ImportError as exc:
        raise OCRProcessingError("OCR/ML modülleri yüklenemedi.", details=str(exc)) from exc

    # 1. Görüntü ön işleme
    processed_input: Any = str(resolved_image)
    if hasattr(image_processor, "preprocess_image"):
        pre_res = image_processor.preprocess_image(str(resolved_image))
        if pre_res is not None:
            processed_input = pre_res

    # 2. OCR ile metin ve güven skoru çıkarma
    ocr_lines: Optional[list[dict[str, Any]]] = None
    if hasattr(receipt_ocr, "extract_text"):
        ocr_lines = receipt_ocr.extract_text(processed_input)

    # 3. Metinlerden işletme, tarih, toplam ve ürünleri ayrıştırma
    parsed_data: Optional[dict[str, Any]] = None
    if ocr_lines is not None and hasattr(receipt_parser, "parse_receipt"):
        parsed_data = receipt_parser.parse_receipt(ocr_lines)

    # Eğer OCR/ML modülü henüz kodlanmamışsa (None döndürdüyse) örnek önizleme verisi döndür
    if not parsed_data or not isinstance(parsed_data, dict):
        parsed_data = {
            "merchant": "Örnek Market (OCR Bekleniyor)",
            "date": date.today().isoformat(),
            "total": 135.00,
            "image_path": relative_path,
            "items": [
                {
                    "name": "Süt 1L",
                    "quantity": 2.0,
                    "unit_price": 45.00,
                    "total_price": 90.00,
                    "category": "Gıda",
                    "confidence": 0.95,
                },
                {
                    "name": "Tam Buğday Ekmek",
                    "quantity": 3.0,
                    "unit_price": 15.00,
                    "total_price": 45.00,
                    "category": "Gıda",
                    "confidence": 0.92,
                },
            ],
        }

    # 4. Ürün kategorilerini ML ile tahmin et (kategori boşsa veya varsayılansa)
    if hasattr(categorizer_ml, "predict_category"):
        for item in parsed_data.get("items", []):
            if isinstance(item, dict):
                current_cat = str(item.get("category") or "").strip()
                if not current_cat or current_cat == DEFAULT_CATEGORY:
                    predicted = categorizer_ml.predict_category(str(item.get("name") or item.get("product_name") or ""))
                    if predicted and isinstance(predicted, str):
                        item["category"] = predicted.strip()

    # 5. Fiş doğrulayıcı (receipt_validator) kontrolü
    if hasattr(receipt_validator, "validate_receipt"):
        validation_res = receipt_validator.validate_receipt(parsed_data)
        if isinstance(validation_res, dict):
            if validation_res.get("is_valid") is False:
                err_msg = validation_res.get("error") or "Fiş verisi doğrulama kontrolünden geçemedi."
                raise ReceiptValidationError(str(err_msg))
            if "data" in validation_res and isinstance(validation_res["data"], dict):
                parsed_data = validation_res["data"]
            if "warnings" in validation_res and isinstance(validation_res["warnings"], list):
                parsed_data["warnings"] = validation_res["warnings"]

    parsed_data["image_path"] = relative_path
    return parsed_data


# ============================================================================
# 1. Fiş Tarama ve Önizleme (Veritabanına Doğrudan Kaydetmez!)
# ============================================================================

@execution_timer
def scan_receipt(image_path: str | Path | Any) -> dict[str, Any]:
    """
    OCR işlemini başlatır ve kullanıcı onay ekranında (Streamlit veya HTML)
    gösterilmek üzere yapılandırılmış fiş verisini döndürür.

    Hem kaydedilmiş dosya yolunu (str / Path) hem de doğrudan Streamlit
    st.file_uploader (UploadedFile) / Flask (FileStorage) dosya nesnesini kabul eder.

    Altın Kural: OCR sonucu doğrudan veritabanına kaydedilmez; önce kullanıcı
    onayına sunulur.
    """
    auto_saved_path: Optional[str] = None
    if not isinstance(image_path, (str, Path)):
        from backend.utils.helpers import remove_uploaded_file, save_uploaded_receipt

        auto_saved_path = save_uploaded_receipt(image_path)
        target_path: str | Path = auto_saved_path
    else:
        target_path = image_path

    try:
        resolved = _resolve_image_path(target_path)
        try:
            rel_path = resolved.relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            rel_path = resolved.as_posix()

        raw_receipt = _run_ocr_ml_pipeline(resolved, rel_path)
        validated_receipt = validate_receipt_payload(raw_receipt)
        if isinstance(raw_receipt.get("warnings"), list):
            validated_receipt["warnings"] = [
                str(w).strip() for w in raw_receipt["warnings"] if str(w).strip()
            ]
        return validated_receipt
    except Exception:
        if auto_saved_path is not None:
            from backend.utils.helpers import remove_uploaded_file

            remove_uploaded_file(auto_saved_path)
        raise


# ============================================================================
# 2. Kullanıcı Onayından Geçen Fişi Kaydetme (Tek Transaction & Rollback)
# ============================================================================

@execution_timer
def confirm_receipt(
    receipt_data: dict[str, Any],
    db_path: Optional[str | Path] = None,
) -> int:
    """
    Kullanıcının kontrol edip onayladığı fişi veritabanına kaydeder.

    Altın Kurallar:
    1. Fiş ('receipts'), ürünler ('receipt_items') ve ana harcama ('expenses')
       tek bir veritabanı transaction'ı içinde kaydedilir; hata olursa rollback yapılır.
    2. Fişin genel toplamı 'expenses' tablosuna tek bir ana gider kaydı olarak
       eklenir; ürünlerin toplamı ikinci kez genel giderlere eklenmez.

    Returns:
        Oluşturulan fişin ID'si (receipt_id).
    """
    receipt_schema = ReceiptSchema.from_dict(receipt_data)
    validated = receipt_schema.to_dict()

    # Önce database/crud.py içindeki create_receipt fonksiyonunu dene
    if db_path is None:
        try:
            from database import crud  # type: ignore[import-not-found]

            if hasattr(crud, "create_receipt"):
                ext_receipt_id = crud.create_receipt(validated)
                if isinstance(ext_receipt_id, int) and ext_receipt_id > 0:
                    return ext_receipt_id
        except ImportError:
            pass

    with get_db_session(db_path) as conn:
        # 1. Adım: receipts tablosuna fiş üst bilgisini kaydet
        cursor = conn.execute(
            """
            INSERT INTO receipts (merchant, receipt_date, total_amount, image_path)
            VALUES (?, ?, ?, ?);
            """,
            (
                validated["merchant"],
                validated["date"],
                validated["total"],
                validated.get("image_path"),
            ),
        )
        receipt_id = int(cursor.lastrowid or 0)

        # 2. Adım: receipt_items tablosuna fişteki her bir ürünü kaydet
        for item in validated["items"]:
            conn.execute(
                """
                INSERT INTO receipt_items (
                    receipt_id, product_name, quantity, unit_price, total_price, category
                )
                VALUES (?, ?, ?, ?, ?, ?);
                """,
                (
                    receipt_id,
                    item["name"],
                    item["quantity"],
                    item["unit_price"],
                    item["total_price"],
                    item["category"],
                ),
            )

        # 3. Adım: expenses tablosuna fişin genel toplamını TEK BİR ana harcama olarak kaydet
        description = f"{validated['merchant']} Fişi"
        conn.execute(
            """
            INSERT INTO expenses (date, description, category, amount, type, receipt_id)
            VALUES (?, ?, ?, ?, 'expense', ?);
            """,
            (
                validated["date"],
                description,
                validated["category"],
                validated["total"],
                receipt_id,
            ),
        )

        return receipt_id


def get_receipt_details(
    receipt_id: int,
    db_path: Optional[str | Path] = None,
) -> Optional[dict[str, Any]]:
    """
    Belirtilen fişin üst bilgilerini ve içindeki ürün kalemlerini birlikte döndürür.
    """
    try:
        clean_id = int(receipt_id)
    except (TypeError, ValueError) as exc:
        raise ReceiptValidationError("Geçersiz fiş numarası.") from exc

    with get_db_session(db_path) as conn:
        receipt_row = conn.execute(
            """
            SELECT id, merchant, receipt_date, total_amount, image_path
            FROM receipts
            WHERE id = ?;
            """,
            (clean_id,),
        ).fetchone()

        if not receipt_row:
            return None

        item_rows = conn.execute(
            """
            SELECT id, receipt_id, product_name, quantity, unit_price, total_price, category
            FROM receipt_items
            WHERE receipt_id = ?
            ORDER BY id ASC;
            """,
            (clean_id,),
        ).fetchall()

        receipt_dict = dict(receipt_row)
        receipt_dict["date"] = receipt_dict["receipt_date"]
        receipt_dict["total"] = receipt_dict["total_amount"]
        receipt_dict["items"] = [
            {
                "id": row["id"],
                "receipt_id": row["receipt_id"],
                "name": row["product_name"],
                "product_name": row["product_name"],
                "quantity": row["quantity"],
                "unit_price": row["unit_price"],
                "total_price": row["total_price"],
                "category": row["category"],
            }
            for row in item_rows
        ]
        return receipt_dict
