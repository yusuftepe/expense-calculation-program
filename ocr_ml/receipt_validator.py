"""
Fiş Verisi Doğrulama Modülü (receipt_validator.py).

Ayrıştırılan fiş verisinin tutarlılığını kontrol eder. Kullanıcı her fişi
onay ekranında görüp düzenleyebildiği için doğrulama hoşgörülüdür:
yalnızca kullanılamaz veride (toplam yok) fişi reddeder, diğer sorunları
'warnings' listesiyle kullanıcıya bildirir.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

from backend.schemas import DEFAULT_CATEGORY

SUM_TOLERANCE = 0.05          # Kuruş yuvarlama farkları için mutlak tolerans (TL)
SUM_TOLERANCE_RATIO = 0.01    # Büyük fişlerde %1 göreli tolerans
LOW_CONFIDENCE = 0.60
MAX_REASONABLE_TOTAL = 1_000_000.0
MAX_RECEIPT_AGE_DAYS = 730


def _to_float(value: Any) -> float | None:
    try:
        return round(float(value), 2)
    except (TypeError, ValueError):
        return None


def _clean_items(raw_items: Any, warnings: list[str]) -> list[dict[str, Any]]:
    """Adı boş veya tutarı geçersiz kalemleri ayıklar, eksik alanları tamamlar."""
    cleaned: list[dict[str, Any]] = []
    dropped = 0
    for item in raw_items or []:
        if not isinstance(item, dict):
            dropped += 1
            continue
        name = str(item.get("name") or item.get("product_name") or "").strip()
        total_price = _to_float(item.get("total_price"))
        unit_price = _to_float(item.get("unit_price"))
        quantity = _to_float(item.get("quantity")) or 1.0

        if total_price is None and unit_price is not None:
            total_price = round(unit_price * quantity, 2)
        if unit_price is None and total_price is not None:
            unit_price = round(total_price / quantity, 2) if quantity else total_price

        if not name or not total_price or total_price <= 0 or not unit_price or unit_price <= 0 or quantity <= 0:
            dropped += 1
            continue

        confidence = _to_float(item.get("confidence"))
        confidence = 1.0 if confidence is None else max(0.0, min(1.0, confidence))

        cleaned.append(
            {
                "name": name,
                "quantity": quantity,
                "unit_price": unit_price,
                "total_price": total_price,
                "category": str(item.get("category") or "").strip() or DEFAULT_CATEGORY,
                "confidence": confidence,
            }
        )
    if dropped:
        warnings.append(f"{dropped} satır geçersiz olduğu için ürün listesinden çıkarıldı.")
    return cleaned


def validate_receipt(receipt: dict[str, Any]) -> dict[str, Any]:
    """
    Fiş verisini doğrular ve temizlenmiş halini döndürür.

    Returns:
        {
            "is_valid": bool,
            "error": str | None,       # is_valid False ise neden
            "warnings": list[str],     # kullanıcıya gösterilecek uyarılar
            "data": dict               # temizlenmiş fiş verisi (warnings dahil)
        }
    """
    if not isinstance(receipt, dict):
        return {"is_valid": False, "error": "Fiş verisi bir sözlük (dict) olmalıdır.", "warnings": [], "data": {}}

    warnings: list[str] = list(receipt.get("warnings") or [])
    data = dict(receipt)

    items = _clean_items(receipt.get("items"), warnings)
    data["items"] = items

    total = _to_float(receipt.get("total") if receipt.get("total") is not None else receipt.get("total_amount"))
    if total is None or total <= 0:
        return {
            "is_valid": False,
            "error": "Fişin genel toplamı okunamadı veya geçersiz.",
            "warnings": warnings,
            "data": data,
        }
    data["total"] = total

    if total > MAX_REASONABLE_TOTAL:
        warnings.append("Genel toplam olağandışı yüksek; OCR hatası olabilir.")

    # Ürün toplamı ile genel toplam karşılaştırması
    if items:
        discount = _to_float(receipt.get("discount")) or 0.0
        items_sum = round(sum(item["total_price"] for item in items) - discount, 2)
        tolerance = max(SUM_TOLERANCE, total * SUM_TOLERANCE_RATIO)
        if abs(items_sum - total) > tolerance:
            warnings.append(
                f"Ürün tutarlarının toplamı ({items_sum:.2f}) genel toplamla ({total:.2f}) uyuşmuyor; "
                "eksik veya hatalı okunan satır olabilir."
            )
    else:
        warnings.append("Fişte ürün satırı okunamadı; yalnızca genel toplam kaydedilecek.")

    # Tarih makullük kontrolü
    raw_date = str(receipt.get("date") or "")
    try:
        receipt_date = date.fromisoformat(raw_date)
    except ValueError:
        receipt_date = None
    if receipt_date:
        today = date.today()
        if receipt_date > today + timedelta(days=1):
            warnings.append("Fiş tarihi gelecekte görünüyor; lütfen tarihi kontrol edin.")
        elif receipt_date < today - timedelta(days=MAX_RECEIPT_AGE_DAYS):
            warnings.append("Fiş tarihi 2 yıldan eski görünüyor; lütfen tarihi kontrol edin.")

    low_conf = [item["name"] for item in items if item["confidence"] < LOW_CONFIDENCE]
    if low_conf:
        warnings.append(f"Düşük güvenle okunan ürünler: {', '.join(low_conf[:5])}")

    # Tekrarlanan uyarıları sırayı koruyarak temizle
    data["warnings"] = list(dict.fromkeys(warnings))
    return {"is_valid": True, "error": None, "warnings": data["warnings"], "data": data}
