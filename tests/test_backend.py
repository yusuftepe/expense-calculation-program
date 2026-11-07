"""
Backend Servisleri, Veri Şemaları, Eşzamanlılık ve Route Testleri (tests/test_backend.py).

Bu test modülü; veri doğrulama şemalarını, harcama/bütçe servislerini,
fiş onaylama ve çifte sayım önleme (no double-counting) kuralını,
analitik raporlarını, paralel görev yöneticisini ve Flask uç noktalarını test eder.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest

from backend.concurrency_worker import BackgroundTaskManager
from backend.main import create_app
from backend.schemas import (
    ExpenseValidationError,
    ReceiptValidationError,
    validate_expense_payload,
    validate_receipt_payload,
)
from backend.scraper import ExchangeRateScraper
from backend.services.analytics_service import (
    get_category_expenses,
    get_dashboard_summary,
    get_monthly_expenses,
    get_product_and_merchant_insights,
)
from backend.services.auth_service import reset_failed_attempts
from backend.services.expense_service import (
    add_expense,
    delete_expense,
    get_all_expenses,
    set_monthly_budget,
)
from backend.services.receipt_service import (
    confirm_receipt,
    get_receipt_details,
    scan_receipt,
)
from backend.utils.helpers import allowed_file, format_currency, generate_safe_filename


@pytest.fixture()
def temp_db(tmp_path: Path) -> Path:
    """Her test için izole geçici SQLite veritabanı oluşturur."""
    reset_failed_attempts("local")
    reset_failed_attempts("127.0.0.1")
    return tmp_path / "test_backend.db"


@pytest.fixture()
def unlocked_client(temp_db: Path):
    """Oturum kilidi açılmış hazır Flask test istemcisi döndürür."""
    app = create_app(
        {
            "TESTING": True,
            "SECRET_KEY": "test-secret-key",
            "DB_PATH": str(temp_db),
        }
    )
    with app.test_client() as client:
        client.post("/login", json={"password": "12345678"})
        yield client


# ============================================================================
# 1. Veri Şeması (schemas.py) ve Yardımcı Fonksiyon (helpers.py) Testleri
# ============================================================================

def test_expense_schema_validation() -> None:
    """Tarih normalizasyonu ve virgüllü tutar dönüşümü çalışmalı, hatalı veriler reddedilmeli."""
    valid = validate_expense_payload(
        {
            "date": "06.11.2026",
            "description": "  Market Alışverişi  ",
            "category": "Gıda",
            "amount": "1.250,50".replace(".", ""),
            "type": "expense",
        }
    )
    assert valid["date"] == "2026-11-06"
    assert valid["description"] == "Market Alışverişi"
    assert valid["amount"] == 1250.50

    # Negatif veya sıfır tutar reddedilmeli
    with pytest.raises(ExpenseValidationError):
        validate_expense_payload(
            {"date": "2026-11-06", "description": "Hatalı", "category": "Gıda", "amount": -10}
        )

    # Geçersiz işlem türü reddedilmeli
    with pytest.raises(ExpenseValidationError):
        validate_expense_payload(
            {"date": "2026-11-06", "description": "Hatalı", "category": "Gıda", "amount": 50, "type": "borc"}
        )


def test_receipt_schema_and_primary_category() -> None:
    """Fiş şeması kalemleri doğrulamalı ve baskın kategoriyi doğru hesaplamalı."""
    receipt = validate_receipt_payload(
        {
            "merchant": "Migros",
            "date": "06/11/2026",
            "total": 850.0,
            "items": [
                {"name": "Süt", "quantity": 2, "unit_price": 45.0, "category": "Gıda", "confidence": 0.96},
                {"name": "Deterjan", "quantity": 1, "unit_price": 760.0, "category": "Temizlik", "confidence": 0.91},
            ],
        }
    )
    assert receipt["date"] == "2026-11-06"
    assert receipt["items"][0]["total_price"] == 90.0
    assert receipt["category"] == "Temizlik"  # 760 TL > 90 TL olduğu için baskın kategori Temizlik

    with pytest.raises(ReceiptValidationError):
        validate_receipt_payload({"merchant": "", "date": "2026-11-06", "total": 100, "items": []})


def test_helpers_file_and_currency_utilities() -> None:
    """Dosya uzantı kontrolü, güvenli ad üretimi ve para formatlama doğru çalışmalı."""
    assert allowed_file("fis_foto.JPG") is True
    assert allowed_file("zararli.exe") is False

    safe_name = generate_safe_filename("Türkçe Fiş Adı!.png")
    assert safe_name.startswith("receipt_")
    assert safe_name.endswith(".png")
    assert format_currency(1250.5) == "1.250,50 TL"


# ============================================================================
# 2. Harcama, Fiş ve Analitik Servis Testleri (Çifte Sayım Kuralı Dahil)
# ============================================================================

def test_receipt_confirmation_no_double_counting(temp_db: Path) -> None:
    """
    Altın Kural Testi: Fiş onaylandığında 'expenses' tablosuna yalnızca fişin genel
    toplamı (1 kayıt) eklenmeli; ürün kalemleri 'receipt_items' içinde saklanmalı
    ve genel giderlere ikinci kez eklenmemelidir.
    """
    add_expense(
        {"date": "2026-11-01", "description": "Maaş", "category": "Gelir", "amount": 30000, "type": "income"},
        db_path=temp_db,
    )
    add_expense(
        {"date": "2026-11-02", "description": "Akbil", "category": "Ulaşım", "amount": 200, "type": "expense"},
        db_path=temp_db,
    )

    receipt_id = confirm_receipt(
        {
            "merchant": "BİM",
            "date": "2026-11-06",
            "total": 500.0,
            "items": [
                {"name": "Süt", "quantity": 2, "unit_price": 50.0, "total_price": 100.0, "category": "Gıda"},
                {"name": "Peynir", "quantity": 2, "unit_price": 200.0, "total_price": 400.0, "category": "Gıda"},
            ],
        },
        db_path=temp_db,
    )
    assert receipt_id > 0

    # Toplam gider 200 (Ulaşım) + 500 (Fiş genel toplamı) = 700 TL olmalı (1200 TL değil!)
    summary = get_dashboard_summary(db_path=temp_db)
    assert summary["total_income"] == 30000.0
    assert summary["total_expenses"] == 700.0
    assert summary["balance"] == 29300.0

    # Fiş detayları ve ürün bazlı analizler doğru gelmeli
    details = get_receipt_details(receipt_id, db_path=temp_db)
    assert details is not None
    assert len(details["items"]) == 2

    insights = get_product_and_merchant_insights(db_path=temp_db)
    assert insights["top_merchants"][0]["merchant"] == "BİM"
    assert insights["top_products"][0]["name"] == "Peynir"


def test_budget_exceeded_alert_calculation(temp_db: Path) -> None:
    """Bütçe sınırı aşıldığında analitik servisi exceeded=True ve doğru kullanım yüzdesi döndürmeli."""
    set_monthly_budget(
        {"category": "Yemek", "monthly_limit": 1000.0, "month": 11, "year": 2026},
        db_path=temp_db,
    )
    add_expense(
        {"date": "2026-11-05", "description": "Restoran", "category": "Yemek", "amount": 1200.0, "type": "expense"},
        db_path=temp_db,
    )

    monthly = get_monthly_expenses(year=2026, month=11, db_path=temp_db)
    budget_info = monthly["budget_status"][0]
    assert budget_info["category"] == "Yemek"
    assert budget_info["exceeded"] is True
    assert budget_info["usage_percent"] == 120.0
    assert budget_info["remaining"] == -200.0


# ============================================================================
# 3. Eşzamanlılık (Concurrency) ve Scraper Testleri
# ============================================================================

def test_concurrency_worker_parallel_execution() -> None:
    """BackgroundTaskManager görevleri paralel çalıştırıp sonuçları sıralı döndürmeli."""
    manager = BackgroundTaskManager(max_workers=2)
    try:
        tasks = [
            (lambda a, b: a * b, (3, 4), {}),
            (lambda a, b: a + b, (10, 5), {}),
        ]
        results = manager.run_parallel_tasks(tasks)
        assert results[0]["status"] == "completed" and results[0]["result"] == 12
        assert results[1]["status"] == "completed" and results[1]["result"] == 15
    finally:
        manager.shutdown()


def test_scraper_xml_parsing_and_currency_conversion() -> None:
    """TCMB XML verisi doğru ayrıştırılmalı ve kur dönüşümleri çalışmalı."""
    xml_data = """<?xml version="1.0" encoding="UTF-8"?>
    <Tarih_Date>
        <Currency CurrencyCode="USD"><ForexSelling>35.0000</ForexSelling></Currency>
        <Currency CurrencyCode="EUR"><ForexSelling>38.5000</ForexSelling></Currency>
    </Tarih_Date>"""
    rates = ExchangeRateScraper.parse_tcmb_xml(xml_data)
    assert rates["USD"] == 35.0
    assert rates["EUR"] == 38.5


# ============================================================================
# 4. Flask Route / Entegrasyon Testleri (main.py)
# ============================================================================

def test_flask_expense_and_receipt_routes(unlocked_client, tmp_path: Path) -> None:
    """Flask üzerinden harcama ekleme, fiş tarama, fiş onaylama ve silme uçtan uca çalışmalı."""
    # 1. Manuel harcama ekle
    res_add = unlocked_client.post(
        "/expenses/add",
        json={
            "date": "2026-11-06",
            "description": "İnternet Faturası",
            "category": "Faturalar",
            "amount": 450.0,
            "type": "expense",
        },
    )
    assert res_add.status_code == 201
    expense_id = res_add.get_json()["expense_id"]

    # 2. Fiş görseli yükleyerek /receipts/scan uç noktasını çağır
    fake_image = io.BytesIO(b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xd9")
    res_scan = unlocked_client.post(
        "/receipts/scan?format=json",
        data={"receipt": (fake_image, "test_fis.jpg")},
        content_type="multipart/form-data",
    )
    assert res_scan.status_code == 200
    scanned_receipt = res_scan.get_json()["receipt"]
    assert "merchant" in scanned_receipt and "items" in scanned_receipt

    # 3. Taranan fiş henüz veritabanına kaydedilmemiş olmalı (toplam gider hâlâ 450.0 TL)
    res_before_confirm = unlocked_client.get("/api/analytics").get_json()
    assert res_before_confirm["summary"]["total_expenses"] == 450.0

    # 4. Fişi onayla (/receipts/confirm)
    res_confirm = unlocked_client.post("/receipts/confirm", json=scanned_receipt)
    assert res_confirm.status_code == 201

    # 5. Onay sonrası toplam gider artmış olmalı
    res_after_confirm = unlocked_client.get("/api/analytics").get_json()
    assert res_after_confirm["summary"]["total_expenses"] == 450.0 + float(scanned_receipt["total"])

    # 6. Manuel harcamayı sil
    res_del = unlocked_client.post(f"/expenses/delete/{expense_id}", json={})
    assert res_del.status_code == 200

    # Test sırasında uploads/ içine kaydedilen geçici görseli temizle
    saved_img_path = scanned_receipt.get("image_path")
    if saved_img_path:
        full_img = Path(__file__).resolve().parents[1] / saved_img_path
        if full_img.is_file():
            full_img.unlink()
