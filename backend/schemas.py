"""
Ortak veri şemaları (Interface Contracts), veri doğrulama modelleri ve özel hata sınıfları.

Bu modül; Frontend, Backend, OCR/ML ve Database katmanları arasındaki
veri alışverişinin standart bir sözleşmeye (Data Contract) uymasını sağlar.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Literal, Optional

# İzin verilen işlem türleri ve varsayılan kategoriler
TransactionType = Literal["income", "expense"]
VALID_TRANSACTION_TYPES: tuple[str, ...] = ("income", "expense")
DEFAULT_CATEGORY: str = "Diğer"


# ============================================================================
# 1. Özel Hata Sınıfları (Custom Exceptions)
# ============================================================================

class AppBaseError(Exception):
    """Uygulama genelindeki tüm özel hatalar için temel sınıf."""

    def __init__(self, message: str, details: Optional[str] = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details


class ImageProcessingError(AppBaseError):
    """Fiş fotoğrafı okunamadığında, bozuk olduğunda veya işlenemediğinde fırlatılır."""


class OCRProcessingError(AppBaseError):
    """OCR motoru görselden metin çıkaramadığında fırlatılır."""


class ReceiptValidationError(AppBaseError):
    """Fiş verisi (tarih, tutar, ürünler) doğrulama kurallarından geçemediğinde fırlatılır."""


class ExpenseValidationError(AppBaseError):
    """Manuel gelir/gider veya bütçe verisi geçersiz olduğunda fırlatılır."""


class DatabaseError(AppBaseError):
    """SQLite veritabanı bağlantı, kayıt veya sorgu işlemlerinde hata oluştuğunda fırlatılır."""


class AuthenticationError(AppBaseError):
    """Uygulama kilidi (Application Lock) şifre doğrulaması başarısız olduğunda fırlatılır."""


# ============================================================================
# 2. Ortak Doğrulama Yardımcıları
# ============================================================================

def normalize_date(value: str | date | datetime) -> str:
    """
    Tarih bilgisini standart ISO formatına ('YYYY-MM-DD') dönüştürür.
    'YYYY-MM-DD', 'DD.MM.YYYY' ve 'DD/MM/YYYY' formatlarını destekler.
    """
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Tarih alanı boş bırakılamaz.")

    raw_date = value.strip()
    supported_formats = ("%Y-%m-%d", "%d.%m.%Y", "%d/%m/%Y", "%Y/%m/%d")

    for fmt in supported_formats:
        try:
            parsed = datetime.strptime(raw_date, fmt).date()
            return parsed.isoformat()
        except ValueError:
            continue

    raise ValueError(
        f"Geçersiz tarih formatı: '{raw_date}'. Beklenen format: YYYY-MM-DD (örn. 2026-11-06)."
    )


def parse_positive_float(value: Any, field_name: str, allow_zero: bool = False) -> float:
    """Sayısal değeri float tipine çevirir, 2 ondalık basamağa yuvarlar ve pozitiflik kontrolü yapar."""
    if value is None or (isinstance(value, str) and not value.strip()):
        raise ValueError(f"'{field_name}' alanı boş bırakılamaz.")

    if isinstance(value, str):
        # "125,50" gibi virgüllü girişleri "125.50" formatına normalize et
        cleaned = value.strip().replace("TL", "").replace("₺", "").strip().replace(",", ".")
    else:
        cleaned = value

    try:
        numeric_value = round(float(cleaned), 2)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"'{field_name}' alanı geçerli bir sayı olmalıdır: {value}") from exc

    if allow_zero:
        if numeric_value < 0:
            raise ValueError(f"'{field_name}' alanı negatif olamaz.")
    else:
        if numeric_value <= 0:
            raise ValueError(f"'{field_name}' alanı sıfırdan büyük olmalıdır.")

    return numeric_value


# ============================================================================
# 3. Veri Modelleri (Data Contracts / Dataclasses)
# ============================================================================

@dataclass
class ExpenseSchema:
    """
    Ana gelir/gider kaydı veri şeması (expenses tablosu ve API sözleşmesi).
    """

    date: str
    description: str
    category: str
    amount: float
    type: TransactionType = "expense"
    receipt_id: Optional[int] = None
    id: Optional[int] = None

    def __post_init__(self) -> None:
        try:
            self.date = normalize_date(self.date)
            self.amount = parse_positive_float(self.amount, "amount")
        except ValueError as exc:
            raise ExpenseValidationError(str(exc)) from exc

        self.description = str(self.description or "").strip()
        if not self.description:
            raise ExpenseValidationError("İşlem açıklaması ('description') boş bırakılamaz.")

        self.category = str(self.category or "").strip() or DEFAULT_CATEGORY

        normalized_type = str(self.type or "expense").strip().lower()
        if normalized_type not in VALID_TRANSACTION_TYPES:
            raise ExpenseValidationError(
                f"Geçersiz işlem türü: '{self.type}'. Yalnızca 'income' veya 'expense' olabilir."
            )
        self.type = normalized_type  # type: ignore[assignment]

        if self.receipt_id is not None:
            try:
                self.receipt_id = int(self.receipt_id)
            except (TypeError, ValueError) as exc:
                raise ExpenseValidationError("Geçersiz fiş kimliği ('receipt_id').") from exc

    def to_dict(self) -> dict[str, Any]:
        """Şemayı veritabanı ve JSON yanıtları için standart sözlüğe çevirir."""
        data: dict[str, Any] = {
            "date": self.date,
            "description": self.description,
            "category": self.category,
            "amount": self.amount,
            "type": self.type,
            "receipt_id": self.receipt_id,
        }
        if self.id is not None:
            data["id"] = self.id
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExpenseSchema:
        """Sözlük verisinden doğrulanmış ExpenseSchema nesnesi üretir."""
        if not isinstance(data, dict):
            raise ExpenseValidationError("Harcama verisi bir sözlük (dict) olmalıdır.")

        # 'description' veya 'title' anahtarlarının her ikisini de destekle
        description = data.get("description") or data.get("title") or ""
        return cls(
            id=data.get("id"),
            date=data.get("date", ""),
            description=description,
            category=data.get("category") or DEFAULT_CATEGORY,
            amount=data.get("amount"),  # type: ignore[arg-type]
            type=data.get("type", "expense"),
            receipt_id=data.get("receipt_id"),
        )


@dataclass
class ReceiptItemSchema:
    """
    Fiş içindeki tek bir ürün satırını temsil eden veri şeması (receipt_items tablosu).
    """

    name: str
    unit_price: float
    quantity: float = 1.0
    total_price: Optional[float] = None
    category: str = DEFAULT_CATEGORY
    confidence: float = 1.0
    id: Optional[int] = None
    receipt_id: Optional[int] = None

    def __post_init__(self) -> None:
        self.name = str(self.name or "").strip()
        if not self.name:
            raise ReceiptValidationError("Fiş kaleminde ürün adı ('name') boş olamaz.")

        try:
            self.quantity = parse_positive_float(self.quantity, "quantity")
            self.unit_price = parse_positive_float(self.unit_price, "unit_price")
            if self.total_price is None:
                self.total_price = round(self.quantity * self.unit_price, 2)
            else:
                self.total_price = parse_positive_float(self.total_price, "total_price")
        except ValueError as exc:
            raise ReceiptValidationError(str(exc)) from exc

        self.category = str(self.category or "").strip() or DEFAULT_CATEGORY

        try:
            conf = float(self.confidence if self.confidence is not None else 1.0)
        except (TypeError, ValueError) as exc:
            raise ReceiptValidationError("Güven skoru ('confidence') sayısal olmalıdır.") from exc

        if not (0.0 <= conf <= 1.0):
            raise ReceiptValidationError("Güven skoru ('confidence') 0.0 ile 1.0 arasında olmalıdır.")
        self.confidence = round(conf, 4)

    def to_dict(self) -> dict[str, Any]:
        """OCR/ML ve Database katmanlarıyla tam uyumlu sözlük döndürür."""
        data: dict[str, Any] = {
            "name": self.name,
            "product_name": self.name,  # Database tablosu (product_name) ile uyumluluk
            "quantity": self.quantity,
            "unit_price": self.unit_price,
            "total_price": self.total_price,
            "category": self.category,
            "confidence": self.confidence,
        }
        if self.id is not None:
            data["id"] = self.id
        if self.receipt_id is not None:
            data["receipt_id"] = self.receipt_id
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReceiptItemSchema:
        """Sözlük verisinden doğrulanmış ReceiptItemSchema nesnesi üretir."""
        if not isinstance(data, dict):
            raise ReceiptValidationError("Fiş ürün kalemi bir sözlük (dict) olmalıdır.")

        product_name = data.get("name") or data.get("product_name") or ""
        return cls(
            id=data.get("id"),
            receipt_id=data.get("receipt_id"),
            name=product_name,
            quantity=data.get("quantity", 1.0),
            unit_price=data.get("unit_price"),  # type: ignore[arg-type]
            total_price=data.get("total_price"),
            category=data.get("category") or DEFAULT_CATEGORY,
            confidence=data.get("confidence", 1.0),
        )


@dataclass
class ReceiptSchema:
    """
    OCR/ML modülünden dönen ve kullanıcı onayından sonra kaydedilen fiş şeması.
    """

    merchant: str
    date: str
    total: float
    items: list[ReceiptItemSchema] = field(default_factory=list)
    image_path: Optional[str] = None
    id: Optional[int] = None

    def __post_init__(self) -> None:
        self.merchant = str(self.merchant or "").strip()
        if not self.merchant:
            raise ReceiptValidationError("İşletme/market adı ('merchant') boş bırakılamaz.")

        try:
            self.date = normalize_date(self.date)
            self.total = parse_positive_float(self.total, "total")
        except ValueError as exc:
            raise ReceiptValidationError(str(exc)) from exc

        normalized_items: list[ReceiptItemSchema] = []
        for item in self.items:
            if isinstance(item, ReceiptItemSchema):
                normalized_items.append(item)
            elif isinstance(item, dict):
                normalized_items.append(ReceiptItemSchema.from_dict(item))
            else:
                raise ReceiptValidationError("Fiş ürünleri geçerli bir formatta değil.")
        self.items = normalized_items

    @property
    def primary_category(self) -> str:
        """
        Fişteki ürünlerin toplam tutarına göre baskın kategoriyi belirler.
        Ana harcama kaydı (expenses) oluşturulurken kullanılır.
        """
        if not self.items:
            return DEFAULT_CATEGORY

        category_totals: dict[str, float] = {}
        for item in self.items:
            cat = item.category or DEFAULT_CATEGORY
            category_totals[cat] = category_totals.get(cat, 0.0) + (item.total_price or 0.0)

        return max(category_totals, key=category_totals.get)  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, Any]:
        """Ortak veri sözleşmesine (receipt_data) uygun sözlük döndürür."""
        data: dict[str, Any] = {
            "merchant": self.merchant,
            "date": self.date,
            "receipt_date": self.date,  # Database tablosu (receipt_date) ile uyumluluk
            "total": self.total,
            "total_amount": self.total,  # Database tablosu (total_amount) ile uyumluluk
            "category": self.primary_category,
            "image_path": self.image_path,
            "items": [item.to_dict() for item in self.items],
        }
        if self.id is not None:
            data["id"] = self.id
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReceiptSchema:
        """Sözlük verisinden doğrulanmış ReceiptSchema nesnesi üretir."""
        if not isinstance(data, dict):
            raise ReceiptValidationError("Fiş verisi bir sözlük (dict) olmalıdır.")

        raw_date = data.get("date") or data.get("receipt_date") or ""
        raw_total = data.get("total") if data.get("total") is not None else data.get("total_amount")
        raw_items = data.get("items") or []

        return cls(
            id=data.get("id"),
            merchant=data.get("merchant", ""),
            date=raw_date,
            total=raw_total,  # type: ignore[arg-type]
            items=[
                item if isinstance(item, ReceiptItemSchema) else ReceiptItemSchema.from_dict(item)
                for item in raw_items
            ],
            image_path=data.get("image_path"),
        )


@dataclass
class BudgetSchema:
    """
    Aylık kategori bazlı bütçe limiti şeması (budgets tablosu).
    """

    category: str
    monthly_limit: float
    month: int
    year: int
    id: Optional[int] = None

    def __post_init__(self) -> None:
        self.category = str(self.category or "").strip()
        if not self.category:
            raise ExpenseValidationError("Bütçe kategorisi boş bırakılamaz.")

        try:
            self.monthly_limit = parse_positive_float(self.monthly_limit, "monthly_limit")
            self.month = int(self.month)
            self.year = int(self.year)
        except (TypeError, ValueError) as exc:
            raise ExpenseValidationError(f"Geçersiz bütçe verisi: {exc}") from exc

        if not (1 <= self.month <= 12):
            raise ExpenseValidationError("Ay ('month') değeri 1 ile 12 arasında olmalıdır.")
        if not (2000 <= self.year <= 2100):
            raise ExpenseValidationError("Yıl ('year') değeri geçerli bir yıl olmalıdır.")

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "category": self.category,
            "monthly_limit": self.monthly_limit,
            "month": self.month,
            "year": self.year,
        }
        if self.id is not None:
            data["id"] = self.id
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BudgetSchema:
        if not isinstance(data, dict):
            raise ExpenseValidationError("Bütçe verisi bir sözlük (dict) olmalıdır.")
        return cls(
            id=data.get("id"),
            category=data.get("category", ""),
            monthly_limit=data.get("monthly_limit"),  # type: ignore[arg-type]
            month=data.get("month"),  # type: ignore[arg-type]
            year=data.get("year"),  # type: ignore[arg-type]
        )


# ============================================================================
# 4. Servisler İçin Hızlı Doğrulama Fonksiyonları
# ============================================================================

def validate_expense_payload(data: dict[str, Any]) -> dict[str, Any]:
    """Ham harcama sözlüğünü doğrular ve standart sözlük formatında döndürür."""
    return ExpenseSchema.from_dict(data).to_dict()


def validate_receipt_payload(data: dict[str, Any]) -> dict[str, Any]:
    """Ham fiş sözlüğünü doğrular ve standart sözlük formatında döndürür."""
    return ReceiptSchema.from_dict(data).to_dict()


def validate_budget_payload(data: dict[str, Any]) -> dict[str, Any]:
    """Ham bütçe sözlüğünü doğrular ve standart sözlük formatında döndürür."""
    return BudgetSchema.from_dict(data).to_dict()
