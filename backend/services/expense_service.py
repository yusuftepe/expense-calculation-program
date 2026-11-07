"""
Harcama ve Bütçe Yönetimi Servisi (expense_service.py).

Bu modül; manuel gelir/gider ekleme, listeleme, filtreleme, silme ve
aylık kategori bütçesi tanımlama işlemlerini yönetir.
Database katmanı (database/crud.py) tamamlandığında doğrudan onu çağırır;
henüz boşsa kendi güvenli SQLite yedek (fallback) sorgularını çalıştırır.
"""

from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional

from backend.decorators import execution_timer
from backend.schemas import (
    BudgetSchema,
    DatabaseError,
    ExpenseSchema,
    ExpenseValidationError,
    normalize_date,
)

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH: Path = PROJECT_ROOT / os.getenv("DB_PATH", "database/expenses.db")


def _resolve_db_path(db_path: Optional[str | Path] = None) -> Path:
    target = Path(db_path) if db_path is not None else DEFAULT_DB_PATH
    if not target.is_absolute():
        target = (PROJECT_ROOT / target).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


@contextmanager
def get_db_session(db_path: Optional[str | Path] = None) -> Iterator[sqlite3.Connection]:
    """
    Veritabanı bağlantısı açar, yabancı anahtar (FOREIGN KEY) desteğini etkinleştirir
    ve 5 temel tablonun (expenses, receipts, receipt_items, budgets, app_settings)
    mevcut olduğundan emin olur. Hata durumunda otomatik rollback yapar.
    """
    conn: Optional[sqlite3.Connection] = None
    try:
        if db_path is None:
            try:
                from database.db_connection import get_connection  # type: ignore[import-not-found]

                ext_conn = get_connection()
                if isinstance(ext_conn, sqlite3.Connection):
                    conn = ext_conn
            except (ImportError, AttributeError, TypeError):
                conn = None

        if conn is None:
            conn = sqlite3.connect(_resolve_db_path(db_path))

        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        _ensure_tables_exist(conn)
        yield conn
        conn.commit()
    except sqlite3.Error as exc:
        if conn is not None:
            conn.rollback()
        raise DatabaseError("Veritabanı işlemi sırasında hata oluştu.", details=str(exc)) from exc
    finally:
        if conn is not None:
            conn.close()


def _ensure_tables_exist(conn: sqlite3.Connection) -> None:
    """Taslakta belirlenen 5 ana tabloyu (yoksa) oluşturur."""
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS receipts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            merchant TEXT NOT NULL,
            receipt_date TEXT NOT NULL,
            total_amount REAL NOT NULL CHECK(total_amount > 0),
            image_path TEXT
        );

        CREATE TABLE IF NOT EXISTS receipt_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            receipt_id INTEGER NOT NULL,
            product_name TEXT NOT NULL,
            quantity REAL NOT NULL DEFAULT 1.0 CHECK(quantity > 0),
            unit_price REAL NOT NULL CHECK(unit_price >= 0),
            total_price REAL NOT NULL CHECK(total_price >= 0),
            category TEXT NOT NULL,
            FOREIGN KEY (receipt_id) REFERENCES receipts(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS expenses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            description TEXT NOT NULL,
            category TEXT NOT NULL,
            amount REAL NOT NULL CHECK(amount > 0),
            type TEXT NOT NULL DEFAULT 'expense' CHECK(type IN ('income', 'expense')),
            receipt_id INTEGER,
            FOREIGN KEY (receipt_id) REFERENCES receipts(id) ON DELETE SET NULL
        );

        CREATE TABLE IF NOT EXISTS budgets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            category TEXT NOT NULL,
            monthly_limit REAL NOT NULL CHECK(monthly_limit > 0),
            month INTEGER NOT NULL CHECK(month BETWEEN 1 AND 12),
            year INTEGER NOT NULL CHECK(year >= 2000),
            UNIQUE(category, month, year)
        );

        CREATE TABLE IF NOT EXISTS app_settings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            setting_key TEXT UNIQUE NOT NULL,
            setting_value TEXT NOT NULL
        );
        """
    )


# ============================================================================
# 1. Harcama / Gelir CRUD İşlemleri (Taslak Sözleşmesi)
# ============================================================================

@execution_timer
def add_expense(data: dict[str, Any], db_path: Optional[str | Path] = None) -> int:
    """
    Yeni harcama veya gelir kaydı oluşturur ve oluşturulan kaydın ID'sini döndürür.

    Args:
        data: {'date', 'description', 'category', 'amount', 'type', 'receipt_id'} sözlüğü.
        db_path: Testler için opsiyonel veritabanı dosya yolu.

    Returns:
        Oluşturulan harcama kaydının ID'si (int).
    """
    validated = ExpenseSchema.from_dict(data).to_dict()

    # Önce database/crud.py içindeki create_expense fonksiyonunu dene
    if db_path is None:
        try:
            from database import crud  # type: ignore[import-not-found]

            if hasattr(crud, "create_expense"):
                result_id = crud.create_expense(validated)
                if isinstance(result_id, int) and result_id > 0:
                    return result_id
        except ImportError:
            pass

    with get_db_session(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO expenses (date, description, category, amount, type, receipt_id)
            VALUES (?, ?, ?, ?, ?, ?);
            """,
            (
                validated["date"],
                validated["description"],
                validated["category"],
                validated["amount"],
                validated["type"],
                validated.get("receipt_id"),
            ),
        )
        return int(cursor.lastrowid or 0)


@execution_timer
def get_all_expenses(
    category: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    transaction_type: Optional[str] = None,
    db_path: Optional[str | Path] = None,
) -> list[dict[str, Any]]:
    """
    Bütün harcamaları (veya tarih/kategori filtresine uyanları) tarihe göre
    yeniden eskiye sıralı olarak getirir.
    """
    records: Optional[list[dict[str, Any]]] = None

    if db_path is None and not any([category, start_date, end_date, transaction_type]):
        try:
            from database import crud  # type: ignore[import-not-found]

            if hasattr(crud, "get_expenses"):
                ext_records = crud.get_expenses()
                if isinstance(ext_records, list):
                    records = [dict(r) for r in ext_records]
        except ImportError:
            records = None

    if records is None:
        query = """
            SELECT id, date, description, category, amount, type, receipt_id
            FROM expenses
            WHERE 1=1
        """
        params: list[Any] = []

        if category and category.strip():
            query += " AND LOWER(category) = LOWER(?)"
            params.append(category.strip())

        if start_date and start_date.strip():
            query += " AND date >= ?"
            params.append(normalize_date(start_date))

        if end_date and end_date.strip():
            query += " AND date <= ?"
            params.append(normalize_date(end_date))

        if transaction_type and transaction_type.strip().lower() in ("income", "expense"):
            query += " AND type = ?"
            params.append(transaction_type.strip().lower())

        query += " ORDER BY date DESC, id DESC;"

        with get_db_session(db_path) as conn:
            rows = conn.execute(query, params).fetchall()
            records = [dict(row) for row in rows]

    return records


def get_expense_by_id(
    expense_id: int,
    db_path: Optional[str | Path] = None,
) -> Optional[dict[str, Any]]:
    """Tek bir harcama kaydını ID değerine göre döndürür."""
    try:
        clean_id = int(expense_id)
    except (TypeError, ValueError) as exc:
        raise ExpenseValidationError("Geçersiz harcama ID değeri.") from exc

    with get_db_session(db_path) as conn:
        row = conn.execute(
            """
            SELECT id, date, description, category, amount, type, receipt_id
            FROM expenses
            WHERE id = ?;
            """,
            (clean_id,),
        ).fetchone()
        return dict(row) if row else None


@execution_timer
def delete_expense(
    expense_id: int,
    db_path: Optional[str | Path] = None,
) -> bool:
    """
    Belirtilen ID'ye sahip harcama kaydını siler.

    Returns:
        Kayıt bulundu ve silindiyse True, bulunamadıysa False.
    """
    try:
        clean_id = int(expense_id)
    except (TypeError, ValueError) as exc:
        raise ExpenseValidationError("Silinecek harcama ID değeri geçersiz.") from exc

    if clean_id <= 0:
        raise ExpenseValidationError("Harcama ID değeri pozitif bir tam sayı olmalıdır.")

    if db_path is None:
        try:
            from database import crud  # type: ignore[import-not-found]

            if hasattr(crud, "delete_expense"):
                res = crud.delete_expense(clean_id)
                if isinstance(res, bool):
                    return res
        except ImportError:
            pass

    with get_db_session(db_path) as conn:
        cursor = conn.execute("DELETE FROM expenses WHERE id = ?;", (clean_id,))
        return (cursor.rowcount or 0) > 0


# ============================================================================
# 2. Bütçe (Budgets) Yönetimi İşlemleri
# ============================================================================

def set_monthly_budget(
    data: dict[str, Any],
    db_path: Optional[str | Path] = None,
) -> int:
    """
    Belirli bir kategori, ay ve yıl için aylık bütçe sınırı tanımlar veya günceller.
    """
    validated = BudgetSchema.from_dict(data).to_dict()

    with get_db_session(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO budgets (category, monthly_limit, month, year)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(category, month, year)
            DO UPDATE SET monthly_limit = excluded.monthly_limit;
            """,
            (
                validated["category"],
                validated["monthly_limit"],
                validated["month"],
                validated["year"],
            ),
        )
        return int(cursor.lastrowid or 0)


def get_monthly_budgets(
    year: int,
    month: int,
    db_path: Optional[str | Path] = None,
) -> list[dict[str, Any]]:
    """Belirtilen yıl ve aya ait kategori bütçe limitlerini getirir."""
    with get_db_session(db_path) as conn:
        rows = conn.execute(
            """
            SELECT id, category, monthly_limit, month, year
            FROM budgets
            WHERE year = ? AND month = ?
            ORDER BY category ASC;
            """,
            (int(year), int(month)),
        ).fetchall()
        return [dict(row) for row in rows]
