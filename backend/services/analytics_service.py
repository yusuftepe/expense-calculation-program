"""
Harcama, Bütçe ve Akıllı Ürün/Market Analitik Servisi (analytics_service.py).

Bu modül; Dashboard özet kartlarını (toplam gelir, gider, bakiye), aylık ve
kategori bazlı harcama dağılımlarını, bütçe aşım uyarılarını ve taslakta
belirtilen akıllı ürün/market analizlerini üretir.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from backend.decorators import execution_timer
from backend.services.expense_service import (
    get_all_expenses,
    get_db_session,
    get_monthly_budgets,
)
from backend.utils.helpers import get_current_year_month


# ============================================================================
# 1. Temel Sözleşme Fonksiyonları (Aylık ve Kategori Bazlı Harcamalar)
# ============================================================================

@execution_timer
def get_monthly_expenses(
    year: Optional[int] = None,
    month: Optional[int] = None,
    db_path: Optional[str | Path] = None,
) -> dict[str, Any]:
    """
    Belirtilen yıl ve ay için gelir, gider, bakiye ve günlük/kategori bazlı
    aylık gider analizini döndürür.

    Args:
        year: Yıl bilgisi (boşsa mevcut yıl).
        month: Ay bilgisi 1-12 (boşsa mevcut ay).
        db_path: Testler için opsiyonel veritabanı yolu.
    """
    cur_year, cur_month = get_current_year_month()
    target_year = int(year) if year is not None else cur_year
    target_month = int(month) if month is not None else cur_month
    month_prefix = f"{target_year:04d}-{target_month:02d}"

    with get_db_session(db_path) as conn:
        rows = conn.execute(
            """
            SELECT date, category, amount, type
            FROM expenses
            WHERE substr(date, 1, 7) = ?
            ORDER BY date ASC;
            """,
            (month_prefix,),
        ).fetchall()

    total_income = 0.0
    total_expenses = 0.0
    by_category: dict[str, float] = {}
    by_day: dict[str, float] = {}

    for row in rows:
        amt = float(row["amount"] or 0.0)
        if row["type"] == "income":
            total_income += amt
        else:
            total_expenses += amt
            cat = str(row["category"] or "Diğer")
            day = str(row["date"])
            by_category[cat] = round(by_category.get(cat, 0.0) + amt, 2)
            by_day[day] = round(by_day.get(day, 0.0) + amt, 2)

    sorted_categories = dict(
        sorted(by_category.items(), key=lambda item: item[1], reverse=True)
    )

    return {
        "year": target_year,
        "month": target_month,
        "period": month_prefix,
        "total_income": round(total_income, 2),
        "total_expenses": round(total_expenses, 2),
        "balance": round(total_income - total_expenses, 2),
        "by_category": sorted_categories,
        "daily_expenses": by_day,
        "budget_status": check_budget_status(
            year=target_year,
            month=target_month,
            category_spending=sorted_categories,
            db_path=db_path,
        ),
    }


@execution_timer
def get_category_expenses(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    db_path: Optional[str | Path] = None,
) -> dict[str, float]:
    """
    Kategori bazlı toplam giderleri büyükten küçüğe sıralı sözlük olarak döndürür.
    Örnek: {'Gıda': 1250.5, 'Ulaşım': 300.0}
    """
    expenses = get_all_expenses(
        start_date=start_date,
        end_date=end_date,
        transaction_type="expense",
        db_path=db_path,
    )

    totals: dict[str, float] = {}
    for item in expenses:
        cat = str(item.get("category") or "Diğer")
        amt = float(item.get("amount") or 0.0)
        totals[cat] = round(totals.get(cat, 0.0) + amt, 2)

    return dict(sorted(totals.items(), key=lambda kv: kv[1], reverse=True))


# ============================================================================
# 2. Dashboard Özeti, Aylık Trend ve Bütçe Aşım Uyarıları
# ============================================================================

def get_monthly_trend(
    limit_months: int = 6,
    db_path: Optional[str | Path] = None,
) -> dict[str, dict[str, float]]:
    """
    Son N ayın gelir ve gider toplamlarını kronolojik sırayla döndürür.
    Dashboard üzerindeki 'Aylık Harcama Grafiği' için kullanılır.
    """
    with get_db_session(db_path) as conn:
        rows = conn.execute(
            """
            SELECT substr(date, 1, 7) AS period,
                   type,
                   ROUND(SUM(amount), 2) AS total
            FROM expenses
            GROUP BY period, type
            ORDER BY period ASC;
            """
        ).fetchall()

    trend: dict[str, dict[str, float]] = {}
    for row in rows:
        period = str(row["period"])
        bucket = trend.setdefault(period, {"income": 0.0, "expense": 0.0})
        bucket[str(row["type"])] = float(row["total"] or 0.0)

    if limit_months > 0 and len(trend) > limit_months:
        last_keys = list(trend.keys())[-limit_months:]
        trend = {k: trend[k] for k in last_keys}

    return trend


def check_budget_status(
    year: int,
    month: int,
    category_spending: Optional[dict[str, float]] = None,
    db_path: Optional[str | Path] = None,
) -> list[dict[str, Any]]:
    """
    Belirlenen aylık bütçe limitleri ile gerçekleşen harcamaları karşılaştırır,
    kullanım yüzdesini ('usage_percent') ve bütçe aşımı ('exceeded') uyarılarını üretir.
    """
    budgets = get_monthly_budgets(year=year, month=month, db_path=db_path)
    if not budgets:
        return []

    if category_spending is None:
        month_prefix = f"{int(year):04d}-{int(month):02d}"
        with get_db_session(db_path) as conn:
            rows = conn.execute(
                """
                SELECT category, ROUND(SUM(amount), 2) AS spent
                FROM expenses
                WHERE type = 'expense' AND substr(date, 1, 7) = ?
                GROUP BY category;
                """,
                (month_prefix,),
            ).fetchall()
            category_spending = {str(r["category"]): float(r["spent"] or 0.0) for r in rows}

    status_list: list[dict[str, Any]] = []
    for b in budgets:
        cat = str(b["category"])
        limit_val = float(b["monthly_limit"])
        spent_val = round(float(category_spending.get(cat, 0.0)), 2)
        remaining = round(limit_val - spent_val, 2)
        usage_pct = round((spent_val / limit_val) * 100.0, 1) if limit_val > 0 else 0.0

        status_list.append(
            {
                "category": cat,
                "budget": limit_val,
                "spent": spent_val,
                "remaining": remaining,
                "exceeded": spent_val > limit_val,
                "usage_percent": usage_pct,
            }
        )

    return status_list


@execution_timer
def get_dashboard_summary(db_path: Optional[str | Path] = None) -> dict[str, Any]:
    """
    Dashboard ana sayfası için tüm özet metrikleri, kategori dağılımını,
    aylık trendi ve bütçe uyarılarını tek seferde döndürür.
    """
    with get_db_session(db_path) as conn:
        totals_row = conn.execute(
            """
            SELECT
                COALESCE(SUM(CASE WHEN type = 'income' THEN amount ELSE 0 END), 0.0) AS total_income,
                COALESCE(SUM(CASE WHEN type = 'expense' THEN amount ELSE 0 END), 0.0) AS total_expenses
            FROM expenses;
            """
        ).fetchone()

    total_income = round(float(totals_row["total_income"] if totals_row else 0.0), 2)
    total_expenses = round(float(totals_row["total_expenses"] if totals_row else 0.0), 2)
    balance = round(total_income - total_expenses, 2)

    cur_year, cur_month = get_current_year_month()
    current_month_data = get_monthly_expenses(year=cur_year, month=cur_month, db_path=db_path)

    return {
        "total_income": total_income,
        "total_expenses": total_expenses,
        "balance": balance,
        "category_breakdown": get_category_expenses(db_path=db_path),
        "monthly_trend": get_monthly_trend(db_path=db_path),
        "current_month": current_month_data,
        "budget_alerts": [
            item for item in current_month_data["budget_status"] if item["exceeded"]
        ],
    }


# ============================================================================
# 3. Akıllı Ürün ve Market Bazlı Harcama Analizi (Taslak Sayfa 22)
# ============================================================================

@execution_timer
def get_product_and_merchant_insights(
    top_n: int = 5,
    db_path: Optional[str | Path] = None,
) -> dict[str, Any]:
    """
    Fiş kalemlerinden ('receipt_items') ve fişlerden ('receipts') yararlanarak:
    - En çok para harcanan ilk N ürünü,
    - En çok alışveriş yapılan market/işletmeleri
    analiz eder.
    """
    with get_db_session(db_path) as conn:
        product_rows = conn.execute(
            """
            SELECT
                product_name AS name,
                category,
                ROUND(SUM(quantity), 2) AS total_quantity,
                ROUND(SUM(total_price), 2) AS total_spent,
                COUNT(*) AS purchase_count
            FROM receipt_items
            GROUP BY LOWER(product_name)
            ORDER BY total_spent DESC
            LIMIT ?;
            """,
            (int(top_n),),
        ).fetchall()

        merchant_rows = conn.execute(
            """
            SELECT
                merchant,
                COUNT(*) AS visit_count,
                ROUND(SUM(total_amount), 2) AS total_spent
            FROM receipts
            GROUP BY LOWER(merchant)
            ORDER BY total_spent DESC
            LIMIT ?;
            """,
            (int(top_n),),
        ).fetchall()

    return {
        "top_products": [dict(row) for row in product_rows],
        "top_merchants": [dict(row) for row in merchant_rows],
    }
