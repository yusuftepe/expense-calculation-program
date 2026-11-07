"""
Smart Expense Analyzer — Tek Flask Uygulama Giriş Noktası (backend/main.py).

Bu modül; Flask uygulamasını başlatır, HTML/Jinja2 şablonları ve JSON API
uç noktalarını tanımlar, gelen istekleri ilgili servislere yönlendirir ve
merkezi hata/oturum yönetimini uygular.
"""

from __future__ import annotations

import json
import os
import secrets
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from flask import (
    Flask,
    Response,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    url_for,
)

from backend.decorators import handle_errors, login_required, logger
from backend.schemas import (
    AuthenticationError,
    ExpenseValidationError,
    ImageProcessingError,
    ReceiptValidationError,
)
from backend.scraper import get_exchange_rates
from backend.services.analytics_service import (
    get_category_expenses,
    get_dashboard_summary,
    get_monthly_expenses,
    get_product_and_merchant_insights,
)
from backend.services.auth_service import (
    initialize_default_password,
    is_password_configured,
    is_session_unlocked,
    lock_session,
    unlock_session,
    update_password,
    verify_password,
)
from backend.services.expense_service import (
    add_expense,
    delete_expense,
    get_all_expenses,
    get_monthly_budgets,
    set_monthly_budget,
)
from backend.services.receipt_service import (
    confirm_receipt,
    get_receipt_details,
    scan_receipt,
)
from backend.utils.helpers import (
    DEFAULT_MAX_FILE_SIZE,
    format_currency,
    get_current_year_month,
    remove_uploaded_file,
    save_uploaded_receipt,
)

# .env dosyasındaki ortam değişkenlerini yükle
load_dotenv()

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
TEMPLATES_DIR: Path = PROJECT_ROOT / "frontend" / "templates"
STATIC_DIR: Path = PROJECT_ROOT / "frontend" / "static"


def _wants_json() -> bool:
    """İsteğin JSON yanıtı bekleyip beklemediğini kontrol eder."""
    if request.path.startswith("/api/") or request.is_json or request.args.get("format") == "json":
        return True
    accept = request.headers.get("Accept", "")
    return "application/json" in accept and "text/html" not in accept


def _render_or_fallback(template_name: str, page_title: str, **context: Any) -> str | Response:
    """
    Frontend geliştiricisi ilgili HTML şablonunu doldurduysa render_template()
    ile şablonu çalıştırır; şablon dosyası henüz 0 byte (boş) ise veya JSON
    istenmişse geliştirme/test sürecinin aksamaması için temiz bir yedek çıktı üretir.
    """
    if _wants_json():
        serializable = {
            k: v for k, v in context.items() if not callable(v)
        }
        return jsonify({"status": "ok", "page": page_title, "data": serializable})

    template_path = TEMPLATES_DIR / template_name
    if template_path.is_file() and template_path.stat().st_size > 0:
        return render_template(template_name, page_title=page_title, **context)

    summary_json = json.dumps(context, ensure_ascii=False, indent=2, default=str)
    return (
        f"<!DOCTYPE html><html lang='tr'><head><meta charset='utf-8'>"
        f"<title>Smart Expense Analyzer - {page_title}</title>"
        f"<style>body{{font-family:sans-serif;max-width:900px;margin:2rem auto;padding:0 1rem;background:#f8fafc;color:#0f172a;}}"
        f"nav a{{margin-right:1rem;color:#2563eb;text-decoration:none;font-weight:600;}}"
        f"pre{{background:#1e293b;color:#f8fafc;padding:1rem;border-radius:8px;overflow:auto;}}</style></head>"
        f"<body><nav><a href='/'>Dashboard</a><a href='/expenses'>Harcamalar</a>"
        f"<a href='/receipts/scan'>Fiş Tara</a><a href='/analytics'>Analiz & Raporlar</a>"
        f"<a href='/settings'>Ayarlar</a><a href='/logout'>Kilitle (Çıkış)</a></nav>"
        f"<h1>{page_title}</h1>"
        f"<p><em>Not: <code>frontend/templates/{template_name}</code> henüz boş olduğu için Backend veri önizlemesi gösteriliyor.</em></p>"
        f"<pre>{summary_json}</pre></body></html>"
    )


def _extract_payload() -> dict[str, Any]:
    """Hem HTML form verisini hem de JSON gövdesini tek tip sözlük olarak alır."""
    if request.is_json:
        data = request.get_json(silent=True)
        return dict(data) if isinstance(data, dict) else {}
    return dict(request.form)


def create_app(test_config: Optional[dict[str, Any]] = None) -> Flask:
    """
    Flask uygulama fabrikası (Application Factory).
    Kaynak kodda sabit SECRET_KEY veya parola barındırmaz; '.env' dosyasından
    okur veya çalışma zamanında kriptografik olarak rastgele üretir.
    """
    flask_app = Flask(
        __name__,
        template_folder=str(TEMPLATES_DIR),
        static_folder=str(STATIC_DIR),
    )

    # SECRET_KEY .env içinde yoksa çalışma anında rastgele 64 karakterlik güvenli anahtar üretilir
    env_secret_key = (os.getenv("SECRET_KEY") or "").strip()
    runtime_secret_key = env_secret_key if env_secret_key else secrets.token_hex(32)

    flask_app.config.from_mapping(
        SECRET_KEY=runtime_secret_key,
        MAX_CONTENT_LENGTH=DEFAULT_MAX_FILE_SIZE,
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        SESSION_COOKIE_SECURE=False,
        DB_PATH=os.getenv("DB_PATH", str(PROJECT_ROOT / "database" / "expenses.db")),
    )

    if test_config:
        flask_app.config.update(test_config)

    flask_app.jinja_env.filters["currency"] = format_currency

    def _db_path() -> Optional[str]:
        return flask_app.config.get("DB_PATH")

    @flask_app.before_request
    def _ensure_initialized() -> None:
        if not flask_app.config.get("_DB_INITIALIZED"):
            try:
                initial_pw = flask_app.config.get("INITIAL_PASSWORD")
                initialize_default_password(default_password=initial_pw, db_path=_db_path())
            except Exception as exc:
                logger.warning("Başlangıç şifre kontrolü atlandı: %s", exc)
            flask_app.config["_DB_INITIALIZED"] = True

    # ========================================================================
    # 1. Uygulama Kilidi (Application Lock: /login & /logout)
    # ========================================================================

    @flask_app.route("/login", methods=["GET", "POST"])
    @handle_errors(redirect_endpoint="login")
    def login() -> Any:
        password_ready = is_password_configured(db_path=_db_path())

        if request.method == "GET":
            if is_session_unlocked():
                return redirect(url_for("dashboard"))
            return _render_or_fallback(
                "login.html",
                "Uygulama Kilidi",
                is_first_setup=not password_ready,
            )

        payload = _extract_payload()
        password = str(payload.get("password") or "")
        client_ip = request.remote_addr or "local"

        # İlk kurulum: Henüz veritabanında şifre yoksa kullanıcının girdiği ilk şifreyi kaydet
        if not password_ready:
            initialize_default_password(default_password=password, db_path=_db_path())
            unlock_session()
            next_url = request.args.get("next") or payload.get("next") or url_for("dashboard")
            if _wants_json():
                return jsonify(
                    {
                        "status": "ok",
                        "message": "İlk uygulama şifreniz oluşturuldu ve giriş yapıldı.",
                        "redirect": next_url,
                    }
                )
            flash("Uygulama şifreniz oluşturuldu ve kilidiniz açıldı.", "success")
            return redirect(str(next_url))

        if verify_password(password, client_key=client_ip, db_path=_db_path()):
            unlock_session()
            next_url = request.args.get("next") or payload.get("next") or url_for("dashboard")
            if _wants_json():
                return jsonify({"status": "ok", "message": "Giriş başarılı.", "redirect": next_url})
            flash("Hoş geldiniz! Uygulama kilidi açıldı.", "success")
            return redirect(str(next_url))

        raise AuthenticationError("Uygulama şifresi hatalı. Lütfen tekrar deneyin.")

    @flask_app.route("/logout", methods=["GET", "POST"])
    def logout() -> Any:
        lock_session()
        if _wants_json():
            return jsonify({"status": "ok", "message": "Uygulama kilitlendi."})
        flash("Uygulama güvenli şekilde kilitlendi.", "info")
        return redirect(url_for("login"))

    # ========================================================================
    # 2. Ana Sayfa / Dashboard (GET /)
    # ========================================================================

    @flask_app.route("/", methods=["GET"])
    @login_required
    @handle_errors()
    def dashboard() -> Any:
        summary = get_dashboard_summary(db_path=_db_path())
        recent_expenses = get_all_expenses(db_path=_db_path())[:10]
        rates = get_exchange_rates()
        return _render_or_fallback(
            "dashboard.html",
            "Dashboard",
            summary=summary,
            recent_expenses=recent_expenses,
            exchange_rates=rates,
        )

    # ========================================================================
    # 3. Harcamalar Listesi, Ekleme ve Silme (/expenses)
    # ========================================================================

    @flask_app.route("/expenses", methods=["GET"])
    @login_required
    @handle_errors()
    def expenses_list() -> Any:
        category = request.args.get("category")
        start_date = request.args.get("start_date")
        end_date = request.args.get("end_date")
        tx_type = request.args.get("type")

        expenses = get_all_expenses(
            category=category,
            start_date=start_date,
            end_date=end_date,
            transaction_type=tx_type,
            db_path=_db_path(),
        )
        categories = list(get_category_expenses(db_path=_db_path()).keys())
        return _render_or_fallback(
            "expenses.html",
            "Harcama ve Gelir Listesi",
            expenses=expenses,
            categories=categories,
            filters={
                "category": category,
                "start_date": start_date,
                "end_date": end_date,
                "type": tx_type,
            },
        )

    @flask_app.route("/expenses/add", methods=["POST"])
    @login_required
    @handle_errors(redirect_endpoint="expenses_list")
    def expenses_add() -> Any:
        payload = _extract_payload()
        expense_id = add_expense(payload, db_path=_db_path())

        if _wants_json():
            return (
                jsonify(
                    {
                        "status": "ok",
                        "message": "Kayıt başarıyla eklendi.",
                        "expense_id": expense_id,
                    }
                ),
                201,
            )

        flash("Harcama/gelir kaydı başarıyla eklendi.", "success")
        return redirect(url_for("expenses_list"))

    @flask_app.route("/expenses/delete/<int:expense_id>", methods=["POST"])
    @login_required
    @handle_errors(redirect_endpoint="expenses_list")
    def expenses_delete(expense_id: int) -> Any:
        deleted = delete_expense(expense_id, db_path=_db_path())
        if not deleted:
            raise ExpenseValidationError(f"Silinecek kayıt bulunamadı (ID: {expense_id}).")

        if _wants_json():
            return jsonify({"status": "ok", "message": "Kayıt silindi.", "expense_id": expense_id})

        flash("Kayıt başarıyla silindi.", "success")
        return redirect(url_for("expenses_list"))

    # ========================================================================
    # 4. Fiş Tarama ve Kullanıcı Onayı (/receipts/scan & /receipts/confirm)
    # ========================================================================

    @flask_app.route("/receipts/scan", methods=["GET", "POST"])
    @login_required
    @handle_errors(redirect_endpoint="receipts_scan")
    def receipts_scan() -> Any:
        if request.method == "GET":
            pending = session.get("pending_receipt")
            return _render_or_fallback(
                "receipt_scanner.html",
                "Fiş Tarama ve Kontrol",
                receipt=pending,
            )

        uploaded_file = request.files.get("receipt")
        if uploaded_file is None:
            raise ImageProcessingError("Lütfen yüklenecek bir fiş fotoğrafı seçin ('receipt').")

        saved_path = save_uploaded_receipt(uploaded_file)
        try:
            scanned_data = scan_receipt(saved_path)
        except Exception:
            remove_uploaded_file(saved_path)
            raise

        session["pending_receipt"] = scanned_data

        if _wants_json():
            return jsonify(
                {
                    "status": "ok",
                    "message": "Fiş başarıyla okundu. Lütfen bilgileri kontrol edip onaylayın.",
                    "receipt": scanned_data,
                }
            )

        flash("Fiş okundu. Lütfen bilgileri kontrol edip onaylayın.", "info")
        for warning_msg in scanned_data.get("warnings") or []:
            flash(str(warning_msg), "warning")
        return _render_or_fallback(
            "receipt_scanner.html",
            "Fiş Tarama ve Kontrol",
            receipt=scanned_data,
        )

    @flask_app.route("/receipts/confirm", methods=["POST"])
    @login_required
    @handle_errors(redirect_endpoint="receipts_scan")
    def receipts_confirm() -> Any:
        if request.is_json:
            receipt_payload = request.get_json(silent=True) or {}
        else:
            form_data = dict(request.form)
            pending = session.get("pending_receipt") or {}

            items_raw = form_data.get("items_json")
            if items_raw:
                try:
                    parsed_items = json.loads(items_raw)
                except ValueError as exc:
                    raise ReceiptValidationError("Ürün kalemleri JSON formatı geçersiz.") from exc
            else:
                parsed_items = pending.get("items", [])

            receipt_payload = {
                "merchant": form_data.get("merchant") or pending.get("merchant"),
                "date": form_data.get("date") or pending.get("date"),
                "total": form_data.get("total") or pending.get("total"),
                "image_path": form_data.get("image_path") or pending.get("image_path"),
                "items": parsed_items,
            }

        receipt_id = confirm_receipt(receipt_payload, db_path=_db_path())
        session.pop("pending_receipt", None)

        if _wants_json():
            return (
                jsonify(
                    {
                        "status": "ok",
                        "message": "Fiş ve harcama kaydı başarıyla oluşturuldu.",
                        "receipt_id": receipt_id,
                    }
                ),
                201,
            )

        flash("Fiş onaylandı ve harcamalarınıza eklendi.", "success")
        return redirect(url_for("dashboard"))

    @flask_app.route("/receipts/<int:receipt_id>", methods=["GET"])
    @login_required
    @handle_errors(redirect_endpoint="expenses_list")
    def receipt_detail(receipt_id: int) -> Any:
        details = get_receipt_details(receipt_id, db_path=_db_path())
        if not details:
            raise ReceiptValidationError(f"Fiş kaydı bulunamadı (ID: {receipt_id}).")
        return jsonify({"status": "ok", "receipt": details})

    # ========================================================================
    # 5. Analitik Ekranı ve JSON API (/analytics & /api/analytics)
    # ========================================================================

    @flask_app.route("/analytics", methods=["GET"])
    @flask_app.route("/reports", methods=["GET"])
    @login_required
    @handle_errors()
    def analytics_page() -> Any:
        cur_year, cur_month = get_current_year_month()
        year = request.args.get("year", default=cur_year, type=int)
        month = request.args.get("month", default=cur_month, type=int)

        monthly_report = get_monthly_expenses(year=year, month=month, db_path=_db_path())
        category_report = get_category_expenses(db_path=_db_path())
        insights = get_product_and_merchant_insights(db_path=_db_path())

        return _render_or_fallback(
            "reports.html",
            "Harcama Analizleri ve Raporlar",
            monthly_report=monthly_report,
            category_report=category_report,
            insights=insights,
        )

    @flask_app.route("/api/analytics", methods=["GET"])
    @login_required
    @handle_errors()
    def api_analytics() -> Any:
        cur_year, cur_month = get_current_year_month()
        year = request.args.get("year", default=cur_year, type=int)
        month = request.args.get("month", default=cur_month, type=int)

        return jsonify(
            {
                "status": "ok",
                "summary": get_dashboard_summary(db_path=_db_path()),
                "monthly": get_monthly_expenses(year=year, month=month, db_path=_db_path()),
                "categories": get_category_expenses(db_path=_db_path()),
                "insights": get_product_and_merchant_insights(db_path=_db_path()),
                "exchange_rates": get_exchange_rates(),
            }
        )

    # ========================================================================
    # 6. Ayarlar: Şifre Değiştirme ve Bütçe Limitleri (/settings)
    # ========================================================================

    @flask_app.route("/settings", methods=["GET"])
    @login_required
    @handle_errors()
    def settings_page() -> Any:
        cur_year, cur_month = get_current_year_month()
        budgets = get_monthly_budgets(year=cur_year, month=cur_month, db_path=_db_path())
        return _render_or_fallback(
            "settings.html",
            "Uygulama ve Bütçe Ayarları",
            budgets=budgets,
            year=cur_year,
            month=cur_month,
        )

    @flask_app.route("/settings/password", methods=["POST"])
    @login_required
    @handle_errors(redirect_endpoint="settings_page")
    def settings_update_password() -> Any:
        payload = _extract_payload()
        old_pw = str(payload.get("old_password") or "")
        new_pw = str(payload.get("new_password") or "")

        update_password(old_pw, new_pw, db_path=_db_path())

        if _wants_json():
            return jsonify({"status": "ok", "message": "Uygulama şifresi başarıyla güncellendi."})

        flash("Uygulama şifreniz başarıyla güncellendi.", "success")
        return redirect(url_for("settings_page"))

    @flask_app.route("/settings/budget", methods=["POST"])
    @login_required
    @handle_errors(redirect_endpoint="settings_page")
    def settings_update_budget() -> Any:
        payload = _extract_payload()
        cur_year, cur_month = get_current_year_month()
        payload.setdefault("year", cur_year)
        payload.setdefault("month", cur_month)

        budget_id = set_monthly_budget(payload, db_path=_db_path())

        if _wants_json():
            return jsonify(
                {
                    "status": "ok",
                    "message": "Aylık bütçe sınırı kaydedildi.",
                    "budget_id": budget_id,
                }
            )

        flash("Aylık kategori bütçesi güncellendi.", "success")
        return redirect(url_for("settings_page"))

    return flask_app


# Doğrudan `flask --app backend/main.py run` veya `python -m backend.main` için uygulama nesnesi
app = create_app()

if __name__ == "__main__":
    debug_mode = os.getenv("FLASK_DEBUG", "True").lower() in ("1", "true", "yes")
    port = int(os.getenv("PORT", "5000"))
    app.run(host="127.0.0.1", port=port, debug=debug_mode)
