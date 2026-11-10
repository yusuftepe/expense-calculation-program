"""
Backend dekoratörleri (Uygulama kilidi kontrolü, merkezi hata yönetimi,
performans ölçümü ve geçici hata tekrar deneme).

Bu modül; COMP 303 dersinin 'Decorators' gereksinimini karşılar ve
Flask route'ları ile servis fonksiyonlarının güvenliğini sağlar.
"""

from __future__ import annotations

import functools
import logging
import time
from typing import Any, Callable, Optional, TypeVar

from flask import flash, has_request_context, jsonify, redirect, request, url_for

from backend.schemas import (
    AppBaseError,
    AuthenticationError,
    DatabaseError,
    ExpenseValidationError,
    ImageProcessingError,
    OCRProcessingError,
    ReceiptValidationError,
)

# Uygulama genelinde kullanılacak logger yapılandırması
logger = logging.getLogger("expense_tracker")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s in %(module)s.%(funcName)s: %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

F = TypeVar("F", bound=Callable[..., Any])


def _wants_json_response() -> bool:
    """İsteğin bir API/JSON yanıtı bekleyip beklemediğini belirler."""
    if not has_request_context():
        return False
    if request.path.startswith("/api/") or request.is_json or request.args.get("format") == "json":
        return True
    accept_header = request.headers.get("Accept", "")
    return "application/json" in accept_header and "text/html" not in accept_header


# ============================================================================
# 1. Uygulama Kilidi (Application Lock) Dekoratörü
# ============================================================================

def login_required(func: F) -> F:
    """
    Korumalı sayfalara ve API uç noktalarına şifre girilmeden erişilmesini engeller.
    Hareketsizlik süresi (5-10 dk) dolmuşsa oturumu otomatik kilitleyip
    Flask'ta '/login' ekranına yönlendirir, Streamlit/Python çağrılarında ise
    AuthenticationError fırlatır.
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        from backend.services.auth_service import is_session_unlocked

        if not is_session_unlocked():
            if not has_request_context():
                logger.info("Kilitli oturum erişim denemesi engellendi: %s", func.__name__)
                raise AuthenticationError(
                    "Uygulama kilitli veya oturum süresi doldu. Lütfen şifrenizi girin."
                )

            logger.info("Kilitli oturum erişim denemesi engellendi: %s", request.path)
            if _wants_json_response():
                return (
                    jsonify(
                        {
                            "status": "error",
                            "error_type": "AuthenticationRequired",
                            "message": "Uygulama kilitli veya oturum süresi doldu. Lütfen şifrenizi girin.",
                        }
                    ),
                    401,
                )

            flash("Devam etmek için lütfen uygulama şifrenizi girin.", "warning")
            return redirect(url_for("login", next=request.path))

        return func(*args, **kwargs)

    return wrapper  # type: ignore[return-value]


# ============================================================================
# 2. Merkezi Hata Yönetimi Dekoratörü
# ============================================================================

def handle_errors(redirect_endpoint: Optional[str] = None) -> Callable[[F], F]:
    """
    Route ve servis fonksiyonlarında oluşan özel hataları (ImageProcessingError,
    OCRProcessingError, ReceiptValidationError, DatabaseError vb.) yakalar.

    Kullanıcıya asla ham Python traceback veya dosya yolu göstermez; teknik
    detayları loglar ve anlaşılır bir hata mesajı döndürür.

    Args:
        redirect_endpoint: HTML isteklerinde hata sonrası yönlendirilecek Flask endpoint adı.
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except (
                ExpenseValidationError,
                ReceiptValidationError,
                ImageProcessingError,
            ) as exc:
                logger.warning(
                    "Doğrulama/Görsel hatası (%s): %s | Detay: %s",
                    exc.__class__.__name__,
                    exc.message,
                    exc.details,
                )
                return _format_error_response(exc.message, exc.__class__.__name__, 400, redirect_endpoint)
            except AuthenticationError as exc:
                logger.warning("Kimlik doğrulama hatası: %s", exc.message)
                return _format_error_response(exc.message, "AuthenticationError", 401, redirect_endpoint or "login")
            except OCRProcessingError as exc:
                logger.error("OCR işleme hatası: %s | Detay: %s", exc.message, exc.details)
                return _format_error_response(exc.message, "OCRProcessingError", 422, redirect_endpoint)
            except DatabaseError as exc:
                logger.error("Veritabanı hatası: %s | Detay: %s", exc.message, exc.details)
                return _format_error_response(
                    "Veritabanı işlemi sırasında bir sorun oluştu. Lütfen tekrar deneyin.",
                    "DatabaseError",
                    500,
                    redirect_endpoint,
                )
            except AppBaseError as exc:
                logger.error("Uygulama hatası: %s | Detay: %s", exc.message, exc.details)
                return _format_error_response(exc.message, exc.__class__.__name__, 400, redirect_endpoint)
            except Exception as exc:
                logger.exception("Beklenmeyen sistem hatası (%s): %s", func.__name__, exc)
                return _format_error_response(
                    "İşlem sırasında beklenmeyen bir hata oluştu.",
                    "InternalServerError",
                    500,
                    redirect_endpoint,
                )

        return wrapper  # type: ignore[return-value]

    return decorator


def _format_error_response(
    user_message: str,
    error_type: str,
    status_code: int,
    redirect_endpoint: Optional[str] = None,
) -> Any:
    """İstek türüne göre JSON yanıtı, Flash mesajlı yönlendirme veya Python sözlüğü üretir."""
    if not has_request_context():
        return {
            "status": "error",
            "error_type": error_type,
            "message": user_message,
            "status_code": status_code,
        }

    if _wants_json_response() or redirect_endpoint is None:
        return (
            jsonify(
                {
                    "status": "error",
                    "error_type": error_type,
                    "message": user_message,
                }
            ),
            status_code,
        )

    flash(user_message, "danger")
    return redirect(url_for(redirect_endpoint))


# ============================================================================
# 3. Performans Ölçüm ve Tekrar Deneme (Retry) Dekoratörleri
# ============================================================================

def execution_timer(func: F) -> F:
    """
    Fonksiyonun çalışma süresini milisaniye (ms) cinsinden ölçer ve loglar.
    Özellikle OCR tarama ve veri analizi işlemlerinin performans takibi için kullanılır.
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            logger.info("'%s' fonksiyonu %.2f ms içinde tamamlandı.", func.__name__, elapsed_ms)

    return wrapper  # type: ignore[return-value]


def retry_on_failure(
    max_attempts: int = 3,
    delay_seconds: float = 0.25,
    exceptions: tuple[type[BaseException], ...] = (DatabaseError, OSError),
) -> Callable[[F], F]:
    """
    Geçici veritabanı kilitlenmeleri veya ağ hatalarında fonksiyonu belirtilen
    sayıda otomatik olarak yeniden dener.
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exc: Optional[BaseException] = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    last_exc = exc
                    logger.warning(
                        "'%s' başarısız oldu (Deneme %d/%d): %s",
                        func.__name__,
                        attempt,
                        max_attempts,
                        exc,
                    )
                    if attempt < max_attempts:
                        time.sleep(delay_seconds)
            if last_exc is not None:
                raise last_exc
            return None

        return wrapper  # type: ignore[return-value]

    return decorator
