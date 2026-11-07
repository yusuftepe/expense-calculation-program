"""
Uygulama Kilidi (Application Lock) ve Şifre Yönetimi Servisi.

Bu modül; tek kullanıcılı yerel uygulama şifresinin Werkzeug ile hashlenerek
SQLite ('app_settings' tablosu) üzerinde saklanmasını, doğrulanmasını,
değiştirilmesini, hatalı giriş sınırlandırmasını (rate-limit) ve oturum
(session) kilitleme işlemlerini yönetir.

Güvenlik Kuralı: Kaynak kod içerisinde hiçbir sabit şifre veya gizli anahtar
bulunmaz. Şifre yalnızca ilk kurulumda kullanıcı tarafından (veya yerel .env
üzerinden) belirlenir ve veritabanında sadece hash olarak saklanır.
"""

from __future__ import annotations

import os
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

from flask import session
from werkzeug.security import check_password_hash, generate_password_hash

from backend.schemas import AuthenticationError, DatabaseError

# Proje kök dizini ve varsayılan veritabanı yolu
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH: Path = PROJECT_ROOT / os.getenv("DB_PATH", "database/expenses.db")

# Uygulama kilidi güvenlik parametreleri (Kaynak kodda sabit şifre tutulmaz!)
PASSWORD_SETTING_KEY: str = "password_hash"
MIN_PASSWORD_LENGTH: int = 4
MAX_FAILED_ATTEMPTS: int = 5
LOCKOUT_DURATION_SECONDS: int = 60
DEFAULT_SESSION_TIMEOUT_MINUTES: int = int(os.getenv("SESSION_TIMEOUT_MINUTES", "10"))

# Hatalı giriş denemelerini takip eden bellek içi (in-memory) durum sözlüğü
_failed_attempts: dict[str, dict[str, float | int]] = {}


def _resolve_db_path(db_path: Optional[str | Path] = None) -> Path:
    """Veritabanı dosya yolunu çözümler ve üst klasörün varlığını garanti eder."""
    target = Path(db_path) if db_path is not None else DEFAULT_DB_PATH
    if not target.is_absolute():
        target = (PROJECT_ROOT / target).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


@contextmanager
def _get_db_connection(db_path: Optional[str | Path] = None) -> Iterator[sqlite3.Connection]:
    """
    Öncelikle database.db_connection.get_connection() fonksiyonunu kullanmayı dener;
    henüz tamamlanmamışsa doğrudan SQLite bağlantısı açar ve app_settings tablosunu hazırlar.
    """
    conn: Optional[sqlite3.Connection] = None
    try:
        if db_path is None:
            try:
                from database.db_connection import get_connection  # type: ignore[import-not-found]

                external_conn = get_connection()
                if isinstance(external_conn, sqlite3.Connection):
                    conn = external_conn
            except (ImportError, AttributeError, TypeError):
                conn = None

        if conn is None:
            resolved_path = _resolve_db_path(db_path)
            conn = sqlite3.connect(resolved_path)

        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS app_settings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                setting_key TEXT UNIQUE NOT NULL,
                setting_value TEXT NOT NULL
            );
            """
        )
        yield conn
        conn.commit()
    except sqlite3.Error as exc:
        if conn is not None:
            conn.rollback()
        raise DatabaseError("Veritabanı ayarları okunurken veya yazılırken hata oluştu.", details=str(exc)) from exc
    finally:
        if conn is not None:
            conn.close()


# ============================================================================
# 1. Şifre Hash Saklama, İlk Kurulum, Doğrulama ve Güncelleme
# ============================================================================

def get_stored_password_hash(db_path: Optional[str | Path] = None) -> Optional[str]:
    """app_settings tablosunda kayıtlı olan password_hash değerini döndürür."""
    with _get_db_connection(db_path) as conn:
        row = conn.execute(
            "SELECT setting_value FROM app_settings WHERE setting_key = ? LIMIT 1;",
            (PASSWORD_SETTING_KEY,),
        ).fetchone()
        return str(row["setting_value"]) if row else None


def is_password_configured(db_path: Optional[str | Path] = None) -> bool:
    """Veritabanında tanımlanmış bir uygulama şifresi olup olmadığını döndürür."""
    return bool(get_stored_password_hash(db_path=db_path))


def initialize_default_password(
    default_password: Optional[str] = None,
    db_path: Optional[str | Path] = None,
) -> bool:
    """
    Veritabanında henüz bir uygulama şifresi kayıtlı değilse, yalnızca parametre olarak
    verilen veya yerel '.env' dosyasındaki 'APP_LOCK_PASSWORD' değişkeninde tanımlanan
    şifreyi Werkzeug ile hashleyerek app_settings tablosuna kaydeder.

    Kaynak kod içinde sabit bir şifre bulunmaz; hem parametre hem de '.env' boşsa
    şifre oluşturmaz ve False döndürür (kullanıcı ilk açılış ekranında belirler).
    """
    if is_password_configured(db_path=db_path):
        return False

    raw_secret = default_password if default_password is not None else os.getenv("APP_LOCK_PASSWORD")
    if not raw_secret or not str(raw_secret).strip():
        return False

    initial_secret = str(raw_secret).strip()
    if len(initial_secret) < MIN_PASSWORD_LENGTH:
        raise AuthenticationError(
            f"Uygulama şifresi en az {MIN_PASSWORD_LENGTH} karakter olmalıdır."
        )

    password_hash = generate_password_hash(initial_secret)
    with _get_db_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO app_settings (setting_key, setting_value)
            VALUES (?, ?)
            ON CONFLICT(setting_key) DO UPDATE SET setting_value = excluded.setting_value;
            """,
            (PASSWORD_SETTING_KEY, password_hash),
        )
    return True


def _check_rate_limit(client_key: str) -> None:
    """Çok sayıda başarısız şifre denemesinde geçici bekleme süresini kontrol eder."""
    state = _failed_attempts.get(client_key)
    if not state:
        return

    locked_until = float(state.get("locked_until", 0.0))
    now = time.time()
    if locked_until > now:
        remaining = int(round(locked_until - now))
        raise AuthenticationError(
            f"Çok fazla hatalı şifre denemesi yapıldı. Lütfen {remaining} saniye sonra tekrar deneyin."
        )

    if locked_until > 0 and now >= locked_until:
        _failed_attempts.pop(client_key, None)


def _record_failed_attempt(client_key: str) -> None:
    """Başarısız giriş denemesini kaydeder ve sınır aşıldıysa geçici kilit uygular."""
    state = _failed_attempts.setdefault(client_key, {"count": 0, "locked_until": 0.0})
    count = int(state["count"]) + 1
    state["count"] = count

    if count >= MAX_FAILED_ATTEMPTS:
        state["locked_until"] = time.time() + LOCKOUT_DURATION_SECONDS


def reset_failed_attempts(client_key: str = "local") -> None:
    """Başarılı giriş sonrasında veya testlerde hatalı deneme sayacını sıfırlar."""
    _failed_attempts.pop(client_key, None)


def verify_password(
    plain_password: str,
    client_key: str = "local",
    db_path: Optional[str | Path] = None,
) -> bool:
    """
    Kullanıcının girdiği şifreyi app_settings tablosundaki hash ile karşılaştırır.
    Çok sayıda hatalı deneme varsa AuthenticationError fırlatır.
    """
    _check_rate_limit(client_key)

    if not plain_password or not isinstance(plain_password, str):
        _record_failed_attempt(client_key)
        return False

    stored_hash = get_stored_password_hash(db_path=db_path)
    if not stored_hash:
        # Yerel .env içinde APP_LOCK_PASSWORD tanımlıysa oradan yüklemeyi dene
        initialize_default_password(db_path=db_path)
        stored_hash = get_stored_password_hash(db_path=db_path)

    if not stored_hash:
        raise AuthenticationError(
            "Henüz bir uygulama şifresi belirlenmemiş. Lütfen önce ilk kurulum şifrenizi oluşturun."
        )

    is_correct = check_password_hash(stored_hash, plain_password)
    if is_correct:
        reset_failed_attempts(client_key)
        return True

    _record_failed_attempt(client_key)
    return False


def update_password(
    old_password: str,
    new_password: str,
    db_path: Optional[str | Path] = None,
) -> bool:
    """
    Eski şifreyi doğruladıktan sonra yeni uygulama şifresini hashleyerek kaydeder.
    """
    if not old_password or not new_password:
        raise AuthenticationError("Eski şifre ve yeni şifre alanları boş bırakılamaz.")

    cleaned_new = new_password.strip()
    if len(cleaned_new) < MIN_PASSWORD_LENGTH:
        raise AuthenticationError(
            f"Yeni şifre en az {MIN_PASSWORD_LENGTH} karakter uzunluğunda olmalıdır."
        )

    if not verify_password(old_password, db_path=db_path):
        raise AuthenticationError("Mevcut şifrenizi hatalı girdiniz.")

    new_hash = generate_password_hash(cleaned_new)
    with _get_db_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO app_settings (setting_key, setting_value)
            VALUES (?, ?)
            ON CONFLICT(setting_key) DO UPDATE SET setting_value = excluded.setting_value;
            """,
            (PASSWORD_SETTING_KEY, new_hash),
        )
    return True


# ============================================================================
# 2. Flask Session (Oturum Açma, Kilitleme ve Hareketsizlik Kontrolü)
# ============================================================================

def unlock_session() -> None:
    """Şifre doğrulandıktan sonra Flask oturumunun kilidini açar ve zaman damgasını günceller."""
    session["unlocked"] = True
    session["last_activity"] = time.time()


def lock_session() -> None:
    """Kullanıcı çıkış yaptığında veya süre dolduğunda uygulama oturumunu kilitler."""
    session.pop("unlocked", None)
    session.pop("last_activity", None)


def is_session_unlocked(timeout_minutes: int = DEFAULT_SESSION_TIMEOUT_MINUTES) -> bool:
    """
    Oturumun açık olup olmadığını ve hareketsizlik süresinin (5-10 dk) aşılıp aşılmadığını
    kontrol eder. Süre aşılmışsa oturumu otomatik kilitler.
    """
    if not session.get("unlocked"):
        return False

    last_activity = session.get("last_activity")
    now = time.time()

    if last_activity is not None:
        elapsed_seconds = now - float(last_activity)
        if elapsed_seconds > (timeout_minutes * 60):
            lock_session()
            return False

    session["last_activity"] = now
    return True
