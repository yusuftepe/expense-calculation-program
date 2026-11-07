"""
Uygulama Kilidi (Application Lock) ve Kimlik Doğrulama Testleri (tests/test_auth.py).

Bu test modülü; şifre hashleme, doğrulama, şifre değiştirme, hatalı giriş
sınırlandırması (rate-limit) ve Flask oturum korumasını (@login_required) doğrular.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from backend.main import create_app
from backend.schemas import AuthenticationError
from backend.services.auth_service import (
    MAX_FAILED_ATTEMPTS,
    get_stored_password_hash,
    initialize_default_password,
    is_password_configured,
    reset_failed_attempts,
    update_password,
    verify_password,
)


@pytest.fixture()
def temp_db(tmp_path: Path) -> Path:
    """Her test için izole, geçici bir SQLite veritabanı yolu üretir."""
    reset_failed_attempts("local")
    reset_failed_attempts("127.0.0.1")
    return tmp_path / "test_auth.db"


@pytest.fixture()
def client(temp_db: Path):
    """Test veritabanına bağlı Flask test istemcisi döndürür."""
    app = create_app(
        {
            "TESTING": True,
            "DB_PATH": str(temp_db),
        }
    )
    with app.test_client() as test_client:
        yield test_client


def test_initialize_and_verify_password(temp_db: Path) -> None:
    """İlk kurulumda şifre oluşturulabilmeli, hash olarak saklanmalı ve doğrulanabilmeli."""
    assert is_password_configured(db_path=temp_db) is False

    created = initialize_default_password("TestParola123!", db_path=temp_db)
    assert created is True
    assert is_password_configured(db_path=temp_db) is True

    # İkinci kez çağrıldığında mevcut şifreyi ezmemeli
    created_again = initialize_default_password("BaskaSifre", db_path=temp_db)
    assert created_again is False

    stored_hash = get_stored_password_hash(db_path=temp_db)
    assert stored_hash is not None
    assert stored_hash != "TestParola123!"  # Düz metin olarak saklanmamalı

    assert verify_password("TestParola123!", db_path=temp_db) is True
    assert verify_password("yanlisSifre", db_path=temp_db) is False


def test_update_password_flow(temp_db: Path) -> None:
    """Eski şifre doğrulandığında yeni şifre kaydedilmeli, yanlış eski şifre reddedilmeli."""
    initialize_default_password("eskiSifre123", db_path=temp_db)

    with pytest.raises(AuthenticationError):
        update_password("hataliEski", "yeniSifre456", db_path=temp_db)

    updated = update_password("eskiSifre123", "yeniSifre456", db_path=temp_db)
    assert updated is True
    assert verify_password("eskiSifre123", db_path=temp_db) is False
    reset_failed_attempts()
    assert verify_password("yeniSifre456", db_path=temp_db) is True


def test_rate_limit_lockout_on_repeated_failures(temp_db: Path) -> None:
    """Çok sayıda hatalı şifre denemesinde geçici kilit (AuthenticationError) uygulanmalı."""
    initialize_default_password("dogruSifre", db_path=temp_db)
    client_key = "test_brute_force_client"
    reset_failed_attempts(client_key)

    for _ in range(MAX_FAILED_ATTEMPTS):
        assert verify_password("yanlis", client_key=client_key, db_path=temp_db) is False

    # Sınır aşıldıktan sonraki denemede doğru şifre girilse bile kilit hatası vermeli
    with pytest.raises(AuthenticationError):
        verify_password("dogruSifre", client_key=client_key, db_path=temp_db)

    reset_failed_attempts(client_key)


def test_protected_routes_require_login(client) -> None:
    """Şifre girilmeden korumalı sayfalara ve API uçlarına erişilememeli."""
    res_html = client.get("/")
    assert res_html.status_code == 302
    assert "/login" in res_html.headers["Location"]

    res_api = client.get("/api/analytics")
    assert res_api.status_code == 401
    assert res_api.get_json()["error_type"] == "AuthenticationRequired"


def test_first_time_setup_and_login_logout_cycle(client) -> None:
    """
    Veritabanında henüz şifre yokken ilk /login isteği şifreyi oluşturmalı,
    sonraki girişlerde ise bu şifreyi doğrulamalıdır.
    """
    # 1. İlk kurulum: Kullanıcı kendi şifresini belirler
    first_setup = client.post("/login", json={"password": "BenimGizliSifrem"})
    assert first_setup.status_code == 200
    assert first_setup.get_json()["status"] == "ok"

    # 2. Korumalı API artık açık olmalı
    res_api = client.get("/api/analytics")
    assert res_api.status_code == 200

    # 3. Çıkış yap (Kilitle)
    res_logout = client.post("/logout", json={})
    assert res_logout.status_code == 200

    # 4. Yanlış şifre artık reddedilmeli
    bad_login = client.post("/login", json={"password": "hatali"})
    assert bad_login.status_code == 401

    # 5. İlk belirlenen şifre ile tekrar giriş yapılabilmeli
    ok_login = client.post("/login", json={"password": "BenimGizliSifrem"})
    assert ok_login.status_code == 200
