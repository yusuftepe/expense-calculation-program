"""Smart Expense Analyzer Backend Paketi."""

from typing import Any

__all__ = ["app", "create_app"]


def __getattr__(name: str) -> Any:
    """
    Streamlit veya diğer modüller 'backend.*' alt paketlerini import ettiğinde
    Flask uygulamasının gereksiz yere bellekte başlatılmasını önlemek için
    'app' ve 'create_app' nesnelerini yalnızca istendiğinde (lazy) yükler.
    """
    if name in ("app", "create_app"):
        from backend.main import app, create_app

        return app if name == "app" else create_app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
