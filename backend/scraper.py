"""
Döviz Kuru ve Finansal Referans Veri Çekici (scraper.py).

Bu modül; COMP 303 dersinin 'Web Scraping / Harici Veri Çekme' ve 'OOP'
gereksinimlerini karşılar. TCMB günlük döviz kurlarını XML üzerinden çeker,
önbelleğe (TTL cache) alır ve çevrimdışı (offline) durumlarda uygulamanın
kesintisiz çalışması için güvenli yedek (fallback) kurları kullanır.
"""

from __future__ import annotations

import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

from backend.decorators import execution_timer, logger

TCMB_TODAY_XML_URL: str = "https://www.tcmb.gov.tr/kurlar/today.xml"

# İnternet bağlantısı olmadığında veya sunucu yanıt vermediğinde kullanılacak yedek kurlar
FALLBACK_EXCHANGE_RATES: dict[str, float] = {
    "TRY": 1.0,
    "USD": 34.50,
    "EUR": 37.20,
    "GBP": 44.10,
}


@dataclass
class ExchangeRateSnapshot:
    """Belirli bir anda çekilen döviz kuru verilerini temsil eden veri sınıfı."""

    base_currency: str
    rates: dict[str, float]
    source: str
    updated_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_currency": self.base_currency,
            "rates": dict(self.rates),
            "source": self.source,
            "updated_at": self.updated_at,
        }


class ExchangeRateScraper:
    """
    TCMB XML servisinden güncel döviz kurlarını çeken, önbellekte tutan
    ve TL <-> Döviz dönüşümlerini hesaplayan sınıf.
    """

    def __init__(
        self,
        url: str = TCMB_TODAY_XML_URL,
        cache_ttl_seconds: int = 3600,
        timeout_seconds: float = 3.0,
    ) -> None:
        self.url = url
        self.cache_ttl_seconds = max(0, int(cache_ttl_seconds))
        self.timeout_seconds = float(timeout_seconds)
        self._cached_snapshot: Optional[ExchangeRateSnapshot] = None
        self._cache_timestamp: float = 0.0

    def _is_cache_valid(self) -> bool:
        if self._cached_snapshot is None:
            return False
        return (time.time() - self._cache_timestamp) < self.cache_ttl_seconds

    @staticmethod
    def parse_tcmb_xml(xml_content: str | bytes) -> dict[str, float]:
        """
        TCMB today.xml içeriğini ayrıştırarak {'USD': ..., 'EUR': ..., 'GBP': ...}
        sözlüğünü döndürür.
        """
        rates: dict[str, float] = {"TRY": 1.0}
        root = ET.fromstring(xml_content)

        for currency_node in root.findall("Currency"):
            code = (currency_node.attrib.get("CurrencyCode") or "").strip().upper()
            if code not in ("USD", "EUR", "GBP"):
                continue

            # Önce ForexSelling (Döviz Satış), yoksa BanknoteSelling değerini al
            raw_val = (
                currency_node.findtext("ForexSelling")
                or currency_node.findtext("BanknoteSelling")
                or ""
            ).strip()

            if raw_val:
                try:
                    rates[code] = round(float(raw_val.replace(",", ".")), 4)
                except ValueError:
                    continue

        if len(rates) <= 1:
            raise ValueError("XML içeriğinden geçerli döviz kuru ayrıştırılamadı.")

        return rates

    @execution_timer
    def get_latest_rates(self, force_refresh: bool = False) -> dict[str, Any]:
        """
        Güncel döviz kurlarını döndürür. Önbellek geçerliyse ağ isteği yapmadan
        önbellekten döner; ağ hatası durumunda güvenli yedek kurları kullanır.
        """
        if not force_refresh and self._is_cache_valid() and self._cached_snapshot is not None:
            return self._cached_snapshot.to_dict()

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            req = urllib.request.Request(
                self.url,
                headers={"User-Agent": "SmartExpenseAnalyzer/1.0"},
            )
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as response:
                xml_bytes = response.read()

            parsed_rates = self.parse_tcmb_xml(xml_bytes)
            for key, fallback_val in FALLBACK_EXCHANGE_RATES.items():
                parsed_rates.setdefault(key, fallback_val)

            snapshot = ExchangeRateSnapshot(
                base_currency="TRY",
                rates=parsed_rates,
                source="TCMB",
                updated_at=now_str,
            )
        except (urllib.error.URLError, TimeoutError, ET.ParseError, ValueError, OSError) as exc:
            logger.info("Canlı döviz kuru alınamadı, yedek kurlar kullanılıyor: %s", exc)
            snapshot = ExchangeRateSnapshot(
                base_currency="TRY",
                rates=dict(FALLBACK_EXCHANGE_RATES),
                source="fallback",
                updated_at=now_str,
            )

        self._cached_snapshot = snapshot
        self._cache_timestamp = time.time()
        return snapshot.to_dict()

    def convert_to_try(self, amount: float, currency: str = "TRY") -> float:
        """Verilen döviz tutarını güncel kur üzerinden Türk Lirasına (TRY) çevirir."""
        clean_currency = (currency or "TRY").strip().upper()
        numeric_amount = float(amount)
        if clean_currency == "TRY":
            return round(numeric_amount, 2)

        rates = self.get_latest_rates()["rates"]
        rate = float(rates.get(clean_currency) or FALLBACK_EXCHANGE_RATES.get(clean_currency, 1.0))
        return round(numeric_amount * rate, 2)

    def convert_from_try(self, amount_try: float, target_currency: str = "USD") -> float:
        """Türk Lirası (TRY) tutarını hedef para birimine (USD, EUR, GBP) çevirir."""
        clean_target = (target_currency or "USD").strip().upper()
        numeric_try = float(amount_try)
        if clean_target == "TRY":
            return round(numeric_try, 2)

        rates = self.get_latest_rates()["rates"]
        rate = float(rates.get(clean_target) or FALLBACK_EXCHANGE_RATES.get(clean_target, 1.0))
        if rate <= 0:
            return 0.0
        return round(numeric_try / rate, 2)


# Uygulama genelinde kullanılacak varsayılan scraper nesnesi
currency_scraper = ExchangeRateScraper()


def get_exchange_rates(force_refresh: bool = False) -> dict[str, Any]:
    """Servis katmanı ve Dashboard için güncel döviz kuru bilgilerini döndürür."""
    return currency_scraper.get_latest_rates(force_refresh=force_refresh)
