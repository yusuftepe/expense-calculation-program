"""
Fiş Ayrıştırma, Kategori Tahmini ve Doğrulama Testleri (tests/test_parser.py).

Bu testler Tesseract gerektirmez; OCR çıktısını taklit eden metin satırlarıyla
receipt_parser, categorizer_ml, receipt_validator ve text_utils modüllerini test eder.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from backend.schemas import DEFAULT_CATEGORY, OCRProcessingError, validate_receipt_payload
from ocr_ml.categorizer_ml import predict_category
from ocr_ml.receipt_parser import (
    UNKNOWN_MERCHANT,
    extract_date,
    extract_total,
    parse_receipt,
)
from ocr_ml.receipt_validator import validate_receipt
from ocr_ml.text_utils import find_trailing_amount, fold, parse_amount

SAMPLE_RECEIPT = """\
ŞOK MARKETLER TİC. A.Ş.
ATATÜRK MAH. CUMHURİYET CAD. NO:12
KADIKÖY V.D. 1234567890
TARİH : 05.11.2026   SAAT : 14:32
FİŞ NO : 0045
2 AD X 22,50
SÜT 1L           %1     *45,00
TAM BUĞDAY EKMEK %1     *15,00
0,750 KG X 60,00
DOMATES          %1     *45,00
ÇAMAŞIR DETERJANI %20   *189,90
KAŞAR PEYNİRİ    %1    *129,50
ANTEP FISTIĞI    %1     *89,90
TOPKDV                   *35,42
TOPLAM                  *514,30
NAKİT                   *600,00
PARA ÜSTÜ                *85,70
"""


# ============================================================================
# 1. Metin yardımcıları (text_utils.py)
# ============================================================================

@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("*45,00", 45.0),
        ("1.250,50", 1250.5),
        ("135.00 TL", 135.0),
        ("12,5", 12.5),
        ("45 ,00", 45.0),
        ("-5,00", -5.0),
        ("abc", None),
        ("", None),
    ],
)
def test_parse_amount(raw: str, expected: float | None) -> None:
    assert parse_amount(raw) == expected


def test_find_trailing_amount_ignores_dates_and_times() -> None:
    assert find_trailing_amount("TARİH : 05.11.2026")[0] is None
    assert find_trailing_amount("07.11.2026 14:32")[0] is None
    assert find_trailing_amount("DOMATES »1 *45 ,00") == (45.0, "DOMATES »1")


def test_find_trailing_amount_does_not_merge_quantity_into_price() -> None:
    """Boşluk binlik ayracı kabul edilmez: '1 330,00' -> 330.0 olmalı (1330.0 değil)."""
    assert find_trailing_amount("SU 1 330,00")[0] == 330.0


def test_fold_handles_turkish_case() -> None:
    assert fold("ÇAMAŞIR İPLİĞİ") == "camasir ipligi"
    assert fold("IŞIK") == "isik"


# ============================================================================
# 2. Fiş ayrıştırma (receipt_parser.py)
# ============================================================================

def test_parse_receipt_full_sample() -> None:
    receipt = parse_receipt(SAMPLE_RECEIPT)

    assert receipt["merchant"] == "ŞOK MARKETLER TİC. A.Ş."
    assert receipt["date"] == "2026-11-05"
    assert receipt["total"] == 514.30

    names = [item["name"] for item in receipt["items"]]
    assert names == [
        "SÜT 1L",
        "TAM BUĞDAY EKMEK",
        "DOMATES",
        "ÇAMAŞIR DETERJANI",
        "KAŞAR PEYNİRİ",
        "ANTEP FISTIĞI",
    ]
    items_sum = round(sum(item["total_price"] for item in receipt["items"]), 2)
    assert items_sum == receipt["total"]


def test_quantity_line_applies_to_next_item() -> None:
    receipt = parse_receipt(SAMPLE_RECEIPT)
    by_name = {item["name"]: item for item in receipt["items"]}

    assert by_name["SÜT 1L"]["quantity"] == 2.0
    assert by_name["SÜT 1L"]["unit_price"] == 22.5
    assert by_name["DOMATES"]["quantity"] == 0.75
    assert by_name["DOMATES"]["unit_price"] == 60.0
    assert by_name["TAM BUĞDAY EKMEK"]["quantity"] == 1.0


def test_payment_and_tax_lines_are_not_items() -> None:
    names = " ".join(item["name"] for item in parse_receipt(SAMPLE_RECEIPT)["items"])
    for keyword in ("TOPKDV", "TOPLAM", "NAKİT", "PARA ÜSTÜ"):
        assert keyword not in names


def test_ocr_vat_remnants_are_stripped_from_names() -> None:
    """OCR '%1' işaretini '»1', 'X1', '41' gibi okursa ürün adında kalmamalı."""
    text = "MARKET\nTARİH: 01.11.2026\nSÜT 1L 1 *45,00\nEKMEK 41 *15,00\nPEYNİR »1 *90,00\nDETERJAN 420 — *100,00\nTOPLAM *250,00"
    names = [item["name"] for item in parse_receipt(text)["items"]]
    assert names == ["SÜT 1L", "EKMEK", "PEYNİR", "DETERJAN"]


def test_keywords_inside_product_names_do_not_drop_items() -> None:
    """'kasa' -> 'KAŞAR', 'fis' -> 'FISTIK' gibi kelime içi eşleşmeler ürünü elememeli."""
    text = "MARKET\nKAŞAR PEYNİRİ *129,50\nFISTIK *40,00\nTELEFON KILIFI *99,00\nTOPLAM *268,50"
    names = [item["name"] for item in parse_receipt(text)["items"]]
    assert names == ["KAŞAR PEYNİRİ", "FISTIK", "TELEFON KILIFI"]


def test_discount_lines_are_recorded_separately() -> None:
    text = "MARKET\nÇAY 1KG *200,00\nİNDİRİM -20,00\nTOPLAM *180,00"
    receipt = parse_receipt(text)
    assert [item["name"] for item in receipt["items"]] == ["ÇAY 1KG"]
    assert receipt["discount"] == 20.0
    assert receipt["total"] == 180.0


@pytest.mark.parametrize(
    ("lines", "expected"),
    [
        (["TARİH: 07.11.2026"], "2026-11-07"),
        (["07/11/2026 14:32"], "2026-11-07"),
        (["2026-11-07"], "2026-11-07"),
        (["07.11.26"], "2026-11-07"),
        (["32.13.2026"], None),
        (["FİŞ NO: 0045"], None),
    ],
)
def test_extract_date(lines: list[str], expected: str | None) -> None:
    assert extract_date(lines) == expected


def test_extract_total_prefers_grand_total_and_ignores_vat() -> None:
    lines = ["ARA TOPLAM *120,00", "KDV TOPLAM *10,00", "GENEL TOPLAM *110,00", "TOPLAM KDV *9,00"]
    assert extract_total(lines) == 110.0


def test_missing_total_is_computed_from_items() -> None:
    receipt = parse_receipt("BİM\nSÜT *30,00\nEKMEK *10,00")
    assert receipt["total"] == 40.0
    assert any("hesaplandı" in warning for warning in receipt["warnings"])


def test_missing_date_and_merchant_produce_warnings() -> None:
    receipt = parse_receipt("123\nSÜT *30,00\nTOPLAM *30,00")
    assert receipt["merchant"] == UNKNOWN_MERCHANT
    assert receipt["date"] == date.today().isoformat()
    assert len(receipt["warnings"]) == 2


def test_unreadable_receipt_raises() -> None:
    with pytest.raises(OCRProcessingError):
        parse_receipt("asdf qwer\n1234")
    with pytest.raises(OCRProcessingError):
        parse_receipt([])


def test_parse_receipt_accepts_ocr_line_dicts() -> None:
    lines = [
        {"line_no": 0, "text": "A101", "confidence": 0.95},
        {"line_no": 1, "text": "MAKARNA *25,00", "confidence": 0.41},
        {"line_no": 2, "text": "TOPLAM *25,00", "confidence": 0.9},
    ]
    receipt = parse_receipt(lines)
    assert receipt["items"][0]["confidence"] == 0.41


# ============================================================================
# 3. Kategori tahmini (categorizer_ml.py)
# ============================================================================

@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("SÜT 1L", "Gıda"),
        ("SUT 1L", "Gıda"),                 # OCR diakritik kaybı
        ("TOST EKMEGI", "Gıda"),            # tamlamada asıl isim sonda
        ("ANTEP FISTIĞI", "Gıda"),          # ünsüz yumuşaması
        ("PEYN. BEYAZ", "Gıda"),            # kısaltma
        ("ÇAMAŞIR SUYU", "Temizlik"),       # çok kelimeli ifade
        ("DETERIAN 4KG", "Temizlik"),       # OCR harf hatası
        ("COCA COLA 1L", "İçecek"),
        ("KEDI MAMASI", "Evcil Hayvan"),
        ("PIL AA 4LU", "Elektronik"),
        ("XYZ 123", DEFAULT_CATEGORY),
        ("", DEFAULT_CATEGORY),
    ],
)
def test_predict_category(name: str, expected: str) -> None:
    assert predict_category(name) == expected


# ============================================================================
# 4. Doğrulama (receipt_validator.py)
# ============================================================================

def _valid_receipt(**overrides) -> dict:
    receipt = {
        "merchant": "MARKET",
        "date": date.today().isoformat(),
        "total": 60.0,
        "items": [
            {"name": "SÜT", "quantity": 2, "unit_price": 22.5, "total_price": 45.0, "confidence": 0.9},
            {"name": "EKMEK", "quantity": 1, "unit_price": 15.0, "total_price": 15.0, "confidence": 0.9},
        ],
    }
    receipt.update(overrides)
    return receipt


def test_validator_accepts_consistent_receipt() -> None:
    result = validate_receipt(_valid_receipt())
    assert result["is_valid"] is True
    assert result["warnings"] == []
    assert all(item["category"] == DEFAULT_CATEGORY for item in result["data"]["items"])


def test_validator_rejects_missing_total() -> None:
    result = validate_receipt(_valid_receipt(total=None))
    assert result["is_valid"] is False
    assert result["error"]


def test_validator_warns_on_sum_mismatch() -> None:
    result = validate_receipt(_valid_receipt(total=100.0))
    assert result["is_valid"] is True
    assert any("uyuşmuyor" in warning for warning in result["warnings"])


def test_validator_drops_invalid_items() -> None:
    receipt = _valid_receipt(total=45.0)
    receipt["items"] = [
        {"name": "SÜT", "quantity": 2, "unit_price": 22.5, "total_price": 45.0},
        {"name": "", "total_price": 10.0},
        {"name": "BOZUK", "total_price": 0},
        "geçersiz",
    ]
    result = validate_receipt(receipt)
    assert [item["name"] for item in result["data"]["items"]] == ["SÜT"]
    assert any("3 satır" in warning for warning in result["warnings"])


def test_validator_warns_on_future_date_and_low_confidence() -> None:
    receipt = _valid_receipt(date=(date.today() + timedelta(days=30)).isoformat())
    receipt["items"][0]["confidence"] = 0.3
    warnings = validate_receipt(receipt)["warnings"]
    assert any("gelecekte" in warning for warning in warnings)
    assert any("Düşük güven" in warning for warning in warnings)


def test_parsed_receipt_passes_backend_schema() -> None:
    """Parser + kategori + doğrulayıcı çıktısı backend ReceiptSchema'dan geçmeli."""
    receipt = parse_receipt(SAMPLE_RECEIPT)
    for item in receipt["items"]:
        item["category"] = predict_category(item["name"])
    validated = validate_receipt(receipt)
    assert validated["is_valid"] is True

    payload = validate_receipt_payload(validated["data"])
    assert payload["total"] == 514.30
    assert payload["category"] == "Gıda"
    assert len(payload["items"]) == 6


# ============================================================================
# 5. Gerçek OCR hatalarına dayanıklılık (benchmark'ta görülen vakalar)
# ============================================================================

@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("TOPLAM *2,019,03", 2019.03),     # binlik nokta virgül okunmuş
        ("TOPLAM *1.,005,02", 1005.02),    # çift ayraç
        ("TUVALET KAĞIDI *424 22", 424.22),  # ondalık virgül kaybolmuş
        ("SÜT *1O5,00", 105.0),            # O -> 0
        ("PEYNİR *4l,50", 41.5),           # l -> 1
    ],
)
def test_amount_ocr_errors(line: str, expected: float) -> None:
    assert find_trailing_amount(line)[0] == expected


def test_space_decimal_requires_star_prefix() -> None:
    """'*' öneki olmadan '15 00' tutar sayılmamalı (ürün adındaki sayılarla karışır)."""
    assert find_trailing_amount("EKMEK 15 00")[0] is None


@pytest.mark.parametrize(
    "remnant", ["1", "91", "YI", "x1", "X20", "xX1", "410", "420", "820", "S1", "»1", "X20 —"]
)
def test_more_vat_remnants_are_stripped(remnant: str) -> None:
    receipt = parse_receipt(f"MARKET\nMAKARNA {remnant} *25,00\nTOPLAM *25,00")
    assert receipt["items"][0]["name"] == "MAKARNA"


def test_garbled_keywords_are_recognized() -> None:
    """'KDV' -> 'KOV', 'TOPLAM' -> 'TOPLAN', 'TOPKDV' -> 'TOPKODOV' okunsa da ürün sayılmamalı."""
    text = "OPET\nKURŞUNSUZ 95 *1.661,65\nARA TOPLAN *1.661,65\nTOPKODOV *131,36\nTOPKOV *131,36\nGENEL TOPLAN *1.661,65"
    receipt = parse_receipt(text)
    assert [item["name"] for item in receipt["items"]] == ["KURŞUNSUZ 95"]
    assert receipt["total"] == 1661.65


def test_short_words_are_not_fuzzy_matched_to_keywords() -> None:
    """'KEDİ' kelimesi 'KREDİ' sanılıp ürün listesinden atılmamalı."""
    receipt = parse_receipt("MARKET\nKEDİ MAMASI *26,85\nTOPLAM *26,85")
    assert [item["name"] for item in receipt["items"]] == ["KEDİ MAMASI"]


def test_date_with_letter_o_instead_of_zero() -> None:
    assert extract_date(["TARİH ; O1.12.2O26 SAAT : 17:27"]) == "2026-12-01"


def test_digit_o_inside_product_name_is_fixed() -> None:
    receipt = parse_receipt("ECZANE\nPAROL 5O0MG *45,00\nTOPLAM *45,00")
    assert receipt["items"][0]["name"] == "PAROL 500MG"


def test_reconcile_star_read_as_digit() -> None:
    """'*192,23' -> '2192,23' okunmuşsa toplamla uzlaştırılarak düzeltilmeli."""
    receipt = parse_receipt("MARKET\nSOĞAN *72,71\nDİŞ MACUNU 2192,23\nYOĞURT *531,39\nTOPLAM *796,33")
    assert [item["total_price"] for item in receipt["items"]] == [72.71, 192.23, 531.39]
    assert any("düzeltildi" in warning for warning in receipt["warnings"])


def test_reconcile_inserted_digit() -> None:
    receipt = parse_receipt("KAFE\nMOCHA *247,51\nTOST *5094,64\nTOPLAM *842,15")
    assert receipt["items"][1]["total_price"] == 594.64
    assert receipt["items"][1]["confidence"] <= 0.5


def test_reconcile_uses_quantity_line() -> None:
    receipt = parse_receipt("OPET\n37,67 LT X 43,59\nKURŞUNSUZ 95 *1.642,64\nTOPLAM *1.642,04")
    item = receipt["items"][0]
    assert item["total_price"] == 1642.04
    assert item["quantity"] == 37.67 and item["unit_price"] == 43.59
    assert "_line_total" not in item


def test_reconcile_leaves_ambiguous_cases_untouched() -> None:
    """Birden fazla kalem düzeltmeye uyuyorsa hiçbirine dokunulmamalı."""
    receipt = parse_receipt("MARKET\nA1 ÜRÜN *211,00\nB1 ÜRÜN *211,00\nTOPLAM *222,00")
    assert [item["total_price"] for item in receipt["items"]] == [211.0, 211.0]
    assert not any("düzeltildi" in warning for warning in receipt["warnings"])


def test_gram_suffix_misread_as_six_is_fixed_only_at_name_end() -> None:
    receipt = parse_receipt("M\nMAKARNA 5006 *25,00\nVİDA 5006 ADET *3,00\nTOPLAM *28,00")
    assert [item["name"] for item in receipt["items"]] == ["MAKARNA 500G", "VİDA 5006 ADET"]


def test_merchant_named_sok_is_not_mistaken_for_street() -> None:
    """'ŞOK' market adı, adres kısaltması 'Sok.' sanılıp atlanmamalı."""
    receipt = parse_receipt("ŞOK. MARKETLER TİC. A.Ş.\nGÜL SOKAK NO:5\nSÜT *30,00\nTOPLAM *30,00")
    assert receipt["merchant"].startswith("ŞOK")


@pytest.mark.parametrize(
    ("name", "expected"),
    [("AYÇİÇEK YAĞI 1L", "Gıda"), ("PİLLERİ", "Elektronik"), ("ELMALARI", "Gıda")],
)
def test_categorizer_handles_possessive_suffixes(name: str, expected: str) -> None:
    assert predict_category(name) == expected


# ============================================================================
# 6. Gerçek fişlerden çıkan kalıplar (örnekler sahte değerlerle yeniden yazıldı)
# ============================================================================

def test_greeting_line_is_not_merchant() -> None:
    receipt = parse_receipt("HOŞ GELDİNİZ\nÖRNEK ALIŞVERİŞ MERKEZLERİ A.Ş.\nSÜT *30,00\nTOPLAM *30,00")
    assert receipt["merchant"].startswith("ÖRNEK")


def test_low_confidence_noise_line_is_not_merchant() -> None:
    lines = [
        {"text": "VER and dci maa endamn ala", "confidence": 0.11},
        {"text": "ÖRNEK AKARYAKIT LTD", "confidence": 0.94},
        {"text": "TOPLAM *100,00", "confidence": 0.9},
    ]
    assert parse_receipt(lines)["merchant"] == "ÖRNEK AKARYAKIT LTD"


def test_pos_slip_total_on_next_line_and_synthesized_item() -> None:
    """POS slibi: tutar 'İŞLEM TUTARI' altında, ₺ '©' okunmuş, ürün satırı yok."""
    text = "ÖRNEK AKARYAKIT LTD\nTARİH:05.11.2026\nSN :ABC123456789 > HGN/HNS VER:1.00.33\nİŞLEM TUTARI\n1.234,50 ©"
    receipt = parse_receipt(text)
    assert receipt["total"] == 1234.50
    assert receipt["items_synthesized"] is True
    assert [item["total_price"] for item in receipt["items"]] == [1234.50]
    assert predict_category(receipt["items"][0]["name"]) == "Ulaşım"


def test_pos_section_total_with_space_thousands_does_not_override_toplam() -> None:
    text = "MARKET\nSÜT *1.050,25\nTOPLAM *1050,25\nKREDİ KARTI *1050,25\nTUTAR\n1 050,25 TL"
    assert parse_receipt(text)["total"] == 1050.25


def test_subtotal_abbreviation_and_discount_from_difference() -> None:
    """'ARA TOP.' kısaltması; indirim satırı okunamazsa ara toplam - toplam farkı indirimdir."""
    text = "MARKET\nEKMEK *100,00\nÇÖREK *50,00\nARA TOP. *150,00\nÇÖREK *49, 5-D\nTOPLAM *100,05"
    receipt = parse_receipt(text)
    assert receipt["subtotal"] == 150.0
    assert receipt["discount"] == 49.95
    assert sorted(item["total_price"] for item in receipt["items"]) == [50.0, 100.0]


def test_only_subtotal_read_uses_discount_or_payment_line() -> None:
    """İri puntolu TOPLAM okunamadıysa: ara toplam - indirim, yoksa ödeme satırı."""
    with_discount = "RESTORAN\nMENÜ *210,00\nARA TOPLAM *210,00\nTUTAR IND. *-10,00\nKredi Karti Odeme *200,00"
    assert parse_receipt(with_discount)["total"] == 200.0
    with_payment = "RESTORAN\nMENÜ *400,00\nARA TOPLAM *400,00\nKredi Karti Odeme *210,00"
    assert parse_receipt(with_payment)["total"] == 210.0


def test_total_corrected_by_payment_line_when_it_matches_items() -> None:
    text = "KAFE\nMOCHA *597,69\nSU *334,17\nLATTE *73,16\nTOPLAM *1.605,02\nKREDİ KARTI *1.005,02"
    receipt = parse_receipt(text)
    assert receipt["total"] == 1005.02


def test_barcode_lines_and_orphan_item_names() -> None:
    """Ürün adı üstte, fiyat barkodla başlayan alt satırda (buruşuk fiş)."""
    text = "MARKET\nKAŞAR PEYNİRİ\n8680596220826 X01 *424,99\nYUMURTA 10LU %01 *99,95\n6370\nTOPLAM *524,94"
    names = [item["name"] for item in parse_receipt(text)["items"]]
    assert names == ["KAŞAR PEYNİRİ", "YUMURTA 10LU"]


def test_garbled_quantity_line_is_not_an_item() -> None:
    receipt = parse_receipt("MARKET\npoşsa KG X 579,00\nPEYNİR *424,99\nTOPLAM *424,99")
    assert [item["total_price"] for item in receipt["items"]] == [424.99]


def test_statement_dates_are_not_receipt_date() -> None:
    lines = ["TARIH 30", "İLK HESAP KESİM TARİHİ: 28/11/2026", "TARİH: 30/10/2026 SAAT: 15:14"]
    assert extract_date(lines) == "2026-10-30"


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("TARİH : O1 12.2026", "2026-12-01"),   # ayraç boşluk, O -> 0
        ("TARİH : 66.08.2025", "2025-08-06"),   # 0 -> 6
        ("TARİH : 26.64.2025", "2025-04-26"),
        ("TARİH : 26.08,2025", "2025-08-26"),   # virgül ayraç
    ],
)
def test_date_ocr_repairs(line: str, expected: str) -> None:
    assert extract_date([line]) == expected


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("WİNSTON SLENDER BLUE : X0— *120,00 >", 120.0),  # satır sonu çöp
        ("ADE Xx01 x46,95 | NV", 46.95),
        ("KREDİ KARTI x1050,25 9", 1050.25),
        ("8680596220826 OTRA X01 .424.99", 424.99),          # '*' -> '.'
        ("Ur RA TOP *1 070,18", 1070.18),                     # '*' sonrası boşluklu binlik
        ("SN :ABC123456789 > HGN/HNS VER:1.00.33", None),     # kelime içindeki sayı fiyat değil
    ],
)
def test_real_world_amount_patterns(line: str, expected: float | None) -> None:
    assert find_trailing_amount(line)[0] == expected
