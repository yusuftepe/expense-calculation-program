"""
Ürün Kategori Tahmin Modülü (categorizer_ml.py).

Fiş kalemlerinin adına bakarak harcama kategorisi tahmin eder.

Yaklaşım: Türkçe anahtar kelime sözlüğü + bulanık eşleştirme (difflib).
OCR hataları ("SÜT" -> "SUT", "DETERJAN" -> "DETERIAN") ve kısaltmalar
("PEYN.", "DET.") için diakritiksiz karşılaştırma ve önek eşleşmesi yapılır.
Etiketli veri biriktiğinde bu modül bir sınıflandırıcıyla (ör. scikit-learn
TF-IDF + LogisticRegression) değiştirilebilir; dış arayüz predict_category aynı kalır.
"""

from __future__ import annotations

import difflib
import os
import re
from functools import lru_cache

from backend.schemas import DEFAULT_CATEGORY
from ocr_ml.text_utils import fold

CATEGORY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "Gıda": (
        "süt", "yoğurt", "ayran", "peynir", "kaşar", "tereyağ", "yumurta", "ekmek", "simit",
        "un", "makarna", "pirinç", "bulgur", "mercimek", "nohut", "fasulye", "şeker", "tuz",
        "yağ", "zeytin", "domates", "salatalık", "biber", "patates", "soğan", "sarımsak",
        "elma", "muz", "portakal", "mandalina", "limon", "çilek", "karpuz", "üzüm", "meyve",
        "sebze", "et", "kıyma", "tavuk", "piliç", "balık", "sucuk", "salam", "sosis",
        "bal", "reçel", "çikolata", "bisküvi", "kek", "gofret", "cips", "kraker", "dondurma",
        "salça", "sos", "ketçap", "mayonez", "baharat", "un", "nişasta", "kakao", "lokum",
        "helva", "tahin", "pekmez", "fındık", "fıstık", "leblebi", "ceviz", "badem", "kuruyemiş", "yulaf", "müsli",
        "konserve", "turşu", "zeytinyağ", "ayçiçek", "margarin", "krema", "labne", "lor", "kefir",
    ),
    "İçecek": (
        "su", "maden suyu", "soda", "kola", "gazoz", "meyve suyu", "çay", "kahve", "nescafe",
        "enerji içeceği", "ice tea", "limonata", "şalgam", "bira", "şarap", "rakı", "votka",
        "viski", "redbull", "fanta", "sprite", "pepsi", "coca", "nektar",
    ),
    "Temizlik": (
        "deterjan", "çamaşır", "bulaşık", "yumuşatıcı", "çamaşır suyu", "sabun", "temizleyici",
        "yüzey", "cam sil", "kireç", "tuvalet kağıdı", "kağıt havlu", "peçete", "çöp poşeti",
        "çöp torbası", "sünger", "bez", "tablet", "ariel", "omo", "fairy", "pril", "domestos",
        "cif", "vileda", "selpak", "solo", "familia", "mendil", "ıslak mendil",
    ),
    "Kişisel Bakım": (
        "şampuan", "saç kremi", "duş jeli", "diş macunu", "diş fırçası", "deodorant",
        "parfüm", "krem", "losyon", "tıraş", "jilet", "pamuk", "ped", "tampon", "bebek bezi",
        "makyaj", "ruj", "maskara", "oje", "kolonya", "güneş kremi",
    ),
    "Sağlık": (
        "eczane", "ilaç", "vitamin", "ağrı kesici", "parol", "aspirin", "majezik", "termometre",
        "maske", "yara bandı", "şurup", "tablet ilaç", "takviye",
    ),
    "Ulaşım": (
        "benzin", "motorin", "dizel", "akaryakıt", "lpg", "otopark", "köprü", "otoyol", "hgs",
        "ogs", "taksi", "otobüs", "metro", "akbil", "istanbulkart", "bilet", "uçak", "tren",
        "kurşunsuz", "opet", "shell", "petrol ofisi",
    ),
    "Restoran & Kafe": (
        "menü", "döner", "kebap", "lahmacun", "pide", "pizza", "hamburger", "burger", "tost",
        "sandviç", "dürüm", "köfte", "çorba", "tatlı", "baklava", "künefe", "latte", "cappuccino",
        "americano", "espresso", "mocha", "frappe", "servis", "garson", "kuver",
    ),
    "Giyim": (
        "tişört", "t-shirt", "gömlek", "pantolon", "kot", "etek", "elbise", "ceket", "mont",
        "kazak", "hırka", "ayakkabı", "bot", "terlik", "çorap", "iç çamaşırı", "atlet", "şort",
        "eşofman", "sweat", "kemer", "şapka", "atkı", "eldiven",
    ),
    "Elektronik": (
        "kulaklık", "şarj", "kablo", "adaptör", "pil", "batarya", "usb", "telefon", "tablet pc",
        "bilgisayar", "laptop", "mouse", "klavye", "monitör", "hoparlör", "powerbank", "ampul",
        "led", "uzatma",
    ),
    "Ev & Yaşam": (
        "tabak", "bardak", "çatal", "kaşık", "bıçak", "tencere", "tava", "saklama kabı",
        "streç film", "alüminyum folyo", "pişirme kağıdı", "havlu", "nevresim", "yastık",
        "battaniye", "mum", "vazo", "çerçeve", "askı", "kova", "paspas",
    ),
    "Evcil Hayvan": ("kedi maması", "köpek maması", "mama", "kum", "kedi kumu", "pati"),
    "Tütün": (
        "sigara", "tütün", "puro", "winston", "marlboro", "parliament", "camel", "kent", "davidoff",
        "chesterfield", "lark", "muratti", "elidor", "pall mall", "samsun", "tekel", "iqos", "heets",
    ),
    "Kırtasiye": (
        "kalem", "defter", "silgi", "kağıt", "dosya", "zarf", "yapıştırıcı", "makas", "boya",
        "kitap", "dergi", "gazete", "a4",
    ),
}


# Diakritiksiz yazılmış iyelik / çoğul ekleri (uzundan kısaya)
POSSESSIVE_SUFFIXES: tuple[str, ...] = ("leri", "lari", "ler", "lar", "si", "su", "yu", "yi", "i", "u")


@lru_cache(maxsize=1)
def _keyword_index() -> list[tuple[str, str]]:
    """(diakritiksiz anahtar kelime, kategori) çiftleri; uzun ifadeler önce."""
    pairs = {
        (fold(keyword), category)
        for category, keywords in CATEGORY_KEYWORDS.items()
        for keyword in keywords
    }
    return sorted(pairs, key=lambda pair: (-len(pair[0]), pair[0], pair[1]))


def _tokenize(folded_name: str) -> list[str]:
    return [token for token in re.split(r"[^a-z0-9]+", folded_name) if token]


def _match_token(token: str, single_keywords: list[tuple[str, str]]) -> str | None:
    """Tek kelimeyi sözlükle eşleştirir; eşleşme yoksa None döner."""
    # Tam eşleşme; Türkçe iyelik/çoğul ekleri atılarak da denenir ("YAĞI" -> yağ, "PİLLERİ" -> pil)
    variants = [token] + [
        token[: -len(suffix)]
        for suffix in POSSESSIVE_SUFFIXES
        if token.endswith(suffix) and len(token) - len(suffix) >= 2
    ]
    for variant in variants:
        for keyword, category in single_keywords:
            if variant == keyword:
                return category
    if len(token) < 4:
        return None

    # Önek: kısaltma ("PEYN" -> peynir) veya ekli hal ("DETERJANI" -> deterjan)
    for keyword, category in single_keywords:
        if len(keyword) >= 4 and (keyword.startswith(token) or token.startswith(keyword)):
            return category

    # Ünsüz yumuşaması: son harf değişir ("FISTIĞI" -> fıstık, "EKMEĞİ" -> ekmek)
    for keyword, category in single_keywords:
        if len(keyword) >= 5 and len(os.path.commonprefix([token, keyword])) >= len(keyword) - 1:
            return category

    # Bulanık eşleşme (OCR harf hataları)
    vocabulary = {kw: cat for kw, cat in single_keywords if len(kw) >= 4}
    close = difflib.get_close_matches(token, vocabulary.keys(), n=1, cutoff=0.85)
    return vocabulary[close[0]] if close else None


def predict_category(product_name: str) -> str:
    """
    Ürün adından kategori tahmin eder.

    Eşleştirme sırası:
        1. Çok kelimeli ifade tam geçiyor mu ("çamaşır suyu", "kedi maması")
        2. Kelimeler sondan başa taranır; Türkçe tamlamalarda asıl isim sondadır
           ("tost ekmeği" -> ekmek). Her kelime için: tam eşleşme, önek
           ("PEYN" -> peynir), ünsüz yumuşaması ("FISTIĞI" -> fıstık),
           bulanık eşleşme (OCR harf hataları, benzerlik >= 0.85).

    Returns:
        Kategori adı; eşleşme yoksa DEFAULT_CATEGORY ("Diğer").
    """
    folded = fold(str(product_name or ""))
    tokens = _tokenize(folded)
    if not tokens:
        return DEFAULT_CATEGORY

    padded = f" {' '.join(tokens)} "
    index = _keyword_index()

    for keyword, category in index:
        if " " in keyword and f" {keyword} " in padded:
            return category

    single_keywords = [(kw, cat) for kw, cat in index if " " not in kw]
    for token in reversed(tokens):
        if token.isdigit():
            continue
        category = _match_token(token, single_keywords)
        if category:
            return category

    return DEFAULT_CATEGORY


def predict_categories(product_names: list[str]) -> list[str]:
    """Birden fazla ürün için toplu kategori tahmini."""
    return [predict_category(name) for name in product_names]
