"""
En İyi Okumayı Seçme Modülü (receipt_reader.py).

Tek bir ön işleme her fotoğrafa uymaz: gölgeli fişte eşikleme, gürültülü fişte
gürültü giderme, ters çekilmiş fişte 180 derece döndürme gerekir. Bu modül
farklı "görünüm" (kırpma, yön, eğiklik) ve "filtre" kombinasyonlarını OCR'dan
geçirir ve fişin kendi sağlamasını kullanarak en iyi okumayı seçer:

    ürün tutarları toplamı - indirim == genel toplam   ->  okuma tutarlı

İlk denemede (en olası kombinasyon) tutarlı okuma bulunursa diğerleri hiç
çalıştırılmaz; zor fotoğraflarda kalan adaylar paralel denenir.

RapidOCR motoru kullanılıyorsa (varsayılan) filtre varyantı denenmez: derin öğrenme modeli
ön işlemeyi kendisi yapar, yalnızca görünümler (kırpma / yön) denenir. Filtre varyantları
Tesseract yedeği içindir.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

from backend.schemas import OCRProcessingError
from ocr_ml import image_processor as ip
from ocr_ml.pii_redactor import find_pii_boxes, redact_lines
from ocr_ml.receipt_ocr import ENGINE_RAPIDOCR, OCRTimeoutError, extract_lines_with_boxes, get_engine_name
from ocr_ml.receipt_parser import UNKNOWN_MERCHANT, parse_receipt

MAX_PARALLEL = 4

# Tesseract'ın OpenMP iş parçacıkları, paralel çalışan birden fazla tesseract sürecinde
# birbirini yavaşlatır; tek iş parçacığı ölçümlerde %30-40 daha hızlı. Yalnızca alt
# süreçleri (tesseract) etkiler; kullanıcı açıkça ayarladıysa dokunulmaz.
os.environ.setdefault("OMP_THREAD_LIMIT", "1")
CONSISTENT_BONUS = 10.0
CONSISTENCY_TOLERANCE = 0.02  # TL
# Tek fiş için toplam OCR süresi bütçesi. Temiz fişler ~0,5 sn'de biter; bütçe yalnızca
# fiş olmayan / okunamaz görsellerde son çare denemelerinin uzamasını engeller.
TIME_BUDGET_SECONDS = 15.0
MIN_CALL_SECONDS = 2.0


@dataclass
class Candidate:
    view_name: str
    filter_name: str
    processed: np.ndarray
    scale: float
    lines: list[dict[str, Any]]
    score: float
    consistent: bool
    timed_out: bool = False
    parsed: Optional[dict[str, Any]] = None


@dataclass
class ReadResult:
    """read_best çıktısı."""

    view: np.ndarray                      # dik, kırpılmış, eğikliği düzeltilmiş renkli görsel
    processed: np.ndarray                 # seçilen filtre uygulanmış OCR girdisi
    lines: list[dict[str, Any]]           # seçilen OCR satırları (kelime kutularıyla, ham metin)
    pii_boxes: list[dict[str, Any]]       # 'view' koordinatlarında kişisel veri kutuları
    score: float
    consistent: bool
    view_name: str
    filter_name: str
    tried: list[str] = field(default_factory=list)


def score_lines(lines: list[dict[str, Any]]) -> tuple[float, bool]:
    """
    OCR satırlarının ne kadar "fiş gibi" okunduğunu puanlar.

    Returns:
        (puan, tutarlı_mı). Tutarlı: genel toplam okundu ve ürün toplamıyla uyuşuyor.
    """
    score, consistent, _ = _score_and_parse(lines)
    return score, consistent


def _score_and_parse(lines: list[dict[str, Any]]) -> tuple[float, bool, Optional[dict[str, Any]]]:
    if not lines:
        return -1.0, False, None
    confidence = sum(line["confidence"] for line in lines) / len(lines)
    try:
        masked = redact_lines([line["text"] for line in lines])
        receipt = parse_receipt(
            [{"text": text, "confidence": line["confidence"]} for line, text in zip(lines, masked)]
        )
    except OCRProcessingError:
        return confidence, False, None

    items = receipt["items"]
    total = receipt["total"]
    total_read = receipt.get("total_source") == "ocr"
    items_sum = round(sum(item["total_price"] for item in items) - receipt.get("discount", 0.0), 2)
    # Seçim için sıkı tolerans: tek rakamlık OCR hatası (162,60 -> 162,68) "tutarlı" sayılmasın
    consistent = bool(
        total_read and items and not receipt.get("items_synthesized")
        and abs(items_sum - total) <= CONSISTENCY_TOLERANCE
    )

    score = confidence
    score += 3.0 if total_read else 0.0
    score += 1.0 if receipt.get("date_found") else 0.0
    score += 1.0 if receipt["merchant"] != UNKNOWN_MERCHANT else 0.0
    score += 0.0 if receipt.get("items_synthesized") else min(len(items), 10) * 0.3
    score += CONSISTENT_BONUS if consistent else 0.0
    return score, consistent, receipt


class _View:
    """Bir görünümün (kırpma + yön + eğiklik) tembel hesaplanan hali."""

    def __init__(self, name: str, builder: Callable[[], np.ndarray]) -> None:
        self.name = name
        self._builder = builder
        self._color: Optional[np.ndarray] = None
        self._gray: Optional[np.ndarray] = None
        self._noise: Optional[float] = None
        self.scale = 1.0

    @property
    def color(self) -> np.ndarray:
        if self._color is None:
            self._color = ip.deskew(self._builder())
        return self._color

    @property
    def gray(self) -> np.ndarray:
        if self._gray is None:
            self._gray, self.scale = ip.resize_for_ocr(ip.to_grayscale(self.color))
        return self._gray

    @property
    def shaded(self) -> bool:
        """Görünümde belirgin gölge / dengesiz ışık var mı?"""
        return ip.estimate_shading(ip.to_grayscale(self.color)) > ip.SHADED_RATIO

    @property
    def noisy(self) -> bool:
        """Görünüm (kırpma sonrası, orijinal çözünürlükte) yoğun gürültülü mü?"""
        if self._noise is None:
            self._noise = ip.estimate_noise(ip.to_grayscale(self.color))
        return self._noise > ip.NOISY_SIGMA


# RapidOCR derin öğrenme modeli kendi ön işlemesini yapar: görünüm filtresiz, renkli verilir
ORIGINAL = "original"


def _evaluate(view: _View, filter_name: str, deadline: float, cap: Optional[float] = None) -> Candidate:
    if filter_name == ORIGINAL:
        processed, scale, engine = view.color, 1.0, ENGINE_RAPIDOCR
    else:
        processed, scale, engine = ip.FILTERS[filter_name](view.gray), view.scale, "tesseract"
    remaining = max(MIN_CALL_SECONDS, deadline - time.monotonic())
    if cap is not None:
        remaining = min(remaining, cap)
    try:
        lines = extract_lines_with_boxes(processed, timeout=remaining, engine=engine)
    except OCRTimeoutError:
        return Candidate(view.name, filter_name, processed, scale, [], -1.0, False, timed_out=True)
    score, consistent, parsed = _score_and_parse(lines)
    return Candidate(view.name, filter_name, processed, scale, lines, score, consistent, parsed=parsed)


def _build_views(image: np.ndarray) -> tuple[list[_View], list[_View]]:
    """
    Denenecek görünümleri olasılık sırasıyla üretir.

    Returns:
        (olası görünümler, son çare görünümler). Metin satırları yataysa olası yönler
        0/180, dikeyse 90/270'tir; diğer yönler ve kırpılmamış görüntü son çaredir.
    """
    corners = ip.find_receipt_corners(image)
    if corners is None:
        corners = ip.find_receipt_by_color(image)
    if corners is not None:
        base, prefix = ip.crop_receipt(image, corners), "kirpik"
    else:
        # Fiş arka planla aynı renkteyse (beyaz masa) köşe bulunamaz; metin bölgesine kırp
        text_crop = ip.crop_to_text(image)
        base, prefix = (text_crop, "metin") if text_crop is not None else (image, "tam")

    horizontal = ip.text_lines_horizontal(base)
    if horizontal is None:  # metin yönü ölçülemediyse en-boy oranına göre tahmin et
        height, width = base.shape[:2]
        horizontal = height >= width
    likely, unlikely = ([0, 180], [90, 270]) if horizontal else ([90, 270], [0, 180])

    def make(deg: int) -> _View:
        return _View(f"{prefix}_{deg}", lambda: ip.rotate_quadrant(base, deg))

    fallback = [make(deg) for deg in unlikely]
    if base is not image:
        fallback.append(_View("tam_0", lambda: image))
    return [make(deg) for deg in likely], fallback


def _run_stage(jobs: list[tuple[_View, str]], deadline: float) -> list[Candidate]:
    if len(jobs) == 1:
        # Tek deneme bütçenin yarısından fazlasını tüketip diğer adaylara yer bırakmamalı
        return [_evaluate(*jobs[0], deadline, cap=TIME_BUDGET_SECONDS / 2)]
    # Görünümleri ana iş parçacığında hazırla (tembel özellikler eşzamanlı yazılmasın)
    for view, _ in jobs:
        _ = view.gray
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL) as pool:
        return list(pool.map(lambda job: _evaluate(*job, deadline), jobs))


def _supplement_missing_fields(best: Candidate, candidates: list[Candidate]) -> list[dict[str, Any]]:
    """
    En iyi okumada tarih veya işletme adı yoksa, aynı görünümün diğer okumalarından tamamlar.
    (Bir filtre toplamı, diğeri tarihi okuyabilir.) Eklenen satırların kelime kutusu yoktur.
    """
    lines = list(best.lines)
    parsed = best.parsed or {}
    others = [
        cand.parsed for cand in sorted(candidates, key=lambda c: -c.score)
        if cand is not best and cand.parsed and cand.view_name == best.view_name
    ]
    if parsed and not parsed.get("date_found"):
        donor = next((p for p in others if p.get("date_found")), None)
        if donor:
            year, month, day = donor["date"].split("-")
            lines.append({"text": f"TARİH: {day}.{month}.{year}", "confidence": 0.5, "words": []})
    if parsed and parsed.get("merchant") == UNKNOWN_MERCHANT:
        donor = next((p for p in others if p.get("merchant") != UNKNOWN_MERCHANT), None)
        if donor:
            lines.insert(0, {"text": donor["merchant"], "confidence": 0.5, "words": []})
    return lines


def _finalize(candidates: list[Candidate], all_views: list[_View], extra_pii_pass: bool) -> ReadResult:
    """En iyi okumayı seçer, eksik alanları tamamlar ve kişisel veri kutularını toplar."""
    best = max(candidates, key=lambda cand: cand.score)
    if not best.lines:
        if all(cand.timed_out for cand in candidates):
            raise OCRTimeoutError("Fiş okuma zaman aşımına uğradı. Lütfen daha net bir fotoğraf yükleyin.")
        raise OCRProcessingError(
            "Fiş görselinde okunabilir metin bulunamadı. Lütfen daha net ve aydınlık bir fotoğraf yükleyin."
        )

    view = next(v for v in all_views if v.name == best.view_name)
    lines = _supplement_missing_fields(best, candidates)

    # Sansür için seçilen görünümü en az iki farklı filtreyle oku: bir filtre kart satırını
    # okuyamazsa diğeri yakalasın (saklanan görselde kart numarası kalmasın)
    same_view = [cand for cand in candidates if cand.view_name == best.view_name]
    if extra_pii_pass and len(same_view) < 2:
        extra_filter = "adaptive" if best.filter_name != "adaptive" else "denoised"
        candidates.append(_evaluate(view, extra_filter, time.monotonic() + TIME_BUDGET_SECONDS / 2))

    # Aynı görünümün tüm filtrelerindeki kişisel veri kutularını birleştir (bir filtre kaçırsa diğeri yakalar)
    pii_boxes: list[dict[str, Any]] = []
    for cand in candidates:
        if cand.view_name != best.view_name:
            continue
        for box in find_pii_boxes(cand.lines):
            pii_boxes.append(
                {
                    "label": box["label"],
                    "left": box["left"] / cand.scale,
                    "top": box["top"] / cand.scale,
                    "width": box["width"] / cand.scale,
                    "height": box["height"] / cand.scale,
                }
            )

    return ReadResult(
        view=view.color,
        processed=best.processed,
        lines=lines,
        pii_boxes=pii_boxes,
        score=best.score,
        consistent=best.consistent,
        view_name=best.view_name,
        filter_name=best.filter_name,
        tried=[f"{c.view_name}/{c.filter_name}:{c.score:.1f}" for c in candidates],
    )


def _read_best_rapidocr(views: list[_View], fallback: list[_View]) -> ReadResult:
    """
    RapidOCR ile okuma: filtre varyantı gerekmez (model ön işlemeyi kendisi yapar), yalnızca
    görünümler (yön / kırpma) denenir. Her aşamada tutarlı okuma bulunursa durulur.
    """
    stages = [[views[0]], [views[1]], fallback]
    candidates: list[Candidate] = []
    deadline = time.monotonic() + TIME_BUDGET_SECONDS
    for number, stage in enumerate(stages):
        # İlk iki aşama (olası yön + ters yön) her zaman denenir; bütçe yalnızca son çareyi keser
        if number >= 2 and deadline - time.monotonic() < MIN_CALL_SECONDS:
            break
        candidates.extend(_evaluate(view, ORIGINAL, deadline) for view in stage)
        if any(cand.consistent for cand in candidates):
            break
    return _finalize(candidates, views + fallback, extra_pii_pass=False)


def read_best(image: np.ndarray) -> ReadResult:
    """
    Görseldeki fişi en iyi şekilde okur.

    Aşamalar (her aşamada tutarlı okuma bulunursa durulur):
        1. Olası yön + görüntü koşuluna en uygun filtre (temiz fotoğrafların çoğu burada biter)
        2. Olası yönde diğer filtreler + karşı yön (180 derece)
        3. Son çare: 90/270 derece ve kırpılmamış görüntü

    Raises:
        OCRProcessingError: Tesseract çalışmazsa veya hiçbir varyantta metin bulunamazsa.
    """
    views, fallback = _build_views(image)
    primary, opposite = views
    if get_engine_name() == ENGINE_RAPIDOCR:
        return _read_best_rapidocr(views, fallback)

    # Görüntü koşuluna göre en olası filtreyle başla:
    #   gürültülü -> gürültü giderme (eşikleme gürültüyü beneklere çevirir)
    #   gölgeli   -> arka plan dengeleme
    #   diğer     -> uyarlanabilir eşikleme
    # Gerçek telefon fotoğraflarında (gölge + sensör gürültüsü + buruşukluk) en iyi sonucu
    # gürültü giderme filtresi veriyor (11 gerçek fişte ölçüldü); arka plan dengelemeyi de içerir.
    if primary.noisy or primary.shaded:
        order = ["denoised", "adaptive", "normalized"]
    else:
        order = ["adaptive", "normalized", "denoised"]
    stages: list[list[tuple[_View, str]]] = [
        [(primary, order[0])],
        [(primary, order[1]), (primary, order[2]), (opposite, order[0]), (opposite, order[1])],
        [(view, "adaptive") for view in fallback] + [(view, "normalized") for view in fallback[:2]],
    ]

    deadline = time.monotonic() + TIME_BUDGET_SECONDS
    candidates: list[Candidate] = []
    for stage in stages:
        if candidates and deadline - time.monotonic() < MIN_CALL_SECONDS:
            break  # süre bütçesi doldu; eldeki en iyi okumayla devam et
        candidates.extend(_run_stage(stage, deadline))
        if any(cand.consistent for cand in candidates):
            break

    return _finalize(candidates, views + fallback, extra_pii_pass=True)
