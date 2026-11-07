"""
Fiş Görseli Ön İşleme Modülü (image_processor.py).

Telefonla çekilen fiş fotoğraflarını Tesseract'ın daha doğru okuyabileceği
hale getirir:
    - EXIF yönü düzeltme (OpenCV otomatik uygular)
    - Fişi arka plandan (masa vb.) ayırıp perspektif düzeltme
    - Eğiklik (skew) düzeltme, 90/180 derece döndürme
    - Boyut normalizasyonu
    - Farklı koşullar için filtre varyantları: uyarlanabilir eşikleme,
      gölge/ışık dengeleme, gürültü giderme

preprocess_image, backend'in çağırdığı giriş noktasıdır; en iyi okumayı seçme
işini receipt_reader modülüne devreder.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union

import cv2
import numpy as np

from backend.schemas import ImageProcessingError

ImageInput = Union[str, Path, np.ndarray]

# Tesseract en iyi sonucu harf yüksekliği ~30 px civarında verir;
# fiş fotoğrafları için bu kabaca 1400-2000 px genişliğe denk gelir.
TARGET_MIN_WIDTH = 1400
MAX_WIDTH = 2400
MAX_DESKEW_ANGLE = 15.0

# Sıkıştırma bombası koruması: 5 MB'lık bir PNG bile bellekte GB'larca yer kaplayabilir.
# Çözünürlük, görsel açılmadan önce dosya başlığından kontrol edilir.
MAX_PIXELS = 120_000_000        # üzeri reddedilir (~11000 x 11000)
DECODE_TARGET_PIXELS = 25_000_000  # üzeri küçültülerek açılır (OCR için fazlasıyla yeterli)


class PreparedImage(np.ndarray):
    """
    OCR'a hazır görsel (numpy dizisi) + önceden hesaplanmış OCR satırları.

    preprocess_image en iyi okumayı seçmek için OCR'ı zaten çalıştırır; sonucu
    'ocr_lines' özelliğinde taşır, böylece receipt_ocr.extract_text OCR'ı
    ikinci kez çalıştırmaz. Normal bir numpy dizisi gibi kullanılabilir.
    """

    ocr_lines: Optional[list[dict[str, Any]]]

    def __new__(cls, array: np.ndarray, ocr_lines: Optional[list[dict[str, Any]]] = None) -> "PreparedImage":
        obj = np.asarray(array).view(cls)
        obj.ocr_lines = ocr_lines
        return obj

    def __array_finalize__(self, obj: Any) -> None:
        self.ocr_lines = getattr(obj, "ocr_lines", None)


def load_image(image: ImageInput) -> np.ndarray:
    """
    Görseli BGR numpy dizisi olarak yükler.

    np.fromfile + cv2.imdecode kullanılır; böylece Türkçe karakter içeren
    dosya yolları (Windows dahil) sorunsuz okunur. Telefon fotoğraflarındaki
    EXIF yön bilgisi OpenCV tarafından otomatik uygulanır.
    """
    if isinstance(image, np.ndarray):
        if image.size == 0:
            raise ImageProcessingError("Boş görsel verisi işlenemez.")
        return np.asarray(image)

    path = Path(image)
    if not path.is_file():
        raise ImageProcessingError(f"Fiş görseli bulunamadı: '{path.name}'")

    flags = _decode_flags_for(path)
    try:
        buffer = np.fromfile(str(path), dtype=np.uint8)
        decoded = cv2.imdecode(buffer, flags)
    except (OSError, cv2.error) as exc:
        raise ImageProcessingError("Fiş görseli okunamadı.", details=str(exc)) from exc

    if decoded is None or decoded.size == 0:
        raise ImageProcessingError(
            "Fiş görseli açılamadı. Dosya bozuk veya desteklenmeyen bir formatta olabilir."
        )
    return decoded


def _decode_flags_for(path: Path) -> int:
    """
    Dosya başlığındaki çözünürlüğe göre güvenli açma bayrağını seçer.
    Çok büyük görselleri reddeder, büyükleri küçültülmüş açar (JPEG'de bellek de azalır).
    """
    import warnings

    from PIL import Image

    too_large = ImageProcessingError(
        "Fiş görselinin çözünürlüğü çok yüksek. Lütfen normal bir telefon fotoğrafı yükleyin."
    )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", Image.DecompressionBombWarning)
            with Image.open(path) as header:  # yalnızca başlık okunur, piksel açılmaz
                width, height = header.size
    except Image.DecompressionBombError as exc:
        raise too_large from exc
    except Exception:  # noqa: BLE001 - başlık okunamazsa karar OpenCV'ye kalır
        return cv2.IMREAD_COLOR

    pixels = width * height
    if pixels > MAX_PIXELS:
        raise too_large
    if pixels <= DECODE_TARGET_PIXELS:
        return cv2.IMREAD_COLOR
    # Hedefin altına inen en küçük küçültme oranını seç
    for factor, flag in ((2, cv2.IMREAD_REDUCED_COLOR_2), (4, cv2.IMREAD_REDUCED_COLOR_4)):
        if pixels / (factor * factor) <= DECODE_TARGET_PIXELS:
            return flag
    return cv2.IMREAD_REDUCED_COLOR_8


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Renkli görseli tek kanallı gri tona çevirir."""
    if image.ndim == 2:
        return image
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def resize_scale(width: int) -> float:
    """Görsel genişliğine göre OCR için uygulanacak ölçek katsayısı."""
    if width < TARGET_MIN_WIDTH:
        return TARGET_MIN_WIDTH / width
    if width > MAX_WIDTH:
        return MAX_WIDTH / width
    return 1.0


# Tesseract en iyi sonucu büyük harf yüksekliği ~30 px civarında verir. Fiş karenin dar bir
# şeridindeyse görsel genişliğine göre ölçeklemek harfleri çok küçük bırakır; bu yüzden ölçek
# mümkün olduğunda doğrudan ölçülen harf yüksekliğinden hesaplanır.
TARGET_CHAR_HEIGHT = 32.0
# Tesseract süresi piksel sayısıyla büyür; gri tonlu filtrede 27 MP görsel ~18 sn sürüyor.
MAX_OCR_SIDE = 5000
MAX_OCR_PIXELS = 12_000_000


def _char_components(gray: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Harf benzeri bağlı bileşenlerin istatistiklerini (x, y, w, h) döndürür.
    Hız için görsel en fazla 2000 px'e küçültülür; dönen ölçek katsayısıyla geri çevrilir.
    """
    height, width = gray.shape[:2]
    shrink = min(1.0, 2000.0 / max(height, width))
    small = gray if shrink == 1.0 else cv2.resize(
        gray, (int(width * shrink), int(height * shrink)), interpolation=cv2.INTER_AREA
    )
    binary = cv2.adaptiveThreshold(
        cv2.medianBlur(small, 3), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 12
    )
    count, _, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    # Yönden bağımsız: 90 derece dönük metinde harfin "yüksekliği" genişliğidir
    max_size = max(small.shape[:2]) * 0.08
    chars = [
        (x, y, w, h)
        for x, y, w, h, area in stats[1:count]
        if 4 <= max(w, h) <= max_size and 0.15 <= w / h <= 1 / 0.15 and area >= 0.15 * w * h
    ]
    return np.array(chars, dtype=np.float32).reshape(-1, 4), shrink


def estimate_char_height(gray: np.ndarray) -> Optional[float]:
    """Metindeki tipik harf yüksekliği (piksel); yeterli harf bulunamazsa None."""
    chars, shrink = _char_components(gray)
    if len(chars) < 20:
        return None
    return float(np.median(np.maximum(chars[:, 2], chars[:, 3]))) / shrink


def crop_to_text(image: np.ndarray) -> Optional[np.ndarray]:
    """
    Görseli metin içeren bölgeye kırpar (fiş arka planla aynı renkteyse köşe tespiti
    çalışmaz; bu durumda harf yoğunluğuna göre kırpılır). Kırpmaya gerek yoksa None.
    """
    gray = to_grayscale(image)
    chars, shrink = _char_components(gray)
    if len(chars) < 20:
        return None
    char_h = float(np.median(np.maximum(chars[:, 2], chars[:, 3])))
    # Harf kutularını genişletip birleştir; en büyük metin bloğunu al. (Yüzdelikle uç kırpmak
    # satır sonundaki seyrek fiyat sütununu da keser.) Bloktan uzak tek tük gürültü dışarıda kalır.
    shrunk_h, shrunk_w = int(gray.shape[0] * shrink) + 1, int(gray.shape[1] * shrink) + 1
    canvas = np.zeros((shrunk_h, shrunk_w), np.uint8)
    for cx, cy, cw, chh in chars.astype(int):
        cv2.rectangle(canvas, (cx, cy), (cx + cw, cy + chh), 255, thickness=-1)
    gap = max(3, int(4 * char_h))
    canvas = cv2.dilate(canvas, cv2.getStructuringElement(cv2.MORPH_RECT, (gap, gap)))
    count, _, stats, _ = cv2.connectedComponentsWithStats(canvas, 8)
    if count < 2:
        return None
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    bx, by, bw, bh, _ = stats[biggest]
    x0, y0, x1, y1 = bx + gap / 2, by + gap / 2, bx + bw - gap / 2, by + bh - gap / 2
    margin = 2 * char_h
    height, width = gray.shape[:2]
    left = max(0, int((x0 - margin) / shrink))
    top = max(0, int((y0 - margin) / shrink))
    right = min(width, int((x1 + margin) / shrink))
    bottom = min(height, int((y1 + margin) / shrink))
    if (right - left) * (bottom - top) > 0.8 * width * height or right - left < 50 or bottom - top < 50:
        return None
    return image[top:bottom, left:right]


def ocr_scale(gray: np.ndarray) -> float:
    """OCR öncesi ölçek: harf yüksekliği ölçülebiliyorsa ~32 px'e, ölçülemiyorsa genişliğe göre."""
    char_h = estimate_char_height(gray)
    scale = TARGET_CHAR_HEIGHT / char_h if char_h else resize_scale(gray.shape[1])
    scale = float(np.clip(scale, 0.5, 4.0))
    height, width = gray.shape[:2]
    scale = min(scale, MAX_OCR_SIDE / max(height, width), (MAX_OCR_PIXELS / (height * width)) ** 0.5)
    return scale


def resize_for_ocr(gray: np.ndarray) -> tuple[np.ndarray, float]:
    """Görseli OCR ölçeğine getirir; (görsel, ölçek) döndürür."""
    scale = ocr_scale(gray)
    if abs(scale - 1.0) < 0.05:
        return gray, 1.0
    height, width = gray.shape[:2]
    interpolation = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
    return cv2.resize(gray, (int(round(width * scale)), int(round(height * scale))), interpolation=interpolation), scale


def normalize_size(gray: np.ndarray) -> np.ndarray:
    """Küçük görselleri büyütür, çok büyük görselleri küçültür (oran korunur)."""
    height, width = gray.shape[:2]
    scale = resize_scale(width)
    if scale == 1.0:
        return gray
    interpolation = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
    return cv2.resize(gray, (int(round(width * scale)), int(round(height * scale))), interpolation=interpolation)


# ============================================================================
# Geometri: fiş tespiti, döndürme, eğiklik
# ============================================================================

def _order_corners(points: np.ndarray) -> np.ndarray:
    """Dört köşeyi sol-üst, sağ-üst, sağ-alt, sol-alt sırasına dizer."""
    pts = points.reshape(4, 2).astype(np.float32)
    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1).ravel()
    return np.array(
        [pts[np.argmin(sums)], pts[np.argmin(diffs)], pts[np.argmax(sums)], pts[np.argmax(diffs)]],
        dtype=np.float32,
    )


def find_receipt_corners(image: np.ndarray) -> Optional[np.ndarray]:
    """
    Fotoğrafta fişin (arka plandan belirgin şekilde açık renkli dörtgen) köşelerini bulur.

    Returns:
        Orijinal görsel koordinatlarında 4x2 köşe dizisi; fiş arka plandan
        ayrıştırılamıyorsa (ör. fiş tüm kareyi kaplıyorsa) None.
    """
    gray = to_grayscale(image)
    height, width = gray.shape[:2]
    scale = 800.0 / max(height, width)
    small = cv2.resize(gray, (max(1, int(width * scale)), max(1, int(height * scale))), interpolation=cv2.INTER_AREA)
    blurred = cv2.GaussianBlur(small, (7, 7), 0)
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    area_ratio = area / float(small.shape[0] * small.shape[1])
    if not 0.12 <= area_ratio <= 0.90:
        return None

    # Gölge sınırı yanlış tespiti: gölgeli fişte aydınlık bölge görüntünün 3-4 kenarına dayanır,
    # masadaki fiş ise en fazla 2 kenara (uzun fişte üst-alt) değer.
    x, y, w, h = cv2.boundingRect(contour)
    margin = 3
    touching = sum(
        (x <= margin, y <= margin, x + w >= small.shape[1] - margin, y + h >= small.shape[0] - margin)
    )
    if touching >= 3:
        return None
    # Fiş dolu bir dörtgendir; girintili / düzensiz bölge fiş değildir
    hull_area = cv2.contourArea(cv2.convexHull(contour))
    if hull_area <= 0 or area / hull_area < 0.9:
        return None

    # Fiş içi, dışından belirgin şekilde açık olmalı (aksi halde gölge vb. yanlış tespit)
    region = np.zeros_like(mask)
    cv2.drawContours(region, [contour], -1, 255, thickness=-1)
    inside = float(small[region > 0].mean())
    outside = float(small[region == 0].mean()) if (region == 0).any() else inside
    if inside - outside < 35:
        return None

    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
    corners = approx if len(approx) == 4 else cv2.boxPoints(cv2.minAreaRect(contour))
    return _order_corners(np.asarray(corners, dtype=np.float32) / scale)


def find_receipt_by_color(image: np.ndarray) -> Optional[np.ndarray]:
    """
    Renkli fotoğrafta fişi renk doygunluğuyla ayırır: fiş kâğıdı az doygun ve parlak, el ve
    ahşap masa doygun, mat ve kumaş karanlıktır. Fişi tutan el, katlama çizgileri veya yandaki
    başka kâğıtlar dörtgen tespitini bozduğunda kullanılır.

    Doygunluk eşiği her fotoğrafta parlak piksellerden Otsu ile seçilir (sıcak ışıkta kâğıdın
    doygunluğu 60'ı aşabilir). Bölge eksene hizalıdır; küçük eğiklik sonra düzeltilir.

    Returns:
        Fiş bölgesinin 4 köşesi (orijinal koordinatlarda) veya None.
    """
    if image.ndim != 3 or image.shape[2] < 3:
        return None
    height, width = image.shape[:2]
    scale = 800.0 / max(height, width)
    small = cv2.resize(image[:, :, :3], (max(1, int(width * scale)), max(1, int(height * scale))), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    saturation, value = hsv[:, :, 1], hsv[:, :, 2]
    if float(np.median(saturation)) < 12:  # renksiz (gri tonlu) görselde bu yöntem anlamsız
        return None
    bright = saturation[value > 80]
    if bright.size < 0.05 * saturation.size:
        return None
    sat_threshold, _ = cv2.threshold(bright.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    sat_threshold = float(np.clip(sat_threshold, 35, 80))

    mask = ((saturation < sat_threshold) & (value > 80)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3)))   # ince köprüleri kes
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 41)))  # katlama çizgilerini kapat

    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if count < 2:
        return None
    order = np.argsort(-stats[1:, 4]) + 1
    x, y, w, h, _ = stats[order[0]]
    x0, y0, x1, y1 = x, y, x + w, y + h
    # Katlamayla ayrılmış parçalar: ana fişle aynı sütundaki büyük bileşenler
    for idx in order[1:]:
        ox, oy, ow, oh, area = stats[idx]
        if area < 0.01 * mask.size:
            break
        overlap = min(x1, ox + ow) - max(x0, ox)
        if overlap > 0.6 * min(w, ow) and ow < 1.4 * w:
            x0, y0, x1, y1 = min(x0, ox), min(y0, oy), max(x1, ox + ow), max(y1, oy + oh)

    # Yandaki kâğıt yığınını at: kâğıt pikseli yoğun sütunlara daralt
    columns = (mask[y0:y1, x0:x1] > 0).mean(axis=0)
    dense = np.where(columns > 0.45)[0]
    if len(dense) > 10:
        x0, x1 = x0 + int(dense[0]), x0 + int(dense[-1]) + 1

    box_area = (x1 - x0) * (y1 - y0)
    if box_area > 0.7 * mask.size or box_area < 0.04 * mask.size or min(x1 - x0, y1 - y0) < 30:
        return None  # ör. beyaz masa: bütün kare "kâğıt" görünür
    pad = int(0.02 * max(x1 - x0, y1 - y0))
    left, top = max(0, x0 - pad) / scale, max(0, y0 - pad) / scale
    right, bottom = min(mask.shape[1], x1 + pad) / scale, min(mask.shape[0], y1 + pad) / scale
    right, bottom = min(right, width - 1), min(bottom, height - 1)
    return np.array([[left, top], [right, top], [right, bottom], [left, bottom]], dtype=np.float32)


def crop_receipt(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Fişi köşelerinden kesip perspektifini düzelterek dikdörtgen görüntüye çevirir."""
    tl, tr, br, bl = corners
    width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    if width < 50 or height < 50:
        return image
    target = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(corners.astype(np.float32), target)
    return cv2.warpPerspective(image, matrix, (width, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def rotate_quadrant(image: np.ndarray, degrees: int) -> np.ndarray:
    """Görseli saat yönünde 0/90/180/270 derece döndürür."""
    codes = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
    return image if degrees % 360 == 0 else cv2.rotate(image, codes[degrees % 360])


def _horizontal_line_score(binary: np.ndarray) -> float:
    """Yatay kalınlaştırma sonrası uzun-yatay (metin satırı) blokların yazı alanına oranı."""
    height, width = binary.shape
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(9, width // 60), 1))
    dilated = cv2.dilate(binary, kernel)
    count, _, stats, _ = cv2.connectedComponentsWithStats(dilated, 8)
    line_area = 0
    for idx in range(1, count):
        _, _, w, h, area = stats[idx]
        if w >= 4 * h and w > width * 0.08 and h < height * 0.2:
            line_area += area
    return line_area / max(1.0, float(np.count_nonzero(dilated)))


def text_lines_horizontal(image: np.ndarray) -> Optional[bool]:
    """
    Metin satırlarının yatay mı (0/180 derece) dikey mi (90/270 derece) olduğunu bulur.

    Satırlar yatay yönde kalınlaştırıldığında yatay metin uzun bantlara dönüşür,
    dikey metin dönüşmez; aynı ölçüm görselin devriği için de yapılıp karşılaştırılır.
    Ölçümlerde (gölge, gürültü, soluk baskı, masa üstü) farklar çok büyüktür (~0.95'e 0.00).

    Returns:
        True: yatay, False: dikey, None: karar verilemedi (metin çok az).
    """
    gray = to_grayscale(image)
    height, width = gray.shape[:2]
    scale = 1000.0 / max(height, width)
    small = cv2.resize(gray, (max(1, int(width * scale)), max(1, int(height * scale))), interpolation=cv2.INTER_AREA)
    if estimate_noise(small) > 6:
        small = cv2.fastNlMeansDenoising(small, None, h=15)
    small = normalize_background(small)
    _, binary = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    horizontal = _horizontal_line_score(binary)
    vertical = _horizontal_line_score(np.ascontiguousarray(binary.T))
    if max(horizontal, vertical) < 0.15 or abs(horizontal - vertical) < 0.1:
        return None
    return horizontal > vertical


def estimate_skew_angle(gray: np.ndarray) -> float:
    """
    Metin satırlarının eğiklik açısını (derece) tahmin eder.

    Yazı pikselleri kalınlaştırılıp satır blokları oluşturulur; her bloğun
    minimum alanlı dikdörtgen açısının medyanı alınır. Güvenilir tahmin
    yapılamazsa 0.0 döner.
    """
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, gray.shape[1] // 40), 3))
    dilated = cv2.dilate(binary, kernel, iterations=1)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    angles: list[float] = []
    min_width = gray.shape[1] * 0.15
    for contour in contours:
        (_, _), (w, h), angle = cv2.minAreaRect(contour)
        if w < h:
            w, h = h, w
            angle += 90.0
        if w < min_width or h <= 0 or w / h < 3:
            continue
        # OpenCV açıyı [0, 90) aralığında verir; [-45, 45) aralığına taşı
        if angle >= 45.0:
            angle -= 90.0
        if angle < -45.0:
            angle += 90.0
        angles.append(angle)

    if len(angles) < 3:
        return 0.0
    median_angle = float(np.median(angles))
    if abs(median_angle) > MAX_DESKEW_ANGLE:
        return 0.0
    return median_angle


def deskew(image: np.ndarray) -> np.ndarray:
    """Görseli (gri veya renkli) tahmin edilen eğiklik açısı kadar döndürür; 0.3 dereceden küçükse dokunmaz."""
    angle = estimate_skew_angle(normalize_size(to_grayscale(image)))
    if abs(angle) < 0.3:
        return image
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    return cv2.warpAffine(
        image, matrix, (width, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )


# ============================================================================
# Filtre varyantları
# ============================================================================

NOISY_SIGMA = 10.0  # bu değerin üzerindeki görseller "gürültülü" kabul edilir


SHADED_RATIO = 0.07  # arka plan aydınlığı değişkenliği bu oranın üzerindeyse "gölgeli"


def estimate_shading(gray: np.ndarray) -> float:
    """
    Arka plan aydınlığının değişkenliği (std / ortalama). Ölçümler: düz ışıkta
    çekilmiş fiş ~0.005-0.05, gölgeli / tek taraftan aydınlanmış fiş ~0.09+.
    """
    height, width = gray.shape[:2]
    scale = 600.0 / max(height, width)
    small = cv2.resize(gray, (max(1, int(width * scale)), max(1, int(height * scale))), interpolation=cv2.INTER_AREA)
    background = cv2.medianBlur(cv2.dilate(small, np.ones((9, 9), np.uint8)), 31).astype(np.float32)
    return float(background.std() / max(float(background.mean()), 1.0))


def estimate_noise(gray: np.ndarray) -> float:
    """
    Görüntü gürültüsünün standart sapmasını tahmin eder (Immerkær yöntemi).
    Ölçümler: temiz fiş fotoğrafı ~0.2-4, yoğun gürültülü (karanlık ortam) ~16+.
    """
    height, width = gray.shape[:2]
    if height < 3 or width < 3:
        return 0.0
    kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)
    response = np.abs(cv2.filter2D(gray.astype(np.float32), -1, kernel))
    return float(response.sum() * np.sqrt(0.5 * np.pi) / (6.0 * (width - 2) * (height - 2)))

def binarize(gray: np.ndarray) -> np.ndarray:
    """
    Gürültüyü azaltır ve uyarlanabilir eşikleme ile siyah-beyaz görsel üretir.
    Temiz ve orta kaliteli fotoğraflar için varsayılan filtre.
    """
    denoised = cv2.medianBlur(gray, 3)
    block_size = max(31, (gray.shape[1] // 40) | 1)  # tek sayı olmalı
    return cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, 15
    )


def normalize_background(gray: np.ndarray) -> np.ndarray:
    """
    Gölge ve dengesiz ışığı giderir: arka plan tahmin edilip görüntü ona bölünür.
    Sonuç gri tondur; eşiklemeyi Tesseract'ın kendi Otsu yöntemine bırakır.
    Soluk termal baskı ve gölgeli fotoğraflarda eşiklemeden daha iyi sonuç verir.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    background = cv2.medianBlur(cv2.dilate(gray, kernel), 31)
    normalized = cv2.divide(gray, background, scale=255)
    return cv2.normalize(normalized, None, 0, 255, cv2.NORM_MINMAX)


def denoise_strong(gray: np.ndarray) -> np.ndarray:
    """Yoğun gürültülü (karanlık ortam, düşük kaliteli kamera) görseller için güçlü gürültü giderme + Otsu."""
    denoised = cv2.fastNlMeansDenoising(gray, None, h=25, templateWindowSize=7, searchWindowSize=21)
    normalized = normalize_background(denoised)
    _, binary = cv2.threshold(normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


FILTERS = {
    "adaptive": binarize,
    "normalized": normalize_background,
    "denoised": denoise_strong,
}


def preprocess_image(image_path: ImageInput) -> PreparedImage:
    """
    Fiş görselini OCR için hazırlar ve en iyi okumayı seçer.

    Adımlar: yükleme -> fiş tespiti/perspektif -> yön ve eğiklik -> filtre varyantları
    -> OCR -> fiş sağlamasıyla (ürün toplamı = genel toplam) en iyi okumanın seçimi
    -> kişisel veri sansürü.

    Dosya yolu verilirse, düzeltilmiş (dik, kırpılmış) ve kişisel verileri
    kapatılmış görsel dosyanın üzerine yazılır; EXIF/GPS bilgisi silinir.
    Böylece uploads/ altında sansürsüz görsel kalmaz.

    Args:
        image_path: Görsel dosya yolu veya hazır numpy dizisi.

    Returns:
        OCR'a hazır tek kanallı görsel (PreparedImage); seçilen OCR satırlarını taşır.

    Raises:
        ImageProcessingError: Görsel okunamazsa veya çok küçükse.
        OCRProcessingError: Tesseract çalışmazsa.
    """
    image = load_image(image_path)
    height, width = image.shape[:2]
    if width < 50 or height < 50:
        raise ImageProcessingError(
            "Fiş görseli çok küçük. Lütfen fişin tamamını net şekilde gösteren bir fotoğraf yükleyin."
        )

    from ocr_ml.receipt_reader import read_best

    result = read_best(image)

    if not isinstance(image_path, np.ndarray):
        from ocr_ml.pii_redactor import write_redacted_image

        write_redacted_image(Path(image_path), result.view, result.pii_boxes)

    return PreparedImage(result.processed, ocr_lines=result.lines)
