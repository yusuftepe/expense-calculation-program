# Smart Expense Analyzer (`expense-calculation-program`)

Kişisel harcamaları takip etmek, fiş fotoğraflarından otomatik harcama verisi çıkarmak, bütçe limitlerini izlemek ve kategori/ürün bazlı finansal analizler sunmak amacıyla geliştirilen modüler Python web uygulaması.

## Proje Amacı ve Mimari

Kullanıcıların gelir ve giderlerini tek bir merkezden yönetebilmesini, market/mağaza fişlerini tarayarak harcama kalemlerini dijitalleştirmesini ve aylık bütçe hedeflerine göre harcama alışkanlıklarını grafiklerle analiz edebilmesini sağlar.

* **Merkezi Backend Orkestrasyonu:** `Frontend`, `OCR/ML` ve `Database` katmanları birbirleriyle doğrudan konuşmaz; tüm veri akışını ve doğrulamayı `Backend` (`backend/main.py` ve `backend/services/`) yönetir.
* **Yerel Uygulama Kilidi (Application Lock):** Uygulama yerel ortamda kişisel finans verilerini korumak için açılışta şifre sorar (`Werkzeug` ile hashlenerek SQLite `app_settings` tablosunda saklanır) ve hareketsizlik durumunda oturumu otomatik kilitler. Kaynak kodda sabit şifre tutulmaz; ilk açılışta kullanıcı kendi şifresini belirler.
* **Kullanıcı Onaylı OCR ve Çifte Sayım Önleme:** Fiş tarama (`OCR/ML`) çıktıları doğrudan veritabanına yazılmaz; önce kullanıcı onay/düzenleme ekranına sunulur. Onaylanan fişin genel toplamı `expenses` tablosuna **tek bir ana gider** olarak yazılırken, fiş kalemleri `receipt_items` tablosunda alt detay olarak saklanır.

## Modül Geliştirme Durumu

| Modül | Klasör | Durum | Açıklama |
| :--- | :--- | :--- | :--- |
| **1. Backend & API** | `backend/` | **Tamamlandı** | Flask route'ları, `Application Lock` (`auth_service`), `expense_service`, `receipt_service`, `analytics_service`, `schemas.py`, `decorators.py`, `concurrency_worker.py`, `scraper.py` ve yedek SQLite/şablon mekanizması hazır. |
| **2. Frontend & Analitik** | `frontend/` | **Geliştiriliyor** | `templates/` altındaki Jinja2 HTML şablonları, `static/` (CSS/JS) ve `analytics.py` grafik bileşenleri eklenecek. |
| **3. OCR / ML** | `ocr_ml/` | **Geliştiriliyor** | `image_processor.py`, `receipt_ocr.py`, `receipt_parser.py`, `receipt_validator.py` ve `categorizer_ml.py` fonksiyonları doldurulacak. |
| **4. Database** | `database/` | **Geliştiriliyor** | `schema.sql`, `db_connection.py`, `models.py` ve `crud.py` fonksiyonları doldurulacak. |
| **5. Birim Testleri** | `tests/` | **Backend Hazır** | `test_auth.py` ve `test_backend.py` (13 test) tamamlandı; `test_database.py`, `test_ocr.py`, `test_parser.py`, `test_analytics.py` ilgili modüllerle birlikte eklenecek. |

## Klasör Yapısı

```text
expense-calculation-program/
├── backend/                  # 1. Kişi: Flask uygulaması, route'lar ve iş mantığı (TAMAMLANDI)
│   ├── main.py               # Tek Flask uygulama giriş noktası ve tüm endpoint'ler
│   ├── schemas.py            # Ortak veri şemaları (Dataclasses) ve özel hata sınıfları
│   ├── decorators.py         # @login_required, @handle_errors, @execution_timer, @retry_on_failure
│   ├── concurrency_worker.py # ThreadPoolExecutor tabanlı paralel görev ve toplu fiş işleme yöneticisi
│   ├── scraper.py            # TCMB güncel döviz kuru çekici (TTL önbellekli ve offline yedekli)
│   ├── services/             # Servis katmanı
│   │   ├── auth_service.py       # Werkzeug şifre hashleme, doğrulama, rate-limit ve session kilidi
│   │   ├── expense_service.py    # Harcama ve aylık bütçe işlemleri
│   │   ├── receipt_service.py    # OCR boru hattı çağrısı ve tek transaction ile fiş onaylama
│   │   └── analytics_service.py  # Aylık, kategori, bütçe aşımı ve ürün/market analizleri
│   └── utils/
│       └── helpers.py        # Güvenli dosya adı üretimi, görsel doğrulama ve para formatlama
├── database/                 # 4. Kişi: SQLite bağlantısı, modeller ve CRUD işlemleri
│   ├── db_connection.py      # Veritabanı bağlantı yöneticisi (get_connection)
│   ├── models.py             # Veritabanı modelleri
│   ├── crud.py               # Ekleme, okuma, güncelleme ve silme fonksiyonları
│   └── schema.sql            # 5 Tablo: expenses, receipts, receipt_items, budgets, app_settings
├── frontend/                 # 2. Kişi: Kullanıcı arayüzü ve görselleştirme bileşenleri
│   ├── analytics.py          # Grafik ve analiz hazırlama modülü
│   ├── templates/            # Jinja2 HTML şablonları (base, login, dashboard, expenses, receipt_scanner, reports, settings)
│   └── static/               # CSS (style, auth, dashboard), JS (main, charts) ve görseller
├── ocr_ml/                   # 3. Kişi: Fiş okuma, görüntü işleme ve kategori tahmini
│   ├── image_processor.py    # Görüntü iyileştirme ve ön işleme (preprocess_image)
│   ├── receipt_ocr.py        # Görselden metin ve güven skoru çıkarma (extract_text)
│   ├── receipt_parser.py     # Metinleri işletme, tarih, ürün ve fiyatlara ayrıştırma (parse_receipt)
│   ├── receipt_validator.py  # Çıkarılan verilerin tutarlılık kontrolü (validate_receipt)
│   └── categorizer_ml.py     # Ürün adından kategori tahmini (predict_category)
├── tests/                    # Birim ve entegrasyon testleri (pytest)
├── uploads/                  # Yüklenen fiş görselleri (.gitignore ile Git takibi dışındadır)
├── .env.example              # Ortam değişkenleri şablonu
├── .gitignore                # Git tarafından yoksayılan dosya kuralları
├── requirements.txt          # Python kütüphane bağımlılıkları
└── README.md                 # Proje dokümantasyonu ve entegrasyon kılavuzu
```

## Kurulum ve Çalıştırma

1. **Depoyu klonlayın ve proje dizinine gidin:**
   ```bash
   git clone <repo-url>
   cd expense-calculation-program
   ```

2. **Sanal ortamı (virtual environment) oluşturun ve etkinleştirin:**
   ```bash
   python -m venv .venv
   # macOS / Linux:
   source .venv/bin/activate
   # Windows:
   .venv\Scripts\activate
   ```

3. **Gerekli kütüphaneleri yükleyin:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Ortam değişkenlerini yapılandırın:**
   `.env.example` dosyasını `.env` adıyla kopyalayın:
   ```bash
   cp .env.example .env
   ```

5. **Uygulamayı başlatın:**
   ```bash
   python -m backend.main
   # veya:
   flask --app backend/main.py run
   ```
   Tarayıcıda `127.0.0.1:5000` adresini açın. İlk açılışta `/login` ekranında belirleyeceğiniz şifre yerel veritabanınıza (`app_settings`) hashlenerek kaydedilir.

6. **Testleri çalıştırın:**
   ```bash
   pytest -v
   ```

## Backend Endpoint Tablosu (`backend/main.py`)

| Method | Endpoint | Açıklama | Şablon / Yanıt |
| :--- | :--- | :--- | :--- |
| `GET`, `POST` | `/login` | İlk kurulum şifresi belirleme ve uygulama kilidi açma | `login.html` / JSON |
| `GET`, `POST` | `/logout` | Oturumu kilitleme ve çıkış | `/login` yönlendirmesi |
| `GET` | `/` | Ana sayfa özet kartları, bütçe uyarıları, son harcamalar ve kurlar | `dashboard.html` |
| `GET` | `/expenses` | Tarih, kategori ve tür (`income`/`expense`) filtreli harcama listesi | `expenses.html` |
| `POST` | `/expenses/add` | Yeni manuel gelir veya gider ekleme | `/expenses` veya JSON |
| `POST` | `/expenses/delete/<id>` | Harcama kaydını silme | `/expenses` veya JSON |
| `GET`, `POST` | `/receipts/scan` | Fiş fotoğrafı yükleme (`name="receipt"`) ve OCR önizlemesi üretme | `receipt_scanner.html` |
| `POST` | `/receipts/confirm` | Kullanıcının onayladığı/düzelttiği fişi tek transaction ile kaydetme | `/` veya JSON |
| `GET` | `/receipts/<id>` | Fiş üst bilgisi ve ürün kalemlerini görüntüleme | JSON |
| `GET` | `/analytics` (`/reports`) | Aylık, kategori ve ürün/market bazlı detaylı analiz ekranı | `reports.html` |
| `GET` | `/api/analytics` | Grafikler için tüm analiz verilerini JSON formatında döndürme | JSON |
| `GET` | `/settings` | Şifre değiştirme ve aylık bütçe tanımlama ekranı | `settings.html` |
| `POST` | `/settings/password` | Eski şifre doğrulamasıyla uygulama şifresini güncelleme | `/settings` veya JSON |
| `POST` | `/settings/budget` | Kategori bazlı aylık bütçe limiti belirleme | `/settings` veya JSON |

## Modüller Arası Entegrasyon Sözleşmesi (Interface Contract)

Diğer modüller (`ocr_ml/`, `database/`, `frontend/`) geliştirilirken `backend/` ile %100 uyumlu çalışması için aşağıdaki standart veri yapıları ve fonksiyon imzaları kullanılır:

### 1. Ortak Fiş Veri Formatı (`receipt_data`)
```python
receipt_data = {
    "merchant": "Migros",          # str
    "date": "2026-11-06",          # str (YYYY-MM-DD, DD.MM.YYYY veya DD/MM/YYYY)
    "total": 850.00,               # float (> 0)
    "items": [
        {
            "name": "Süt",         # str ('name' veya 'product_name' desteklenir)
            "quantity": 2.0,       # float (> 0)
            "unit_price": 45.00,   # float (> 0)
            "total_price": 90.00,  # float (verilmezse quantity * unit_price hesaplanır)
            "category": "Gıda",    # str
            "confidence": 0.96     # float (0.0 - 1.0)
        }
    ]
}
```

### 2. OCR / ML Modülü (`ocr_ml/`) Beklenen Fonksiyonlar
* `image_processor.py` -> `preprocess_image(image_path: str)`
* `receipt_ocr.py` -> `extract_text(image_path: str) -> list[dict]`
* `receipt_parser.py` -> `parse_receipt(ocr_results: list[dict]) -> dict`
* `receipt_validator.py` -> `validate_receipt(data: dict) -> dict`
* `categorizer_ml.py` -> `predict_category(product_name: str) -> str`

### 3. Database Modülü (`database/`) Beklenen Fonksiyonlar
* `db_connection.py` -> `get_connection()`
* `crud.py` -> `create_expense(data: dict) -> int`, `create_receipt(receipt_data: dict) -> int`, `get_expenses() -> list[dict]`, `get_receipt_items(receipt_id: int) -> list[dict]`, `delete_expense(expense_id: int) -> bool`
