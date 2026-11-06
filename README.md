# Smart Expense Analyzer (expense-calculation-program)

Kişisel harcamaları takip etmek, fiş fotoğraflarından otomatik harcama verisi çıkarmak, bütçe limitlerini izlemek ve kategori/ürün bazlı finansal analizler sunmak amacıyla geliştirilen modüler Python web uygulaması.

## Proje Amacı

Kullanıcıların gelir ve giderlerini tek bir merkezden yönetebilmesini, market/mağaza fişlerini tarayarak harcama kalemlerini dijitalleştirmesini ve aylık bütçe hedeflerine göre harcama alışkanlıklarını grafiklerle analiz edebilmesini sağlar. Yerel ortamda kişisel finans verilerini korumak için tek kullanıcılı uygulama kilidi (**Application Lock**) içerir.

## Planlanan Özellikler

- **Uygulama Kilidi (Application Lock):** Uygulama açılışında şifre koruması, oturum yönetimi ve hareketsizlik durumunda otomatik kilitleme.
- **Manuel Gelir & Gider Yönetimi:** Tarih, açıklama, kategori ve tutar bilgileriyle gelir/gider ekleme, filtreleme ve silme.
- **Akıllı Fiş Tarama (OCR & ML):** Fiş fotoğraflarından işletme adı, tarih, ürünler ve fiyatların çıkarılması, kullanıcı onayı/düzenlemesi sonrasında veritabanına kaydedilmesi.
- **Otomatik Kategorizasyon:** Harcama ve fiş kalemlerinin içeriklerine göre kategorilere ayrılması.
- **Bütçe ve Analitik Paneli (Dashboard):** Toplam gelir, gider, bakiye özetleri, kategori dağılım grafikleri, aylık harcama eğilimleri, ürün/market bazlı raporlar ve bütçe aşım uyarıları.

## Kullanılan Teknolojiler

- **Backend:** Python, Flask, Werkzeug, python-dotenv
- **Veritabanı:** SQLite (`sqlite3`)
- **Veri Analizi ve Görselleştirme:** Pandas, NumPy, Matplotlib, Plotly
- **Görüntü İşleme ve OCR / ML:** OpenCV, Pillow, OCR & Kategori Tahmin Modülleri
- **Frontend:** HTML5, CSS3, JavaScript, Jinja2 Şablon Motoru
- **Test:** Pytest

## Klasör Yapısı

```text
expense-calculation-program/
├── backend/                  # Flask uygulaması, route'lar ve iş mantığı
│   ├── main.py               # Uygulama ana giriş noktası
│   ├── schemas.py            # Ortak veri şemaları ve doğrulama modelleri
│   ├── decorators.py         # Oturum/şifre kontrolü ve yardımcı dekoratörler
│   ├── concurrency_worker.py # Paralel / arka plan görev yöneticisi
│   ├── scraper.py            # Harici veri çekme yardımcı modülü
│   ├── services/             # Servis katmanı (auth, expense, receipt, analytics)
│   └── utils/                # Yardımcı fonksiyonlar (helpers.py)
├── database/                 # SQLite bağlantısı, modeller ve CRUD işlemleri
│   ├── db_connection.py      # Veritabanı bağlantı yöneticisi
│   ├── models.py             # Veritabanı modelleri
│   ├── crud.py               # Ekleme, okuma, güncelleme ve silme fonksiyonları
│   └── schema.sql            # Tablo şemaları (expenses, receipts, receipt_items, budgets, app_settings)
├── frontend/                 # Kullanıcı arayüzü ve görselleştirme bileşenleri
│   ├── analytics.py          # Grafik ve analiz hazırlama modülü
│   ├── templates/            # Jinja2 HTML şablonları (login, dashboard, expenses, receipt_scanner, reports, settings)
│   └── static/               # CSS, JS ve statik görsel dosyaları
├── ocr_ml/                   # Fiş okuma, görüntü işleme ve kategori tahmini
│   ├── image_processor.py    # Görüntü iyileştirme ve ön işleme
│   ├── receipt_ocr.py        # Görselden metin çıkarma
│   ├── receipt_parser.py     # Metinleri ürün, fiyat ve tarihe ayrıştırma
│   ├── receipt_validator.py  # Çıkarılan verilerin tutarlılık kontrolü
│   └── categorizer_ml.py     # Ürün ve harcama kategorisi tahmini
├── tests/                    # Birim ve entegrasyon testleri (pytest)
├── uploads/                  # Yüklenen fiş görselleri (Git takibi dışındadır)
├── .env.example              # Örnek ortam değişkenleri şablonu
├── .gitignore                # Git tarafından yoksayılan dosya kuralları
├── requirements.txt          # Python kütüphane bağımlılıkları
└── README.md                 # Proje dokümantasyonu
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
   `.env.example` dosyasını `.env` adıyla kopyalayın ve içindeki ayarları düzenleyin:
   ```bash
   cp .env.example .env
   ```

5. **Testleri çalıştırın:**
   ```bash
   pytest -v
   ```

## Ekip Görev Dağılımı

| Modül | Klasör | Sorumluluk Alanı |
| :--- | :--- | :--- |
| **1. Backend** | `backend/` | Flask route yönetimi, uygulama kilidi (`auth_service`), servis orkestrasyonu ve modüller arası entegrasyon |
| **2. Frontend & Analitik** | `frontend/` | HTML/CSS/Jinja2 ekranları, kullanıcı etkileşimleri, grafik ve rapor görünümleri |
| **3. OCR / ML** | `ocr_ml/` | Fiş görseli ön işleme, metin çıkarma, ürün/fiyat ayrıştırma, doğrulama ve kategori tahmini |
| **4. Database** | `database/` | SQLite şema tasarımı (`schema.sql`), bağlantı/transaction yönetimi ve CRUD işlemleri |
