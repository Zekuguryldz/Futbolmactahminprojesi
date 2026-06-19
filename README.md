# ⚽ Futbol Maç Sonucu Tahmin Projesi

Uluslararası futbol maçı verilerinden yola çıkarak **"ev sahibi takım kazanır mı?"** sorusunu makine öğrenmesiyle tahmin eden uçtan uca bir proje. Veri yükleme → keşifsel analiz (EDA) → istatistik → görselleştirme → özellik mühendisliği → model eğitimi → metrik karşılaştırma → özellik önemi → karar ağacı görselleştirme → özet rapor adımlarından oluşan 11 adımlı bir pipeline ve bunu interaktif kullanmak için bir **Streamlit arayüzü** içerir.

## Kullanılan Modeller

- Karar Ağacı (Decision Tree)
- Random Forest
- Lojistik Regresyon (Logistic Regression)

Her üç model de aynı `max_depth` ve aynı train/test bölmesi ile adil şekilde karşılaştırılır; Accuracy, Precision, Recall, F1, AUC-ROC ve 5-fold Cross-Validation skorları raporlanır.

## Veri Setleri

| Dosya | İçerik |
|---|---|
| `international_matches1.csv` | Tüm uluslararası maçlar (ana veri seti) |
| `world_cups1.csv` | Dünya Kupası edisyonları (yıl, gol, maç sayısı) |
| `world_cup_matches1.csv` | Dünya Kupası maç detayları |
| `2022_world_cup_matches1.csv` | 2022 Katar Dünya Kupası maçları |

## Kurulum

```bash
pip install -r requirements.txt
```

## Çalıştırma

### Pipeline (11 adım, sırayla)

```bash
python3 01_veri_yukleme.py
python3 02_eda.py
python3 03_istatistiksel_analiz.py
python3 04_gorsellestirme.py
python3 05_ozellik_muhendisligi.py
python3 06_model_egitimi.py
python3 07_metrikler.py
python3 08_model_karsilastirma.py
python3 09_ozellik_onemi.py
python3 10_karar_agaci.py
python3 11_ozet_rapor.py
```

Ortak veri hazırlık mantığı (`tur_tipi`, encoding, train/test split) `ortak.py` modülünde toplanmıştır; `05`–`11` arası adımlar bunu import eder.

### İnteraktif Arayüz (Streamlit)

```bash
streamlit run streamlit_app.py
```

4 sekme sunar:
- **🔮 Tahmin** — iki takım seçip seçtiğin modelle maç sonucu olasılığını görürsün
- **📊 Veri Keşfi (EDA)** — yıl/turnuva filtreli gol dağılımı, ev sahibi kazanma oranı, takım istatistikleri
- **📈 Model Karşılaştırma** — tam metrik tablosu, ROC eğrileri, Confusion Matrix
- **🌳 Özellik Önemi & Karar Ağacı** — Random Forest özellik önemleri, interaktif karar ağacı görseli

## Proje Yapısı

```
01-11_*.py          Pipeline adımları (sırayla çalıştırılır)
ortak.py             Ortak veri hazırlık fonksiyonları (DRY)
streamlit_app.py      İnteraktif web arayüzü
*.csv                Veri setleri
*.png                Pipeline çalıştırıldığında üretilen grafikler
DOKUMAN.md            Adım adım detaylı teknik dokümantasyon
```

## Notlar

- `Home_Enc` / `Away_Enc` özellikleri `LabelEncoder` ile kodlanmıştır; takım isimleri arasında sıralı bir ilişki yoktur, bu basitlik amacıyla tercih edilmiş bir yaklaşımdır (One-Hot Encoding daha doğru bir alternatif olurdu).
- Random Forest ve Karar Ağacı, overfitting karşılaştırmasının adil olması için aynı `max_depth` değeriyle eğitilir.
