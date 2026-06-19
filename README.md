# ⚽ Futbol Maç Sonucu Tahmin Projesi

🔗 **Canlı Demo:** [futbolmactahminprojesi-bxrx8x9jyn9yplm7m2jawb.streamlit.app](https://futbolmactahminprojesi-bxrx8x9jyn9yplm7m2jawb.streamlit.app/)

Uluslararası futbol maçı verilerinden yola çıkarak **"ev sahibi takım kazanır mı?"** sorusunu makine öğrenmesiyle tahmin eden uçtan uca bir proje. Veri yükleme → keşifsel analiz (EDA) → istatistik → görselleştirme → özellik mühendisliği → model eğitimi → metrik karşılaştırma → özellik önemi → karar ağacı görselleştirme → özet rapor adımlarından oluşan 11 adımlı bir pipeline ve bunu interaktif kullanmak için bir **Streamlit arayüzü** içerir.

## Kullanılan Modeller

- Karar Ağacı (Decision Tree)
- Random Forest
- Lojistik Regresyon (Logistic Regression)

Her üç model de aynı `max_depth` ve aynı train/test bölmesi ile adil şekilde karşılaştırılır; Accuracy, Precision, Recall, F1, AUC-ROC ve 5-fold Cross-Validation skorları raporlanır.

## Özellikler (Features)

Modeller, her maçtan ÖNCEKİ bilgiyle (leak-free, kronolojik) hesaplanan şu özelliklerle eğitilir:

- **Elo_Fark** — ev sahibi ve deplasman takımının maç öncesi Elo rating farkı (en belirleyici özellik)
- **Home_Form / Away_Form** — takımların son 5 maçtaki ortalama puanı
- **H2H_Ev_Galibiyet_Orani** — bu eşleşmedeki geçmiş ev sahibi galibiyet oranı
- **Home Stadium or Not**, **Tur_Tipi** — stadyum ve turnuva bilgisi

`Adım 12` (`12_gelismis_ozellikler.py`), bu özelliklerin eski (sadece takım ID'si tabanlı) özellik setine göre Test Accuracy'yi ~%57-60'tan ~%69'a çıkardığını ölçen karşılaştırma deneyini içerir.

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

### Pipeline (12 adım, sırayla)

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
python3 12_gelismis_ozellikler.py
```

Ortak veri hazırlık mantığı (`tur_tipi`, Elo/form/h2h özellik mühendisliği, encoding, train/test split) `ortak.py` modülünde toplanmıştır; `06`–`12` arası adımlar bunu import eder.

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
01-12_*.py           Pipeline adımları (sırayla çalıştırılır)
ortak.py              Ortak veri hazırlık + Elo/form/h2h özellik mühendisliği (DRY)
streamlit_app.py       İnteraktif web arayüzü
*.csv                 Veri setleri
*.png                 Pipeline çalıştırıldığında üretilen grafikler
DOKUMAN.md             Adım adım detaylı teknik dokümantasyon
```

## Notlar

- `Home_Enc` / `Away_Enc` (ham takım ID'si) özellikleri hâlâ `ortak.py` içinde hesaplanır ve `ESKI_OZELLIKLER` olarak saklanır (Adım 12'deki kıyaslama deneyi için), ancak canlı pipeline artık bunları değil, Elo/form/h2h tabanlı özellikleri kullanır.
- Random Forest ve Karar Ağacı, overfitting karşılaştırmasının adil olması için aynı `max_depth` değeriyle eğitilir.
- Elo/form/h2h hesaplaması kronolojik sırada yapılır; her satır sadece o maçtan ÖNCEKİ bilgiyi kullanır (data leakage yok).

## Geliştirme Notu

Bu projenin geliştirilmesinde (vibe coding) **Claude (Anthropic)** desteği alınmıştır.
