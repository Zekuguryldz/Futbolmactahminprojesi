# Futbol Maç Sonucu Tahmin Projesi — Tam Dokümantasyon

## Projenin Amacı

Bu proje, uluslararası futbol maçlarına ait geçmiş verilerden yola çıkarak **"ev sahibi takım kazanır mı?"** sorusunu makine öğrenmesi ile tahmin etmeyi amaçlar. Veri yükleme → keşifsel analiz → istatistik → görselleştirme → özellik mühendisliği → model eğitimi → metrik hesaplama → karşılaştırma → özellik önemi → karar ağacı görselleştirme → özet rapor şeklinde 11 adımlı bir pipeline oluşturulmuştur.

## Kullanılan Veri Setleri

| Dosya | İçerik |
|---|---|
| `international_matches1.csv` | Tüm uluslararası maçlar (ana veri seti) |
| `world_cups1.csv` | Dünya Kupası edisyonları (yıl, gol, maç sayısı) |
| `world_cup_matches1.csv` | Dünya Kupası maç detayları (aşama bilgisiyle) |
| `2022_world_cup_matches1.csv` | 2022 Katar Dünya Kupası maçları |

---

## ADIM 1 — Veri Yükleme (`01_veri_yukleme.py`)

### Ne Yapar?
4 CSV dosyasını pandas ile belleğe yükler ve temel veri kalitesi kontrollerini gerçekleştirir.

### Kod Mantığı

```python
df_cups    = pd.read_csv("world_cups1.csv")
df_matches = pd.read_csv("world_cup_matches1.csv")
df_2022    = pd.read_csv("2022_world_cup_matches1.csv")
df_intl    = pd.read_csv("international_matches1.csv")
```
`pd.read_csv()` — CSV dosyasını satır/sütun yapısındaki bir DataFrame'e dönüştürür.

### Kontroller ve Neden Yapılır?

| Kontrol | Kod | Neden? |
|---|---|---|
| Boyut (satır x sütun) | `df.shape` | Veri ne kadar büyük, kaç özellik var? |
| Sütun isimleri | `df.columns` | Hangi değişkenlerle çalışacağız? |
| İlk 3 satır | `df.head(3)` | Veri nasıl görünüyor, formatı doğru mu? |
| Veri tipleri | `df.dtypes` | Sayısal olması gereken sütun metin mi gelmiş? |
| Eksik değer | `df.isnull().sum().sum()` | Boş hücre var mı? Model bozulmasın. |
| Tekrar eden satır | `df.duplicated().sum()` | Aynı maç iki kez mi girilmiş? |

---

## ADIM 2 — Keşifsel Veri Analizi / EDA (`02_eda.py`)

### Ne Yapar?
Verinin genel özelliklerini, dağılımlarını ve örüntülerini keşfeder. Model kurmadan önce "veriyi tanımak" aşamasıdır.

### Maç Sonucu Fonksiyonu

```python
def sonuc_belirle(row):
    if row["Home Goals"] > row["Away Goals"]:
        return "Ev Sahibi Kazandı"
    elif row["Home Goals"] < row["Away Goals"]:
        return "Deplasman Kazandı"
    else:
        return "Beraberlik"

df["Sonuc"] = df.apply(sonuc_belirle, axis=1)
```

- `apply(fonksiyon, axis=1)` — Her satıra (maça) bu fonksiyonu uygular.
- Fonksiyon, ev ve deplasman gollerini karşılaştırarak 3 sınıftan birini döner.
- Böylece ham gol sayılarından kategorik bir **hedef değişken** türetilmiş olur.

### Neden EDA Yapılır?
- Hangi sonuç (ev/deplasman/beraberlik) daha sık olduğunu görmek için.
- Hangi turnuvalarda kaç maç oynandığını anlamak için.
- Gol dağılımının şeklini (sağa mı, sola mı çarpık?) görmek için.
- Ev sahibi avantajının gerçek mi olduğunu sorgulamak için.

---

## ADIM 3 — İstatistiksel Analiz (`03_istatistiksel_analiz.py`)

### Ne Yapar?
Sayısal istatistik araçları kullanarak veriyi sayısal olarak özetler.

### Merkezi Eğilim Ölçüleri

```python
df['Home Goals'].mean()    # Ortalama: tüm değerlerin toplamı / sayısı
df['Home Goals'].median()  # Medyan: sıralı dizinin ortasındaki değer
df['Home Goals'].std()     # Standart sapma: değerlerin ortalamadan ne kadar saptığı
```

- **Ortalama vs Medyan farkı önemlidir:** Gol dağılımı sağa çarpık (çok az maçta 10+ gol) olduğunda ortalama medyandan büyük çıkar. Medyan daha sağlıklı merkezi eğilimi gösterir.

### Ev Sahipliği Etkisi

```python
ev = df.groupby("Home Stadium or Not")[["Home Goals","Away Goals","Toplam Gol"]].mean()
```

- `groupby()` — Veriyi bir sütunun değerlerine göre gruplar.
- Kendi stadında oynayan takımın gol ortalamalarını, nötr sahada oynayan takımın ortalamasıyla karşılaştırır.
- **Algoritma mantığı:** Ev sahibi avantajı var mı? Bu soru, istatistik ile doğrulanır.

### Korelasyon Matrisi

```python
df[["Home Goals","Away Goals","Home Stadium or Not","Toplam Gol"]].corr()
```

- Korelasyon -1 ile +1 arasında değer alır.
- +1 → güçlü pozitif ilişki, -1 → güçlü negatif ilişki, 0 → ilişki yok.
- **Neden önemli?** İlişkili özellikler modeli kafa karıştırabilir (multicollinearity).

### Dünya Kupası Gol Trendi

```python
df_cups["Mac Basi Gol"] = df_cups["Goals Scored"] / df_cups["Matches Played"]
```

- Her edisyondaki toplam golü maç sayısına böler.
- Futbolun yıllar içinde daha az gollü bir hal alıp almadığını gösterir.

### Yüzdelik Dilimler

```python
np.percentile(df['Toplam Gol'], p)
```

- %75 → Maçların %75'inde toplam gol bu değerin altında.
- **Ne işe yarar?** Aykırı değerleri anlamaya ve veriyi kesimlere ayırmaya yarar.

---

## ADIM 4 — Görselleştirme (`04_gorsellestirme.py`)

### Ne Yapar?
9 farklı grafik oluşturarak analizleri görsel hale getirir ve `veri_analizi.png` olarak kaydeder.

### Kullanılan Kütüphaneler

```python
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
```

- **matplotlib** — Python'un temel grafik kütüphanesi.
- **seaborn** — matplotlib üzerine kurulu, istatistiksel grafikler için.
- **gridspec** — Birden fazla grafiği ızgara düzeninde yerleştirmek için.

### GridSpec Düzeni

```python
fig = plt.figure(figsize=(20, 16))
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)
```

3x3 = 9 grafik alanı oluşturur. `hspace` ve `wspace` grafiklerin dikey/yatay aralığıdır.

### 9 Grafiğin Açıklaması

| # | Grafik Türü | Ne Gösterir? | Neden Kullanıldı? |
|---|---|---|---|
| 1 | Pasta grafiği | Maç sonucu dağılımı (%) | Oranları göstermenin en kolay yolu |
| 2 | Histogram | Gol sayısı yoğunluk dağılımı | Dağılım şeklini (Poisson?) görmek için |
| 3 | Kutu grafiği (boxplot) | Ev sahipliği avantajı | Medyan ve aykırı değerleri karşılaştırmak için |
| 4 | Çubuk grafik | Yıllık DK toplam gol | Tarihsel trendi görmek için |
| 5 | Çizgi grafik | Maç başına gol trendi | Zaman serisi değişimi için |
| 6 | Yatay çubuk | Top 10 turnuva | Sıralı kategorileri göstermek için |
| 7 | Yatay çubuk | Aşamaya göre ort. gol | Gruplama sonucu karşılaştırma için |
| 8 | Isı haritası (heatmap) | Korelasyon matrisi | Sayısal ilişkileri renk kodlu göstermek için |
| 9 | Gruplu çubuk | Sonuca göre ortalama gol | İki grubu yan yana karşılaştırmak için |

---

## ADIM 5 — Özellik Mühendisliği (`05_ozellik_muhendisligi.py`)

### Ne Yapar?
Ham veriyi makine öğrenmesi modeline beslenebilecek sayısal özelliklere dönüştürür.

### 1. Turnuva Tipi Özelliği

```python
def tur_tipi(t):
    t = str(t).lower()
    if "fifa" in t or "world cup" in t:
        return 2
    elif any(k in t for k in ["copa", "euro", "africa", "asian", "confederation"]):
        return 1
    return 0
```

- Turnuva adı metnini analiz ederek 3 seviyeli ordinal değişken oluşturur.
- **Neden?** "FIFA World Cup" ve "Friendly" aynı öneme sahip maçlar değildir; model bunu öğrenmeli.
- **Algoritma:** Kural tabanlı metin eşleştirme (rule-based text matching).

### 2. LabelEncoder — Kategorik → Sayısal

```python
le = LabelEncoder()
df["Home_Enc"] = le.fit_transform(df["Home Team"])
df["Away_Enc"] = le.fit_transform(df["Away Team"])
```

- Takım isimlerini (metin) sıralı sayılara çevirir. "Argentina" → 5, "Brazil" → 12 gibi.
- **Neden?** Makine öğrenmesi algoritmaları metin değil sayı işler.
- **Sınırlılık:** LabelEncoder sıralı ilişki ima eder (5 < 12), ancak takımlar arasında böyle bir sıra yoktur. Daha doğru yaklaşım One-Hot Encoding olurdu; burada basitlik için LabelEncoder tercih edilmiştir.

### 3. Hedef Değişken (Target)

```python
df["Target"] = (df["Sonuc"] == "Ev Sahibi Kazandi").astype(int)
```

- 3 sınıflı (`Ev Sahibi / Deplasman / Beraberlik`) problemi 2 sınıflı (binary) hale getirir.
- `True` → 1 (kazandı), `False` → 0 (kazanmadı).
- **Neden binary?** Binary sınıflandırma daha basit, yorumlanması daha kolay.

### 4. Train/Test Bölme

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

- Veriyi %80 eğitim / %20 test olarak böler.
- `stratify=y` — Her iki bölümde de hedef sınıfların oranı eşit kalır (örn. %43 kazandı oranı her iki kümede de korunur).
- `random_state=42` — Aynı bölme tekrar üretilebilsin diye sabit bir rastgelelik tohumu.

---

## ADIM 6 — Model Eğitimi (`06_model_egitimi.py`)

### Ne Yapar?
3 farklı makine öğrenmesi algoritması eğitir ve her birinin performansını raporlar.

### Kullanılan 3 Algoritma

#### a) Karar Ağacı (Decision Tree)

```python
DecisionTreeClassifier(max_depth=5, random_state=42)
```

- **Nasıl çalışır?** Veriyi sorularla böler: "Home_Enc <= 47.5 mi?" → Evet → "Home Stadium or Not <= 0.5 mi?" → ... Ağaç dallanarak her yaprağa bir sınıf atar.
- **max_depth=5** — Ağaç en fazla 5 seviye derine iner. Sınır koymazsak ezber (overfitting) yapar.
- **Avantaj:** Yorumlanabilir, kural üretir.
- **Dezavantaj:** Tek ağaç, aşırı ezbere eğilimli.

#### b) Random Forest (Rastgele Orman)

```python
RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
```

- **Nasıl çalışır?** 100 farklı karar ağacı eğitir. Her ağaç, verinin rastgele bir alt kümesinde ve rastgele seçilmiş özelliklerde çalışır. Sonuç: 100 ağacın **oylama** sonucu (majority voting).
- **n_estimators=100** — 100 ağaç oluştur.
- **n_jobs=-1** — Tüm CPU çekirdeklerini paralel kullan.
- **Avantaj:** Tek ağaçtan çok daha güçlü, aşırı öğrenmeye karşı dirençli (ensemble yöntemi).

#### c) Lojistik Regresyon (Logistic Regression)

```python
LogisticRegression(max_iter=1000, random_state=42)
```

- **Nasıl çalışır?** Her özelliğe bir ağırlık (w) atar: `P(kazandı) = sigmoid(w1*Home_Enc + w2*Away_Enc + w3*Stadium + w4*Tur_Tipi + b)`. Sigmoid fonksiyonu çıktıyı 0-1 arasına sıkıştırır ve bu olasılık 0.5'ten büyükse "kazandı" der.
- **max_iter=1000** — Ağırlıkların optimize edilmesi için maksimum iterasyon sayısı.
- **Avantaj:** Hızlı, basit, olasılık çıktısı verir.
- **Dezavantaj:** Doğrusal olmayan ilişkileri yakalayamaz.

### Eğitim ve Değerlendirme Döngüsü

```python
for isim, model in modeller.items():
    model.fit(X_train, y_train)       # Model eğitimi
    y_pred = model.predict(X_test)    # Test verisi üzerinde tahmin
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc  = accuracy_score(y_test, y_pred)
    print(f"Overfitting: {train_acc - test_acc:+.4f}")
```

- `fit()` — Modeli eğitim verisiyle eğitir (parametreler öğrenilir).
- `predict()` — Yeni veriye tahmin uygular.
- **Overfitting farkı** = Train Acc − Test Acc. Yüksekse model eğitimi ezberlemiş, genelleme yapamıyor.

---

## ADIM 7 — Metrikler (`07_metrikler.py`)

### Ne Yapar?
4 temel sınıflandırma metriğini hesaplar, hem tablo hem de grafik olarak sunar.

### 4 Temel Metrik

```
Accuracy  = (TP + TN) / Toplam
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1 Score  = 2 * (Precision * Recall) / (Precision + Recall)
```

| Metrik | Ne Ölçer? | Ne Zaman Önemli? |
|---|---|---|
| **Accuracy** | Genel doğruluk | Sınıflar dengeli olduğunda |
| **Precision** | "Kazandı" dediğimizin ne kadarı gerçekten kazandı? | Yanlış pozitifin maliyeti yüksekse |
| **Recall** | Gerçekten kazananların ne kadarını yakaladık? | Yanlış negatifin maliyeti yüksekse |
| **F1 Score** | Precision ve Recall'un harmonik ortalaması | İkisi arasında denge istendiğinde |

### Radar Grafiği

```python
angles = np.linspace(0, 2 * np.pi, len(metrik_isimleri), endpoint=False).tolist()
ax2.plot(angles, degerler, "o-", linewidth=2, color=renk)
ax2.fill(angles, degerler, alpha=0.10, color=renk)
```

- 4 metriği dairesel eksenlere yerleştirerek modellerin "güçlü ve zayıf yanlarını" tek bakışta görmeyi sağlar.
- `np.linspace(0, 2π, 4)` — Daireyi 4 eşit parçaya böler (her parça bir metrik).

---

## ADIM 8 — Model Karşılaştırma & Overfitting (`08_model_karsilastirma.py`)

### Ne Yapar?
Modelleri çok boyutlu olarak karşılaştırır: ROC eğrisi, cross-validation, öğrenme eğrisi, confusion matrix.

### AUC-ROC

```python
y_prob = model.predict_proba(X_test)[:, 1]   # Kazanma olasılıkları
roc_auc_score(y_test, y_prob)
fpr, tpr, _ = roc_curve(y_test, y_prob)
```

- `predict_proba()` — Modelin her örnek için olasılık çıkarması.
- **ROC eğrisi:** Eşik değeri 0→1 arasında değiştirildiğinde False Positive Rate vs True Positive Rate grafiği.
- **AUC (Alan altındaki alan):** 1.0 = mükemmel, 0.5 = rastgele tahmin.
- Accuracy'den daha güvenilirdir çünkü sınıf dengesizliğinden etkilenmez.

### Cross-Validation (Çapraz Doğrulama)

```python
cv_sc = cross_val_score(model, X, y, cv=5, scoring="accuracy", n_jobs=-1)
```

- Veriyi 5 eşit parçaya böler. Her seferinde 4 parça eğitim, 1 parça test olarak kullanılır.
- 5 farklı test sonucunun ortalaması alınır.
- **Neden?** Tek bir train/test bölmesi şansa bağlı olabilir. 5-fold daha güvenilir bir performans tahmini verir.

### Öğrenme Eğrisi (Learning Curve)

```python
train_sizes, train_sc, val_sc = learning_curve(
    modeller["Random Forest"], X, y,
    train_sizes=np.linspace(0.1, 1.0, 8), cv=5
)
```

- Eğitim veri miktarı arttıkça modelin performansının nasıl değiştiğini gösterir.
- **Tipik örüntüler:**
  - Train acc yüksek, val acc düşük → Overfitting (daha fazla veri veya regularization gerekli)
  - İkisi de düşük → Underfitting (daha güçlü model gerekli)
  - İkisi yakın ve yüksek → İdeal model

### Confusion Matrix

```python
cm = confusion_matrix(y_test, sonuclar[isim]["y_pred"])
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
```

```
                 TAHMİN
              Kazanmadı  Kazandı
GERÇEK  Kazanmadı  [ TN  |  FP ]
        Kazandı    [ FN  |  TP ]
```

- **TN (True Negative):** Kazanmadı dedi, gerçekten kazanmadı.
- **TP (True Positive):** Kazandı dedi, gerçekten kazandı.
- **FP (False Positive):** Kazandı dedi ama kazanmadı (yanlış alarm).
- **FN (False Negative):** Kazanmadı dedi ama kazandı (kaçırılan).

---

## ADIM 9 — Özellik Önemi (`09_ozellik_onemi.py`)

### Ne Yapar?
Random Forest'in hangi özelliğe ne kadar önem verdiğini çıkarır ve görselleştirir.

### Özellik Önemi Nasıl Hesaplanır?

```python
importance = pd.Series(rf.feature_importances_, index=ozellikler).sort_values(ascending=False)
```

Random Forest'te her ağaç, bir özelliği kullanarak veriyi böldüğünde **Gini safsızlığını** ne kadar azalttığını ölçer. 100 ağaçtaki bu katkıların ortalaması alınarak her özelliğe bir önem skoru atanır. Tüm skorların toplamı 1'dir.

### ASCII Bar Gösterimi

```python
bar = "█" * int(skor * 50)
print(f"  {oz:<25}: {skor:.4f}  {bar}")
```

Skoru 50 ile çarpıp tam sayıya yuvarlar → bu kadar blok karakter basar. Hızlı görsel karşılaştırma sağlar.

### Yorumlama
- En yüksek skora sahip özellik, modelin tahminlerinde en belirleyici faktördür.
- Düşük skorlu özellikler çıkarılabilir (boyut küçültme).

---

## ADIM 10 — Karar Ağacı Görselleştirme (`10_karar_agaci.py`)

### Ne Yapar?
Karar ağacını görsel ağaç diyagramı ve metin kuralları olarak sunar.

### Metin Kuralları

```python
dt_kisa = DecisionTreeClassifier(max_depth=3, random_state=42)
dt_kisa.fit(X_train, y_train)
kural_metni = export_text(dt_kisa, feature_names=ozellikler)
```

Çıktı şöyle görünür:
```
|--- Home_Enc <= 47.50
|   |--- Home Stadium or Not <= 0.50
|   |   |--- class: 0 (Kazanmadı)
|   |--- Home Stadium or Not >  0.50
|   |   |--- class: 1 (Kazandı)
```

Her satır bir **karar kuralı**dır. Model bu if-else zinciriyle çalışır.

### Grafik Parametreleri

```python
plot_tree(
    dt,
    feature_names=ozellikler,
    class_names=["Kazanmadi", "Kazandi"],
    filled=True,     # Düğümleri sınıfa göre renklendir
    rounded=True,    # Yuvarlak kenar
    fontsize=9,
    max_depth=3,     # Yalnızca ilk 3 seviyeyi göster (okunabilirlik)
)
```

### Karar Ağacı Düğüm Bilgisi

Her düğümde şunlar yazılır:

| Bilgi | Anlamı |
|---|---|
| `gini = 0.48` | Safsızlık ölçüsü (0=tam saf, 0.5=en karışık) |
| `samples = 2400` | Bu düğüme düşen örnek sayısı |
| `value = [1400, 1000]` | [Kazanmadı sayısı, Kazandı sayısı] |
| `class = Kazanmadi` | Çoğunluk sınıfı (yaprak tahmin sonucu) |

---

## ADIM 11 — Özet Rapor (`11_ozet_rapor.py`)

### Ne Yapar?
Tüm adımları bir araya getirerek 3 modelin bütün metriklerini tek tabloda özetler ve ısı haritası olarak görselleştirir.

### Özet Tablo Yapısı

```python
ozet = pd.DataFrame(sonuclar).T
```

Her satır bir model, her sütun bir metriktir:

| | Train Acc | Test Acc | Precision | Recall | F1 Score | AUC-ROC | CV Mean | CV Std | Overfit |
|---|---|---|---|---|---|---|---|---|---|
| Karar Ağacı | ... | ... | ... | ... | ... | ... | ... | ... | ... |
| Random Forest | ... | ... | ... | ... | ... | ... | ... | ... | ... |
| Lojistik Reg. | ... | ... | ... | ... | ... | ... | ... | ... | ... |

### En İyi Model Seçimi

```python
en_iyi_auc = max(sonuclar, key=lambda k: sonuclar[k]["AUC-ROC"])
en_iyi_f1  = max(sonuclar, key=lambda k: sonuclar[k]["F1 Score"])
en_iyi_acc = max(sonuclar, key=lambda k: sonuclar[k]["Test Acc"])
```

Farklı metrikler farklı modeli en iyi seçebilir — bu tasarımlı bir karşılaştırmadır.

### Overfitting Değerlendirme Mantığı

```python
fark = v["Train Acc"] - v["Test Acc"]
if fark > 0.05:    → "YÜKSEK overfitting riski"
elif fark > 0.02:  → "ORTA overfitting"
else:              → "Düşük overfitting — model iyi genelliyor"
```

Bu eşik değerleri (0.05, 0.02) makul pratik kurallardır. Projenin veri büyüklüğüne ve iş gereksinimlerine göre ayarlanabilir.

---

## Pipeline Özeti — Kavramsal Akış

```
CSV Dosyaları
      |
      v
[01] Veri Yükleme          → Pandas DataFrame'ler
      |
      v
[02] EDA                   → Dağılım, oran, kategorik analiz
      |
      v
[03] İstatistiksel Analiz  → Ortalama, std, korelasyon, yüzdelik
      |
      v
[04] Görselleştirme        → 9 grafik → veri_analizi.png
      |
      v
[05] Özellik Mühendisliği  → LabelEncoder, Tur_Tipi, Target, %80/%20 bölme
      |
      v
[06] Model Eğitimi         → Karar Ağacı + Random Forest + Lojistik Reg.
      |
      v
[07] Metrikler             → Accuracy / Precision / Recall / F1 + Radar
      |
      v
[08] Model Karşılaştırma   → ROC, CV, Öğrenme Eğrisi, Confusion Matrix
      |
      v
[09] Özellik Önemi         → Hangi özellik daha belirleyici?
      |
      v
[10] Karar Ağacı Görsel    → Ağaç diyagramı + if-else kuralları
      |
      v
[11] Özet Rapor            → Tüm metrikler tek tabloda + ısı haritası
```

---

## Kullanılan Kütüphaneler

| Kütüphane | Versiyon | Ne İçin? |
|---|---|---|
| `pandas` | — | Veri yükleme, gruplama, tablo işlemleri |
| `numpy` | — | Sayısal hesaplama, yüzdelik, linspace |
| `matplotlib` | — | Grafik çizimi (temel) |
| `seaborn` | — | İstatistiksel grafikler (heatmap, boxplot) |
| `sklearn.preprocessing` | — | LabelEncoder |
| `sklearn.model_selection` | — | train_test_split, cross_val_score, learning_curve |
| `sklearn.tree` | — | DecisionTreeClassifier, plot_tree, export_text |
| `sklearn.ensemble` | — | RandomForestClassifier |
| `sklearn.linear_model` | — | LogisticRegression |
| `sklearn.metrics` | — | accuracy, precision, recall, f1, roc_auc, confusion_matrix |

---

## Çalıştırma Sırası

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
