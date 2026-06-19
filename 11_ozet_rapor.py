import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)
from ortak import model_verisi_hazirla
import warnings
warnings.filterwarnings("ignore")

print("=" * 60)
print("ADIM 11 — OZET RAPOR")
print("=" * 60)

# --- Veri hazırlık ---
df, X, y, X_train, X_test, y_train, y_test = model_verisi_hazirla()

modeller = {
    "Karar Agaci"       : DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest"     : RandomForestClassifier(n_estimators=100, max_depth=5,
                                                  random_state=42, n_jobs=-1),
    "Lojistik Regresyon": LogisticRegression(max_iter=1000, random_state=42),
}

# --- Tüm metrikleri hesapla ---
sonuclar = {}
for isim, model in modeller.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    cv     = cross_val_score(model, X, y, cv=5, scoring="accuracy", n_jobs=-1)
    sonuclar[isim] = {
        "Train Acc" : accuracy_score(y_train, model.predict(X_train)),
        "Test Acc"  : accuracy_score(y_test, y_pred),
        "Precision" : precision_score(y_test, y_pred, zero_division=0),
        "Recall"    : recall_score(y_test, y_pred, zero_division=0),
        "F1 Score"  : f1_score(y_test, y_pred, zero_division=0),
        "AUC-ROC"   : roc_auc_score(y_test, y_prob),
        "CV Mean"   : cv.mean(),
        "CV Std"    : cv.std(),
        "Overfit"   : accuracy_score(y_train, model.predict(X_train)) - accuracy_score(y_test, y_pred),
    }

# --- Tablo yazdır ---
ozet = pd.DataFrame(sonuclar).T
print("\n[Tam Metrik Tablosu]")
print(ozet.to_string(float_format=lambda x: f"{x:.4f}"))

# --- En iyi model ---
en_iyi_auc  = max(sonuclar, key=lambda k: sonuclar[k]["AUC-ROC"])
en_iyi_f1   = max(sonuclar, key=lambda k: sonuclar[k]["F1 Score"])
en_iyi_acc  = max(sonuclar, key=lambda k: sonuclar[k]["Test Acc"])
print(f"\n=> En yuksek AUC-ROC : {en_iyi_auc}")
print(f"=> En yuksek F1 Score: {en_iyi_f1}")
print(f"=> En yuksek Accuracy: {en_iyi_acc}")

# --- Overfitting yorumu ---
print("\n[Overfitting Degerlendirme]")
for isim, v in sonuclar.items():
    fark = v["Overfit"]
    if fark > 0.05:
        durum = "YUKSEK overfitting riski"
    elif fark > 0.02:
        durum = "ORTA overfitting"
    else:
        durum = "Dusuk overfitting — model iyi genelliyor"
    print(f"  {isim:<22}: {durum} ({fark:+.4f})")

# --- Görselleştirme: Özet ısı haritası ---
fig, ax = plt.subplots(figsize=(10, 4))
metrikler_goster = ["Test Acc", "Precision", "Recall", "F1 Score", "AUC-ROC", "CV Mean"]
ozet_num = ozet[metrikler_goster].astype(float)
import seaborn as sns
sns.heatmap(ozet_num, annot=True, fmt=".4f", cmap="YlGn",
            linewidths=0.5, ax=ax, vmin=0.4, vmax=0.85)
ax.set_title("Model Metrik Karsilastirma Tablosu", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("ozet_rapor.png", bbox_inches="tight")
plt.show()
print("=> ozet_rapor.png kaydedildi.")

print("\n" + "=" * 60)
print("PROJE TAMAMLANDI")
print("=" * 60)
print("""
Calistirma sirasi:
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
""")
