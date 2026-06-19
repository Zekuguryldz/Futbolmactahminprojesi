"""Deney: Elo rating + form + head-to-head özellikleri ve 3 sınıflı hedef ile
mevcut (4 özellik, ikili hedef) modeli ne kadar geçiyoruz, ölçer.
Elo/form/h2h hesaplama mantığı ortak.py'de (ozellik_muhendisligi_ekle); burada
sadece deneyin A/B/C/D karşılaştırması yapılır."""
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from ortak import veri_yukle, train_test_ayir, model_verisi_hazirla, ESKI_OZELLIKLER, OZELLIKLER
import warnings
warnings.filterwarnings("ignore")

print("=" * 60)
print("ADIM 12 — GELiSMiS OZELLiKLER DENEYi")
print("=" * 60)

df = veri_yukle()  # Elo/Form/H2H ozellikleri ortak.py icinde otomatik eklenir

print(f"\n[Yeni Ozellikler Ornek]")
print(df[["Home Team", "Away Team", "Home_Elo", "Away_Elo", "Home_Form", "Away_Form",
          "H2H_Ev_Galibiyet_Orani"]].tail(5).to_string(index=False))


def sonuc_3sinif(row):
    if row["Home Goals"] > row["Away Goals"]:
        return 2
    elif row["Home Goals"] < row["Away Goals"]:
        return 0
    return 1


df["Target3"] = df.apply(sonuc_3sinif, axis=1)

modeller_fab = lambda: {
    "Karar Agaci"       : DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest"     : RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1),
    "Lojistik Regresyon": LogisticRegression(max_iter=1000, random_state=42),
}

sonuclar = {}

# --- A) Baseline: eski 4 ozellik, ikili hedef (pinlenmis, ortak.py'deki guncel
#        varsayilan OZELLIKLER degisse bile bu deney sabit kalsin diye) ---
_, _, _, X_train_b, X_test_b, y_train_b, y_test_b = model_verisi_hazirla(ozellikler=ESKI_OZELLIKLER)
for isim, model in modeller_fab().items():
    model.fit(X_train_b, y_train_b)
    sonuclar.setdefault(isim, {})["A) Baseline (4 ozellik, ikili)"] = accuracy_score(y_test_b, model.predict(X_test_b))

# --- B) Yeni ozellikler + ikili hedef (sadece ozellik etkisini izole et) ---
X_yeni = df[OZELLIKLER]
y_ikili = df["Target"]
X_train_c, X_test_c, y_train_c, y_test_c = train_test_ayir(X_yeni, y_ikili)
for isim, model in modeller_fab().items():
    model.fit(X_train_c, y_train_c)
    sonuclar.setdefault(isim, {})["B) Yeni ozellikler + ikili hedef"] = accuracy_score(y_test_c, model.predict(X_test_c))

# --- C) Yeni ozellikler + 3 sinifli hedef (native) ve ikiliye indirgenmis hali ---
y_3 = df["Target3"]
X_train_d, X_test_d, y_train_d, y_test_d = train_test_ayir(X_yeni, y_3)
for isim, model in modeller_fab().items():
    model.fit(X_train_d, y_train_d)
    y_pred_d = model.predict(X_test_d)
    sonuclar.setdefault(isim, {})["C) Yeni ozellik + 3-sinif (native acc)"] = accuracy_score(y_test_d, y_pred_d)
    y_test_bin = (y_test_d == 2).astype(int)
    y_pred_bin = (y_pred_d == 2).astype(int)
    sonuclar.setdefault(isim, {})["D) Ayni model, ikiliye indirgenmis (baseline ile kiyaslanabilir)"] = accuracy_score(y_test_bin, y_pred_bin)

ozet = pd.DataFrame(sonuclar).T
print("\n[Karsilastirma Tablosu — Test Accuracy]")
print(ozet.to_string(float_format=lambda x: f"{x:.4f}"))

print("\n[Naif Baseline'lar (referans icin)]")
print(f"  Ikili hedef cogunluk sinifi tahmini : {max(y_ikili.mean(), 1-y_ikili.mean()):.4f}")
print(f"  3-sinif hedef cogunluk sinifi tahmini: {y_3.value_counts(normalize=True).max():.4f}")

print("\n=> Adim 12 tamamlandi. Sonuc: Elo/Form/H2H ozellikleri ortak.py'deki")
print("   varsayilan OZELLIKLER setine alindi; 06-11 arasi adimlar artik bunu kullaniyor.")
