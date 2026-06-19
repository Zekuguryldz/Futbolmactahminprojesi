"""Deney: Hucum/savunma gucu, dinlenme gunu ve mac onemi (eleme/dostluk)
ozelliklerinin, mevcut (Elo+Form+H2H) ozellik setine gercekten ek deger
katip katmadigini olcer."""
import pandas as pd
from collections import defaultdict, deque
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from ortak import veri_yukle, train_test_ayir, OZELLIKLER
import warnings
warnings.filterwarnings("ignore")

print("=" * 60)
print("ADIM 13 — iLERi OZELLiKLER DENEYi (Hucum/Savunma, Dinlenme, Mac Onemi)")
print("=" * 60)

df = veri_yukle().sort_values("Date").reset_index(drop=True)
df["Date_dt"] = pd.to_datetime(df["Date"])

# --- Hucum/Savunma gucu (son 5 mactaki ortalama atilan/yenen gol, leak-free) ---
gol_atilan = defaultdict(lambda: deque(maxlen=5))
gol_yenen = defaultdict(lambda: deque(maxlen=5))
home_hucum, away_hucum = [], []
home_savunma, away_savunma = [], []

# --- Dinlenme gunu (son mactan bu yana gecen gun, leak-free) ---
son_mac_tarihi = {}
home_dinlenme, away_dinlenme = [], []

for _, row in df.iterrows():
    h, a = row["Home Team"], row["Away Team"]

    home_hucum.append(sum(gol_atilan[h]) / len(gol_atilan[h]) if gol_atilan[h] else 1.3)
    away_hucum.append(sum(gol_atilan[a]) / len(gol_atilan[a]) if gol_atilan[a] else 1.3)
    home_savunma.append(sum(gol_yenen[h]) / len(gol_yenen[h]) if gol_yenen[h] else 1.3)
    away_savunma.append(sum(gol_yenen[a]) / len(gol_yenen[a]) if gol_yenen[a] else 1.3)

    tarih = row["Date_dt"]
    home_dinlenme.append(min((tarih - son_mac_tarihi[h]).days, 365) if h in son_mac_tarihi else 365)
    away_dinlenme.append(min((tarih - son_mac_tarihi[a]).days, 365) if a in son_mac_tarihi else 365)
    son_mac_tarihi[h] = tarih
    son_mac_tarihi[a] = tarih

    gol_atilan[h].append(row["Home Goals"])
    gol_yenen[h].append(row["Away Goals"])
    gol_atilan[a].append(row["Away Goals"])
    gol_yenen[a].append(row["Home Goals"])

df["Home_Hucum"] = home_hucum
df["Away_Hucum"] = away_hucum
df["Home_Savunma"] = home_savunma
df["Away_Savunma"] = away_savunma
df["Hucum_Savunma_Fark"] = (df["Home_Hucum"] - df["Away_Savunma"]) - (df["Away_Hucum"] - df["Home_Savunma"])
df["Home_Dinlenme"] = home_dinlenme
df["Away_Dinlenme"] = away_dinlenme
df["Dinlenme_Fark"] = df["Home_Dinlenme"] - df["Away_Dinlenme"]

# --- Mac onemi: eleme (qualification) mi, dostluk mu ---
df["Is_Qualification"] = df["Tournament"].str.contains("qualification", case=False, na=False).astype(int)
df["Is_Friendly"] = (df["Tournament"] == "Friendly").astype(int)

YENI_EK_OZELLIKLER = ["Hucum_Savunma_Fark", "Dinlenme_Fark", "Is_Qualification", "Is_Friendly"]
TUM_OZELLIKLER = OZELLIKLER + YENI_EK_OZELLIKLER

print(f"\n[Ek Ozellikler Ornek]")
print(df[["Home Team", "Away Team", "Hucum_Savunma_Fark", "Dinlenme_Fark",
          "Is_Qualification", "Is_Friendly"]].tail(5).to_string(index=False))

modeller_fab = lambda: {
    "Karar Agaci"       : DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest"     : RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1),
    "Lojistik Regresyon": LogisticRegression(max_iter=1000, random_state=42),
}

sonuclar = {}
y = df["Target"]

# --- E) Mevcut baseline (Elo+Form+H2H) ---
X_e = df[OZELLIKLER]
X_train_e, X_test_e, y_train_e, y_test_e = train_test_ayir(X_e, y)
for isim, model in modeller_fab().items():
    model.fit(X_train_e, y_train_e)
    y_pred = model.predict(X_test_e)
    y_prob = model.predict_proba(X_test_e)[:, 1]
    sonuclar.setdefault(isim, {})["E) Mevcut (Elo+Form+H2H)"] = accuracy_score(y_test_e, y_pred)
    sonuclar.setdefault(isim, {})["E) AUC-ROC"] = roc_auc_score(y_test_e, y_prob)

# --- F) Mevcut + Hucum/Savunma + Dinlenme + Mac Onemi ---
X_f = df[TUM_OZELLIKLER]
X_train_f, X_test_f, y_train_f, y_test_f = train_test_ayir(X_f, y)
for isim, model in modeller_fab().items():
    model.fit(X_train_f, y_train_f)
    y_pred = model.predict(X_test_f)
    y_prob = model.predict_proba(X_test_f)[:, 1]
    sonuclar.setdefault(isim, {})["F) +Hucum/Savunma+Dinlenme+MacOnemi"] = accuracy_score(y_test_f, y_pred)
    sonuclar.setdefault(isim, {})["F) AUC-ROC"] = roc_auc_score(y_test_f, y_prob)

ozet = pd.DataFrame(sonuclar).T
print("\n[Karsilastirma Tablosu]")
print(ozet.to_string(float_format=lambda x: f"{x:.4f}"))

# --- Random Forest ile yeni ozelliklerin onem siralamasi ---
rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
rf.fit(X_train_f, y_train_f)
onem = pd.Series(rf.feature_importances_, index=TUM_OZELLIKLER).sort_values(ascending=False)
print("\n[Tum Ozelliklerin Onem Siralamasi — Random Forest]")
print(onem.to_string(float_format=lambda x: f"{x:.4f}"))

print("\n=> Adim 13 tamamlandi.")
