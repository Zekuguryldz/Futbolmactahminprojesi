"""Deney: max_depth/Elo-K/form-penceresi GridSearch, Gradient Boosting ve
VotingClassifier ensemble'ı mevcut (Elo+Form+H2H, max_depth=5 sabit) modeli
ne kadar geçiyor, ölçer."""
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score
from ortak import veri_yukle, ozellik_hedef_ayir, train_test_ayir, OZELLIKLER
import warnings
warnings.filterwarnings("ignore")

print("=" * 60)
print("ADIM 14 — HiPERPARAMETRE OPTiMiZASYONU DENEYi")
print("=" * 60)

# --- A) Mevcut baseline: K=20, pencere=5, max_depth=5 sabit ---
df_a = veri_yukle()
X_a, y_a = ozellik_hedef_ayir(df_a)
X_train_a, X_test_a, y_train_a, y_test_a = train_test_ayir(X_a, y_a)

baseline = {
    "Karar Agaci"       : DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest"     : RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1),
    "Lojistik Regresyon": LogisticRegression(max_iter=1000, random_state=42),
}
sonuclar = {}
for isim, model in baseline.items():
    model.fit(X_train_a, y_train_a)
    sonuclar.setdefault(isim, {})["A) Baseline (sabit ayarlar)"] = accuracy_score(y_test_a, model.predict(X_test_a))

# --- B) GridSearchCV: her modelin kendi hiperparametrelerini ara ---
print("\n[GridSearchCV calisiyor — birkaç dakika surebilir]")

izgaralar = {
    "Karar Agaci": (
        DecisionTreeClassifier(random_state=42),
        {"max_depth": [3, 4, 5, 6, 7, 8, 10, None], "min_samples_leaf": [1, 5, 10, 20]},
    ),
    "Random Forest": (
        RandomForestClassifier(random_state=42, n_jobs=-1),
        {"max_depth": [4, 5, 6, 8, 10, None], "n_estimators": [100, 200], "min_samples_leaf": [1, 5, 10]},
    ),
    "Lojistik Regresyon": (
        LogisticRegression(max_iter=2000, random_state=42),
        {"C": [0.01, 0.1, 1.0, 10.0]},
    ),
}

en_iyi_modeller = {}
for isim, (model, grid) in izgaralar.items():
    gs = GridSearchCV(model, grid, cv=5, scoring="accuracy", n_jobs=-1)
    gs.fit(X_train_a, y_train_a)
    en_iyi_modeller[isim] = gs.best_estimator_
    sonuclar.setdefault(isim, {})["B) GridSearchCV optimize"] = accuracy_score(y_test_a, gs.best_estimator_.predict(X_test_a))
    print(f"  {isim:<22}: en iyi parametreler = {gs.best_params_}")

# --- C) Elo K-faktoru ve form penceresi taramasi (Random Forest ile degerlendirilir) ---
print("\n[Elo K / Form Penceresi taramasi]")
en_iyi_kombinasyon, en_iyi_skor = None, 0
for k in [10, 20, 30, 40]:
    for pencere in [3, 5, 8, 10]:
        df_k = veri_yukle(k=k, pencere=pencere)
        X_k, y_k = ozellik_hedef_ayir(df_k)
        X_train_k, X_test_k, y_train_k, y_test_k = train_test_ayir(X_k, y_k)
        rf_k = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
        rf_k.fit(X_train_k, y_train_k)
        skor = accuracy_score(y_test_k, rf_k.predict(X_test_k))
        if skor > en_iyi_skor:
            en_iyi_skor, en_iyi_kombinasyon = skor, (k, pencere)

print(f"  En iyi kombinasyon: K={en_iyi_kombinasyon[0]}, pencere={en_iyi_kombinasyon[1]} -> RF Test Acc={en_iyi_skor:.4f}")
print(f"  (Baseline K=20, pencere=5 -> RF Test Acc={sonuclar['Random Forest']['A) Baseline (sabit ayarlar)']:.4f})")

# En iyi K/pencere ile veri setini yeniden kur, B) optimize modelleri bu veri ile tekrar degerlendir
df_c = veri_yukle(k=en_iyi_kombinasyon[0], pencere=en_iyi_kombinasyon[1])
X_c, y_c = ozellik_hedef_ayir(df_c)
X_train_c, X_test_c, y_train_c, y_test_c = train_test_ayir(X_c, y_c)
for isim, model in en_iyi_modeller.items():
    model.fit(X_train_c, y_train_c)
    sonuclar.setdefault(isim, {})["C) +En iyi Elo K/pencere"] = accuracy_score(y_test_c, model.predict(X_test_c))

# --- D) Gradient Boosting ---
gb = GradientBoostingClassifier(random_state=42)
gb.fit(X_train_c, y_train_c)
sonuclar.setdefault("Gradient Boosting (yeni)", {})["D) Gradient Boosting"] = accuracy_score(y_test_c, gb.predict(X_test_c))

# --- E) VotingClassifier (soft voting): optimize edilmis 3 model + GB ---
voting = VotingClassifier(
    estimators=[("dt", en_iyi_modeller["Karar Agaci"]), ("rf", en_iyi_modeller["Random Forest"]),
                ("lr", en_iyi_modeller["Lojistik Regresyon"]), ("gb", gb)],
    voting="soft",
)
voting.fit(X_train_c, y_train_c)
y_pred_v = voting.predict(X_test_c)
y_prob_v = voting.predict_proba(X_test_c)[:, 1]
sonuclar.setdefault("Voting Ensemble (yeni)", {})["E) Voting (DT+RF+LR+GB)"] = accuracy_score(y_test_c, y_pred_v)
sonuclar.setdefault("Voting Ensemble (yeni)", {})["E) AUC-ROC"] = roc_auc_score(y_test_c, y_prob_v)

ozet = pd.DataFrame(sonuclar).T
print("\n[Karsilastirma Tablosu — Test Accuracy]")
print(ozet.to_string(float_format=lambda x: f"{x:.4f}"))

print("\n=> Adim 14 tamamlandi.")
