import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, learning_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix
from ortak import model_verisi_hazirla
import warnings
warnings.filterwarnings("ignore")

sns.set_theme(style="whitegrid")

print("=" * 55)
print("ADIM 8 — MODEL KARSILASTIRMA & OVERFiTTiNG")
print("=" * 55)

# --- Veri hazırlık ---
df, X, y, X_train, X_test, y_train, y_test = model_verisi_hazirla()

modeller = {
    "Karar Agaci"       : DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest"     : RandomForestClassifier(n_estimators=100, max_depth=5,
                                                  random_state=42, n_jobs=-1),
    "Lojistik Regresyon": LogisticRegression(max_iter=1000, random_state=42),
}

sonuclar = {}
for isim, model in modeller.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    sonuclar[isim] = {
        "train_acc": accuracy_score(y_train, model.predict(X_train)),
        "test_acc" : accuracy_score(y_test, y_pred),
        "auc"      : roc_auc_score(y_test, y_prob),
        "y_pred"   : y_pred,
        "y_prob"   : y_prob,
    }

isimler = list(sonuclar.keys())
renkler = ["#3498db", "#2ecc71", "#e74c3c"]

# --- Konsol özet ---
print(f"\n{'Model':<22} {'Train':>7} {'Test':>7} {'AUC':>7} {'Overfit':>9}")
print("-" * 56)
for isim, v in sonuclar.items():
    print(f"{isim:<22} {v['train_acc']:>7.4f} {v['test_acc']:>7.4f} "
          f"{v['auc']:>7.4f} {v['train_acc']-v['test_acc']:>+9.4f}")

# --- Grafik 1: Karşılaştırma & Overfitting ---
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle("Model Karsilastirma & Overfitting Analizi",
             fontsize=14, fontweight="bold")

# Train vs Test
ax = axes[0, 0]
x = np.arange(len(isimler))
w = 0.35
ax.bar(x - w/2, [sonuclar[i]["train_acc"] for i in isimler],
       w, label="Train", color="#3498db", alpha=0.85)
ax.bar(x + w/2, [sonuclar[i]["test_acc"] for i in isimler],
       w, label="Test", color="#e74c3c", alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(isimler, rotation=15, ha="right")
ax.set_ylim(0.4, 1.05)
ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
ax.set_ylabel("Dogruluk")
ax.set_title("Train vs Test (Overfitting)")
ax.legend()

# Test Accuracy
ax = axes[0, 1]
bars = ax.bar(isimler, [sonuclar[i]["test_acc"] for i in isimler],
              color=renkler, edgecolor="white", width=0.5)
for bar, isim in zip(bars, isimler):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
            f"{sonuclar[isim]['test_acc']:.4f}", ha="center", fontsize=9, fontweight="bold")
ax.set_ylim(0.4, 0.80)
ax.set_ylabel("Dogruluk")
ax.set_title("Test Accuracy")
ax.tick_params(axis="x", rotation=15)

# AUC-ROC
ax = axes[0, 2]
bars = ax.bar(isimler, [sonuclar[i]["auc"] for i in isimler],
              color=renkler, edgecolor="white", width=0.5)
for bar, isim in zip(bars, isimler):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
            f"{sonuclar[isim]['auc']:.4f}", ha="center", fontsize=9, fontweight="bold")
ax.set_ylim(0.5, 0.85)
ax.set_ylabel("AUC-ROC")
ax.set_title("AUC-ROC Skoru")
ax.tick_params(axis="x", rotation=15)

# ROC Eğrileri
ax = axes[1, 0]
for isim, renk in zip(isimler, renkler):
    fpr, tpr, _ = roc_curve(y_test, sonuclar[isim]["y_prob"])
    ax.plot(fpr, tpr, color=renk, linewidth=2,
            label=f"{isim} (AUC={sonuclar[isim]['auc']:.3f})")
ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Egrileri")
ax.legend(fontsize=8)

# Cross-Validation kutu
ax = axes[1, 1]
cv_all = []
for isim, model in modeller.items():
    cv_sc = cross_val_score(model, X, y, cv=5, scoring="accuracy", n_jobs=-1)
    cv_all.append(cv_sc)
bp = ax.boxplot(cv_all, labels=isimler, patch_artist=True, boxprops=dict(alpha=0.7))
for patch, renk in zip(bp["boxes"], renkler):
    patch.set_facecolor(renk)
ax.set_ylabel("CV Dogrulugu (5-fold)")
ax.set_title("Cross-Validation Dagilimi")
ax.tick_params(axis="x", rotation=15)

# Öğrenme Eğrisi (Random Forest)
ax = axes[1, 2]
train_sizes, train_sc, val_sc = learning_curve(
    modeller["Random Forest"], X, y,
    train_sizes=np.linspace(0.1, 1.0, 8),
    cv=5, scoring="accuracy", n_jobs=-1
)
ax.plot(train_sizes, train_sc.mean(axis=1), "o-", color="#3498db", label="Train", linewidth=2)
ax.fill_between(train_sizes,
                train_sc.mean(axis=1) - train_sc.std(axis=1),
                train_sc.mean(axis=1) + train_sc.std(axis=1), alpha=0.2, color="#3498db")
ax.plot(train_sizes, val_sc.mean(axis=1), "o-", color="#e74c3c", label="Dogrulama", linewidth=2)
ax.fill_between(train_sizes,
                val_sc.mean(axis=1) - val_sc.std(axis=1),
                val_sc.mean(axis=1) + val_sc.std(axis=1), alpha=0.2, color="#e74c3c")
ax.set_xlabel("Egitim Ornek Sayisi")
ax.set_ylabel("Dogruluk")
ax.set_title("Ogrenme Egrisi (Random Forest)")
ax.legend()

plt.tight_layout()
plt.savefig("model_karsilastirma.png", bbox_inches="tight")
plt.show()
print("=> model_karsilastirma.png kaydedildi.")

# --- Grafik 2: Confusion Matrix ---
fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
fig2.suptitle("Karmasiklik Matrisi (Confusion Matrix)", fontsize=13, fontweight="bold")
for ax, isim in zip(axes2, isimler):
    cm = confusion_matrix(y_test, sonuclar[isim]["y_pred"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Kazanmadi", "Kazandi"],
                yticklabels=["Kazanmadi", "Kazandi"])
    ax.set_title(isim)
    ax.set_xlabel("Tahmin")
    ax.set_ylabel("Gercek")
plt.tight_layout()
plt.savefig("confusion_matrix.png", bbox_inches="tight")
plt.show()
print("=> confusion_matrix.png kaydedildi.")

# --- Overfitting yorumu ---
print("\n[Overfitting Degerlendirme]")
for isim, v in sonuclar.items():
    fark = v["train_acc"] - v["test_acc"]
    if fark > 0.05:
        durum = "YUKSEK overfitting riski"
    elif fark > 0.02:
        durum = "ORTA overfitting"
    else:
        durum = "Dusuk overfitting - model iyi genelliyor"
    print(f"  {isim:<22}: {durum} (fark={fark:+.4f})")

print("\n=> Siradaki: 09_ozellik_onemi.py")
