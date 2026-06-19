import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report
)
from ortak import model_verisi_hazirla
import warnings
warnings.filterwarnings("ignore")

print("=" * 55)
print("ADIM 7 — PRECISION / RECALL / ACCURACY / F1")
print("=" * 55)

# --- Veri hazırlık ---
df, X, y, X_train, X_test, y_train, y_test = model_verisi_hazirla()

modeller = {
    "Karar Agaci"       : DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest"     : RandomForestClassifier(n_estimators=100, max_depth=5,
                                                  random_state=42, n_jobs=-1),
    "Lojistik Regresyon": LogisticRegression(max_iter=1000, random_state=42),
}

# --- Metrikleri hesapla ---
sonuclar = {}
print(f"\n{'Model':<22} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8}")
print("-" * 62)

for isim, model in modeller.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)

    sonuclar[isim] = {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1 Score": f1}
    print(f"{isim:<22} {acc:>9.4f} {prec:>10.4f} {rec:>8.4f} {f1:>8.4f}")

# --- Detaylı rapor ---
print("\n[Detayli Siniflandirma Raporlari]")
for isim, model in modeller.items():
    y_pred = model.predict(X_test)
    print(f"\n-- {isim} --")
    print(classification_report(y_test, y_pred,
                                target_names=["Kazanmadi", "Kazandi"],
                                digits=4))

# --- Görselleştirme: Gruplu çubuk ---
isimler         = list(sonuclar.keys())
metrik_isimleri = ["Accuracy", "Precision", "Recall", "F1 Score"]
renkler         = ["#3498db", "#2ecc71", "#e74c3c"]

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Precision / Recall / Accuracy / F1 Karsilastirma",
             fontsize=13, fontweight="bold")

# Sol: gruplu çubuk
ax = axes[0]
x  = np.arange(len(metrik_isimleri))
w  = 0.25
for idx, (isim, renk) in enumerate(zip(isimler, renkler)):
    degerler = [sonuclar[isim][m] for m in metrik_isimleri]
    bars = ax.bar(x + idx * w, degerler, w, label=isim, color=renk, alpha=0.85)
    for bar, val in zip(bars, degerler):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)
ax.set_xticks(x + w)
ax.set_xticklabels(metrik_isimleri)
ax.set_ylim(0, 0.95)
ax.set_ylabel("Skor")
ax.set_title("4 Metrik — 3 Model")
ax.legend()

# Sağ: radar grafiği
ax2 = fig.add_subplot(122, polar=True)
angles = np.linspace(0, 2 * np.pi, len(metrik_isimleri), endpoint=False).tolist()
angles += angles[:1]
for isim, renk in zip(isimler, renkler):
    degerler = [sonuclar[isim][m] for m in metrik_isimleri] + [sonuclar[isim][metrik_isimleri[0]]]
    ax2.plot(angles, degerler, "o-", linewidth=2, color=renk, label=isim)
    ax2.fill(angles, degerler, alpha=0.10, color=renk)
ax2.set_thetagrids(np.degrees(angles[:-1]), metrik_isimleri)
ax2.set_ylim(0, 1)
ax2.set_title("Radar Grafigi", pad=18, fontweight="bold")
ax2.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1))

plt.tight_layout()
plt.savefig("metrikler.png", bbox_inches="tight")
plt.show()
print("=> metrikler.png kaydedildi. Siradaki: 08_model_karsilastirma.py")
