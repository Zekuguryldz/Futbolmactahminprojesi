import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from ortak import model_verisi_hazirla, OZELLIKLER
import warnings
warnings.filterwarnings("ignore")

print("=" * 55)
print("ADIM 9 — OZELLiK ONEMi (Random Forest)")
print("=" * 55)

# --- Veri hazırlık ---
df, X, y, X_train, X_test, y_train, y_test = model_verisi_hazirla()
ozellikler = OZELLIKLER

# --- Random Forest eğit ---
rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

# --- Özellik önemleri ---
importance = pd.Series(rf.feature_importances_, index=ozellikler).sort_values(ascending=False)

print("\n[Ozellik Onem Skorlari]")
for oz, skor in importance.items():
    bar = "█" * int(skor * 50)
    print(f"  {oz:<25}: {skor:.4f}  {bar}")

print(f"\nEn onemli ozellik : {importance.idxmax()} ({importance.max():.4f})")
print(f"En az onemli      : {importance.idxmin()} ({importance.min():.4f})")

# --- Görselleştirme ---
fig, ax = plt.subplots(figsize=(9, 5))
colors = ["#9b59b6" if v == importance.max() else "#bdc3c7" for v in importance.values]
importance.sort_values().plot(kind="barh", color=colors[::-1], ax=ax)
ax.set_title("Random Forest — Ozellik Onemi", fontsize=13, fontweight="bold")
ax.set_xlabel("Onem Skoru")
for i, (val, name) in enumerate(zip(importance.sort_values(), importance.sort_values().index)):
    ax.text(val + 0.002, i, f"{val:.4f}", va="center", fontsize=10)
plt.tight_layout()
plt.savefig("ozellik_onemi.png", bbox_inches="tight")
plt.show()
print("=> ozellik_onemi.png kaydedildi. Siradaki: 10_karar_agaci.py")
