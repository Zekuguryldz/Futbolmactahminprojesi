import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from ortak import model_verisi_hazirla, OZELLIKLER
import warnings
warnings.filterwarnings("ignore")

print("=" * 55)
print("ADIM 10 — KARAR AGACI GORSELLESTiRME")
print("=" * 55)

# --- Veri hazırlık ---
df, X, y, X_train, X_test, y_train, y_test = model_verisi_hazirla()
ozellikler = OZELLIKLER

# --- Karar Ağacı eğit ---
dt = DecisionTreeClassifier(max_depth=5, random_state=42)
dt.fit(X_train, y_train)

# --- Metin formatında kural çıktısı (ilk 3 seviye) ---
print("\n[Karar Agaci Kurallari — max_depth=3]")
dt_kisa = DecisionTreeClassifier(max_depth=3, random_state=42)
dt_kisa.fit(X_train, y_train)
kural_metni = export_text(dt_kisa, feature_names=ozellikler)
print(kural_metni)

# --- Ağaç görseli ---
fig, ax = plt.subplots(figsize=(22, 9))
plot_tree(
    dt,
    feature_names=ozellikler,
    class_names=["Kazanmadi", "Kazandi"],
    filled=True,
    rounded=True,
    fontsize=9,
    ax=ax,
    max_depth=3,
)
ax.set_title("Karar Agaci (max_depth=5 egitildi, ilk 3 seviye gosteriliyor)",
             fontsize=13, fontweight="bold")
plt.savefig("karar_agaci.png", bbox_inches="tight")
plt.show()
print("=> karar_agaci.png kaydedildi.")

print("""
[Karar Agaci Nasil Okunur?]
  - Her dugum bir kural sorar: ozellik <= esik_degeri?
  - Sol dal  = EVET (True)
  - Sag dal  = HAYIR (False)
  - Yaprak   = tahmin sinifi (Kazandi / Kazanmadi)
  - samples  = o dugume dusen ornek sayisi
  - value    = [Kazanmadi_sayisi, Kazandi_sayisi]
  - gini     = safsizlik olcusu (0=tam saf, 0.5=en karisik)
""")

print("=> Tum adimlar tamamlandi! Ozet icin 11_ozet_rapor.py")
