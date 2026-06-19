import pandas as pd

print("=" * 55)
print("ADIM 2 — KEŞİFSEL VERİ ANALİZİ (EDA)")
print("=" * 55)

df = pd.read_csv("international_matches1.csv")

# --- Genel istatistik ---
print("\n[Sayısal Sütunların İstatistikleri]")
print(df.describe())

# --- Maç sonucu sütunu oluştur ---
def sonuc_belirle(row):
    if row["Home Goals"] > row["Away Goals"]:
        return "Ev Sahibi Kazandı"
    elif row["Home Goals"] < row["Away Goals"]:
        return "Deplasman Kazandı"
    else:
        return "Beraberlik"

df["Sonuc"] = df.apply(sonuc_belirle, axis=1)

# --- Maç sonucu dağılımı ---
print("\n[Maç Sonucu Dağılımı]")
print(df["Sonuc"].value_counts())
print(f"\nEv sahibi kazanma oranı : {(df['Sonuc'] == 'Ev Sahibi Kazandı').mean():.2%}")
print(f"Deplasman kazanma oranı  : {(df['Sonuc'] == 'Deplasman Kazandı').mean():.2%}")
print(f"Beraberlik oranı         : {(df['Sonuc'] == 'Beraberlik').mean():.2%}")

# --- Turnuva dağılımı ---
print("\n[Turnuva Türleri — Top 10]")
print(df["Tournament"].value_counts().head(10))

# --- Ev sahipliği dağılımı ---
print("\n[Ev Sahipliği Değişkeni (Home Stadium or Not)]")
print(df["Home Stadium or Not"].value_counts())
print("  (1 = Kendi stadı, 0 = Nötr/Deplasman)")

# --- Gol aralıkları ---
print("\n[Gol Aralığı Dağılımı — Ev Sahibi]")
print(df["Home Goals"].value_counts().sort_index().head(10))

print("\n[Gol Aralığı Dağılımı — Deplasman]")
print(df["Away Goals"].value_counts().sort_index().head(10))

print("\n=> EDA tamamlandı. Sıradaki: 03_istatistiksel_analiz.py")
