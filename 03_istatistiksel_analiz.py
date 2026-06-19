import pandas as pd
import numpy as np

print("=" * 55)
print("ADIM 3 — İSTATİSTİKSEL ANALİZ")
print("=" * 55)

df = pd.read_csv("international_matches1.csv")
df_cups = pd.read_csv("world_cups1.csv")

df["Toplam Gol"] = df["Home Goals"] + df["Away Goals"]

# --- Merkezi eğilim ölçüleri ---
print("\n[Gol İstatistikleri]")
print(f"  Ev sahibi — Ortalama : {df['Home Goals'].mean():.2f}")
print(f"  Ev sahibi — Medyan   : {df['Home Goals'].median():.2f}")
print(f"  Ev sahibi — Std      : {df['Home Goals'].std():.2f}")
print(f"  Deplasman — Ortalama : {df['Away Goals'].mean():.2f}")
print(f"  Deplasman — Medyan   : {df['Away Goals'].median():.2f}")
print(f"  Deplasman — Std      : {df['Away Goals'].std():.2f}")
print(f"  Toplam gol (tüm maçlar): {df['Toplam Gol'].sum():,}")
print(f"  Rekor (tek maç)       : {df['Toplam Gol'].max()}")

# --- Ev sahipliği etkisi ---
print("\n[Ev Sahipliği Etkisi]")
ev = df.groupby("Home Stadium or Not")[["Home Goals", "Away Goals", "Toplam Gol"]].mean().round(3)
ev.index = ["Nötr/Deplasman (0)", "Kendi Stadı (1)"]
print(ev)

# --- Korelasyon ---
print("\n[Korelasyon Matrisi]")
print(df[["Home Goals", "Away Goals", "Home Stadium or Not", "Toplam Gol"]].corr().round(3))

# --- Turnuvaya göre ortalama gol ---
print("\n[Turnuvaya Göre Ortalama Gol — Top 10]")
tur_gol = df.groupby("Tournament")["Toplam Gol"].mean().sort_values(ascending=False)
print(tur_gol.head(10).round(2))

# --- Dünya Kupası trendi ---
print("\n[Dünya Kupası — Maç Başına Gol Trendi]")
df_cups["Mac Basi Gol"] = (df_cups["Goals Scored"] / df_cups["Matches Played"]).round(2)
print(df_cups[["Year", "Goals Scored", "Matches Played", "Mac Basi Gol"]].to_string(index=False))

# --- Yüzdelik dilimler ---
print("\n[Toplam Gol Yüzdelik Dilimleri]")
for p in [25, 50, 75, 90, 95, 99]:
    print(f"  %{p:3d} : {np.percentile(df['Toplam Gol'], p):.1f} gol")

print("\n=> Istatistiksel analiz tamamlandi. Siradaki: 04_gorsellestirme.py")
