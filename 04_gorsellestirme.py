import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 120

print("=" * 55)
print("ADIM 4 — GÖRSELLEŞTİRME")
print("=" * 55)

df      = pd.read_csv("international_matches1.csv")
df_cups = pd.read_csv("world_cups1.csv")
df_wc   = pd.read_csv("world_cup_matches1.csv")

def sonuc_belirle(row):
    if row["Home Goals"] > row["Away Goals"]:
        return "Ev Sahibi Kazandi"
    elif row["Home Goals"] < row["Away Goals"]:
        return "Deplasman Kazandi"
    else:
        return "Beraberlik"

df["Sonuc"] = df.apply(sonuc_belirle, axis=1)

fig = plt.figure(figsize=(20, 16))
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

# 1 — Maç sonucu pasta
ax1 = fig.add_subplot(gs[0, 0])
counts = df["Sonuc"].value_counts()
ax1.pie(counts, labels=counts.index, autopct="%1.1f%%",
        colors=["#2ecc71", "#e74c3c", "#3498db"], startangle=140)
ax1.set_title("Mac Sonucu Dagilimi", fontweight="bold")

# 2 — Gol dağılımı histogram
ax2 = fig.add_subplot(gs[0, 1])
ax2.hist(df["Home Goals"], bins=range(0, 15), alpha=0.6,
         color="#2ecc71", label="Ev Sahibi", density=True)
ax2.hist(df["Away Goals"], bins=range(0, 15), alpha=0.6,
         color="#e74c3c", label="Deplasman", density=True)
ax2.set_xlabel("Gol Sayisi")
ax2.set_ylabel("Yogunluk")
ax2.set_title("Gol Dagilimi", fontweight="bold")
ax2.legend()

# 3 — Ev sahipliği kutu grafiği
ax3 = fig.add_subplot(gs[0, 2])
ev_data = [
    df[df["Home Stadium or Not"] == 1]["Home Goals"].values,
    df[df["Home Stadium or Not"] == 0]["Home Goals"].values,
]
ax3.boxplot(ev_data, labels=["Kendi Stadi", "Notr/Deplasman"],
            patch_artist=True,
            boxprops=dict(facecolor="#3498db", alpha=0.6))
ax3.set_ylabel("Ev Sahibi Gol")
ax3.set_title("Ev Sahipligi Avantaji", fontweight="bold")

# 4 — DK yıllık toplam gol
ax4 = fig.add_subplot(gs[1, 0])
ax4.bar(df_cups["Year"], df_cups["Goals Scored"],
        color="#9b59b6", edgecolor="white")
ax4.set_xlabel("Yil")
ax4.set_ylabel("Toplam Gol")
ax4.set_title("Dunya Kupasi Yillik Gol", fontweight="bold")
ax4.tick_params(axis="x", rotation=45)

# 5 — Maç başına gol trendi
ax5 = fig.add_subplot(gs[1, 1])
df_cups["gol_per_mac"] = df_cups["Goals Scored"] / df_cups["Matches Played"]
ax5.plot(df_cups["Year"], df_cups["gol_per_mac"],
         marker="o", color="#e67e22", linewidth=2)
ax5.set_xlabel("Yil")
ax5.set_ylabel("Mac Basina Gol")
ax5.set_title("Mac Basina Gol Trendi", fontweight="bold")
ax5.tick_params(axis="x", rotation=45)

# 6 — Top 10 turnuva
ax6 = fig.add_subplot(gs[1, 2])
top_tur = df["Tournament"].value_counts().head(10)
ax6.barh(top_tur.index[::-1], top_tur.values[::-1], color="#1abc9c")
ax6.set_xlabel("Mac Sayisi")
ax6.set_title("Top 10 Turnuva", fontweight="bold")

# 7 — WC aşamaya göre gol
ax7 = fig.add_subplot(gs[2, 0])
stage_goals = df_wc.groupby("Stage").agg(
    Ort_Gol=("Home Goals", lambda x: (x + df_wc.loc[x.index, "Away Goals"]).mean())
).sort_values("Ort_Gol")
ax7.barh(stage_goals.index, stage_goals["Ort_Gol"], color="#e74c3c")
ax7.set_xlabel("Ort. Gol/Mac")
ax7.set_title("WC: Asamaya Gore Ort. Gol", fontweight="bold")

# 8 — Korelasyon ısı haritası
ax8 = fig.add_subplot(gs[2, 1])
num_cols = df[["Home Goals", "Away Goals", "Home Stadium or Not"]].copy()
num_cols["Toplam Gol"] = num_cols["Home Goals"] + num_cols["Away Goals"]
sns.heatmap(num_cols.corr(), annot=True, fmt=".2f",
            cmap="coolwarm", ax=ax8, square=True, linewidths=0.5)
ax8.set_title("Korelasyon Matrisi", fontweight="bold")

# 9 — Sonuca göre ortalama gol
ax9 = fig.add_subplot(gs[2, 2])
sonuc_gol = df.groupby("Sonuc")[["Home Goals", "Away Goals"]].mean()
x = np.arange(len(sonuc_gol))
w = 0.35
ax9.bar(x - w/2, sonuc_gol["Home Goals"], w, label="Ev Sahibi", color="#2ecc71")
ax9.bar(x + w/2, sonuc_gol["Away Goals"], w, label="Deplasman", color="#e74c3c")
ax9.set_xticks(x)
ax9.set_xticklabels(sonuc_gol.index, rotation=15, ha="right")
ax9.set_ylabel("Ortalama Gol")
ax9.set_title("Sonuca Gore Ort. Gol", fontweight="bold")
ax9.legend()

plt.suptitle("Dunya Kupasi & Uluslararasi Mac Veri Analizi",
             fontsize=15, fontweight="bold", y=1.01)
plt.savefig("gorsellestirme.png", bbox_inches="tight")
plt.show()
print("=> gorsellestirme.png kaydedildi.")
print("=> Siradaki: 05_ozellik_muhendisligi.py")
