import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from ortak import tur_tipi

print("=" * 55)
print("ADIM 5 — OZELLiK MUHENDiSLiGi")
print("=" * 55)

df = pd.read_csv("international_matches1.csv")

# --- Hedef değişken: maç sonucu ---
def sonuc_belirle(row):
    if row["Home Goals"] > row["Away Goals"]:
        return "Ev Sahibi Kazandi"
    elif row["Home Goals"] < row["Away Goals"]:
        return "Deplasman Kazandi"
    else:
        return "Beraberlik"

df["Sonuc"] = df.apply(sonuc_belirle, axis=1)

# --- Yeni özellik: Turnuva tipi ---
# FIFA/Dünya Kupası = 2, Kıta şampiyonası = 1, Diğer = 0 (bkz. ortak.py)
df["Tur_Tipi"] = df["Tournament"].apply(tur_tipi)

print("\n[Turnuva Tipi Dağılımı]")
print(df["Tur_Tipi"].value_counts())
print("  (2=FIFA/WC, 1=Kita Sampiyonasi, 0=Diger)")

# --- Kategorik → Sayısal: LabelEncoder ---
le = LabelEncoder()
df["Home_Enc"] = le.fit_transform(df["Home Team"])
df["Away_Enc"] = le.fit_transform(df["Away Team"])

print("\n[LabelEncoder Örnek]")
print(df[["Home Team", "Home_Enc", "Away Team", "Away_Enc"]].head(5).to_string(index=False))

# --- İkili hedef: Ev sahibi kazandı mı? (1=Evet, 0=Hayır) ---
df["Target"] = (df["Sonuc"] == "Ev Sahibi Kazandi").astype(int)

print("\n[Hedef Değişken Dağılımı]")
print(df["Target"].value_counts())
print(f"Ev sahibi kazanma orani: {df['Target'].mean():.2%}")

# --- Özellik seti ---
ozellikler = ["Home_Enc", "Away_Enc", "Home Stadium or Not", "Tur_Tipi"]
X = df[ozellikler]
y = df["Target"]

print(f"\n[Ozellikler] : {ozellikler}")
print(f"[X sekli]    : {X.shape}")
print(f"[y sekli]    : {y.shape}")

# --- Train / Test bölme ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n[Train / Test Bolme]")
print(f"  Egitim : {X_train.shape[0]} ornek")
print(f"  Test   : {X_test.shape[0]} ornek")
print(f"  Test orani: %20")

print("\n=> Ozellik muhendisligi tamamlandi. Siradaki: 06_model_egitimi.py")
