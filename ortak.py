"""05-11 arası adımlarda tekrar eden veri hazırlık mantığı."""
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

VERI_DOSYASI = "international_matches1.csv"
OZELLIKLER = ["Home_Enc", "Away_Enc", "Home Stadium or Not", "Tur_Tipi"]


def tur_tipi(t):
    t = str(t).lower()
    if "fifa" in t or "world cup" in t:
        return 2
    elif any(k in t for k in ["copa", "euro", "africa", "asian", "confederation"]):
        return 1
    return 0


def veri_yukle(dosya=VERI_DOSYASI):
    df = pd.read_csv(dosya)
    df["Tur_Tipi"] = df["Tournament"].apply(tur_tipi)
    le = LabelEncoder()
    df["Home_Enc"] = le.fit_transform(df["Home Team"])
    df["Away_Enc"] = le.fit_transform(df["Away Team"])
    df["Target"] = (df["Home Goals"] > df["Away Goals"]).astype(int)
    return df


def veri_ve_encoderlar_yukle(dosya=VERI_DOSYASI):
    """veri_yukle ile aynı mantık, ek olarak Home/Away LabelEncoder nesnelerini de döner
    (yeni bir takım çiftini tahmin için kodlamak isteyen arayüzler için)."""
    df = pd.read_csv(dosya)
    df["Tur_Tipi"] = df["Tournament"].apply(tur_tipi)
    le_home = LabelEncoder()
    le_away = LabelEncoder()
    df["Home_Enc"] = le_home.fit_transform(df["Home Team"])
    df["Away_Enc"] = le_away.fit_transform(df["Away Team"])
    df["Target"] = (df["Home Goals"] > df["Away Goals"]).astype(int)
    return df, le_home, le_away


def ozellik_hedef_ayir(df, ozellikler=OZELLIKLER):
    return df[ozellikler], df["Target"]


def train_test_ayir(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def model_verisi_hazirla(dosya=VERI_DOSYASI, ozellikler=OZELLIKLER):
    """veri_yukle + ozellik_hedef_ayir + train_test_ayir adımlarını tek seferde yapar."""
    df = veri_yukle(dosya)
    X, y = ozellik_hedef_ayir(df, ozellikler)
    X_train, X_test, y_train, y_test = train_test_ayir(X, y)
    return df, X, y, X_train, X_test, y_train, y_test
