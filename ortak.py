"""05-12 arası adımlarda tekrar eden veri hazırlık mantığı."""
import pandas as pd
from collections import defaultdict, deque
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

VERI_DOSYASI = "international_matches1.csv"

# Adım 12 deneyinde doğrulanan, gerçek tahmin gücü taşıyan özellikler (Elo/form/h2h).
# Eski (Adım 5-11'in ilk halinde kullanılan) özellik seti, kıyaslama amaçlı saklanıyor.
ESKI_OZELLIKLER = ["Home_Enc", "Away_Enc", "Home Stadium or Not", "Tur_Tipi"]
OZELLIKLER = ["Elo_Fark", "Home_Form", "Away_Form", "H2H_Ev_Galibiyet_Orani",
              "Home Stadium or Not", "Tur_Tipi"]

ELO_K = 20
ELO_BASLANGIC = 1500.0
FORM_PENCERE = 5


def tur_tipi(t):
    t = str(t).lower()
    if "fifa" in t or "world cup" in t:
        return 2
    elif any(k in t for k in ["copa", "euro", "africa", "asian", "confederation"]):
        return 1
    return 0


def ozellik_muhendisligi_ekle(df, k=ELO_K, pencere=FORM_PENCERE):
    """Elo rating, form (son N maç puanı) ve head-to-head özelliklerini kronolojik
    sırada, leak-free şekilde (her satıra sadece o maçtan ÖNCEKİ bilgiyle) ekler.
    `k` (Elo K-faktörü) ve `pencere` (form penceresi) hiperparametre aramasında
    farklı değerlerle denenebilsin diye parametrik bırakıldı.
    Döndürülen `guncel_durum`, en son Elo/form/h2h değerlerini içerir — yeni bir
    maç için tahmin yaparken (örn. Streamlit arayüzü) kullanılır."""
    df = df.sort_values("Date").reset_index(drop=True)

    elo = defaultdict(lambda: ELO_BASLANGIC)
    form = defaultdict(lambda: deque(maxlen=pencere))
    h2h_ev_galibiyet = defaultdict(int)
    h2h_toplam = defaultdict(int)

    home_elo, away_elo = [], []
    home_form, away_form = [], []
    h2h_orani = []

    for _, row in df.iterrows():
        h, a = row["Home Team"], row["Away Team"]

        h_elo, a_elo = elo[h], elo[a]
        home_elo.append(h_elo)
        away_elo.append(a_elo)

        home_form.append(sum(form[h]) / len(form[h]) if form[h] else 1.0)
        away_form.append(sum(form[a]) / len(form[a]) if form[a] else 1.0)

        pair = (h, a)
        h2h_orani.append(h2h_ev_galibiyet[pair] / h2h_toplam[pair] if h2h_toplam[pair] else 0.5)

        if row["Home Goals"] > row["Away Goals"]:
            sonuc_h, puan_h, puan_a = 1.0, 3, 0
        elif row["Home Goals"] < row["Away Goals"]:
            sonuc_h, puan_h, puan_a = 0.0, 0, 3
        else:
            sonuc_h, puan_h, puan_a = 0.5, 1, 1

        beklenen_h = 1 / (1 + 10 ** ((a_elo - h_elo) / 400))
        elo[h] = h_elo + k * (sonuc_h - beklenen_h)
        elo[a] = a_elo + k * ((1 - sonuc_h) - (1 - beklenen_h))
        form[h].append(puan_h)
        form[a].append(puan_a)

        h2h_toplam[pair] += 1
        if sonuc_h == 1.0:
            h2h_ev_galibiyet[pair] += 1

    df["Home_Elo"] = home_elo
    df["Away_Elo"] = away_elo
    df["Elo_Fark"] = df["Home_Elo"] - df["Away_Elo"]
    df["Home_Form"] = home_form
    df["Away_Form"] = away_form
    df["H2H_Ev_Galibiyet_Orani"] = h2h_orani

    guncel_durum = {
        "elo": dict(elo),
        "form": {takim: list(puanlar) for takim, puanlar in form.items()},
        "h2h_ev_galibiyet": dict(h2h_ev_galibiyet),
        "h2h_toplam": dict(h2h_toplam),
    }
    return df, guncel_durum


def veri_yukle(dosya=VERI_DOSYASI, k=ELO_K, pencere=FORM_PENCERE):
    df = pd.read_csv(dosya)
    df["Tur_Tipi"] = df["Tournament"].apply(tur_tipi)
    le = LabelEncoder()
    df["Home_Enc"] = le.fit_transform(df["Home Team"])
    df["Away_Enc"] = le.fit_transform(df["Away Team"])
    df["Target"] = (df["Home Goals"] > df["Away Goals"]).astype(int)
    df, _ = ozellik_muhendisligi_ekle(df, k=k, pencere=pencere)
    return df


def veri_ve_encoderlar_yukle(dosya=VERI_DOSYASI):
    """veri_yukle ile aynı mantık, ek olarak Home/Away LabelEncoder nesnelerini ve
    Elo/form/h2h'nin güncel (en son maç sonrası) durumunu da döner — tahmin
    arayüzlerinin yeni bir maç için özellik üretebilmesi için."""
    df = pd.read_csv(dosya)
    df["Tur_Tipi"] = df["Tournament"].apply(tur_tipi)
    le_home = LabelEncoder()
    le_away = LabelEncoder()
    df["Home_Enc"] = le_home.fit_transform(df["Home Team"])
    df["Away_Enc"] = le_away.fit_transform(df["Away Team"])
    df["Target"] = (df["Home Goals"] > df["Away Goals"]).astype(int)
    df, guncel_durum = ozellik_muhendisligi_ekle(df)
    return df, le_home, le_away, guncel_durum


def tahmin_ozellikleri_hesapla(ev_takim, deplasman_takim, guncel_durum, stadyum, tur_tipi_kodu):
    """Streamlit gibi arayüzlerde, henüz oynanmamış bir maç için OZELLIKLER
    listesindeki değerleri günel Elo/form/h2h durumuna göre üretir."""
    elo = guncel_durum["elo"]
    form = guncel_durum["form"]
    h2h_ev_galibiyet = guncel_durum["h2h_ev_galibiyet"]
    h2h_toplam = guncel_durum["h2h_toplam"]

    h_elo = elo.get(ev_takim, ELO_BASLANGIC)
    a_elo = elo.get(deplasman_takim, ELO_BASLANGIC)

    h_form_listesi = form.get(ev_takim, [])
    a_form_listesi = form.get(deplasman_takim, [])
    h_form = sum(h_form_listesi) / len(h_form_listesi) if h_form_listesi else 1.0
    a_form = sum(a_form_listesi) / len(a_form_listesi) if a_form_listesi else 1.0

    pair = (ev_takim, deplasman_takim)
    toplam = h2h_toplam.get(pair, 0)
    h2h_orani = h2h_ev_galibiyet.get(pair, 0) / toplam if toplam else 0.5

    return pd.DataFrame([{
        "Elo_Fark": h_elo - a_elo,
        "Home_Form": h_form,
        "Away_Form": a_form,
        "H2H_Ev_Galibiyet_Orani": h2h_orani,
        "Home Stadium or Not": stadyum,
        "Tur_Tipi": tur_tipi_kodu,
    }])[OZELLIKLER]


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
