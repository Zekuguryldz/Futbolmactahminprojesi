import pandas as pd

print("=" * 55)
print("ADIM 1 — VERİ YÜKLEME")
print("=" * 55)

# 4 CSV dosyasını yükle
df_cups    = pd.read_csv("world_cups1.csv")
df_matches = pd.read_csv("world_cup_matches1.csv")
df_2022    = pd.read_csv("2022_world_cup_matches1.csv")
df_intl    = pd.read_csv("international_matches1.csv")

# --- Boyut ---
print("\n[Satır x Sütun]")
print(f"  world_cups1            : {df_cups.shape}")
print(f"  world_cup_matches1     : {df_matches.shape}")
print(f"  2022_world_cup_matches1: {df_2022.shape}")
print(f"  international_matches1 : {df_intl.shape}")

# --- Sütun isimleri ---
for isim, df in [("world_cups1", df_cups),
                 ("world_cup_matches1", df_matches),
                 ("2022_world_cup_matches1", df_2022),
                 ("international_matches1", df_intl)]:
    print(f"\n[{isim}] Sütunlar:")
    print(f"  {list(df.columns)}")

# --- İlk 3 satır ---
print("\n--- international_matches1 ilk 3 satır ---")
print(df_intl.head(3).to_string())

# --- Veri tipleri ---
print("\n--- international_matches1 Veri Tipleri ---")
print(df_intl.dtypes)

# --- Eksik değer ---
print("\n--- Eksik Değer Kontrolü ---")
for isim, df in [("world_cups1", df_cups),
                 ("international_matches1", df_intl)]:
    toplam = df.isnull().sum().sum()
    print(f"  {isim}: {toplam} eksik değer")

# --- Tekrar eden satır ---
print("\n--- Tekrar Eden Satır Kontrolü ---")
for isim, df in [("world_cups1", df_cups),
                 ("international_matches1", df_intl)]:
    tekrar = df.duplicated().sum()
    print(f"  {isim}: {tekrar} tekrar")

print("\n=> Veri yükleme tamamlandı. Sıradaki: 02_eda.py")
