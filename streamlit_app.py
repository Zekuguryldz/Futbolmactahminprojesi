import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)
from sklearn.model_selection import cross_val_score
from ortak import (
    veri_ve_encoderlar_yukle, ozellik_hedef_ayir, train_test_ayir,
    tur_tipi, tahmin_ozellikleri_hesapla, OZELLIKLER
)
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Futbol Maç Tahmin Projesi", layout="wide")
sns.set_theme(style="whitegrid")

RENKLER = {"Karar Agaci": "#3498db", "Random Forest": "#2ecc71", "Lojistik Regresyon": "#e74c3c"}


@st.cache_data
def veri_hazirla():
    df, le_home, le_away, guncel_durum = veri_ve_encoderlar_yukle()
    df["Yil"] = pd.to_datetime(df["Date"]).dt.year
    return df, le_home, le_away, guncel_durum


@st.cache_resource
def modelleri_egit(_X_train, _y_train):
    modeller = {
        "Karar Agaci": DecisionTreeClassifier(max_depth=5, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1),
        "Lojistik Regresyon": LogisticRegression(max_iter=1000, random_state=42),
    }
    for model in modeller.values():
        model.fit(_X_train, _y_train)
    return modeller


df, le_home, le_away, guncel_durum = veri_hazirla()
X, y = ozellik_hedef_ayir(df)
X_train, X_test, y_train, y_test = train_test_ayir(X, y)
modeller = modelleri_egit(X_train, y_train)

st.title("⚽ Futbol Maç Sonucu Tahmin Projesi")
st.caption("Karar Ağacı · Random Forest · Lojistik Regresyon — international_matches1.csv üzerinde eğitildi")

tab1, tab2, tab3, tab4 = st.tabs([
    "🔮 Tahmin", "📊 Veri Keşfi (EDA)", "📈 Model Karşılaştırma", "🌳 Özellik Önemi & Karar Ağacı"
])

# ============================================================
# TAB 1 — TAHMİN
# ============================================================
with tab1:
    st.subheader("Maç Sonucu Tahmini")
    col1, col2 = st.columns(2)
    with col1:
        ev_takim = st.selectbox("Ev Sahibi Takım", sorted(le_home.classes_), index=0)
    with col2:
        deplasman_takim = st.selectbox("Deplasman Takımı", sorted(le_away.classes_), index=1)

    turnuva = st.selectbox("Turnuva", sorted(df["Tournament"].unique()))
    stadyum = st.radio("Ev sahibi kendi stadyumunda mı oynuyor?", ["Evet", "Hayır"], horizontal=True)
    model_secimi = st.selectbox("Model", list(modeller.keys()))

    if st.button("Tahmin Et", type="primary"):
        if ev_takim == deplasman_takim:
            st.warning("Ev sahibi ve deplasman takımı aynı olamaz.")
        else:
            ozellik_df = tahmin_ozellikleri_hesapla(
                ev_takim, deplasman_takim, guncel_durum,
                stadyum=1 if stadyum == "Evet" else 0,
                tur_tipi_kodu=tur_tipi(turnuva),
            )

            model = modeller[model_secimi]
            olasilik = model.predict_proba(ozellik_df)[0][1]
            tahmin = "Ev Sahibi Kazanır" if olasilik >= 0.5 else "Ev Sahibi Kazanmaz"

            st.metric("Tahmin", tahmin)
            st.progress(float(olasilik))
            st.write(f"**{ev_takim}** kazanma olasılığı: **{olasilik:.1%}**  |  "
                     f"**{deplasman_takim}/Berabere** olasılığı: **{1 - olasilik:.1%}**")
            h_elo = guncel_durum["elo"].get(ev_takim, 1500.0)
            a_elo = guncel_durum["elo"].get(deplasman_takim, 1500.0)
            st.caption(f"Model: {model_secimi} — Güncel Elo: {ev_takim}={h_elo:.0f}, "
                       f"{deplasman_takim}={a_elo:.0f} — Turnuva tipi kodu: {tur_tipi(turnuva)} "
                       f"(0=Diğer, 1=Kıta Şampiyonası, 2=FIFA/Dünya Kupası)")

# ============================================================
# TAB 2 — EDA
# ============================================================
with tab2:
    st.subheader("Veri Keşfi")
    yil_min, yil_max = int(df["Yil"].min()), int(df["Yil"].max())
    secili_yil = st.slider("Yıl Aralığı", yil_min, yil_max, (yil_min, yil_max))
    turnuva_secimi = st.multiselect("Turnuva (boş = tümü)", sorted(df["Tournament"].unique()))

    df_f = df[(df["Yil"] >= secili_yil[0]) & (df["Yil"] <= secili_yil[1])]
    if turnuva_secimi:
        df_f = df_f[df_f["Tournament"].isin(turnuva_secimi)]

    c1, c2, c3 = st.columns(3)
    c1.metric("Toplam Maç", f"{len(df_f):,}")
    c2.metric("Ev Sahibi Kazanma Oranı", f"{df_f['Target'].mean():.1%}")
    c3.metric("Ortalama Toplam Gol", f"{(df_f['Home Goals'] + df_f['Away Goals']).mean():.2f}")

    colA, colB = st.columns(2)
    with colA:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(df_f["Home Goals"], color="#3498db", label="Ev Sahibi Gol", kde=False, ax=ax, alpha=0.6, discrete=True)
        sns.histplot(df_f["Away Goals"], color="#e74c3c", label="Deplasman Gol", kde=False, ax=ax, alpha=0.6, discrete=True)
        ax.set_title("Gol Dağılımı")
        ax.legend()
        st.pyplot(fig)

    with colB:
        yillik = df_f.groupby("Yil")["Target"].mean()
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.plot(yillik.index, yillik.values, color="#2ecc71")
        ax2.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
        ax2.set_title("Yıllara Göre Ev Sahibi Kazanma Oranı")
        ax2.set_ylabel("Kazanma Oranı")
        st.pyplot(fig2)

    st.markdown("**En Çok Maç Oynayan 10 Ev Sahibi Takım**")
    top_takim = df_f["Home Team"].value_counts().head(10)
    fig3, ax3 = plt.subplots(figsize=(8, 3.5))
    top_takim.sort_values().plot(kind="barh", ax=ax3, color="#9b59b6")
    st.pyplot(fig3)

# ============================================================
# TAB 3 — MODEL KARŞILAŞTIRMA
# ============================================================
with tab3:
    st.subheader("Model Karşılaştırma")

    sonuclar = {}
    for isim, model in modeller.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        cv = cross_val_score(model, X, y, cv=5, scoring="accuracy", n_jobs=-1)
        sonuclar[isim] = {
            "Train Acc": accuracy_score(y_train, model.predict(X_train)),
            "Test Acc": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1 Score": f1_score(y_test, y_pred, zero_division=0),
            "AUC-ROC": roc_auc_score(y_test, y_prob),
            "CV Mean": cv.mean(),
            "CV Std": cv.std(),
        }
        sonuclar[isim]["Overfit"] = sonuclar[isim]["Train Acc"] - sonuclar[isim]["Test Acc"]

    ozet = pd.DataFrame(sonuclar).T
    st.dataframe(ozet.style.format("{:.4f}").background_gradient(cmap="YlGn", axis=0), width="stretch")

    colA, colB = st.columns(2)
    with colA:
        fig, ax = plt.subplots(figsize=(6, 4))
        metrikler_goster = ["Test Acc", "Precision", "Recall", "F1 Score", "AUC-ROC"]
        ozet[metrikler_goster].plot(kind="bar", ax=ax,
                                     color=["#3498db", "#2ecc71", "#e74c3c", "#f39c12", "#9b59b6"])
        ax.set_title("Metrik Karşılaştırma")
        ax.legend(loc="lower right", fontsize=8)
        plt.xticks(rotation=15)
        st.pyplot(fig)

    with colB:
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        for isim, model in modeller.items():
            y_prob = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            ax2.plot(fpr, tpr, label=f"{isim} (AUC={sonuclar[isim]['AUC-ROC']:.3f})", color=RENKLER[isim])
        ax2.plot([0, 1], [0, 1], "k--", alpha=0.5)
        ax2.set_xlabel("False Positive Rate")
        ax2.set_ylabel("True Positive Rate")
        ax2.set_title("ROC Eğrileri")
        ax2.legend(fontsize=8)
        st.pyplot(fig2)

    secilen_cm = st.selectbox("Confusion Matrix için model seç", list(modeller.keys()), key="cm_model")
    cm = confusion_matrix(y_test, modeller[secilen_cm].predict(X_test))
    fig3, ax3 = plt.subplots(figsize=(4, 3.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax3,
                xticklabels=["Kazanmadı", "Kazandı"], yticklabels=["Kazanmadı", "Kazandı"])
    ax3.set_xlabel("Tahmin")
    ax3.set_ylabel("Gerçek")
    st.pyplot(fig3)

# ============================================================
# TAB 4 — ÖZELLİK ÖNEMİ & KARAR AĞACI
# ============================================================
with tab4:
    st.subheader("Random Forest — Özellik Önemi")
    rf = modeller["Random Forest"]
    importance = pd.Series(rf.feature_importances_, index=OZELLIKLER).sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(8, 3.5))
    colors = ["#9b59b6" if v == importance.max() else "#bdc3c7" for v in importance.values]
    importance.sort_values().plot(kind="barh", color=colors[::-1], ax=ax)
    ax.set_title("Özellik Önem Skorları")
    st.pyplot(fig)

    st.subheader("Karar Ağacı Görselleştirme")
    derinlik = st.slider("Gösterilecek derinlik", 1, 5, 3)
    dt = modeller["Karar Agaci"]
    fig2, ax2 = plt.subplots(figsize=(18, 7))
    plot_tree(dt, feature_names=OZELLIKLER, class_names=["Kazanmadi", "Kazandi"],
              filled=True, rounded=True, fontsize=8, ax=ax2, max_depth=derinlik)
    st.pyplot(fig2)

    with st.expander("Metin formatında kurallar (ilk 3 seviye)"):
        dt_kisa = DecisionTreeClassifier(max_depth=3, random_state=42).fit(X_train, y_train)
        st.text(export_text(dt_kisa, feature_names=OZELLIKLER))
