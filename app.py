import gradio as gr
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

sns.set_theme(style="whitegrid")

RENKLER = {"Karar Agaci": "#3498db", "Random Forest": "#2ecc71", "Lojistik Regresyon": "#e74c3c"}

# ============================================================
# VERİ + MODEL HAZIRLIĞI (bir kere, uygulama başında)
# ============================================================
df, le_home, le_away, guncel_durum = veri_ve_encoderlar_yukle()
df["Yil"] = pd.to_datetime(df["Date"]).dt.year

X, y = ozellik_hedef_ayir(df)
X_train, X_test, y_train, y_test = train_test_ayir(X, y)

modeller = {
    "Karar Agaci": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1),
    "Lojistik Regresyon": LogisticRegression(max_iter=1000, random_state=42),
}
for model in modeller.values():
    model.fit(X_train, y_train)

TAKIMLAR = sorted(le_home.classes_)
TURNUVALAR = sorted(df["Tournament"].unique())
YIL_MIN, YIL_MAX = int(df["Yil"].min()), int(df["Yil"].max())


# ============================================================
# TAB 1 — TAHMİN
# ============================================================
def tahmin_et(ev_takim, deplasman_takim, turnuva, stadyum, model_secimi):
    if ev_takim == deplasman_takim:
        gr.Warning("Ev sahibi ve deplasman takımı aynı olamaz.")
        return None, ""

    ozellik_df = tahmin_ozellikleri_hesapla(
        ev_takim, deplasman_takim, guncel_durum,
        stadyum=1 if stadyum == "Evet" else 0,
        tur_tipi_kodu=tur_tipi(turnuva),
    )

    model = modeller[model_secimi]
    olasilik = float(model.predict_proba(ozellik_df)[0][1])
    etiketler = {"Ev Sahibi Kazanır": olasilik, "Ev Sahibi Kazanmaz": 1 - olasilik}

    h_elo = guncel_durum["elo"].get(ev_takim, 1500.0)
    a_elo = guncel_durum["elo"].get(deplasman_takim, 1500.0)
    caption = (f"Model: {model_secimi} — Güncel Elo: {ev_takim}={h_elo:.0f}, "
               f"{deplasman_takim}={a_elo:.0f} — Turnuva tipi kodu: {tur_tipi(turnuva)} "
               f"(0=Diğer, 1=Kıta Şampiyonası, 2=FIFA/Dünya Kupası)")
    return etiketler, caption


# ============================================================
# TAB 2 — EDA
# ============================================================
def eda_guncelle(yil_baslangic, yil_bitis, turnuva_secimi):
    df_f = df[(df["Yil"] >= yil_baslangic) & (df["Yil"] <= yil_bitis)]
    if turnuva_secimi:
        df_f = df_f[df_f["Tournament"].isin(turnuva_secimi)]

    toplam_mac = f"**Toplam Maç:** {len(df_f):,}"
    kazanma_orani = f"**Ev Sahibi Kazanma Oranı:** {df_f['Target'].mean():.1%}"
    ortalama_gol = f"**Ortalama Toplam Gol:** {(df_f['Home Goals'] + df_f['Away Goals']).mean():.2f}"

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(df_f["Home Goals"], color="#3498db", label="Ev Sahibi Gol", kde=False, ax=ax, alpha=0.6, discrete=True)
    sns.histplot(df_f["Away Goals"], color="#e74c3c", label="Deplasman Gol", kde=False, ax=ax, alpha=0.6, discrete=True)
    ax.set_title("Gol Dağılımı")
    ax.legend()

    yillik = df_f.groupby("Yil")["Target"].mean()
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.plot(yillik.index, yillik.values, color="#2ecc71")
    ax2.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    ax2.set_title("Yıllara Göre Ev Sahibi Kazanma Oranı")
    ax2.set_ylabel("Kazanma Oranı")

    top_takim = df_f["Home Team"].value_counts().head(10)
    fig3, ax3 = plt.subplots(figsize=(8, 3.5))
    top_takim.sort_values().plot(kind="barh", ax=ax3, color="#9b59b6")
    ax3.set_title("En Çok Maç Oynayan 10 Ev Sahibi Takım")

    return toplam_mac, kazanma_orani, ortalama_gol, fig, fig2, fig3


# ============================================================
# TAB 3 — MODEL KARŞILAŞTIRMA (statik, bir kere hesaplanır)
# ============================================================
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
ozet_tablo = ozet.round(4).reset_index().rename(columns={"index": "Model"})

fig_metrik, ax_metrik = plt.subplots(figsize=(6, 4))
metrikler_goster = ["Test Acc", "Precision", "Recall", "F1 Score", "AUC-ROC"]
ozet[metrikler_goster].plot(kind="bar", ax=ax_metrik,
                             color=["#3498db", "#2ecc71", "#e74c3c", "#f39c12", "#9b59b6"])
ax_metrik.set_title("Metrik Karşılaştırma")
ax_metrik.legend(loc="lower right", fontsize=8)
plt.setp(ax_metrik.get_xticklabels(), rotation=15)

fig_roc, ax_roc = plt.subplots(figsize=(6, 4))
for isim, model in modeller.items():
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    ax_roc.plot(fpr, tpr, label=f"{isim} (AUC={sonuclar[isim]['AUC-ROC']:.3f})", color=RENKLER[isim])
ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.5)
ax_roc.set_xlabel("False Positive Rate")
ax_roc.set_ylabel("True Positive Rate")
ax_roc.set_title("ROC Eğrileri")
ax_roc.legend(fontsize=8)


def confusion_matrix_ciz(secilen_cm):
    cm = confusion_matrix(y_test, modeller[secilen_cm].predict(X_test))
    fig, ax = plt.subplots(figsize=(4, 3.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Kazanmadı", "Kazandı"], yticklabels=["Kazanmadı", "Kazandı"])
    ax.set_xlabel("Tahmin")
    ax.set_ylabel("Gerçek")
    return fig


# ============================================================
# TAB 4 — ÖZELLİK ÖNEMİ & KARAR AĞACI
# ============================================================
rf = modeller["Random Forest"]
importance = pd.Series(rf.feature_importances_, index=OZELLIKLER).sort_values(ascending=False)

fig_onem, ax_onem = plt.subplots(figsize=(8, 3.5))
colors = ["#9b59b6" if v == importance.max() else "#bdc3c7" for v in importance.values]
importance.sort_values().plot(kind="barh", color=colors[::-1], ax=ax_onem)
ax_onem.set_title("Özellik Önem Skorları")

dt = modeller["Karar Agaci"]


def karar_agaci_ciz(derinlik):
    fig, ax = plt.subplots(figsize=(18, 7))
    plot_tree(dt, feature_names=OZELLIKLER, class_names=["Kazanmadi", "Kazandi"],
              filled=True, rounded=True, fontsize=8, ax=ax, max_depth=derinlik)
    return fig


dt_kisa = DecisionTreeClassifier(max_depth=3, random_state=42).fit(X_train, y_train)
KURALLAR_METNI = export_text(dt_kisa, feature_names=OZELLIKLER)


# ============================================================
# ARAYÜZ
# ============================================================
with gr.Blocks(title="Futbol Maç Tahmin Projesi") as demo:
    gr.Markdown("# ⚽ Futbol Maç Sonucu Tahmin Projesi")
    gr.Markdown("Karar Ağacı · Random Forest · Lojistik Regresyon — international_matches1.csv üzerinde eğitildi")

    with gr.Tab("🔮 Tahmin"):
        with gr.Row():
            ev_takim = gr.Dropdown(TAKIMLAR, value=TAKIMLAR[0], label="Ev Sahibi Takım")
            deplasman_takim = gr.Dropdown(TAKIMLAR, value=TAKIMLAR[1], label="Deplasman Takımı")
        turnuva = gr.Dropdown(TURNUVALAR, value=TURNUVALAR[0], label="Turnuva")
        stadyum = gr.Radio(["Evet", "Hayır"], value="Evet", label="Ev sahibi kendi stadyumunda mı oynuyor?")
        model_secimi = gr.Dropdown(list(modeller.keys()), value=list(modeller.keys())[0], label="Model")
        tahmin_buton = gr.Button("Tahmin Et", variant="primary")

        tahmin_cikti = gr.Label(label="Tahmin")
        tahmin_caption = gr.Markdown()

        tahmin_buton.click(
            tahmin_et,
            inputs=[ev_takim, deplasman_takim, turnuva, stadyum, model_secimi],
            outputs=[tahmin_cikti, tahmin_caption],
        )

    with gr.Tab("📊 Veri Keşfi (EDA)"):
        with gr.Row():
            yil_baslangic = gr.Slider(YIL_MIN, YIL_MAX, value=YIL_MIN, step=1, label="Başlangıç Yılı")
            yil_bitis = gr.Slider(YIL_MIN, YIL_MAX, value=YIL_MAX, step=1, label="Bitiş Yılı")
        turnuva_secimi = gr.Dropdown(TURNUVALAR, value=[], multiselect=True, label="Turnuva (boş = tümü)")

        with gr.Row():
            toplam_mac_md = gr.Markdown()
            kazanma_orani_md = gr.Markdown()
            ortalama_gol_md = gr.Markdown()

        with gr.Row():
            gol_dagilimi_plot = gr.Plot(label="Gol Dağılımı")
            yillik_kazanma_plot = gr.Plot(label="Yıllara Göre Ev Sahibi Kazanma Oranı")
        top_takim_plot = gr.Plot(label="En Çok Maç Oynayan 10 Ev Sahibi Takım")

        eda_inputs = [yil_baslangic, yil_bitis, turnuva_secimi]
        eda_outputs = [toplam_mac_md, kazanma_orani_md, ortalama_gol_md,
                       gol_dagilimi_plot, yillik_kazanma_plot, top_takim_plot]
        for girdi in eda_inputs:
            girdi.change(eda_guncelle, inputs=eda_inputs, outputs=eda_outputs)

        demo.load(eda_guncelle, inputs=eda_inputs, outputs=eda_outputs)

    with gr.Tab("📈 Model Karşılaştırma"):
        gr.Dataframe(value=ozet_tablo, label="Metrik Özeti")
        with gr.Row():
            gr.Plot(value=fig_metrik, label="Metrik Karşılaştırma")
            gr.Plot(value=fig_roc, label="ROC Eğrileri")

        secilen_cm = gr.Dropdown(list(modeller.keys()), value=list(modeller.keys())[0],
                                  label="Confusion Matrix için model seç")
        cm_plot = gr.Plot(value=confusion_matrix_ciz(list(modeller.keys())[0]))
        secilen_cm.change(confusion_matrix_ciz, inputs=secilen_cm, outputs=cm_plot)

    with gr.Tab("🌳 Özellik Önemi & Karar Ağacı"):
        gr.Markdown("### Random Forest — Özellik Önemi")
        gr.Plot(value=fig_onem)

        gr.Markdown("### Karar Ağacı Görselleştirme")
        derinlik = gr.Slider(1, 5, value=3, step=1, label="Gösterilecek derinlik")
        agac_plot = gr.Plot(value=karar_agaci_ciz(3))
        derinlik.change(karar_agaci_ciz, inputs=derinlik, outputs=agac_plot)

        with gr.Accordion("Metin formatında kurallar (ilk 3 seviye)", open=False):
            gr.Markdown(f"```\n{KURALLAR_METNI}\n```")

if __name__ == "__main__":
    demo.launch()
