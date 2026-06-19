from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from ortak import model_verisi_hazirla
import warnings
warnings.filterwarnings("ignore")

print("=" * 55)
print("ADIM 6 — MODEL EGiTiMi")
print("=" * 55)

# --- Veri hazırlık (kısa) ---
df, X, y, X_train, X_test, y_train, y_test = model_verisi_hazirla()

# --- 3 modeli tanımla ---
modeller = {
    "Karar Agaci"       : DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest"     : RandomForestClassifier(n_estimators=100, max_depth=5,
                                                  random_state=42, n_jobs=-1),
    "Lojistik Regresyon": LogisticRegression(max_iter=1000, random_state=42),
}

# --- Eğit ve değerlendir ---
for isim, model in modeller.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc  = accuracy_score(y_test, y_pred)

    print(f"\n{'='*40}")
    print(f"  {isim}")
    print(f"{'='*40}")
    print(f"  Train Accuracy : {train_acc:.4f}")
    print(f"  Test  Accuracy : {test_acc:.4f}")
    print(f"  Overfitting    : {train_acc - test_acc:+.4f}")
    print(f"\n  Siniflandirma Raporu:")
    print(classification_report(y_test, y_pred,
                                target_names=["Kazanmadi (0)", "Kazandi (1)"],
                                digits=4))

print("=> Model egitimi tamamlandi. Siradaki: 07_metrikler.py")
