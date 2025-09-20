import pandas as pd
import dagshub
import mlflow
import mlflow.sklearn
import joblib
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ==============================================================================
# 1. KONFIGURASI DAGSHUB
# ==============================================================================
dagshub.init(repo_owner='riolintang-0', repo_name='Eksperimen_SML_Ulasan_RS_Semarang', mlflow=True)

def load_kamus(file_path):
    try:
        df = pd.read_csv(file_path)
        return set(df['word'].astype(str).tolist())
    except FileNotFoundError: return set()
KAMUS_POSITIF = load_kamus('kamus/positive.csv')
KAMUS_NEGATIF = load_kamus('kamus/negative.csv')
df = pd.read_csv('data_processed/dataset_rs_processed.csv')
df.dropna(subset=['ulasan'], inplace=True)
def label_sentiment(ulasan):
    ulasan_split = str(ulasan).split()
    skor_positif = sum(1 for kata in ulasan_split if kata in KAMUS_POSITIF)
    skor_negatif = sum(1 for kata in ulasan_split if kata in KAMUS_NEGATIF)
    if skor_positif > skor_negatif: return 1
    elif skor_negatif > skor_positif: return 0
    else: return -1
df['sentiment'] = df['ulasan'].apply(label_sentiment)
df_final = df[df['sentiment'] != -1].copy()
X = df_final['ulasan']
y = df_final['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
param_grid = {'alpha': [0.1, 0.5, 1.0], 'fit_prior': [True, False]}

# Mulai MLflow Run
with mlflow.start_run():
    mlflow.set_tag("mlflow.runName", "advanced_tuning_run")
    grid_search = GridSearchCV(estimator=MultinomialNB(), param_grid=param_grid, scoring='f1_macro', cv=5, n_jobs=-1)
    grid_search.fit(X_train_vec, y_train)
    best_model = grid_search.best_estimator_

    # Hitung waktu prediksi (METRIK CUSTOM #1)
    start_time = time.time()
    y_pred = best_model.predict(X_test_vec)
    end_time = time.time()
    prediction_time = end_time - start_time

    # Hitung metrik standar
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    # Logging manual
    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    # LOGGING METRIK CUSTOM
    mlflow.log_metric("prediction_time_seconds", prediction_time)

    # Buat & log confusion matrix (METRIK CUSTOM #2)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues')
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')
    mlflow.log_figure(fig, "confusion_matrix.png")

    # Log model
    # 1. Simpan model ke file lokal
    model_filename = "best_model.joblib"
    joblib.dump(best_model, model_filename)

    # 2. Log file tersebut sebagai artefak ke DagsHub
    mlflow.log_artifact(model_filename, artifact_path="model")


print("\nEksperimen advance selesai. Periksa hasilnya di tab 'Experiments' repositori DagsHub Anda.")