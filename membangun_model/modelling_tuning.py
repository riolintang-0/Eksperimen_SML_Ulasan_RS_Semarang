import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def load_kamus(file_path):
    """Membaca file CSV kamus yang memiliki header ('word', 'weight')."""
    try:
        df = pd.read_csv(file_path)
        if 'word' not in df.columns:
            print(f"Error: File {file_path} tidak memiliki kolom 'word'.")
            return set()
        return set(df['word'].astype(str).tolist())
    except FileNotFoundError:
        print(f"Error: File kamus tidak ditemukan di {file_path}")
        return set()

# 1. Memuat Kamus & Data (Sama seperti sebelumnya)
print("Memuat kamus dan data...")
KAMUS_POSITIF = load_kamus('kamus/positive.csv')
KAMUS_NEGATIF = load_kamus('kamus/negative.csv')
df = pd.read_csv('data_processed/dataset_rs_processed.csv')
df.dropna(subset=['ulasan'], inplace=True)

# 2. Membuat Label Sentimen (Sama seperti sebelumnya)
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

# 3. Vectorization (Sama seperti sebelumnya)
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ==============================================================================
# 4. HYPERPARAMETER TUNING & MANUAL LOGGING (BAGIAN BARU)
# ==============================================================================
print("Memulai hyperparameter tuning dengan GridSearchCV...")

# Definisikan hyperparameter yang akan diuji
param_grid = {
    'alpha': [0.1, 0.5, 1.0, 1.5, 2.0],
    'fit_prior': [True, False]
}

# Inisialisasi GridSearchCV
grid_search = GridSearchCV(
    estimator=MultinomialNB(),
    param_grid=param_grid,
    scoring='f1_macro', # Metrik untuk evaluasi cross-validation
    cv=5, # 5-fold cross-validation
    n_jobs=-1 # Gunakan semua core CPU
)

# Mulai MLflow run secara manual
with mlflow.start_run():

    # Latih model dengan GridSearchCV
    grid_search.fit(X_train_vec, y_train)

    # Dapatkan model dan parameter terbaik
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    print(f"Parameter terbaik ditemukan: {best_params}")

    # Lakukan prediksi dengan model terbaik
    y_pred = best_model.predict(X_test_vec)

    # Hitung metrik evaluasi final
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    
    print(f"Metrik di data test: Akurasi={accuracy:.4f}, F1-Score={f1:.4f}")

    # Log parameter terbaik secara manual
    mlflow.log_params(best_params)

    # Log metrik final secara manual
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    
    # Log model terbaik secara manual
    mlflow.sklearn.log_model(best_model, "best_model")

print("\nEksperimen tuning selesai. Jalankan 'mlflow ui' untuk melihat hasilnya.")