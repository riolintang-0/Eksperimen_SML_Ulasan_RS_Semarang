import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


mlflow.set_experiment("Sentiment Analysis - Baseline")

# Aktifkan MLflow Autologging
mlflow.sklearn.autolog()

# Muat data dan buat label (sama seperti sebelumnya)
df = pd.read_csv('data_processed/dataset_rs_processed.csv')
df.dropna(subset=['ulasan'], inplace=True)

KAMUS_POSITIF = set(pd.read_csv('kamus/positive.csv')['word'].astype(str).tolist())
KAMUS_NEGATIF = set(pd.read_csv('kamus/negative.csv')['word'].astype(str).tolist())

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

# Latih model di dalam blok 'with mlflow.start_run()'
with mlflow.start_run() as run:
    print(f"Starting run with ID: {run.info.run_id}")
    
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)
    
    print("Model training finished. Autolog should save the artifacts.")

print("\nEksperimen selesai. Periksa hasilnya di MLflow UI.")