import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Set nama eksperimen
mlflow.set_experiment("Sentiment Analysis - Tuning")

# Muat data dan buat label
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
X_test_vec = vectorizer.transform(X_test)

# Definisikan hyperparameter yang akan diuji
param_grid = {
    'alpha': [0.1, 0.5, 1.0],
    'fit_prior': [True, False]
}

grid_search = GridSearchCV(estimator=MultinomialNB(), param_grid=param_grid, scoring='f1_macro', cv=3)

# Mulai MLflow run secara manual
with mlflow.start_run():
    grid_search.fit(X_train_vec, y_train)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    y_pred = best_model.predict(X_test_vec)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    
    # Logging manual
    mlflow.log_params(best_params)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    

    mlflow.sklearn.log_model(best_model, "best_model")
    # --------------------------------------------------

print("\nEksperimen tuning selesai.")