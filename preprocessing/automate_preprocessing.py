import nltk
import pandas as pd
import re
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
# Pastikan stopwords tersedia
# nltk.download('stopwords')
# nltk.download('punkt_tab')
stop_words = set(stopwords.words('indonesian'))
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
factory = StemmerFactory()
stemmer = factory.create_stemmer()

tqdm.pandas() 

def casefolding(text):
    text = text.lower()                     # merubah kalimat menjadi huruf kecil
    text = re.sub(r'[-+]?[0-9]+','',text)   # menghapus angka
    text = re.sub(r'[^\w\s]',' ',text)       # menghapus simbol dan tanda baca
    text = text.strip()                     # menghapus spasi awal dan akhir

    return text

def normalisasi(text):
    slang_dict = pd.read_csv('new_kamusalay.csv', encoding='latin-1', header=None)
    slang_dict = slang_dict.rename(columns={0: 'original', 1: 'replacement'})

    slang_dict.tail()

    # Mengonversi kamus slang menjadi dictionary untuk pencarian yang lebih cepat
    slang_dict = dict(zip(slang_dict['original'], slang_dict['replacement']))
    # Split the text into words
    words = text.split()
    # Normalize words using the slang dictionary
    normalized_words = [slang_dict.get(word, word) for word in words]
    # Join the normalized words back into a string
    return " ".join(normalized_words)

def tokenisasi(text):
  # Tokenisasi

  text = word_tokenize(text)

  return text

def stopwords_removal(text):
  new_stopwords = ["yg", "dg", "rt", "dgn", "ny", "d", 'klo',
                 'kalo', 'amp', 'biar', 'bikin', 'bilang',
                 'gak', 'ga , ngga', 'krn', 'nya', 'nih', 'sih',
                 'si', 'tau', 'tdk', 'tuh', 'utk', 'ya',
                 'jd', 'jgn', 'sdh', 'aja', 'n', 't','ngga',
                 'hehe', 'pen', 'u', 'nan', 'loh', 'rt',
                 '&amp', 'yah',"nya","muas", "rumah", "sakit", "sultan", "agung",
                 'amino', 'dokter', 'rsud', 'pol', 'elisabeth', 'rs', 'semarang', 'rsi',
                 'pasien', 'baitul', 'izzah', 'rswn', 'rawat', 'pasien', 'primaya','setelah',
                 'saya', 'yang', 'rsjd', 'cipto', 'wa', 'dr','rumah sakit'
                ]

  stopwords_dict = stopwords.words('indonesian') + new_stopwords
  text = [word for word in text if word not in stopwords_dict]
  return text

def stemming(text):
    text = [stemmer.stem(word) for word in text]
    text = " ".join(text)
    return text


def preprocessing(text):
  text = casefolding(text)
  text = normalisasi(text)
  text = tokenisasi(text)
  text = stopwords_removal(text)
  text = stemming(text)

  return text

def run_preprocessing(input_path, output_path):
    """
    Fungsi utama untuk memuat data, menjalankan preprocessing, dan menyimpan hasil.
    """
    print("Memulai proses preprocessing...")

    # Memuat data
    df = pd.read_excel(input_path)
    print(f"Data berhasil dimuat dari {input_path}")

    # Asumsikan kolom ulasan bernama 'ulasan'. Sesuaikan jika berbeda.
    # Menerapkan fungsi preprocessing ke setiap baris di kolom 'ulasan'
    df= df.assign(ulasan=df['ulasan'].progress_apply(preprocessing))
    
    print("Kolom 'ulasan_bersih' berhasil dibuat.")

    # Menyimpan dataframe yang sudah bersih ke file CSV baru
    df.to_csv(output_path, index=False)
    print(f"Preprocessing selesai. Data bersih disimpan di {output_path}")

# Blok ini akan dieksekusi hanya jika file ini dijalankan secara langsung
if __name__ == "__main__":
    # Tentukan path input dan output
    raw_data_path = '../data_raw/dataset_rs.xlsx'
    processed_data_path = './data_processed/dataset_rs_processed.csv'
    
    # Jalankan fungsi utama
    run_preprocessing(raw_data_path, processed_data_path)
