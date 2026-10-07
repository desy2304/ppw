import streamlit as st
import numpy as np
import string
import re
import joblib
import trafilatura

from pathlib import Path
from gensim.models import Word2Vec
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# KONFIGURASI HALAMAN
st.set_page_config(
    page_title="Klasifikasi Berita",
    layout="centered"
)

# PATH MODEL
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"

# LOAD MODEL
@st.cache_resource
def load_model():
    # Memuat model Word2Vec Skip-gram
    model = Word2Vec.load(
        str(MODEL_DIR / "word2vec_skipgram.model")
    )

    # Memuat model Gaussian Naive Bayes
    nb = joblib.load(
        MODEL_DIR / "naive_bayes.pkl"
    )

    return model, nb

model, nb = load_model()

# STOPWORD DAN STEMMING
# Mengambil daftar stopword Bahasa Indonesia
stop_factory = StopWordRemoverFactory()
stopwords = set(stop_factory.get_stop_words())

# Membuat stemmer Bahasa Indonesia
stem_factory = StemmerFactory()
stemmer = stem_factory.create_stemmer()

# PREPROCESSING
def preprocessing(teks):
    # Mengubah teks menjadi huruf kecil
    teks = teks.lower()

    # Menghapus tanda baca
    teks = teks.translate(
        str.maketrans('', '', string.punctuation)
    )

    # Menghapus angka
    teks = re.sub(r'\d+', '', teks)

    # Tokenisasi sederhana berdasarkan spasi
    kata = teks.split()

    # Menghapus stopword
    kata = [
        k for k in kata
        if k not in stopwords
    ]

    # Melakukan stemming
    kata = [
        stemmer.stem(k)
        for k in kata
    ]

    return kata

# MEMBUAT VEKTOR BERITA
def buat_vektor_berita(tokens, model):
    vektor_kata = []

    # Mengambil vektor setiap kata yang terdapat dalam model
    for kata in tokens:
        if kata in model.wv:
            vektor_kata.append(model.wv[kata])

    # Menggunakan rata-rata vektor kata sebagai vektor berita
    if len(vektor_kata) > 0:
        return np.mean(vektor_kata, axis=0)

    # Jika tidak ada kata yang dikenali model
    return np.zeros(model.vector_size)

# MENGAMBIL ISI BERITA DARI URL
def ambil_isi_berita(url):
    try:
        # Mengambil halaman berita dari URL
        downloaded = trafilatura.fetch_url(url)

        if not downloaded:
            return None

        # Mengekstrak isi utama artikel
        isi = trafilatura.extract(downloaded, favor_recall=True)

        return isi

    except Exception:
        return None

# TAMPILAN APLIKASI
st.title("Klasifikasi Berita")

st.write(
    """
    Aplikasi klasifikasi berita menggunakan
    **Word2Vec Skip-gram** dan **Gaussian Naive Bayes**.

    Masukkan URL berita Detik untuk mengetahui apakah berita
    termasuk kategori **Sport** atau **Finance**.
    """
)

st.divider()

# INPUT BERITA
url_berita = st.text_input("Masukkan URL Berita Detik", placeholder="https://sport.detik.com/...")

# PROSES KLASIFIKASI
if st.button(
    "Klasifikasikan Berita",
    use_container_width=True
):
    # Mengecek apakah URL sudah diisi
    if not url_berita.strip():
        st.warning("Silakan masukkan URL berita terlebih dahulu.")

    # Memastikan URL berasal dari Detik.com
    elif "detik.com" not in url_berita.lower():
        st.warning("Silakan masukkan URL berita dari Detik.com.")

    else:
        # Mengambil isi artikel dari URL
        with st.spinner("Mengambil isi berita..."):
            teks_berita = ambil_isi_berita(url_berita)

        if not teks_berita:
            st.error(
                "Isi berita gagal diambil dari URL. "
                "Pastikan URL berita dapat diakses."
            )

        else:
            st.success("Isi berita berhasil diambil.")

            # Menampilkan isi berita yang berhasil diambil
            with st.expander("Lihat Isi Berita"):
                st.write(teks_berita)

            # Melakukan preprocessing
            tokens = preprocessing(teks_berita)

            # Mengecek hasil preprocessing
            if len(tokens) == 0:
                st.error("Isi berita tidak memiliki kata yang dapat diproses.")

            else:
                # Mengubah berita menjadi vektor
                vector = buat_vektor_berita(tokens, model)

                # Mengubah vektor menjadi bentuk 2 dimensi
                vector = vector.reshape(1, -1)

                # Melakukan prediksi menggunakan Naive Bayes
                prediksi = nb.predict(vector)[0]

                # Mengambil probabilitas setiap kelas
                probabilitas = nb.predict_proba(vector)[0]
                kelas = nb.classes_

                # Mengambil probabilitas tertinggi
                probabilitas_tertinggi = np.max(probabilitas)

                st.divider()

                # HASIL KLASIFIKASI
                st.subheader("Hasil Klasifikasi")

                if prediksi == "sport":
                    st.success("Berita termasuk kategori **SPORT**")

                elif prediksi == "finance":
                    st.success("Berita termasuk kategori **FINANCE**")

                else:
                    st.success(f"Berita termasuk kategori **{prediksi.upper()}**")

                # Menampilkan tingkat keyakinan model
                st.metric("Tingkat Keyakinan", f"{probabilitas_tertinggi * 100:.2f}%")

                # PROBABILITAS KELAS
                st.subheader("Probabilitas")

                for nama_kelas, nilai in zip(kelas, probabilitas):
                    st.write(f"**{nama_kelas.upper()}**")
                    st.progress(float(nilai))
                    st.caption(f"{nilai * 100:.2f}%")