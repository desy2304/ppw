import streamlit as st
import numpy as np
import string
import re
import joblib

from gensim.models import Word2Vec
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory


# =========================================================
# KONFIGURASI
# =========================================================

st.set_page_config(
    page_title="Klasifikasi Berita",
    page_icon="📰",
    layout="centered"
)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    model = Word2Vec.load(
        "model/word2vec_skipgram.model"
    )

    nb = joblib.load(
        "model/naive_bayes.pkl"
    )

    return model, nb


model, nb = load_model()


# =========================================================
# STOPWORD DAN STEMMER
# =========================================================

stop_factory = StopWordRemoverFactory()
stopwords = set(stop_factory.get_stop_words())

stem_factory = StemmerFactory()
stemmer = stem_factory.create_stemmer()


# =========================================================
# PREPROCESSING
# =========================================================

def preprocessing(teks):

    # 1. Mengubah teks menjadi huruf kecil
    teks = teks.lower()

    # 2. Menghapus tanda baca
    teks = teks.translate(
        str.maketrans('', '', string.punctuation)
    )

    # 3. Menghapus angka
    teks = re.sub(r'\d+', '', teks)

    # 4. Tokenisasi
    kata = teks.split()

    # 5. Menghapus stopword
    kata = [
        k for k in kata
        if k not in stopwords
    ]

    # 6. Stemming
    kata = [
        stemmer.stem(k)
        for k in kata
    ]

    return kata


# =========================================================
# MEMBUAT VEKTOR BERITA
# =========================================================

def buat_vektor_berita(tokens, model):

    vektor_kata = []

    for kata in tokens:

        if kata in model.wv:
            vektor_kata.append(
                model.wv[kata]
            )

    if len(vektor_kata) > 0:

        return np.mean(
            vektor_kata,
            axis=0
        )

    else:

        return np.zeros(
            model.vector_size
        )


# =========================================================
# TAMPILAN
# =========================================================

st.title("📰 Klasifikasi Berita")

st.write(
    """
    Aplikasi klasifikasi berita menggunakan
    **Word2Vec Skip-gram** dan **Gaussian Naive Bayes**.
    
    Masukkan isi berita untuk mengetahui apakah berita
    termasuk kategori **Sport** atau **Finance**.
    """
)

st.divider()


# =========================================================
# INPUT BERITA
# =========================================================

teks_berita = st.text_area(
    "Masukkan Isi Berita",
    placeholder="Masukkan teks berita di sini...",
    height=250
)


# =========================================================
# TOMBOL PREDIKSI
# =========================================================

if st.button(
    "🔍 Klasifikasikan Berita",
    use_container_width=True
):

    if not teks_berita.strip():

        st.warning(
            "Silakan masukkan isi berita terlebih dahulu."
        )

    else:

        # Preprocessing
        tokens = preprocessing(teks_berita)

        # Mengecek apakah masih ada token
        if len(tokens) == 0:

            st.error(
                "Teks tidak memiliki kata yang dapat diproses."
            )

        else:

            # Membuat vector berita
            vector = buat_vektor_berita(
                tokens,
                model
            )

            # Mengubah menjadi bentuk 2D
            vector = vector.reshape(1, -1)

            # Prediksi
            prediksi = nb.predict(vector)[0]

            # Probabilitas
            probabilitas = nb.predict_proba(vector)[0]

            # Nama kelas
            kelas = nb.classes_

            # Mengambil probabilitas tertinggi
            probabilitas_tertinggi = np.max(
                probabilitas
            )

            st.divider()

            st.subheader("Hasil Klasifikasi")

            if prediksi == "sport":

                st.success(
                    "⚽ Berita termasuk kategori **SPORT**"
                )

            elif prediksi == "finance":

                st.success(
                    "💰 Berita termasuk kategori **FINANCE**"
                )

            else:

                st.success(
                    f"Berita termasuk kategori **{prediksi.upper()}**"
                )

            st.metric(
                "Tingkat Keyakinan",
                f"{probabilitas_tertinggi * 100:.2f}%"
            )

            # Menampilkan probabilitas masing-masing kelas
            st.subheader("Probabilitas")

            for nama_kelas, nilai in zip(
                kelas,
                probabilitas
            ):

                st.write(
                    f"**{nama_kelas.upper()}**"
                )

                st.progress(
                    float(nilai)
                )

                st.caption(
                    f"{nilai * 100:.2f}%"
                )