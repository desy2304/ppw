import streamlit as st
import numpy as np
import string
import re
import joblib
import trafilatura

from pathlib import Path
from urllib.parse import urlparse

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
def load_models():

    # Load model Word2Vec Skip-gram
    skipgram_model = Word2Vec.load(
        str(MODEL_DIR / "word2vec_skipgram.model")
    )

    # Load model Gaussian Naive Bayes
    nb_model = joblib.load(
        MODEL_DIR / "naive_bayes.pkl"
    )

    return skipgram_model, nb_model


skipgram_model, nb_model = load_models()


# LOAD STOPWORD DAN STEMMER

stop_factory = StopWordRemoverFactory()
stop_words = set(stop_factory.get_stop_words())

stem_factory = StemmerFactory()
stemmer = stem_factory.create_stemmer()


# FUNGSI PREPROCESSING

def preprocessing(teks):

    # 1. Lowercase
    teks = teks.lower()

    # 2. Menghapus tanda baca
    teks = teks.translate(
        str.maketrans('', '', string.punctuation)
    )

    # 3. Menghapus angka
    teks = re.sub(r"\d+", " ", teks)

    # 4. Menghapus spasi berlebih
    teks = re.sub(r"\s+", " ", teks).strip()

    # 5. Tokenisasi
    tokens = teks.split()

    # 6. Stopword removal
    tokens = [
        kata
        for kata in tokens
        if kata not in stop_words
    ]

    # 7. Stemming
    tokens = [
        stemmer.stem(kata)
        for kata in tokens
    ]

    return tokens


# FUNGSI DOCUMENT VECTOR / VSM

def document_vector(tokens, model):

    vectors = []

    for word in tokens:

        if word in model.wv:
            vectors.append(
                model.wv[word]
            )

    # Jika tidak ada kata yang dikenal model
    if len(vectors) == 0:

        return np.zeros(
            model.vector_size
        )

    # Rata-rata vector semua kata
    return np.mean(
        vectors,
        axis=0
    )


# FUNGSI VALIDASI URL

def validasi_url(url):

    try:

        parsed_url = urlparse(url)

        domain = parsed_url.netloc.lower()
        path = parsed_url.path.lower()

        # Menghilangkan www jika ada
        domain = domain.replace("www.", "")

        # Mengecek kategori berdasarkan domain
        if domain == "sport.detik.com":

            kategori_url = "sport"

        elif domain == "finance.detik.com":

            kategori_url = "finance"

        else:

            return (
                False,
                None,
                "URL harus berasal dari Detik Sport atau Detik Finance."
            )

        # Memastikan URL merupakan halaman artikel
        if not re.search(r"/d-\d+", path):

            return (
                False,
                None,
                "URL yang dimasukkan bukan URL artikel berita Detik."
            )

        return True, kategori_url, None

    except Exception:

        return (
            False,
            None,
            "Format URL tidak valid."
        )


# FUNGSI MENGAMBIL ARTIKEL DARI URL

def extract_article(url):

    try:

        # Mengambil halaman berita
        downloaded = trafilatura.fetch_url(url)

        if downloaded is None:
            return None

        # Mengekstrak isi utama artikel
        text = trafilatura.extract(
            downloaded,
            favor_recall=True,
            include_comments=False,
            include_tables=False
        )

        return text

    except Exception:

        return None


# FUNGSI VALIDASI ISI ARTIKEL

def validasi_isi(tokens, model):

    # Minimal jumlah token
    if len(tokens) < 20:

        return (
            False,
            "Isi berita terlalu pendek untuk dianalisis."
        )

    # Menghitung kata yang terdapat dalam vocabulary Word2Vec
    kata_dikenal = [
        kata
        for kata in tokens
        if kata in model.wv
    ]

    # Menghitung persentase kata yang dikenal model
    rasio_kata_dikenal = (
        len(kata_dikenal) / len(tokens)
    )

    # Minimal 20% kata harus dikenal model
    if rasio_kata_dikenal < 0.20:

        return (
            False,
            "Isi berita tidak cukup sesuai dengan kosakata model."
        )

    return True, None


# TAMPILAN APLIKASI

st.title("Klasifikasi Berita")

st.write(
    """
    Klasifikasi berita menggunakan
    **Word2Vec Skip-gram** dan **Gaussian Naive Bayes**.

    Masukkan URL berita dari kategori
    **Sport** atau **Finance**.
    """
)

st.divider()


# INPUT URL

url = st.text_input(
    "Masukkan URL berita:",
    placeholder="https://sport.detik.com/..."
)


# TOMBOL KLASIFIKASI

if st.button(
    "Klasifikasi Berita",
    use_container_width=True
):

    # Mengecek input URL

    if not url.strip():

        st.warning(
            "Silakan masukkan URL berita terlebih dahulu."
        )

    else:

        # Validasi URL

        url_valid, kategori_url, pesan_error = validasi_url(url)

        if not url_valid:

            st.warning(pesan_error)

        else:

            # Mengambil artikel

            with st.spinner(
                "Mengambil isi berita..."
            ):

                artikel = extract_article(url)


            # Mengecek hasil scraping

            if artikel is None:

                st.error(
                    "Gagal mengambil isi berita dari URL tersebut."
                )

                st.info(
                    "Pastikan URL merupakan halaman artikel "
                    "berita yang dapat diakses."
                )

            else:

                # Menampilkan isi artikel

                st.success(
                    "Artikel berhasil diambil."
                )

                with st.expander(
                    "Lihat isi berita"
                ):

                    st.write(artikel)


                # PREPROCESSING

                tokens = preprocessing(
                    artikel
                )


                # Validasi isi artikel

                isi_valid, pesan_error = validasi_isi(
                    tokens,
                    skipgram_model
                )


                if not isi_valid:

                    st.warning(
                        pesan_error
                    )

                else:

                    # DOCUMENT VECTOR / VSM

                    vector = document_vector(
                        tokens,
                        skipgram_model
                    )


                    # Mengubah vector menjadi bentuk 2D

                    X_input = np.array(
                        [vector]
                    )


                    # PREDIKSI

                    prediction = nb_model.predict(
                        X_input
                    )[0]


                    # PROBABILITAS

                    probabilities = nb_model.predict_proba(
                        X_input
                    )[0]

                    classes = nb_model.classes_


                    # Mengambil probabilitas Sport dan Finance

                    prob_sport = 0
                    prob_finance = 0

                    for nama_kelas, nilai in zip(
                        classes,
                        probabilities
                    ):

                        if str(nama_kelas).lower() == "sport":

                            prob_sport = nilai

                        elif str(nama_kelas).lower() == "finance":

                            prob_finance = nilai


                    # Menghitung confidence tertinggi

                    confidence = max(
                        prob_sport,
                        prob_finance
                    )


                    # Menghitung selisih probabilitas

                    margin = abs(
                        prob_sport - prob_finance
                    )


                    # HASIL KLASIFIKASI

                    st.divider()

                    st.subheader(
                        "Hasil Klasifikasi"
                    )


                    # Threshold confidence

                    threshold = 0.70

                    # Minimum selisih antar kelas

                    minimum_margin = 0.20


                    if (
                        confidence >= threshold
                        and margin >= minimum_margin
                    ):

                        if str(prediction).lower() == "sport":

                            st.success(
                                "Berita termasuk kategori **SPORT**"
                            )

                        elif str(prediction).lower() == "finance":

                            st.success(
                                "Berita termasuk kategori **FINANCE**"
                            )

                        else:

                            st.warning(
                                f"Kategori: **{str(prediction).upper()}**"
                            )

                    else:

                        st.warning(
                            "Berita tidak cukup kuat untuk "
                            "diklasifikasikan sebagai SPORT atau FINANCE."
                        )


                    # CONFIDENCE

                    st.metric(
                        "Tingkat Keyakinan",
                        f"{confidence * 100:.2f}%"
                    )


                    # PROBABILITAS

                    st.subheader(
                        "Probabilitas Klasifikasi"
                    )

                    probability_data = []

                    for nama_kelas, nilai in zip(
                        classes,
                        probabilities
                    ):

                        probability_data.append({
                            "Kategori": str(nama_kelas).upper(),
                            "Probabilitas": f"{nilai * 100:.2f}%"
                        })

                    st.table(
                        probability_data
                    )