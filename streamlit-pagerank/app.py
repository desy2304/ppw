import streamlit as st
import pandas as pd

from urllib.parse import urlparse
from crawler import web_crawling

import networkx as nx
import streamlit.components.v1 as components
from pyvis.network import Network


# Konfigurasi halaman
st.set_page_config(
    page_title="Web Crawling dan PageRank",
    page_icon="🌐",
    layout="wide"
)


# Judul aplikasi
st.title("🌐 Web Crawling dan Analisis PageRank")

st.write(
    "Aplikasi ini digunakan untuk melakukan web crawling, "
    "menganalisis hubungan antarhalaman website, "
    "dan menentukan halaman yang paling penting "
    "menggunakan algoritma PageRank."
)


# Sidebar pengaturan crawling
st.sidebar.header("Pengaturan Crawling")

max_halaman = st.sidebar.number_input(
    "Maksimal Halaman",
    min_value=1,
    max_value=100,
    value=20,
    step=1
)

max_depth = st.sidebar.number_input(
    "Kedalaman Crawling",
    min_value=0,
    max_value=5,
    value=2,
    step=1
)


# Input URL
st.subheader("Input Website")

url_awal = st.text_input(
    "Masukkan URL Website",
    value="https://quotes.toscrape.com/"
)


# Tombol menjalankan crawling
if st.button("Mulai Crawling", type="primary"):

    # Validasi URL
    parsed_url = urlparse(url_awal)

    if parsed_url.scheme not in ["http", "https"] or not parsed_url.netloc:
        st.error("Masukkan URL yang valid menggunakan http:// atau https://")

    else:
        with st.spinner("Sedang melakukan web crawling..."):

            try:
                # Menjalankan proses crawling
                visited, edges = web_crawling(
                    url_awal,
                    max_halaman=max_halaman,
                    max_depth=max_depth
                )

                # Menyimpan hasil ke session state
                st.session_state["visited"] = visited
                st.session_state["edges"] = edges

                st.success("Proses web crawling berhasil diselesaikan!")

            except Exception as e:
                st.error(f"Terjadi kesalahan: {e}")


# Menampilkan hasil crawling
if "visited" in st.session_state and "edges" in st.session_state:

    visited = st.session_state["visited"]
    edges = st.session_state["edges"]

    st.subheader("Hasil Web Crawling")

    # Menampilkan statistik crawling
    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Jumlah Halaman Dikunjungi",
            len(visited)
        )

    with col2:
        st.metric(
            "Jumlah Hubungan URL",
            len(edges)
        )

    # Membuat DataFrame hasil crawling
    df_crawling = pd.DataFrame(
        sorted(edges),
        columns=["Source", "Target"]
    )

    # Menampilkan tabel
    st.subheader("Tabel Hubungan Antarhalaman")

    st.dataframe(
        df_crawling,
        use_container_width=True
    )
    
    # VISUALISASI GRAPH
    st.subheader("Visualisasi Graph Hasil Web Crawling")

    # Membuat graph berarah
    G = nx.DiGraph()

    # Menambahkan node dan edge
    G.add_nodes_from(visited)
    G.add_edges_from(edges)

    # Menampilkan informasi graph
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Jumlah Node", G.number_of_nodes())

    with col2:
        st.metric("Jumlah Edge", G.number_of_edges())

    # Membuat visualisasi interaktif
    net = Network(
        height="600px",
        width="100%",
        directed=True,
        bgcolor="#ffffff",
        font_color="black",
        cdn_resources="in_line"
    )

    # Mengatur posisi node
    net.barnes_hut()

    # Menambahkan node ke PyVis
    for i, node in enumerate(sorted(G.nodes()), start=1):
        net.add_node(
            node,
            label=f"Node {i}",
            title=node,
            color="#87CEEB",
            size=15
        )

    # Menambahkan edge ke PyVis
    for source, target in G.edges():
        net.add_edge(source, target)

    # Menghasilkan HTML graph
    html_graph = net.generate_html()

    # Menampilkan graph di Streamlit
    components.html(
        html_graph,
        height=620,
        scrolling=True
    )
