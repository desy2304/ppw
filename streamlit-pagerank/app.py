"""Aplikasi Streamlit: crawling, visualisasi graph, dan analisis PageRank."""

from urllib.parse import urlparse

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network

from crawler import web_crawling

st.set_page_config(page_title="Web Crawling dan PageRank", page_icon="🌐", layout="wide")
st.title("🌐 Web Crawling dan Analisis PageRank")
st.write(
    "Aplikasi ini digunakan untuk melakukan web crawling, "
    "menganalisis hubungan antarhalaman website, dan menentukan "
    "halaman yang paling penting menggunakan algoritma PageRank."
)

st.sidebar.header("Pengaturan Crawling")
max_halaman = st.sidebar.number_input("Maksimal Halaman", min_value=1, max_value=100, value=20, step=1)
max_depth = st.sidebar.number_input("Kedalaman Crawling", min_value=0, max_value=5, value=2, step=1)

st.subheader("Input Website")
url_awal = st.text_input("Masukkan URL Website", value="https://quotes.toscrape.com/")

if st.button("Mulai Crawling", type="primary"):
    # Hapus hasil lama agar tidak tampil jika input baru gagal.
    st.session_state.pop("visited", None)
    st.session_state.pop("edges", None)
    parsed_url = urlparse(url_awal)
    if parsed_url.scheme not in ("http", "https") or not parsed_url.netloc:
        st.error("Masukkan URL yang valid menggunakan http:// atau https://")
    else:
        with st.spinner("Sedang melakukan web crawling..."):
            try:
                visited, edges = web_crawling(
                    url_awal, max_halaman=int(max_halaman), max_depth=int(max_depth)
                )
                st.session_state["visited"] = visited
                st.session_state["edges"] = edges
                st.success("Proses web crawling selesai!")
            except Exception as exc:
                st.error(f"Terjadi kesalahan: {exc}")

if "visited" in st.session_state and "edges" in st.session_state:
    visited = st.session_state["visited"]
    edges = st.session_state["edges"]

    # Gunakan hanya tautan yang kedua ujungnya berhasil dikunjungi.
    edges_visited = sorted(
        (source, target)
        for source, target in edges
        if source in visited and target in visited
    )

    G = nx.DiGraph()
    G.add_nodes_from(visited)
    G.add_edges_from(edges_visited)

    st.subheader("Hasil Web Crawling")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Jumlah Halaman Dikunjungi", len(visited))
    with col2:
        st.metric("Jumlah Hubungan URL", len(edges_visited))

    st.subheader("Tabel Hubungan Antarhalaman")
    df_crawling = pd.DataFrame(edges_visited, columns=["Source", "Target"])
    st.dataframe(df_crawling, use_container_width=True, hide_index=True)

    st.subheader("Visualisasi Graph Hasil Web Crawling")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Jumlah Node", G.number_of_nodes())
    with col2:
        st.metric("Jumlah Edge", G.number_of_edges())

    if G.number_of_nodes() == 0:
        st.warning("Tidak ada halaman yang berhasil diakses. Periksa URL dan izin crawling.")
    else:
        net = Network(
            height="600px", width="100%", directed=True,
            bgcolor="#ffffff", font_color="black", cdn_resources="in_line"
        )
        
        net.barnes_hut()
        
        # Menampilkan URL lengkap pada setiap node
        for node in sorted(G.nodes()):
            net.add_node(
                node,
                label=node,
                title=node,
                color="#87CEEB",
                size=15,
                font={"size": 12}
            )
        
        # Menambahkan hubungan antarhalaman
        for source, target in G.edges():
            net.add_edge(source, target)

        components.html(net.generate_html(), height=620, scrolling=True)

        st.subheader("Analisis PageRank")
        try:
            pagerank = nx.pagerank(G, alpha=0.85, max_iter=1000)
        except (nx.PowerIterationFailedConvergence, ImportError) as exc:
            st.error(f"PageRank gagal dihitung: {exc}. Pastikan scipy terpasang.")
        else:
            ranking_pagerank = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)
            df_pagerank = pd.DataFrame(ranking_pagerank, columns=["URL", "PageRank"])
            df_pagerank.insert(0, "Ranking", range(1, len(df_pagerank) + 1))

            st.subheader("Tabel Ranking PageRank")
            st.dataframe(df_pagerank, use_container_width=True, hide_index=True)
            url_tertinggi = df_pagerank.iloc[0]["URL"]
            skor_tertinggi = df_pagerank.iloc[0]["PageRank"]
            st.success(
                f"Halaman dengan PageRank tertinggi: {url_tertinggi}\n\n"
                f"Nilai PageRank: {skor_tertinggi:.6f}"
            )

            st.subheader("Top 10 Halaman Berdasarkan PageRank")
            top_10 = df_pagerank.head(10).copy()
            top_10["Label"] = top_10["URL"].apply(
                lambda url: urlparse(url).path.strip("/") or "Homepage"
            )
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.barh(top_10["Label"], top_10["PageRank"], color="skyblue")
            ax.invert_yaxis()
            ax.set_title("Top 10 Halaman dengan PageRank Tertinggi")
            ax.set_xlabel("Nilai PageRank")
            ax.set_ylabel("Halaman Website")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)