import requests
import time

from bs4 import BeautifulSoup
from collections import deque
from urllib.parse import urljoin, urlparse, urldefrag
from urllib.robotparser import RobotFileParser


# Konfigurasi crawler
USER_AGENT = "PPWPageRankBot/1.0"


# Fungsi untuk mengekstraksi URL dari halaman website
def ekstraksi_url(url, session):

    daftar_url = set()

    try:
        # Mengakses halaman website
        response = session.get(
            url,
            timeout=10,
            allow_redirects=False
        )

        response.raise_for_status()

        # Memastikan konten berupa HTML
        content_type = response.headers.get("Content-Type", "")

        if "text/html" not in content_type.lower():
            return daftar_url

        # Membaca konten HTML
        soup = BeautifulSoup(response.text, "html.parser")

        # Mengambil semua tautan
        for link in soup.find_all("a", href=True):

            # Mengubah URL relatif menjadi URL absolut
            url_lengkap = urljoin(url, link["href"])

            # Menghapus fragment URL
            url_lengkap, _ = urldefrag(url_lengkap)

            # Memastikan URL menggunakan HTTP/HTTPS
            if urlparse(url_lengkap).scheme in ["http", "https"]:
                daftar_url.add(url_lengkap)

    except requests.RequestException as e:
        print(f"Gagal mengakses {url}: {e}")

    return daftar_url


# Fungsi utama Web Crawling
def web_crawling(url_awal, max_halaman=20, max_depth=2):

    visited = set()
    edges = set()

    # Antrean URL
    queue = deque([(url_awal, 0)])
    queued = {url_awal}

    # Mengambil domain awal
    domain_awal = urlparse(url_awal).netloc

    # Membaca robots.txt
    robots = RobotFileParser()
    robots.set_url(urljoin(url_awal, "/robots.txt"))

    try:
        robots.read()
    except Exception as e:
        raise RuntimeError(
            f"Tidak dapat memeriksa robots.txt: {e}"
        ) from e

    # Membuat session
    session = requests.Session()
    session.headers.update({
        "User-Agent": USER_AGENT
    })

    while queue and len(visited) < max_halaman:

        url, depth = queue.popleft()

        # Menghindari URL duplikat
        if url in visited:
            continue

        # Membatasi crawling pada domain awal
        if urlparse(url).netloc != domain_awal:
            continue

        # Memeriksa izin crawling
        if not robots.can_fetch(USER_AGENT, url):
            print(f"Crawling tidak diizinkan: {url}")
            continue

        print(f"Crawling: {url} | Depth: {depth}")

        # Mengambil URL dari halaman
        daftar_url = ekstraksi_url(url, session)

        # Menyimpan halaman yang sudah dikunjungi
        visited.add(url)

        for link in daftar_url:

            # Hanya mengambil URL dari domain yang sama
            if urlparse(link).netloc == domain_awal:

                # Menyimpan hubungan antarhalaman
                edges.add((url, link))

                # Menambahkan URL baru ke antrean
                if (
                    depth < max_depth
                    and link not in visited
                    and link not in queued
                ):
                    queue.append((link, depth + 1))
                    queued.add(link)

        # Memberikan jeda antarpermintaan
        time.sleep(1)

    session.close()

    return visited, edges