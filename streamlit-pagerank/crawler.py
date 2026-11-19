"""Crawler sederhana untuk analisis PageRank pada satu domain."""

import time
from collections import deque
from urllib.parse import urljoin, urlparse, urldefrag
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup

USER_AGENT = "PPWPageRankBot/1.0"


def ekstraksi_url(url, session):
    """Ambil tautan dari halaman HTML; None jika halaman gagal diakses."""
    try:
        response = session.get(url, timeout=10, allow_redirects=False)
        response.raise_for_status()
        if "text/html" not in response.headers.get("Content-Type", "").lower():
            return None
        soup = BeautifulSoup(response.text, "html.parser")
        daftar_url = set()
        for link in soup.find_all("a", href=True):
            url_lengkap, _ = urldefrag(urljoin(url, link["href"]))
            if urlparse(url_lengkap).scheme in ("http", "https"):
                daftar_url.add(url_lengkap)
        return daftar_url
    except requests.RequestException as exc:
        print(f"Gagal mengakses {url}: {exc}")
        return None


def web_crawling(url_awal, max_halaman=20, max_depth=2):
    """Kembalikan (halaman berhasil diakses, semua tautan internal yang ditemukan)."""
    visited = set()
    edges = set()
    queue = deque([(url_awal, 0)])
    queued = {url_awal}
    domain_awal = urlparse(url_awal).netloc

    robots = RobotFileParser()
    robots.set_url(urljoin(url_awal, "/robots.txt"))
    try:
        robots.read()
    except Exception as exc:
        raise RuntimeError(f"Tidak dapat memeriksa robots.txt: {exc}") from exc

    with requests.Session() as session:
        session.headers.update({"User-Agent": USER_AGENT})
        while queue and len(visited) < max_halaman:
            url, depth = queue.popleft()
            if url in visited or urlparse(url).netloc != domain_awal:
                continue
            if not robots.can_fetch(USER_AGENT, url):
                print(f"Crawling tidak diizinkan: {url}")
                continue

            print(f"Crawling: {url} | Depth: {depth}")
            daftar_url = ekstraksi_url(url, session)
            if daftar_url is None:
                continue  # Halaman gagal diakses tidak dihitung sebagai visited

            visited.add(url)
            for link in daftar_url:
                if urlparse(link).netloc != domain_awal:
                    continue
                edges.add((url, link))
                if depth < max_depth and link not in visited and link not in queued:
                    queue.append((link, depth + 1))
                    queued.add(link)
            time.sleep(1)

    return visited, edges
