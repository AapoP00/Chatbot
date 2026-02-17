# indexer.py
import os
import time
import requests
import xml.etree.ElementTree as ET
from typing import List
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
from playwright.sync_api import sync_playwright

# Load environment variables from .env.
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = "https://bittipuisto.fi"
SITEMAP_URL = f"{BASE_URL}/sitemap.xml"

CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_store")
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "bittipuisto")

# Initialize OpenAI client.
client = OpenAI(api_key=OPENAI_API_KEY)

# Create or load persistent Chroma collection
chroma = chromadb.PersistentClient(path=CHROMA_DIR)
collection = chroma.get_or_create_collection(name=COLLECTION_NAME)

def is_xml_url(url: str) -> bool:
    return url.lower().split("?")[0].endswith(".xml")

def fetch_xml(url: str) -> str:
    r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    return r.text

def parse_sitemap(xml_text: str) -> List[str]:
    # Returns all <loc> URLs from both sitemapindex and urlset XML structures.
    root = ET.fromstring(xml_text)
    urls = []
    for node in root.iter():
        if node.tag.endswith("loc") and node.text:
            urls.append(node.text.strip())
    return urls

def fetch_sitemap_urls() -> List[str]:
    print("Fetching sitemap...")
    xml = fetch_xml(SITEMAP_URL)

    first_level = parse_sitemap(xml)
    # Parsing to actial page URLs.
    urls: List[str] = []

    # if many URLs end with .xml, treat them as nested sitemaps
    xml_links = [u for u in first_level if is_xml_url(u)]
    if xml_links:
        # Optionally skip the users sitemap
        xml_links = [u for u in xml_links if "wp-sitemap-users" not in u]

        for sm in xml_links:
            print(f"Reading sitemap: {sm}")
            try:
                sm_xml = fetch_xml(sm)
                leaf_urls = parse_sitemap(sm_xml)
                # Keep only actual page URLs within the base domain.
                urls.extend([u for u in leaf_urls if u.startswith(BASE_URL) and not is_xml_url(u)])
            except Exception as e:
                print(f"Failed reading sitemap {sm}: {e}")
    else:
        # If the main sitemap was already a urlset (contains page URLs directly)
        urls = [u for u in first_level if u.startswith(BASE_URL) and not is_xml_url(u)]

    # dedupe
    urls = list(dict.fromkeys(urls))

    print(f"Found {len(urls)} page URLs from sitemap.")
    return urls

def render_page(browser, url: str) -> str | None:
    # Using playwright to render the page.
    # Return full HTML content.
    page = browser.new_page()
    try:
        page.goto(url, wait_until="domcontentloaded", timeout=30000)
        page.wait_for_timeout(300)
        return page.content()
    except Exception as e:
        print(f"Render failed for {url}: {e}")
        return None
    finally:
        page.close()

def html_to_text(html: str) -> str:
    # Extract visible text from HTML.
    soup = BeautifulSoup(html, "html.parser")

    # Remove unwanted elements.
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
        tag.decompose()

    text = " ".join(soup.get_text(" ").split())
    return text

def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> List[str]:
    # Splitting the long text into chunks.
    chunks = []
    i = 0
    step = max(1, chunk_size - overlap)
    while i < len(text):
        chunk = text[i:i + chunk_size]
        if chunk.strip():
            chunks.append(chunk)
        i += step
    return chunks

def embed_texts(texts: List[str]) -> List[List[float]]:
    # Generating embeddings for a batch of texts.
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return [d.embedding for d in response.data]

def index_url(browser, url: str):
    print(f"Indexing: {url}")

    html = render_page(browser, url)
    if not html:
        return
    
    text = html_to_text(html)

    # Skip small pages.
    if len(text) < 200:
        print("Skipped.")
        return
    
    chunks = chunk_text(text)

    ids = [f"{url}#chunk={i}" for i in range(len(chunks))]
    metadatas = [{"source": url, "chunk_index": i} for i in range(len(chunks))]

    # Batch embeddings to avoid large requests.
    BATCH_SIZE = 32

    for start in range(0, len(chunks), BATCH_SIZE):
        batch_texts = chunks[start:start + BATCH_SIZE]
        batch_ids = ids[start:start + BATCH_SIZE]
        batch_meta = metadatas[start:start + BATCH_SIZE]

        vectors = embed_texts(batch_texts)

        collection.upsert(
            ids=batch_ids,
            documents=batch_texts,
            embeddings=vectors,
            metadatas=batch_meta
        )

def main():
    # Main indexing loop.
    urls = fetch_sitemap_urls()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)

        try:
            for i, url in enumerate(urls, start=1):
                try:
                    index_url(browser, url)
                except Exception as e:
                    print(f"Failed indexing {url}: {e}")
                
                # Small delay for the health of the server.
                time.sleep(0.2)

                if i % 25 == 0:
                    print(f"Progress: {i}/{len(urls)}")
        finally:
            browser.close()

    print("Indexing completed.")
    print(f"Vector store location: {CHROMA_DIR}")

if __name__=="__main__":
    main()