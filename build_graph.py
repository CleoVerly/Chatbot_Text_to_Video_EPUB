import os
import json
import requests
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from neo4j import GraphDatabase
from dotenv import load_dotenv
import time

# ======================================================================================
# 1. KONFIGURASI DAN KONEKSI
# ======================================================================================
load_dotenv()
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

try:
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    driver.verify_connectivity()
    print("Koneksi ke Neo4j berhasil!")
except Exception as e:
    print(f"Gagal terhubung ke Neo4j: {e}")
    exit()

EPUB_FILE_PATH = r"epub_files\الأربعون الشبابية.epub" 

# ======================================================================================
# 2. EXTRACT - Mengekstrak Teks dari EPUB
# ======================================================================================
def extract_text_from_epub(epub_path):
    if not os.path.exists(epub_path):
        print(f"Error: File tidak ditemukan di {epub_path}")
        return None
    print(f"Mengekstrak teks dari {epub_path}...")
    book = epub.read_epub(epub_path)
    items = list(book.get_items_of_type(ebooklib.ITEM_DOCUMENT))
    full_text = ""
    for item in items:
        try:
            content = item.get_body_content()
            if content:
                soup = BeautifulSoup(content, 'html.parser')
                body = soup.find('body')
                if body:
                    text = body.get_text(separator=' ', strip=True)
                    full_text += text + " "
        except Exception as e:
            print(f"  - Peringatan: Melewati item yang rusak (ID: {item.id}). Error: {e}")
            continue
    print("Ekstraksi teks selesai.")
    return full_text

# ======================================================================================
# 3. TRANSFORM - Mengubah Teks menjadi Graf Terstruktur
# ======================================================================================
def extract_graph_from_chunk(chunk_text, api_key):
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    prompt = (
        "You are an expert data analyst. From the following text, extract all significant entities (people, concepts, books, places) "
        "and the relationships between them. Return the result in a strict JSON format. "
        "The JSON object must have two keys: 'entities' and 'relationships'. "
        "Entities should have 'name' and 'type'. Relationships should have 'source', 'target', and 'type'.\n\n"
        f"Text to analyze: '{chunk_text}'\n\nJSON:"
    )
    data = {
        "model": "openai/gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"},
        "temperature": 0.1,
    }
    try:
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data, timeout=60)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        print(f"  - Terjadi error: {e}")
    return None

# ======================================================================================
# 4. LOAD - Memuat Graf dan Membuat Indeks
# ======================================================================================
def load_graph_to_neo4j(session, graph_data, chunk_text, chunk_id, book_title):
    cypher_chunk = """
    MERGE (b:Book {title: $book_title})
    CREATE (c:Chunk {id: $chunk_id, text: $chunk_text})
    MERGE (b)-[:HAS_CHUNK]->(c)
    """
    session.run(cypher_chunk, book_title=book_title, chunk_id=chunk_id, chunk_text=chunk_text)
    entities = graph_data.get('entities', [])
    relationships = graph_data.get('relationships', [])
    for entity in entities:
        cypher_entity = """
        MATCH (c:Chunk {id: $chunk_id})
        MERGE (e:Entity {name: $name})
        ON CREATE SET e.type = $type
        MERGE (c)-[:MENTIONS]->(e)
        """
        session.run(cypher_entity, chunk_id=chunk_id, name=entity['name'], type=entity.get('type', 'Unknown'))
    for rel in relationships:
        cypher_rel = """
        MATCH (source:Entity {name: $source})
        MATCH (target:Entity {name: $target})
        MERGE (source)-[r:RELATIONSHIP {type: $type}]->(target)
        """
        session.run(cypher_rel, source=rel['source'], target=rel['target'], type=rel.get('type', 'RELATED_TO'))

def create_fulltext_index(session):
    """Membuat full-text search index pada node Chunk."""
    index_name = "chunkTextIndex"
    # Cek apakah indeks sudah ada
    result = session.run("SHOW INDEXES YIELD name WHERE name = $name", name=index_name)
    if result.peek() is None:
        print(f"Membuat full-text index '{index_name}'...")
        # Membuat indeks pada properti 'text' dari node 'Chunk'
        session.run("""
        CREATE FULLTEXT INDEX chunkTextIndex FOR (c:Chunk) ON EACH [c.text]
        """)
        print("Indeks berhasil dibuat.")
    else:
        print(f"Full-text index '{index_name}' sudah ada.")

# ======================================================================================
# 5. FUNGSI UTAMA (MAIN EXECUTION)
# ======================================================================================
def main():
    full_text = extract_text_from_epub(EPUB_FILE_PATH)
    if not full_text: return

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    chunks = text_splitter.split_text(full_text)
    print(f"Teks berhasil dipecah menjadi {len(chunks)} potongan (chunks).")

    book_title = os.path.basename(EPUB_FILE_PATH)

    # PERBAIKAN: Menggunakan query yang lebih efisien untuk menghapus data lama
    with driver.session() as session:
        print("Menghapus data lama dari buku ini...")
        # Langkah 1: Hapus semua node Chunk yang terhubung dengan buku ini.
        # Ini akan secara otomatis menghapus semua hubungan yang melekat padanya.
        session.run("""
            MATCH (b:Book {title: $title})-[:HAS_CHUNK]->(c:Chunk)
            DETACH DELETE c
        """, title=book_title)
        
        # Langkah 2: Hapus node Buku itu sendiri setelah semua chunk-nya hilang.
        session.run("""
            MATCH (b:Book {title: $title})
            DETACH DELETE b
        """, title=book_title)
        print("Data lama berhasil dihapus.")

    for i, chunk in enumerate(chunks):
        print(f"\n--- Memproses Chunk {i+1}/{len(chunks)} ---")
        graph_json_str = extract_graph_from_chunk(chunk, OPENROUTER_API_KEY)
        if graph_json_str:
            try:
                graph_data = json.loads(graph_json_str)
                print(f"  - Berhasil mengekstrak {len(graph_data.get('entities',[]))} entitas dan {len(graph_data.get('relationships',[]))} hubungan.")
                with driver.session() as session:
                    load_graph_to_neo4j(session, graph_data, chunk, f"{book_title}_chunk_{i}", book_title)
                print("  - Berhasil memuat graf ke Neo4j.")
            except json.JSONDecodeError:
                print("  - Gagal mem-parse JSON dari LLM.")
        else:
            print("  - Gagal mengekstrak graf dari chunk ini.")
        time.sleep(1) 

    # Membuat full-text index setelah semua data dimuat
    with driver.session() as session:
        create_fulltext_index(session)

    print("\n========================================")
    print("Proses pembuatan Knowledge Graph selesai!")
    print("========================================")
    driver.close()

if __name__ == "__main__":
    main()
