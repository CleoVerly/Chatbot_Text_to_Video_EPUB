import streamlit as st
import os
import requests
import json
from neo4j import GraphDatabase
from dotenv import load_dotenv
import re

# ======================================================================================
# 1. KONFIGURASI HALAMAN DAN KONEKSI NEO4J
# ======================================================================================

st.set_page_config(page_title="Chatbot Knowledge Graph", layout="wide")

# Muat environment variables dari file .env
load_dotenv()

# Kredensial dari file .env
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Inisialisasi driver Neo4j.
@st.cache_resource
def get_neo4j_driver():
    """Membuat dan mengembalikan instance driver Neo4j."""
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
        driver.verify_connectivity()
        print("Koneksi ke Neo4j berhasil!")
        return driver
    except Exception as e:
        st.error(f"Gagal terhubung ke Neo4j: {e}")
        return None

driver = get_neo4j_driver()

# Judul dan subjudul aplikasi
st.title("🤖 Chatbot dengan Knowledge Graph (Neo4j)")
st.markdown("Ajukan pertanyaan tentang buku yang grafnya sudah Anda buat.")

if not driver:
    st.stop()

# ======================================================================================
# 2. FUNGSI-FUNGSI BARU YANG LEBIH CERDAS DAN ANDAL
# ======================================================================================

def generate_cypher_query(question, api_key):
    """
    Menggunakan LLM dengan prompt "few-shot" untuk mengubah pertanyaan pengguna 
    menjadi query Cypher yang andal, baik untuk pertanyaan spesifik maupun umum.
    """
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    
    # Prompt yang lebih cerdas dengan contoh untuk memandu LLM
    prompt = (
        "You are a Neo4j expert. Your task is to convert a user's question into an optimal Cypher query. "
        "The graph schema is: (:Book)-[:HAS_CHUNK]->(:Chunk)-[:MENTIONS]->(:Entity).\n\n"
        "**Instructions & Examples:**\n\n"
        "1.  **For specific questions about entities:** Identify the key entities and find chunks that mention them. Use case-insensitive regex for matching.\n"
        "    -   **User Question:** 'jelaskan mengenai pertempuran di uhud'\n"
        "    -   **Cypher Query:** `MATCH (c:Chunk)-[:MENTIONS]->(e:Entity) WHERE e.name =~ '(?i)pertempuran.*uhud' RETURN c.text AS context LIMIT 10`\n\n"
        "2.  **For questions with multiple entities:** Find chunks that mention ALL entities.\n"
        "    -   **User Question:** 'hadis tentang dedikasi pemuda dalam perang'\n"
        "    -   **Cypher Query:** `MATCH (c:Chunk) WHERE ((c)-[:MENTIONS]->(:Entity {name: 'Hadis'})) AND ((c)-[:MENTIONS]->(:Entity {name: 'Pemuda'})) AND ((c)-[:MENTIONS]->(:Entity {name: 'Perang'})) RETURN c.text AS context LIMIT 15`\n\n"
        "3.  **For general summary questions:** If the question is about the book's content in general (e.g., 'apa isi buku ini', 'ringkasan'), create a query to find the most central topics by counting entity mentions and retrieving chunks related to them.\n"
        "    -   **User Question:** 'apa isi buku ini?'\n"
        "    -   **Cypher Query:** `MATCH (e:Entity)<-[:MENTIONS]-(c:Chunk) WITH e, count(c) AS mentions ORDER BY mentions DESC LIMIT 7 MATCH (e)<-[:MENTIONS]-(chunk:Chunk) WITH e, collect(chunk)[..2] AS chunks UNWIND chunks AS c RETURN c.text AS context`\n\n"
        f"**User's question to convert:** '{question}'\n\n"
        "Provide ONLY the single best Cypher query based on the user's question."
    )
    
    data = {"model": "openai/gpt-4o-mini", "messages": [{"role": "user", "content": prompt}], "temperature": 0.0}
    
    try:
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data, timeout=45)
        response.raise_for_status()
        content = response.json()['choices'][0]['message']['content']
        # Membersihkan kode dari markdown
        if "```" in content:
            content = content.split("```")[1].replace("cypher", "").strip()
        return content
    except (requests.RequestException, KeyError, IndexError) as e:
        st.error(f"Gagal membuat query Cypher: {e}")
        return None

def get_context_from_graph(cypher_query):
    """Menjalankan query Cypher di Neo4j dan mengembalikan hasilnya."""
    if not cypher_query: return []
    with driver.session() as session:
        try:
            result = session.run(cypher_query)
            return [record["context"] for record in result]
        except Exception as e:
            st.error(f"Error menjalankan query Cypher: {e}")
            return []

def get_final_answer(question, context, api_key):
    """Merangkum konteks menjadi jawaban akhir."""
    headers = { "Authorization": f"Bearer {api_key}", "Content-Type": "application/json" }
    prompt = (
        "Anda adalah asisten AI ahli. Berdasarkan konteks dari buku berikut, jawab pertanyaan pengguna dalam Bahasa Indonesia.\n"
        "- Jika pertanyaan meminta ringkasan, rangkumlah topik utama dari konteks yang diberikan.\n"
        "- Jika pertanyaan spesifik, jawablah berdasarkan informasi yang ada dalam konteks.\n"
        "- Jika konteks tidak cukup untuk menjawab, katakan Anda tidak dapat menemukan informasi detail di dalam buku.\n\n"
        f"KONTEKS:\n{' '.join(context)}\n\nPERTANYAAN PENGGUNA:\n{question}\n\nJAWABAN:"
    )
    data = { "model": "openai/gpt-4o-mini", "messages": [{"role": "user", "content": prompt}], "temperature": 0.3 }
    try:
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data, timeout=90)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        return f"Gagal menghasilkan jawaban: {e}"

# ======================================================================================
# 3. ANTARMUKA PENGGUNA (UI) STREAMLIT
# ======================================================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "cypher" in message:
            with st.expander("Lihat Query Cypher yang Digunakan"):
                st.code(message["cypher"], language="cypher")

if prompt := st.chat_input("Tanyakan sesuatu tentang isi buku..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Menerjemahkan pertanyaan ke query graf, mencari di Neo4j, dan merangkum jawaban..."):
            # LANGKAH 1: Buat query yang andal menggunakan LLM yang dipandu
            cypher_query = generate_cypher_query(prompt, OPENROUTER_API_KEY)

            # LANGKAH 2: Dapatkan konteks dari graf
            if cypher_query:
                context = get_context_from_graph(cypher_query)
            else:
                context = []
            
            # LANGKAH 3: Hasilkan jawaban
            if not context:
                response_text = "Maaf, saya tidak dapat menemukan informasi yang relevan di dalam Knowledge Graph untuk pertanyaan tersebut."
            else:
                response_text = get_final_answer(prompt, context, OPENROUTER_API_KEY)

            st.markdown(response_text)
            
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response_text,
                "cypher": cypher_query if cypher_query else "Gagal membuat query."
            })
            st.rerun()
