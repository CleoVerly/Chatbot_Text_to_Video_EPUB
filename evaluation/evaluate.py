import os
import faiss
import numpy as np
import requests
import json
import time
import re

# Impor baru untuk metrik F1-Score dan Cosine Similarity
from sklearn.metrics import f1_score
from scipy.spatial.distance import cosine

from sentence_transformers import SentenceTransformer
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

# ======================================================================================
# KONFIGURASI DAN FUNGSI INTI
# ======================================================================================
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

print("Memuat model embedding (hanya sekali)...")
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
print("Model berhasil dimuat.")

def extract_text_from_epub(epub_path):
    """Mengekstrak teks bersih dari file EPUB dengan penanganan error yang lebih baik."""
    try:
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
            except Exception:
                continue
        return full_text
    except Exception as e:
        print(f"  - Peringatan: Gagal membaca {os.path.basename(epub_path)} karena error: {e}")
        return None

def process_epub(epub_path, model):
    """Memproses file EPUB menjadi indeks FAISS dan potongan teks."""
    text = extract_text_from_epub(epub_path)
    if not text: return None, None
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    if not chunks: return None, None
    embeddings = model.encode(chunks, show_progress_bar=False).astype('float32')
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index, chunks

def is_summary_question(question):
    summary_keywords = ["apa isi buku", "tentang apa", "ringkasan", "rangkuman", "summarize", "buku ini membahas"]
    return any(keyword in question.lower() for keyword in summary_keywords)

def find_relevant_chunks(query, index, chunks, model, top_k=20):
    query_embedding = model.encode([query]).astype('float32')
    _, indices = index.search(query_embedding, top_k)
    return [chunks[i] for i in indices[0]]

def rerank_chunks_with_llm(query, chunks_to_rerank, api_key):
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    chunk_list_str = "".join(f"CHUNK {i+1}:\n\"{chunk}\"\n\n" for i, chunk in enumerate(chunks_to_rerank))
    prompt = (
        "You are a relevance ranking assistant. From the list of chunks below, identify the numbers of the TOP 7 most relevant chunks for answering the user's question. "
        "Return your answer ONLY as a JSON list of numbers. Example: [1, 5, 3, 10, 2, 8, 12]\n\n"
        f"USER'S QUESTION:\n\"{query}\"\n\nLIST OF CHUNKS:\n{chunk_list_str}\n\nJSON list of top 7 chunk numbers:"
    )
    data = {"model": "mistralai/mistral-7b-instruct:free", "messages": [{"role": "user", "content": prompt}], "temperature": 0.0}
    try:
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data, timeout=45)
        response.raise_for_status()
        content = response.json()['choices'][0]['message']['content']
        ranked_indices = json.loads(content.strip())
        return [chunks_to_rerank[i-1] for i in ranked_indices if 0 < i <= len(chunks_to_rerank)]
    except Exception as e:
        print(f"  - Peringatan: Re-ranking gagal ({e}). Menggunakan 7 kandidat teratas.")
        return chunks_to_rerank[:7]

def get_llm_response(query, context, api_key, is_summary=False):
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    instruction = "Based on the following sample of text chunks from a book, provide a general summary of the main topics discussed." if is_summary else "Based on the context, answer the user's original question."
    prompt = (
        f"You are an expert AI assistant. {instruction} "
        f"Answer in Bahasa Indonesia. If the info isn't in the context, say you cannot find the answer in the book.\n\n"
        f"CONTEXT:\n{' '.join(context)}\n\nUSER'S QUESTION:\n{query}\n\nANSWER:"
    )
    data = {"model": "openai/gpt-4o-mini", "messages": [{"role": "user", "content": prompt}], "temperature": 0.3}
    try:
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data, timeout=90)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except Exception:
        return "Gagal menghasilkan jawaban."

# ======================================================================================
# FUNGSI EVALUASI OTOMATIS (DENGAN METRIK BARU)
# ======================================================================================

def evaluate_retrieval(retrieved_context, ideal_context_summary, api_key):
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    prompt = (
        "You are a relevance evaluator. Does the 'RETRIEVED CONTEXT' contain the information described in the 'IDEAL CONTEXT SUMMARY'? "
        "Answer with a single word: 'Yes' or 'No'.\n\n"
        f"IDEAL CONTEXT SUMMARY:\n{ideal_context_summary}\n\n"
        f"RETRIEVED CONTEXT:\n{' '.join(retrieved_context)}\n\n"
        "Answer (Yes/No):"
    )
    data = {"model": "openai/gpt-4o-mini", "messages": [{"role": "user", "content": prompt}], "temperature": 0.0}
    try:
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data, timeout=45)
        response.raise_for_status()
        answer = response.json()['choices'][0]['message']['content'].strip().lower()
        return "yes" in answer
    except Exception:
        return False

def evaluate_faithfulness(generated_answer, ground_truth_answer, api_key):
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    prompt = (
        "You are a factual consistency evaluator. Does the 'GENERATED ANSWER' accurately reflect the information in the 'GROUND TRUTH ANSWER'? "
        "Ignore minor differences in wording, focus on factual correctness. Answer with a single word: 'Yes' or 'No'.\n\n"
        f"GROUND TRUTH ANSWER:\n{ground_truth_answer}\n\n"
        f"GENERATED ANSWER:\n{generated_answer}\n\n"
        "Answer (Yes/No):"
    )
    data = {"model": "openai/gpt-4o-mini", "messages": [{"role": "user", "content": prompt}], "temperature": 0.0}
    try:
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data, timeout=45)
        response.raise_for_status()
        answer = response.json()['choices'][0]['message']['content'].strip().lower()
        return "yes" in answer
    except Exception:
        return False

# --- FUNGSI METRIK BARU ---
def calculate_f1_score(generated, ground_truth):
    """Menghitung F1-Score berdasarkan tumpang tindih kata."""
    gen_tokens = set(re.split(r'\W+', generated.lower()))
    gt_tokens = set(re.split(r'\W+', ground_truth.lower()))
    
    common_tokens = gen_tokens.intersection(gt_tokens)
    
    if not common_tokens:
        return 0.0
        
    precision = len(common_tokens) / len(gen_tokens)
    recall = len(common_tokens) / len(gt_tokens)
    
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

def calculate_semantic_similarity(generated, ground_truth, model):
    """Menghitung kemiripan makna menggunakan embedding."""
    try:
        gen_embedding = model.encode(generated)
        gt_embedding = model.encode(ground_truth)
        
        # Cosine similarity dihitung sebagai 1 - cosine distance
        similarity = 1 - cosine(gen_embedding, gt_embedding)
        return similarity
    except Exception:
        return 0.0

# ======================================================================================
# FUNGSI UTAMA (MAIN EXECUTION)
# ======================================================================================

def main():
    print("Memulai proses evaluasi otomatis...")
    
    try:
        with open("evaluation\golden_dataset.json", "r", encoding="utf-8") as f:
            dataset = json.load(f)
    except FileNotFoundError:
        print("Error: File 'golden_dataset.json' tidak ditemukan.")
        return

    # Inisialisasi skor
    retrieval_hits = 0
    faithfulness_hits = 0
    f1_scores = []
    semantic_scores = []
    total_questions = len(dataset)
    
    epub_files = {item.get("epub_file") for item in dataset if item.get("epub_file")}
    processed_data = {}
    for epub_file in epub_files:
        path = os.path.join("epub_files", epub_file)
        print(f"\nMemproses file: {path}...")
        index, chunks = process_epub(path, embedding_model)
        if index and chunks:
            processed_data[epub_file] = (index, chunks)
            print("Berhasil diproses.")
        else:
            print(f"Gagal memproses file: {path}")

    for i, item in enumerate(dataset):
        print(f"\n--- Menguji Pertanyaan {i+1}/{total_questions}: '{item['question']}' ---")
        
        epub_file = item.get("epub_file")
        if not epub_file or epub_file not in processed_data:
            print("  - Melewati pertanyaan karena file EPUB tidak ada atau gagal diproses.")
            continue
            
        index, chunks = processed_data[epub_file]
        
        summary_flag = is_summary_question(item['question'])
        if summary_flag:
            retrieved_context = chunks[:15]
        else:
            initial_chunks = find_relevant_chunks(item['question'], index, chunks, embedding_model)
            retrieved_context = rerank_chunks_with_llm(item['question'], initial_chunks, OPENROUTER_API_KEY)
        
        generated_answer = get_llm_response(item['question'], retrieved_context, OPENROUTER_API_KEY, is_summary=summary_flag)
        
        # Evaluasi dengan semua metrik
        is_retrieval_hit = evaluate_retrieval(retrieved_context, item['ideal_context_summary'], OPENROUTER_API_KEY)
        is_faithful_hit = evaluate_faithfulness(generated_answer, item['ground_truth_answer'], OPENROUTER_API_KEY)
        f1 = calculate_f1_score(generated_answer, item['ground_truth_answer'])
        semantic_sim = calculate_semantic_similarity(generated_answer, item['ground_truth_answer'], embedding_model)
        
        if is_retrieval_hit:
            retrieval_hits += 1
            print("  - Retrieval: TEPAT")
        else:
            print("  - Retrieval: KURANG TEPAT")
            
        if is_faithful_hit:
            faithfulness_hits += 1
            print("  - Faithfulness: SESUAI FAKTA")
        else:
            print("  - Faithfulness: TIDAK SESUAI FAKTA")
        
        f1_scores.append(f1)
        semantic_scores.append(semantic_sim)
        print(f"  - F1-Score: {f1:.2f}")
        print(f"  - Semantic Similarity: {semantic_sim:.2f}")
        
        time.sleep(1)

    print("\n========================================")
    print("         LAPORAN EVALUASI AKHIR         ")
    print("========================================")
    retrieval_accuracy = (retrieval_hits / total_questions) * 100 if total_questions > 0 else 0
    faithfulness_accuracy = (faithfulness_hits / total_questions) * 100 if total_questions > 0 else 0
    avg_f1 = np.mean(f1_scores) * 100 if f1_scores else 0
    avg_semantic = np.mean(semantic_scores) * 100 if semantic_scores else 0
    
    print(f"Total Pertanyaan Diuji: {total_questions}")
    print(f"Akurasi Pengambilan (Retrieval Hit Rate): {retrieval_accuracy:.2f}%")
    print(f"Akurasi Jawaban (Faithfulness Score): {faithfulness_accuracy:.2f}%")
    print(f"Rata-rata F1-Score: {avg_f1:.2f}%")
    print(f"Rata-rata Kemiripan Semantik: {avg_semantic:.2f}%")
    print("========================================")

if __name__ == "__main__":
    main()
